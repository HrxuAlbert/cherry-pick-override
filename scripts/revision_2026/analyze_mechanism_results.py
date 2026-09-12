#!/usr/bin/env python3
"""Analyze frozen prompt, evidence-order, and reasoning perturbation runs."""

from __future__ import annotations

def _cpo_workspace(_f=__file__):
    """Resolve the release root.

    In the author's tree these scripts live at
    <root>/Writing/V0.2/revision_plan/scripts/, so parents[4] is the root. In
    this release they live at <repo>/scripts/revision_2026/, so walk up until a
    directory containing outputs/revision_2026 is found and fall back to the
    original rule.
    """
    import os, pathlib as _p
    env = os.environ.get("CPO_WORKSPACE")
    if env:
        return _p.Path(env).resolve()
    here = _p.Path(_f).resolve()
    for parent in here.parents:
        if (parent / "outputs" / "revision_2026").is_dir():
            return parent
    return here.parents[4]

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


WORKSPACE = _cpo_workspace()
DIRECTIONAL = {"support", "refute"}
VALID = {*DIRECTIONAL, "insufficient", "conflicting"}
PARSE_FAILURE = "parse_failure"


class MechanismAnalysisError(RuntimeError):
    """Raised when mechanism outputs do not form a complete paired design."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-index", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/mechanisms/analysis",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def normalize(row: dict[str, Any]) -> str:
    if row.get("status") != "ok" or not row.get("parse_ok"):
        return PARSE_FAILURE
    label = str((row.get("parsed") or {}).get("verdict_normal") or "").lower()
    return label if label in VALID else PARSE_FAILURE


def load_variant_rows(manifest_path: Path) -> tuple[str, dict[tuple[str, str], str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest["run_id"])
    raw_path = workspace_path(manifest["output_root"]) / run_id / "raw_results.jsonl"
    if not raw_path.is_file():
        raise MechanismAnalysisError(f"Missing result file for {run_id}: {raw_path}")
    output: dict[tuple[str, str], str] = {}
    with raw_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MechanismAnalysisError(f"Invalid JSON at {raw_path}:{line_number}") from exc
            key = (str(row.get("model_slot")), str(row.get("case_id")))
            if key in output:
                raise MechanismAnalysisError(f"Duplicate completed pair in {raw_path}: {key}")
            output[key] = normalize(row)
    return run_id, output


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(quantile * (len(ordered) - 1))))
    return ordered[index]


def compare(
    baseline: dict[tuple[str, str], str],
    variant: dict[tuple[str, str], str],
    variant_name: str,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if set(baseline) != set(variant):
        missing = sorted(set(baseline) - set(variant))[:5]
        extra = sorted(set(variant) - set(baseline))[:5]
        raise MechanismAnalysisError(
            f"Variant {variant_name} does not match baseline; missing={missing}, extra={extra}"
        )
    by_model: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for (model, case_id), baseline_label in baseline.items():
        by_model[model].append((case_id, baseline_label, variant[(model, case_id)]))
    output: list[dict[str, Any]] = []
    for model, triples in sorted(by_model.items()):
        triples.sort()
        n = len(triples)
        base_directional = [int(base in DIRECTIONAL) for _, base, _ in triples]
        variant_directional = [int(value in DIRECTIONAL) for _, _, value in triples]
        rng = random.Random(seed)
        deltas: list[float] = []
        for _ in range(resamples):
            sample = [rng.randrange(n) for _ in range(n)]
            deltas.append(
                sum(variant_directional[i] - base_directional[i] for i in sample) / n
            )
        output.append(
            {
                "variant": variant_name,
                "model_slot": model,
                "n_cases": n,
                "baseline_directional_rate": sum(base_directional) / n,
                "variant_directional_rate": sum(variant_directional) / n,
                "directional_delta": (sum(variant_directional) - sum(base_directional)) / n,
                "directional_delta_ci_low": percentile(deltas, 0.025),
                "directional_delta_ci_high": percentile(deltas, 0.975),
                "any_label_flip_rate": sum(base != value for _, base, value in triples) / n,
                "support_refute_flip_rate": sum(
                    {base, value} == {"support", "refute"} for _, base, value in triples
                )
                / n,
                "recovery_to_conflicting_rate": sum(
                    base in DIRECTIONAL and value == "conflicting"
                    for _, base, value in triples
                )
                / n,
                "baseline_parse_failure_rate": sum(
                    base == PARSE_FAILURE for _, base, _ in triples
                )
                / n,
                "variant_parse_failure_rate": sum(
                    value == PARSE_FAILURE for _, _, value in triples
                )
                / n,
                "bootstrap_resamples": resamples,
                "seed": seed,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise MechanismAnalysisError(f"Output directory is nonempty; use --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.manifest_index.resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    baseline_name = str(index["baseline_variant"])
    loaded: dict[str, dict[tuple[str, str], str]] = {}
    for variant, metadata in index["variants"].items():
        _, loaded[variant] = load_variant_rows(
            workspace_path(metadata["manifest_path"])
        )
    baseline = loaded[baseline_name]
    rows: list[dict[str, Any]] = []
    for variant in sorted(set(loaded) - {baseline_name}):
        rows.extend(
            compare(
                baseline,
                loaded[variant],
                variant,
                args.bootstrap_resamples,
                args.seed,
            )
        )
    if not rows:
        raise MechanismAnalysisError("No mechanism contrasts were available.")
    write_csv(output_dir / "mechanism_metrics.csv", rows)
    lines = [
        "# Prompt, evidence-order, and reasoning sensitivity\n\n",
        "All comparisons use the same frozen 60-case strict-consensus set and reuse the primary/original/direct run as the baseline.\n\n",
        "| Variant | Model | Directional baseline | Directional variant | Delta | 95% paired bootstrap | Any flip | S/R flip | Recovery to C |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | `{row['model_slot']}` | "
            f"{row['baseline_directional_rate']:.3f} | {row['variant_directional_rate']:.3f} | "
            f"{row['directional_delta']:+.3f} | "
            f"[{row['directional_delta_ci_low']:+.3f}, {row['directional_delta_ci_high']:+.3f}] | "
            f"{row['any_label_flip_rate']:.3f} | {row['support_refute_flip_rate']:.3f} | "
            f"{row['recovery_to_conflicting_rate']:.3f} |\n"
        )
    (output_dir / "summary.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "contrast_rows": len(rows), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
