#!/usr/bin/env python3
"""Compute dataset-defined current-model metrics before human-audit joining.

This is the immediate post-run analysis for the four-model AVeriTeC P0
replication. It requires a complete result for every model/case pair, retains
parse failures in fixed denominators, and does not access human annotations.
"""

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
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_cached_model_landscape import (  # noqa: E402
    LABELS,
    PARSE_FAILURE,
    compute_metrics,
    paired_bootstrap,
)


WORKSPACE = _cpo_workspace()
GOLD_MAP = {
    "Supported": "support",
    "Refuted": "refute",
    "Not Enough Evidence": "insufficient",
    "Conflicting Evidence/Cherrypicking": "conflicting",
}


class CurrentAnalysisError(RuntimeError):
    """Raised when a current-model result set is incomplete or inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE / "Writing/V0.2/revision_plan/current_models_manifest.json",
    )
    parser.add_argument(
        "--model-results",
        type=Path,
        default=WORKSPACE
        / "outputs/revision_2026/current_models_averitec/runs/averitec-p0-current-models-20260818-v1/raw_results_full.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/current_models_averitec/analysis_dataset_defined",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def normalize_result(row: dict[str, Any]) -> str:
    if row.get("status") != "ok" or not row.get("parse_ok"):
        return PARSE_FAILURE
    label = str((row.get("parsed") or {}).get("verdict_normal") or "").lower()
    return label if label in LABELS else PARSE_FAILURE


def load_results(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        raise CurrentAnalysisError(f"Current-model results do not exist: {path}")
    successes: dict[tuple[str, int], dict[str, Any]] = {}
    failures: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CurrentAnalysisError(f"Invalid JSON at {path}:{line_number}") from exc
            if "model_slot" not in row or "case_id" not in row:
                raise CurrentAnalysisError(f"Missing model_slot/case_id at {path}:{line_number}")
            key = (str(row["model_slot"]), int(row["case_id"]))
            if row.get("status") == "ok":
                if key in successes:
                    raise CurrentAnalysisError(f"Duplicate successful output: {key}")
                successes[key] = row
            else:
                failures[key] = row
    return {**failures, **successes}


def source_cases(manifest: dict[str, Any]) -> dict[int, str]:
    dataset = manifest["dataset"]
    if dataset.get("format") == "materialized_json":
        sample_path = workspace_path(dataset["materialized_path"])
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        cases = {int(row["case_id"]): str(row["gold_normal"]) for row in sample}
        expected = int(dataset["expected_total"])
        if len(cases) != expected or any(label not in set(GOLD_MAP.values()) for label in cases.values()):
            raise CurrentAnalysisError("Invalid materialized source cases.")
        return cases
    sample_path = workspace_path(dataset["sample_map_path"])
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    cases: dict[int, str] = {}
    for row in sample:
        case_id = int(row["case_id"])
        if case_id in cases:
            raise CurrentAnalysisError(f"Duplicate source case ID: {case_id}")
        try:
            cases[case_id] = GOLD_MAP[row["label"]]
        except KeyError as exc:
            raise CurrentAnalysisError(f"Unknown source label: {row.get('label')}") from exc
    expected = int(dataset["expected_total"])
    if len(cases) != expected:
        raise CurrentAnalysisError(f"Expected {expected} source cases; found {len(cases)}")
    return cases


def materialize(
    manifest: dict[str, Any],
    cases: dict[int, str],
    indexed_results: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    slots = [str(model["slot"]) for model in manifest["models"] if model.get("enabled")]
    expected_keys = {(slot, case_id) for slot in slots for case_id in cases}
    missing = sorted(expected_keys - set(indexed_results))
    unexpected = sorted(set(indexed_results) - expected_keys)
    if missing or unexpected:
        raise CurrentAnalysisError(
            f"Result set is not the complete frozen matrix: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    output: dict[str, list[dict[str, Any]]] = {}
    dataset_name = str(manifest["dataset"].get("name", "dataset"))
    for slot in slots:
        output[slot] = [
            {
                "dataset": dataset_name,
                "case_id": str(case_id),
                "gold": gold,
                "pred": normalize_result(indexed_results[(slot, case_id)]),
            }
            for case_id, gold in sorted(cases.items())
        ]
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
        raise CurrentAnalysisError(f"Output directory is nonempty; use --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.resolve()
    results_path = args.model_results.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = materialize(manifest, source_cases(manifest), load_results(results_path))

    dataset_name = str(manifest["dataset"].get("name", "dataset"))
    metrics = [
        compute_metrics(dataset_name, slot, rows)
        for slot, rows in predictions.items()
    ]
    confusion: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    for slot, rows in predictions.items():
        normalized.extend({"model_slot": slot, **row} for row in rows)
        counts = Counter((row["gold"], row["pred"]) for row in rows)
        for gold in LABELS:
            for pred in (*LABELS, PARSE_FAILURE):
                confusion.append(
                    {
                        "model_slot": slot,
                        "gold": gold,
                        "prediction": pred,
                        "count": counts[(gold, pred)],
                    }
                )

    contrasts: list[dict[str, Any]] = []
    for slot_a, slot_b in combinations(predictions, 2):
        for metric, subset, outcome in (
            (
                "dataset_defined_cco",
                lambda row: row["gold"] == "conflicting",
                lambda row: int(row["pred"] in {"support", "refute"}),
            ),
            (
                "pure_sr_accuracy",
                lambda row: row["gold"] in {"support", "refute"},
                lambda row: int(row["pred"] == row["gold"]),
            ),
        ):
            n, rate_a, rate_b, delta, low, high = paired_bootstrap(
                predictions[slot_a],
                predictions[slot_b],
                subset,
                outcome,
                args.bootstrap_resamples,
                args.seed,
            )
            contrasts.append(
                {
                    "metric": metric,
                    "model_a": slot_a,
                    "model_b": slot_b,
                    "n_cases": n,
                    "rate_a": rate_a,
                    "rate_b": rate_b,
                    "delta_b_minus_a": delta,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    write_csv(output_dir / "model_metrics.csv", metrics)
    write_csv(output_dir / "confusion_matrices.csv", confusion)
    write_csv(output_dir / "normalized_predictions.csv", normalized)
    write_csv(output_dir / "paired_model_contrasts.csv", contrasts)
    lines = [
        f"# Current-model {dataset_name} results (dataset-defined)\n\n",
        "> This report is audit-independent. Human-confirmed CCO is computed separately after the blinded labels are frozen.\n\n",
        "| Model slot | Dataset CCO | 95% Wilson | Conflict recall | Pure S/R accuracy | False conflict | Insufficient directional | Parse failure |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in metrics:
        lines.append(
            f"| `{row['system']}` | {row['cco_n']}/{row['n_conflicting']} ({row['cco_rate']:.3f}) | "
            f"[{row['cco_wilson_low']:.3f}, {row['cco_wilson_high']:.3f}] | "
            f"{row['conflict_recall']:.3f} | {row['pure_sr_accuracy']:.3f} | "
            f"{row['false_conflict_rate_on_sr']:.3f} | {row['insufficient_directional_rate']:.3f} | "
            f"{row['parse_failure_rate']:.3f} |\n"
        )
    (output_dir / "summary.md").write_text("".join(lines), encoding="utf-8")
    (output_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "human_audit_accessed": False,
                "label_basis": "dataset_defined",
                "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
                "model_results": {"path": str(results_path), "sha256": sha256_file(results_path)},
                "bootstrap_resamples": args.bootstrap_resamples,
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "models": len(metrics), "cases_per_model": len(next(iter(predictions.values()))), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
