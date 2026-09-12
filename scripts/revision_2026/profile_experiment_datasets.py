#!/usr/bin/env python3
"""Create reproducible sample/prompt statistics for both experiment datasets."""

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
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE = _cpo_workspace()
RUNNER_PATH = Path(__file__).resolve().parent / "run_current_models.py"
SPEC = importlib.util.spec_from_file_location("current_runner_for_profile", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class ProfileError(RuntimeError):
    """Raised when a frozen dataset profile cannot be generated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests",
        nargs="+",
        type=Path,
        default=[
            WORKSPACE / "Writing/V0.2/revision_plan/current_models_manifest.json",
            WORKSPACE / "Writing/V0.2/revision_plan/current_models_vitaminc_manifest.json",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/audit_independent/dataset_profiles",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: list[int], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_mean": sum(values) / len(values),
        f"{prefix}_median": percentile(values, 0.5),
        f"{prefix}_p25": percentile(values, 0.25),
        f"{prefix}_p75": percentile(values, 0.75),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise ProfileError(f"Output directory is nonempty; use --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for manifest_path_raw in args.manifests:
        manifest_path = manifest_path_raw.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases, metadata = RUNNER.load_cases(manifest, None)
        prompt_path = RUNNER.resolve_workspace_path(manifest["prompt"]["template_path"])
        template = prompt_path.read_text(encoding="utf-8")
        dataset = str(manifest["dataset"]["name"])
        for case in cases:
            rendered = RUNNER.render_prompt(template, case["claim"], case["evidence"])
            case_rows.append(
                {
                    "dataset": dataset,
                    "case_id": case["case_id"],
                    "label": case["gold_normal"],
                    "split": case.get("split", ""),
                    "sample_group": case.get("sample_group", ""),
                    "claim_characters": len(case["claim"]),
                    "claim_whitespace_tokens": len(case["claim"].split()),
                    "evidence_characters": len(case["evidence"]),
                    "evidence_whitespace_tokens": len(case["evidence"].split()),
                    "evidence_units": int(case.get("evidence_unit_count", 1)),
                    "rendered_prompt_characters": len(rendered),
                }
            )
        input_rows.append(
            {
                "dataset": dataset,
                "manifest_path": str(manifest_path),
                "manifest_sha256": RUNNER.sha256_file(manifest_path),
                "case_count": metadata["case_count"],
                "evidence_order": metadata["evidence_order"],
            }
        )

    profiles: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in case_rows}):
        dataset_rows = [row for row in case_rows if row["dataset"] == dataset]
        for label in ("all", "support", "refute", "insufficient", "conflicting"):
            selected = dataset_rows if label == "all" else [
                row for row in dataset_rows if row["label"] == label
            ]
            profiles.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "n": len(selected),
                    **describe([row["claim_characters"] for row in selected], "claim_chars"),
                    **describe([row["evidence_characters"] for row in selected], "evidence_chars"),
                    **describe([row["evidence_units"] for row in selected], "evidence_units"),
                    **describe([row["rendered_prompt_characters"] for row in selected], "prompt_chars"),
                }
            )
    group_counts = [
        {"dataset": dataset, "sample_group": group, "n": count}
        for (dataset, group), count in sorted(
            Counter((row["dataset"], row["sample_group"]) for row in case_rows).items()
        )
    ]
    write_csv(output_dir / "case_profiles.csv", case_rows)
    write_csv(output_dir / "dataset_profiles.csv", profiles)
    write_csv(output_dir / "sample_group_counts.csv", group_counts)
    (output_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "human_audit_accessed": False,
                "inputs": input_rows,
                "length_units": {
                    "characters": "Python Unicode code points",
                    "whitespace_tokens": "split on whitespace; descriptive only",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Frozen experiment-dataset profiles\n\n",
        "These statistics are generated from the exact materialized inputs used by the current-model runner and do not use human annotations.\n\n",
        "| Dataset | Label | n | Claim chars (median [IQR]) | Evidence chars (median [IQR]) | Evidence units (median [IQR]) | Prompt chars (median [IQR]) |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in profiles:
        lines.append(
            f"| {row['dataset']} | {row['label']} | {row['n']} | "
            f"{row['claim_chars_median']:.0f} [{row['claim_chars_p25']:.0f}, {row['claim_chars_p75']:.0f}] | "
            f"{row['evidence_chars_median']:.0f} [{row['evidence_chars_p25']:.0f}, {row['evidence_chars_p75']:.0f}] | "
            f"{row['evidence_units_median']:.0f} [{row['evidence_units_p25']:.0f}, {row['evidence_units_p75']:.0f}] | "
            f"{row['prompt_chars_median']:.0f} [{row['prompt_chars_p25']:.0f}, {row['prompt_chars_p75']:.0f}] |\n"
        )
    (output_dir / "summary.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "datasets": len(args.manifests), "cases": len(case_rows), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
