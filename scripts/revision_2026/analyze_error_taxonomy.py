#!/usr/bin/env python3
"""Describe dataset-defined CCO errors by structural-validator category."""

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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE = _cpo_workspace()
DIRECTIONAL = {"support", "refute"}


class TaxonomyError(RuntimeError):
    """Raised when cached predictions and structural features do not align."""


def parse_args() -> argparse.Namespace:
    root = WORKSPACE / "Writing/V0.2/code_release/outputs/option_a_exp/strengthening"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--averitec-predictions",
        type=Path,
        default=root / "e1_full_4label_utility/raw_results.jsonl",
    )
    parser.add_argument(
        "--averitec-validator",
        type=Path,
        default=root / "e3_structured_certificate_validator_fewshot/raw_results.jsonl",
    )
    parser.add_argument(
        "--vitaminc-predictions",
        type=Path,
        default=root / "e4_vitaminc_mixed/raw_results.jsonl",
    )
    parser.add_argument(
        "--vitaminc-validator",
        type=Path,
        default=root / "e3_validator_on_e4_vitaminc_mixed/raw_results.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/audit_independent/error_taxonomy",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TaxonomyError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def structural_category(validity: dict[str, Any]) -> str:
    support = bool(validity.get("has_material_support", False))
    refute = bool(validity.get("has_material_refute", False))
    mixed = bool(validity.get("has_material_mixed", False))
    insufficient = bool(validity.get("has_material_insufficient", False))
    if mixed:
        return "material_mixed"
    if support and refute:
        return "separate_support_and_refute"
    if insufficient and (support or refute):
        return "directional_plus_insufficient"
    if insufficient:
        return "insufficient_only"
    if support:
        return "support_only"
    if refute:
        return "refute_only"
    return "no_material_signal"


def joined_rows(
    dataset: str,
    predictions: list[dict[str, Any]],
    validators: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validator_map = {str(row["case_id"]): row.get("validity") or {} for row in validators}
    panels = [
        row for row in predictions if row.get("system") == "panel_3judge_4opt_strong"
    ]
    if len(validator_map) != len(panels):
        raise TaxonomyError(
            f"{dataset} validator/panel counts differ: {len(validator_map)} vs {len(panels)}"
        )
    output: list[dict[str, Any]] = []
    for row in panels:
        case_id = str(row["case_id"])
        if row.get("gold_normal") != "conflicting":
            continue
        validity = validator_map.get(case_id)
        if validity is None:
            raise TaxonomyError(f"{dataset} missing validator case {case_id}")
        category = structural_category(validity)
        for judge in row.get("judge_outputs") or []:
            parsed = judge.get("parsed") or {}
            prediction = str(parsed.get("verdict_normal") or "parse_failure")
            confidence = parsed.get("confidence")
            output.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "model": str(judge.get("model") or "unknown"),
                    "structural_category": category,
                    "prediction": prediction,
                    "is_dataset_cco": int(prediction in DIRECTIONAL),
                    "confidence": float(confidence) if confidence is not None else None,
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
        raise TaxonomyError(f"Output directory is nonempty; use --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = joined_rows(
        "AVeriTeC",
        read_jsonl(args.averitec_predictions.resolve()),
        read_jsonl(args.averitec_validator.resolve()),
    ) + joined_rows(
        "VitaminC-Mixed",
        read_jsonl(args.vitaminc_predictions.resolve()),
        read_jsonl(args.vitaminc_validator.resolve()),
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["structural_category"])].append(row)
    metrics: list[dict[str, Any]] = []
    for (dataset, model, category), group in sorted(grouped.items()):
        cco = [row for row in group if row["is_dataset_cco"]]
        confidences = [row["confidence"] for row in cco if row["confidence"] is not None]
        directions = Counter(row["prediction"] for row in cco)
        metrics.append(
            {
                "dataset": dataset,
                "model": model,
                "structural_category": category,
                "n_conflict_cases": len(group),
                "dataset_cco_n": len(cco),
                "dataset_cco_rate": len(cco) / len(group),
                "cco_support_n": directions["support"],
                "cco_refute_n": directions["refute"],
                "cco_mean_confidence": sum(confidences) / len(confidences) if confidences else "",
                "cco_high_confidence_ge_080_n": sum(value >= 0.8 for value in confidences),
            }
        )
    write_csv(output_dir / "case_model_taxonomy.csv", rows)
    write_csv(output_dir / "taxonomy_metrics.csv", metrics)
    lines = [
        "# Dataset-defined CCO error taxonomy\n\n",
        "> Categories are assigned by the cached structural validator and the target uses dataset conflict labels. Human-confirmed error taxonomy remains pending.\n\n",
        "| Dataset | Model | Structural category | Conflict cases | Dataset CCO | CCO rate | Support / Refute | Mean CCO confidence |\n",
        "|---|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in metrics:
        mean_conf = row["cco_mean_confidence"]
        mean_text = f"{mean_conf:.3f}" if isinstance(mean_conf, float) else "—"
        lines.append(
            f"| {row['dataset']} | `{row['model']}` | `{row['structural_category']}` | "
            f"{row['n_conflict_cases']} | {row['dataset_cco_n']} | "
            f"{row['dataset_cco_rate']:.3f} | {row['cco_support_n']} / {row['cco_refute_n']} | "
            f"{mean_text} |\n"
        )
    (output_dir / "summary.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "human_audit_accessed": False, "case_model_rows": len(rows), "taxonomy_rows": len(metrics), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
