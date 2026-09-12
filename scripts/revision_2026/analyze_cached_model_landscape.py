#!/usr/bin/env python3
"""Build an audit-independent model/error landscape from cached predictions.

This analysis deliberately uses dataset labels only. It never reads the human
audit directory and it labels every CCO estimate as dataset-defined rather than
human-confirmed. The output is intended to provide the historical continuity,
cross-dataset, error-taxonomy, and model-family tables needed by the revision.
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
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


WORKSPACE = _cpo_workspace()
LABELS = ("support", "refute", "insufficient", "conflicting")
PARSE_FAILURE = "parse_failure"
EXPECTED = {
    "AVeriTeC": {"support": 50, "refute": 50, "insufficient": 35, "conflicting": 150},
    "VitaminC-Mixed": {"support": 50, "refute": 50, "insufficient": 50, "conflicting": 100},
}
SYSTEM_ORDER = (
    "single_haiku_3opt",
    "single_haiku_4opt_strong",
    "single_sonnet_4opt_strong",
    "panel_3judge_3opt",
    "panel_3judge_4opt_strong",
)


class AnalysisError(RuntimeError):
    """Raised when cached data violate the frozen analysis contract."""


def parse_args() -> argparse.Namespace:
    root = WORKSPACE / "Writing/V0.2/code_release/outputs/option_a_exp/strengthening"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--averitec",
        type=Path,
        default=root / "e1_full_4label_utility/raw_results.jsonl",
    )
    parser.add_argument(
        "--vitaminc",
        type=Path,
        default=root / "e4_vitaminc_mixed/raw_results.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/audit_independent/cached_landscape",
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AnalysisError(f"Missing cached result file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    return label if label in LABELS else PARSE_FAILURE


def validate_dataset(dataset: str, rows: list[dict[str, Any]]) -> None:
    systems: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        systems[str(row.get("system", ""))].append(row)
    if set(systems) != set(SYSTEM_ORDER):
        raise AnalysisError(
            f"{dataset} systems differ from the frozen five-system cache: "
            f"{sorted(systems)}"
        )
    expected = EXPECTED[dataset]
    expected_n = sum(expected.values())
    for system, system_rows in systems.items():
        ids = [str(row.get("case_id")) for row in system_rows]
        if len(system_rows) != expected_n or len(set(ids)) != expected_n:
            raise AnalysisError(
                f"{dataset}/{system} needs {expected_n} unique cases; "
                f"found {len(system_rows)} rows and {len(set(ids))} IDs."
            )
        counts = Counter(normalize_label(row.get("gold_normal")) for row in system_rows)
        if dict(counts) != expected:
            raise AnalysisError(
                f"{dataset}/{system} label composition mismatch: {dict(counts)}"
            )


def materialize_predictions(
    dataset: str, rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Return top-level systems plus typed panel-member model views."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        base = {
            "case_id": str(row["case_id"]),
            "gold": normalize_label(row.get("gold_normal")),
            "dataset": dataset,
        }
        system = str(row["system"])
        out[system].append({**base, "pred": normalize_label(row.get("verdict_normal"))})
        if system != "panel_3judge_4opt_strong":
            continue
        seen_models: set[str] = set()
        for judge in row.get("judge_outputs") or []:
            model = str(judge.get("model") or "unknown")
            if model in seen_models:
                raise AnalysisError(
                    f"Duplicate panel member {model} for {dataset} case {row['case_id']}"
                )
            seen_models.add(model)
            parsed = judge.get("parsed") or {}
            out[f"panel_member_4opt::{model}"].append(
                {**base, "pred": normalize_label(parsed.get("verdict_normal"))}
            )
    return dict(out)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def safe_rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def macro_f1(rows: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for label in LABELS:
        tp = sum(row["gold"] == label and row["pred"] == label for row in rows)
        fp = sum(row["gold"] != label and row["pred"] == label for row in rows)
        fn = sum(row["gold"] == label and row["pred"] != label for row in rows)
        precision = safe_rate(tp, tp + fp)
        recall = safe_rate(tp, tp + fn)
        values.append(safe_rate(2 * precision * recall, precision + recall))
    return sum(values) / len(values)


def compute_metrics(dataset: str, system: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_conflict = sum(row["gold"] == "conflicting" for row in rows)
    n_sr = sum(row["gold"] in {"support", "refute"} for row in rows)
    n_insufficient = sum(row["gold"] == "insufficient" for row in rows)
    parse_failures = sum(row["pred"] == PARSE_FAILURE for row in rows)
    correct = sum(row["pred"] == row["gold"] for row in rows)
    cco = sum(
        row["gold"] == "conflicting" and row["pred"] in {"support", "refute"}
        for row in rows
    )
    conflict_correct = sum(
        row["gold"] == "conflicting" and row["pred"] == "conflicting" for row in rows
    )
    sr_correct = sum(
        row["gold"] in {"support", "refute"} and row["pred"] == row["gold"]
        for row in rows
    )
    false_conflict = sum(
        row["gold"] in {"support", "refute"} and row["pred"] == "conflicting"
        for row in rows
    )
    insufficient_directional = sum(
        row["gold"] == "insufficient" and row["pred"] in {"support", "refute"}
        for row in rows
    )
    cco_lo, cco_hi = wilson_interval(cco, n_conflict)
    return {
        "dataset": dataset,
        "system": system,
        "label_basis": "dataset_defined",
        "n_total": n,
        "n_parse_ok": n - parse_failures,
        "parse_failures": parse_failures,
        "parse_failure_rate": safe_rate(parse_failures, n),
        "overall_accuracy": safe_rate(correct, n),
        "macro_f1": macro_f1(rows),
        "n_conflicting": n_conflict,
        "cco_n": cco,
        "cco_rate": safe_rate(cco, n_conflict),
        "cco_wilson_low": cco_lo,
        "cco_wilson_high": cco_hi,
        "conflict_recall": safe_rate(conflict_correct, n_conflict),
        "n_pure_sr": n_sr,
        "pure_sr_correct_n": sr_correct,
        "pure_sr_accuracy": safe_rate(sr_correct, n_sr),
        "false_conflict_n": false_conflict,
        "false_conflict_rate_on_sr": safe_rate(false_conflict, n_sr),
        "n_insufficient": n_insufficient,
        "insufficient_directional_n": insufficient_directional,
        "insufficient_directional_rate": safe_rate(insufficient_directional, n_insufficient),
        **{f"pred_{label}_n": sum(row["pred"] == label for row in rows) for label in LABELS},
        "pred_parse_failure_n": parse_failures,
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(quantile * (len(ordered) - 1))))
    return ordered[index]


def paired_bootstrap(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    subset: Callable[[dict[str, Any]], bool],
    outcome: Callable[[dict[str, Any]], int],
    resamples: int,
    seed: int,
) -> tuple[int, float, float, float, float, float]:
    by_a = {row["case_id"]: row for row in rows_a if subset(row)}
    by_b = {row["case_id"]: row for row in rows_b if subset(row)}
    if set(by_a) != set(by_b):
        raise AnalysisError("Paired contrast case sets do not match.")
    ids = sorted(by_a)
    a_values = [outcome(by_a[case_id]) for case_id in ids]
    b_values = [outcome(by_b[case_id]) for case_id in ids]
    n = len(ids)
    rate_a = safe_rate(sum(a_values), n)
    rate_b = safe_rate(sum(b_values), n)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        sample = [rng.randrange(n) for _ in range(n)]
        delta = sum(b_values[i] - a_values[i] for i in sample) / n
        deltas.append(delta)
    return n, rate_a, rate_b, rate_b - rate_a, percentile(deltas, 0.025), percentile(deltas, 0.975)


def build_contrasts(
    all_predictions: dict[str, dict[str, list[dict[str, Any]]]],
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    pairs = [
        ("single_haiku_3opt", "panel_3judge_3opt", "panel_vs_single_3opt"),
        ("single_haiku_3opt", "single_haiku_4opt_strong", "haiku_3opt_to_4opt"),
        ("panel_3judge_3opt", "panel_3judge_4opt_strong", "panel_3opt_to_4opt"),
        ("single_haiku_4opt_strong", "panel_3judge_4opt_strong", "typed_panel_vs_typed_single"),
    ]
    metrics = [
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
    ]
    output: list[dict[str, Any]] = []
    for dataset, systems in all_predictions.items():
        for system_a, system_b, contrast_name in pairs:
            for metric, subset, outcome in metrics:
                n, rate_a, rate_b, delta, low, high = paired_bootstrap(
                    systems[system_a],
                    systems[system_b],
                    subset,
                    outcome,
                    resamples,
                    seed,
                )
                output.append(
                    {
                        "dataset": dataset,
                        "contrast": contrast_name,
                        "metric": metric,
                        "system_a": system_a,
                        "system_b": system_b,
                        "n_cases": n,
                        "rate_a": rate_a,
                        "rate_b": rate_b,
                        "delta_b_minus_a": delta,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_resamples": resamples,
                        "seed": seed,
                    }
                )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_rate(value: float) -> str:
    return f"{100 * value:.1f}"


def write_summary(
    path: Path, metrics: list[dict[str, Any]], contrasts: list[dict[str, Any]]
) -> None:
    lines = [
        "# Audit-independent cached-model landscape\n\n",
        "> Scope: all CCO values below use dataset labels, not the pending human audit.\n\n",
        "## Model/system metrics\n\n",
        "| Dataset | Model/system | Accuracy | Macro-F1 | Dataset CCO | 95% Wilson | Conflict recall | Pure S/R acc. | False conflict | Insufficient directional |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    headline = [row for row in metrics if row["system"] in SYSTEM_ORDER]
    for row in headline:
        lines.append(
            f"| {row['dataset']} | `{row['system']}` | {fmt_rate(row['overall_accuracy'])} | "
            f"{fmt_rate(row['macro_f1'])} | {row['cco_n']}/{row['n_conflicting']} "
            f"({fmt_rate(row['cco_rate'])}) | "
            f"[{fmt_rate(row['cco_wilson_low'])}, {fmt_rate(row['cco_wilson_high'])}] | "
            f"{fmt_rate(row['conflict_recall'])} | {fmt_rate(row['pure_sr_accuracy'])} | "
            f"{fmt_rate(row['false_conflict_rate_on_sr'])} | "
            f"{fmt_rate(row['insufficient_directional_rate'])} |\n"
        )
    lines.extend(
        [
            "\n## Paired contrasts\n\n",
            "| Dataset | Contrast | Metric | A | B | Delta B-A | 95% paired bootstrap | n |\n",
            "|---|---|---|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| {row['dataset']} | `{row['contrast']}` | {row['metric']} | "
            f"{fmt_rate(row['rate_a'])} | {fmt_rate(row['rate_b'])} | "
            f"{100 * row['delta_b_minus_a']:+.1f} pp | "
            f"[{100 * row['ci_low']:+.1f}, {100 * row['ci_high']:+.1f}] pp | "
            f"{row['n_cases']} |\n"
        )

    lookup = {(row["dataset"], row["system"]): row for row in headline}
    av3 = lookup[("AVeriTeC", "panel_3judge_3opt")]
    av4 = lookup[("AVeriTeC", "panel_3judge_4opt_strong")]
    vc3 = lookup[("VitaminC-Mixed", "panel_3judge_3opt")]
    vc4 = lookup[("VitaminC-Mixed", "panel_3judge_4opt_strong")]
    lines.extend(
        [
            "\n## Data-supported provisional findings\n\n",
            f"**Finding 1 (dataset-defined continuity only).** Adding the explicit four-way contract reduces panel directional commitments on dataset-labelled conflicts from {fmt_rate(av3['cco_rate'])}% to {fmt_rate(av4['cco_rate'])}% on AVeriTeC and from {fmt_rate(vc3['cco_rate'])}% to {fmt_rate(vc4['cco_rate'])}% on VitaminC-Mixed; it does not establish the human-confirmed rate.\n\n",
            "**Finding 2 (error trade-off).** The same table reports pure-directional accuracy, false-conflict errors, and insufficient-case directional errors beside CCO, preventing a lower commitment rate from being interpreted as an unconditional improvement.\n\n",
            "**Finding 3 (cross-dataset boundary).** Panel and prompt effects are reported separately for the two substrates; a pattern present on one dataset is not generalized to the other without a paired contrast and an audited construct check.\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise AnalysisError(f"Output directory is nonempty; use --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "AVeriTeC": args.averitec.resolve(),
        "VitaminC-Mixed": args.vitaminc.resolve(),
    }
    cached = {dataset: load_jsonl(path) for dataset, path in input_paths.items()}
    for dataset, rows in cached.items():
        validate_dataset(dataset, rows)

    predictions = {
        dataset: materialize_predictions(dataset, rows) for dataset, rows in cached.items()
    }
    metrics: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    for dataset, systems in predictions.items():
        ordered_systems = sorted(
            systems,
            key=lambda name: (
                SYSTEM_ORDER.index(name) if name in SYSTEM_ORDER else len(SYSTEM_ORDER),
                name,
            ),
        )
        for system in ordered_systems:
            rows = systems[system]
            metrics.append(compute_metrics(dataset, system, rows))
            matrix = Counter((row["gold"], row["pred"]) for row in rows)
            for gold in LABELS:
                for pred in (*LABELS, PARSE_FAILURE):
                    confusion.append(
                        {
                            "dataset": dataset,
                            "system": system,
                            "gold": gold,
                            "prediction": pred,
                            "count": matrix[(gold, pred)],
                        }
                    )

    contrasts = build_contrasts(
        predictions, args.bootstrap_resamples, args.seed
    )
    write_csv(output_dir / "model_metrics.csv", metrics)
    write_csv(output_dir / "confusion_matrices.csv", confusion)
    write_csv(output_dir / "paired_contrasts.csv", contrasts)
    write_summary(output_dir / "summary.md", metrics, contrasts)
    (output_dir / "results.json").write_text(
        json.dumps(
            {"metrics": metrics, "paired_contrasts": contrasts},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "label_basis": "dataset_defined",
                "human_audit_accessed": False,
                "inputs": {
                    dataset: {"path": str(path), "sha256": sha256_file(path)}
                    for dataset, path in input_paths.items()
                },
                "bootstrap_resamples": args.bootstrap_resamples,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "human_audit_accessed": False,
                "metric_rows": len(metrics),
                "contrast_rows": len(contrasts),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
