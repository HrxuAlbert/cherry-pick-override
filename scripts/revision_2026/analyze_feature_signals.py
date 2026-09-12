#!/usr/bin/env python3
"""Audit-independent confidence-versus-structure diagnostic.

The pending human labels are not accessed. Instead, this script uses the
AVeriTeC dataset conflict label as an explicitly provisional target and tests
whether cached structural-validator features add screening information beyond
model confidence. Cross-validation is grouped by case, all preprocessing is
fit inside each training fold, and model effects are present in every model.
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
from typing import Any

import numpy as np


WORKSPACE = _cpo_workspace()
LABELS = {"support", "refute", "insufficient", "conflicting"}
COUNT_FEATURES = ("n_subclaims", "n_material_subclaims", "n_cited_evidence_ids")
BOOLEAN_FEATURES = (
    "has_material_support",
    "has_material_refute",
    "has_material_mixed",
    "has_material_insufficient",
)
FEATURE_SETS = {
    "model_only": (),
    "confidence_only": ("confidence",),
    "structure_only": (*COUNT_FEATURES, *BOOLEAN_FEATURES),
    "confidence_plus_structure": ("confidence", *COUNT_FEATURES, *BOOLEAN_FEATURES),
}


class FeatureAnalysisError(RuntimeError):
    """Raised when the provisional feature analysis cannot be reproduced."""


def parse_args() -> argparse.Namespace:
    cached = WORKSPACE / "Writing/V0.2/code_release/outputs/option_a_exp/strengthening"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=cached / "e1_full_4label_utility/raw_results.jsonl",
    )
    parser.add_argument(
        "--validator",
        type=Path,
        default=cached / "e3_structured_certificate_validator_fewshot/raw_results.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/audit_independent/feature_signals",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FeatureAnalysisError(f"Missing input: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise FeatureAnalysisError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def build_rows(
    prediction_rows: list[dict[str, Any]], validator_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    validators: dict[str, dict[str, Any]] = {}
    for row in validator_rows:
        case_id = str(row.get("case_id"))
        if case_id in validators:
            raise FeatureAnalysisError(f"Duplicate validator case ID: {case_id}")
        validators[case_id] = row.get("validity") or {}

    panels = [
        row for row in prediction_rows if row.get("system") == "panel_3judge_4opt_strong"
    ]
    if len(panels) != 285 or len(validators) != 285:
        raise FeatureAnalysisError(
            f"Expected 285 typed-panel and validator cases; got {len(panels)} and {len(validators)}"
        )

    output: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for panel in panels:
        case_id = str(panel["case_id"])
        gold = str(panel.get("gold_normal") or "").lower()
        if gold not in LABELS:
            raise FeatureAnalysisError(f"Invalid gold label for case {case_id}: {gold}")
        validity = validators.get(case_id)
        if validity is None:
            raise FeatureAnalysisError(f"Missing validator row for case {case_id}")
        for judge in panel.get("judge_outputs") or []:
            model = str(judge.get("model") or "")
            pair = (case_id, model)
            if not model or pair in seen_pairs:
                raise FeatureAnalysisError(f"Missing/duplicate model output for case {case_id}")
            seen_pairs.add(pair)
            parsed = judge.get("parsed") or {}
            pred = str(parsed.get("verdict_normal") or "").lower()
            if pred not in {"support", "refute"}:
                continue
            if gold not in {"support", "refute", "conflicting"}:
                continue
            confidence = parsed.get("confidence")
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError) as exc:
                raise FeatureAnalysisError(
                    f"Directional output lacks confidence for case {case_id}/{model}"
                ) from exc
            if not 0.0 <= confidence_value <= 1.0:
                raise FeatureAnalysisError(
                    f"Confidence outside [0,1] for case {case_id}/{model}"
                )
            feature_row: dict[str, Any] = {
                "case_id": case_id,
                "model": model,
                "gold": gold,
                "prediction": pred,
                "target_dataset_conflict": int(gold == "conflicting"),
                "confidence": confidence_value,
            }
            for name in COUNT_FEATURES:
                try:
                    feature_row[name] = float(validity.get(name, 0) or 0)
                except (TypeError, ValueError) as exc:
                    raise FeatureAnalysisError(
                        f"Invalid validator count {name} for case {case_id}"
                    ) from exc
            for name in BOOLEAN_FEATURES:
                feature_row[name] = float(bool(validity.get(name, False)))
            output.append(feature_row)
    if not output:
        raise FeatureAnalysisError("No eligible directional outputs were found.")
    targets = Counter(row["target_dataset_conflict"] for row in output)
    if min(targets.values()) < 20:
        raise FeatureAnalysisError(f"Too few rows in one target class: {dict(targets)}")
    return output


def grouped_stratified_folds(
    rows: list[dict[str, Any]], folds: int, seed: int
) -> dict[str, int]:
    if folds < 2:
        raise FeatureAnalysisError("At least two folds are required.")
    by_case: dict[str, int] = {}
    for row in rows:
        case_id = row["case_id"]
        target = int(row["target_dataset_conflict"])
        if case_id in by_case and by_case[case_id] != target:
            raise FeatureAnalysisError(f"Case-level target is inconsistent: {case_id}")
        by_case[case_id] = target
    rng = random.Random(seed)
    assignment: dict[str, int] = {}
    for target in (0, 1):
        case_ids = sorted(case_id for case_id, value in by_case.items() if value == target)
        rng.shuffle(case_ids)
        for index, case_id in enumerate(case_ids):
            assignment[case_id] = index % folds
    if len(set(assignment.values())) != folds:
        raise FeatureAnalysisError("Fold construction produced an empty fold.")
    return assignment


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return mean, std if std > 1e-12 else 1.0


def design_matrices(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_set: str,
    models: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    requested = FEATURE_SETS[feature_set]
    names = ["intercept", *[f"model::{model}" for model in models[1:]]]
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    if "confidence" in requested:
        for model in models:
            values = [row["confidence"] for row in train_rows if row["model"] == model]
            stats[("confidence", model)] = _mean_std(values)
        names.append("confidence_z_within_model")
    for feature in COUNT_FEATURES:
        if feature in requested:
            stats[(feature, "all")] = _mean_std([row[feature] for row in train_rows])
            names.append(f"{feature}_z")
    for feature in BOOLEAN_FEATURES:
        if feature in requested:
            names.append(feature)

    def encode(rows: list[dict[str, Any]]) -> np.ndarray:
        matrix: list[list[float]] = []
        for row in rows:
            values = [1.0]
            values.extend(float(row["model"] == model) for model in models[1:])
            if "confidence" in requested:
                mean, std = stats[("confidence", row["model"])]
                values.append((row["confidence"] - mean) / std)
            for feature in COUNT_FEATURES:
                if feature in requested:
                    mean, std = stats[(feature, "all")]
                    values.append((row[feature] - mean) / std)
            for feature in BOOLEAN_FEATURES:
                if feature in requested:
                    values.append(row[feature])
            matrix.append(values)
        return np.asarray(matrix, dtype=float)

    x_train = encode(train_rows)
    x_test = encode(test_rows)
    y_train = np.asarray([row["target_dataset_conflict"] for row in train_rows], dtype=float)
    return x_train, y_train, x_test, names


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(
    x: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 100
) -> np.ndarray:
    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise FeatureAnalysisError("Invalid logistic-regression training matrix.")
    weights = np.zeros(x.shape[1], dtype=float)
    penalty = np.eye(x.shape[1], dtype=float) * l2
    penalty[0, 0] = 0.0
    for _ in range(max_iter):
        probabilities = _sigmoid(x @ weights)
        gradient = x.T @ (probabilities - y) + penalty @ weights
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
        hessian = x.T @ (x * variance[:, None]) + penalty
        hessian += np.eye(x.shape[1]) * 1e-9
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return weights


def average_precision(y_true: list[int], scores: list[float]) -> float:
    positives = sum(y_true)
    if positives == 0:
        return 0.0
    groups: dict[float, list[int]] = defaultdict(list)
    for target, score in zip(y_true, scores):
        groups[float(score)].append(int(target))
    tp = 0
    fp = 0
    previous_recall = 0.0
    area = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        tp += sum(group)
        fp += len(group) - sum(group)
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def auroc(y_true: list[int], scores: list[float]) -> float:
    positives = [score for target, score in zip(y_true, scores) if target == 1]
    negatives = [score for target, score in zip(y_true, scores) if target == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def recall_at_fraction(y_true: list[int], scores: list[float], fraction: float) -> float:
    positives = sum(y_true)
    if positives == 0:
        return 0.0
    k = max(1, math.ceil(len(scores) * fraction))
    selected = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:k]
    return sum(y_true[index] for index in selected) / positives


def cross_validated_predictions(
    rows: list[dict[str, Any]], folds: int, seed: int, l2: float
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    assignment = grouped_stratified_folds(rows, folds, seed)
    models = sorted({row["model"] for row in rows})
    output: list[dict[str, Any]] = []
    feature_names: dict[str, list[str]] = {}
    for fold in range(folds):
        train = [row for row in rows if assignment[row["case_id"]] != fold]
        test = [row for row in rows if assignment[row["case_id"]] == fold]
        train_cases = {row["case_id"] for row in train}
        test_cases = {row["case_id"] for row in test}
        if train_cases & test_cases:
            raise FeatureAnalysisError("Case leakage detected between CV folds.")
        for feature_set in FEATURE_SETS:
            x_train, y_train, x_test, names = design_matrices(
                train, test, feature_set, models
            )
            weights = fit_logistic(x_train, y_train, l2=l2)
            scores = _sigmoid(x_test @ weights)
            feature_names[feature_set] = names
            for row, score in zip(test, scores):
                output.append(
                    {
                        "case_id": row["case_id"],
                        "model": row["model"],
                        "gold": row["gold"],
                        "prediction": row["prediction"],
                        "target_dataset_conflict": row["target_dataset_conflict"],
                        "fold": fold,
                        "feature_set": feature_set,
                        "risk_score": float(score),
                    }
                )
    expected = len(rows) * len(FEATURE_SETS)
    if len(output) != expected:
        raise FeatureAnalysisError(f"Expected {expected} OOF rows; produced {len(output)}")
    return output, feature_names


def metric_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        rows = [row for row in predictions if row["feature_set"] == feature_set]
        targets = [int(row["target_dataset_conflict"]) for row in rows]
        scores = [float(row["risk_score"]) for row in rows]
        output.append(
            {
                "label_basis": "dataset_conflict_proxy",
                "feature_set": feature_set,
                "n_rows": len(rows),
                "n_cases": len({row["case_id"] for row in rows}),
                "n_positive_rows": sum(targets),
                "prevalence": sum(targets) / len(targets),
                "auprc": average_precision(targets, scores),
                "auroc": auroc(targets, scores),
                "recall_at_top_10pct": recall_at_fraction(targets, scores, 0.10),
                "recall_at_top_20pct": recall_at_fraction(targets, scores, 0.20),
                "recall_at_top_30pct": recall_at_fraction(targets, scores, 0.30),
            }
        )
    return output


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(quantile * (len(ordered) - 1))))
    return ordered[index]


def cluster_bootstrap_difference(
    predictions: list[dict[str, Any]], resamples: int, seed: int
) -> dict[str, Any]:
    by_feature: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in predictions:
        by_feature[row["feature_set"]][row["case_id"]].append(row)
    case_ids = sorted(by_feature["confidence_only"])
    if set(case_ids) != set(by_feature["confidence_plus_structure"]):
        raise FeatureAnalysisError("Bootstrap feature sets do not share case IDs.")

    def score(feature_set: str, sampled_ids: list[str]) -> float:
        targets: list[int] = []
        scores: list[float] = []
        for case_id in sampled_ids:
            for row in by_feature[feature_set][case_id]:
                targets.append(int(row["target_dataset_conflict"]))
                scores.append(float(row["risk_score"]))
        return average_precision(targets, scores)

    observed = score("confidence_plus_structure", case_ids) - score(
        "confidence_only", case_ids
    )
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(resamples):
        sampled = [case_ids[rng.randrange(len(case_ids))] for _ in case_ids]
        differences.append(
            score("confidence_plus_structure", sampled)
            - score("confidence_only", sampled)
        )
    return {
        "comparison": "confidence_plus_structure_minus_confidence_only",
        "metric": "AUPRC",
        "unit": "case_cluster",
        "n_cases": len(case_ids),
        "resamples": resamples,
        "seed": seed,
        "observed_difference": observed,
        "ci_low": percentile(differences, 0.025),
        "ci_high": percentile(differences, 0.975),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    source_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    bootstrap: dict[str, Any],
) -> None:
    models = Counter(row["model"] for row in source_rows)
    lines = [
        "# Confidence versus evidence-structure diagnostic\n\n",
        "> Provisional scope: the target is the AVeriTeC dataset conflict label, not the pending human audit. These numbers diagnose the existing cache and cannot serve as the final human-confirmed E4 result.\n\n",
        f"Eligible directional outputs: {len(source_rows)} from {len(models)} models and {len({row['case_id'] for row in source_rows})} cases. Five-fold cross-validation is grouped by case. Model fixed effects are included in every specification.\n\n",
        "| Feature specification | AUPRC | AUROC | Recall @ top 10% | Recall @ top 20% | Recall @ top 30% |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in metrics:
        lines.append(
            f"| `{row['feature_set']}` | {row['auprc']:.3f} | {row['auroc']:.3f} | "
            f"{row['recall_at_top_10pct']:.3f} | {row['recall_at_top_20pct']:.3f} | "
            f"{row['recall_at_top_30pct']:.3f} |\n"
        )
    lines.extend(
        [
            "\n## Paired case-cluster bootstrap\n\n",
            f"Combined minus confidence-only AUPRC: {bootstrap['observed_difference']:+.3f}, "
            f"95% interval [{bootstrap['ci_low']:+.3f}, {bootstrap['ci_high']:+.3f}] "
            f"({bootstrap['resamples']} resamples).\n\n",
            "**Provisional Finding.** Evidence-structure features are evaluated as incremental screening signals under grouped out-of-fold prediction. The finding must be recomputed with frozen human labels before it can support the paper's construct-level claim.\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FeatureAnalysisError(f"Output directory is nonempty; use --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.predictions.resolve()
    validator_path = args.validator.resolve()
    source_rows = build_rows(read_jsonl(prediction_path), read_jsonl(validator_path))
    predictions, feature_names = cross_validated_predictions(
        source_rows, args.folds, args.seed, args.l2
    )
    metrics = metric_rows(predictions)
    bootstrap = cluster_bootstrap_difference(
        predictions, args.bootstrap_resamples, args.seed
    )
    write_csv(output_dir / "cv_metrics.csv", metrics)
    write_csv(output_dir / "oof_predictions.csv", predictions)
    write_summary(output_dir / "summary.md", source_rows, metrics, bootstrap)
    (output_dir / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "label_basis": "dataset_conflict_proxy",
                "human_audit_accessed": False,
                "predictions": {
                    "path": str(prediction_path),
                    "sha256": sha256_file(prediction_path),
                },
                "validator": {
                    "path": str(validator_path),
                    "sha256": sha256_file(validator_path),
                },
                "folds": args.folds,
                "bootstrap_resamples": args.bootstrap_resamples,
                "seed": args.seed,
                "l2": args.l2,
                "feature_names": feature_names,
            },
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
                "eligible_rows": len(source_rows),
                "cases": len({row['case_id'] for row in source_rows}),
                "models": len({row['model'] for row in source_rows}),
                "output_dir": str(output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
