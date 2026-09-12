#!/usr/bin/env python3
"""Merge the E4 decomposition runs and estimate Delta_decomp.

The Claude Sonnet 5 arm was re-run into a separate output directory after the
Anthropic credit balance was restored, because the main run was still appending
to its own JSONL and a minority of rows exceed the atomic-append size. Both runs
share a run_id, so their rows describe the same experiment. They are merged per
(model, case) rather than per request hash, because stage 2's prompt embeds the
stage-1 decomposition and a re-run that decomposes slightly differently produces
a second, equally valid stage-2 call; see ``merge_runs``.

Delta_decomp is the change in directional commitment on gold-Conflicting claims
when the judge first decomposes the evidence into supporting and refuting items
and then issues the verdict with its own decomposition in context, against the
same judge's verdict in the parent run. Unlike the recognition contrast, this is
a *paired* comparison: the same claim is scored under both conditions, so the
bootstrap resamples cases and recomputes the paired difference.

Frozen in 09_PREANALYSIS_ADDENDUM.md §5. No human-audit artifact is read.
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
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORKSPACE = _cpo_workspace()
DIRECTIONAL = {"support", "refute"}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260901
RUN_ID = "averitec-probe-e4-decompose-20260901-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decompose-runs",
        type=Path,
        nargs="+",
        default=[
            WORKSPACE / f"outputs/revision_2026/probes/runs/{RUN_ID}/raw_results_full.jsonl",
            WORKSPACE
            / f"outputs/revision_2026/probes/runs_anthropic_resume/{RUN_ID}/raw_results_full.jsonl",
        ],
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=WORKSPACE
        / "outputs/revision_2026/current_models_averitec/runs"
        / "averitec-p0-current-models-20260818-v1/raw_results_full.jsonl",
    )
    parser.add_argument(
        "--recognition",
        type=Path,
        default=WORKSPACE
        / f"outputs/revision_2026/probes/runs/averitec-probe-e1-recognition-20260901-v1"
        / "raw_results_full.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/probes/analysis",
    )
    parser.add_argument("--merged-out", type=Path, default=None)
    return parser.parse_args()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"corrupt JSONL at {path}:{number}: {exc}") from exc
    return rows


def merge_runs(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge the decomposition runs, keeping one coherent attempt per (model, case).

    Deduplicating on ``request_sha256`` alone is not sufficient here. Stage 2's prompt
    embeds the model's own stage-1 decomposition, so when a case is re-run and the
    decomposition differs even slightly, the two stage-2 calls have different request
    hashes and both survive. That would leave a case with two stage-2 verdicts and no
    principled way to say which decomposition produced which.

    A stage-2 verdict is only interpretable next to the stage-1 output it was
    conditioned on, so the unit kept here is the (model, case) attempt from a single
    source file: the last file that supplies a usable stage-1 for that case wins, and
    its stage-2 row travels with it.
    """
    per_file: dict[str, int] = {}
    run_ids: set[str] = set()
    # (model, case) -> source index -> stage -> row
    attempts: dict[tuple[str, int], dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for index, path in enumerate(paths):
        rows = load_jsonl(path)
        per_file[str(path)] = len(rows)
        for row in rows:
            run_ids.add(str(row.get("run_id")))
            key = (row["model_slot"], int(row["case_id"]))
            stage = row["stage"]
            held = attempts[key][index].get(stage)
            if held is None or (held.get("status") != "ok" and row.get("status") == "ok"):
                attempts[key][index][stage] = row
    if len(run_ids) != 1:
        raise SystemExit(f"refusing to merge rows from different run_ids: {sorted(run_ids)}")

    merged: list[dict[str, Any]] = []
    chosen_source = Counter()
    for key, by_source in attempts.items():
        usable = [
            index
            for index, stages in sorted(by_source.items())
            if stages.get("stage1_decompose", {}).get("status") == "ok"
        ]
        index = usable[-1] if usable else sorted(by_source)[-1]
        chosen_source[str(paths[index])] += 1
        for stage in ("stage1_decompose", "stage2_verdict"):
            row = by_source[index].get(stage)
            if row is not None:
                merged.append(row)

    merged.sort(key=lambda r: (r["model_slot"], int(r["case_id"]), r["stage"]))
    duplicates = Counter((r["model_slot"], int(r["case_id"]), r["stage"]) for r in merged)
    offending = [k for k, v in duplicates.items() if v > 1]
    if offending:
        raise SystemExit(f"merge left duplicate (model, case, stage) rows: {offending[:5]}")

    report = {
        "rows_per_file": per_file,
        "attempts_kept": len(attempts),
        "attempts_by_source": dict(chosen_source),
        "merged_rows": len(merged),
        "run_id": run_ids.pop(),
        "status_counts": dict(Counter(r["status"] for r in merged)),
    }
    return merged, report


def paired_bootstrap(pairs: list[tuple[int, int]], resamples: int, seed: int) -> tuple[float, float]:
    """95% interval for mean(after) - mean(before) over paired cases."""
    if not pairs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(resamples):
        total = 0
        for _ in range(n):
            before, after = pairs[rng.randrange(n)]
            total += after - before
        diffs.append(total / n)
    diffs.sort()
    return (diffs[int(0.025 * len(diffs))], diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))])


def main() -> None:
    args = parse_args()
    merged, merge_report = merge_runs(args.decompose_runs)
    if args.merged_out:
        args.merged_out.parent.mkdir(parents=True, exist_ok=True)
        with args.merged_out.open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    stage2 = {
        (r["model_slot"], int(r["case_id"])): (r.get("parsed") or {}).get("verdict_normal")
        for r in merged
        if r["stage"] == "stage2_verdict" and r.get("status") == "ok"
    }
    stage1 = {
        (r["model_slot"], int(r["case_id"])): r.get("parsed") or {}
        for r in merged
        if r["stage"] == "stage1_decompose" and r.get("status") == "ok"
    }

    baseline: dict[tuple[str, int], str] = {}
    gold: dict[int, str] = {}
    for row in load_jsonl(args.verdicts):
        gold[int(row["case_id"])] = row["gold_normal"]
        if row.get("status") == "ok":
            verdict = (row.get("parsed") or {}).get("verdict_normal")
            if verdict:
                baseline[(row["model_slot"], int(row["case_id"]))] = verdict

    recognition = {
        (r["model_slot"], int(r["case_id"])): (r.get("parsed") or {}).get("recognition")
        for r in load_jsonl(args.recognition)
        if r.get("status") == "ok"
    }

    models = sorted({slot for slot, _ in stage2} | {slot for slot, _ in baseline})
    report: dict[str, Any] = {
        "merge": merge_report,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "case"},
        "models": {},
    }

    for slot in models:
        conflicting = [c for c, g in gold.items() if g == "conflicting"]
        pure = [c for c, g in gold.items() if g in DIRECTIONAL]

        pairs: list[tuple[int, int]] = []
        for case_id in conflicting:
            before, after = baseline.get((slot, case_id)), stage2.get((slot, case_id))
            if before is None or after is None:
                continue
            pairs.append((int(before in DIRECTIONAL), int(after in DIRECTIONAL)))

        n_paired = len(pairs)
        cpo_before = sum(b for b, _ in pairs) / n_paired if n_paired else None
        cpo_after = sum(a for _, a in pairs) / n_paired if n_paired else None
        lo, hi = paired_bootstrap(pairs, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)

        # Secondary: does the decomposition buy conflict recall by over-producing
        # CONFLICTING on one-sided evidence?
        fc_pairs = []
        for case_id in pure:
            before, after = baseline.get((slot, case_id)), stage2.get((slot, case_id))
            if before is None or after is None:
                continue
            fc_pairs.append((int(before == "conflicting"), int(after == "conflicting")))
        fc_before = sum(b for b, _ in fc_pairs) / len(fc_pairs) if fc_pairs else None
        fc_after = sum(a for _, a in fc_pairs) / len(fc_pairs) if fc_pairs else None
        fc_lo, fc_hi = paired_bootstrap(fc_pairs, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)

        # Secondary: do the model's own stage-1 items agree with the E1 probe?
        agree = total = 0
        for case_id in conflicting:
            decomposition = stage1.get((slot, case_id))
            answer = recognition.get((slot, case_id))
            if not decomposition or answer is None:
                continue
            total += 1
            agree += int((decomposition.get("decomposition_two_sided") == "yes") == (answer == "yes"))

        # Did decomposition rescue the cases where the baseline overcommitted?
        rescued = still = 0
        for case_id in conflicting:
            before, after = baseline.get((slot, case_id)), stage2.get((slot, case_id))
            if before is None or after is None or before not in DIRECTIONAL:
                continue
            if after in DIRECTIONAL:
                still += 1
            else:
                rescued += 1

        report["models"][slot] = {
            "n_conflicting_paired": n_paired,
            "cpo_baseline": cpo_before,
            "cpo_decomposed": cpo_after,
            "cpo_baseline_wilson": list(wilson(sum(b for b, _ in pairs), n_paired)) if n_paired else None,
            "cpo_decomposed_wilson": list(wilson(sum(a for _, a in pairs), n_paired)) if n_paired else None,
            "delta_decomp": (None if cpo_after is None else cpo_after - cpo_before),
            "delta_decomp_ci": [lo, hi],
            "baseline_cpo_cases": rescued + still,
            "rescued_by_decomposition": rescued,
            "still_directional": still,
            "n_pure_paired": len(fc_pairs),
            "false_conflict_baseline": fc_before,
            "false_conflict_decomposed": fc_after,
            "false_conflict_delta": (None if fc_after is None else fc_after - fc_before),
            "false_conflict_ci": [fc_lo, fc_hi],
            "decomposition_vs_probe_agreement": (agree / total) if total else None,
            "decomposition_vs_probe_n": total,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e4_decomposition.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"merged {merge_report['merged_rows']} rows into "
          f"{merge_report['attempts_kept']} (model, case) attempts from "
          f"{len(merge_report['rows_per_file'])} files; statuses {merge_report['status_counts']}")
    for source, count in merge_report["attempts_by_source"].items():
        print(f"  {count:>5} attempts from {Path(source).parent.parent.name}")
    print()
    print(f"{'model':<30}{'n':>5}{'CPO base':>10}{'CPO decomp':>12}"
          f"{'Delta':>9}{'95% CI':>22}{'rescued':>9}")
    for slot, d in report["models"].items():
        if d["cpo_decomposed"] is None:
            print(f"{slot:<30}{'--':>5}  no decomposed verdicts")
            continue
        ci = d["delta_decomp_ci"]
        print(f"{slot:<30}{d['n_conflicting_paired']:>5}{d['cpo_baseline']:>10.3f}"
              f"{d['cpo_decomposed']:>12.3f}{d['delta_decomp']:>+9.3f}"
              f"{f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':>22}"
              f"{d['rescued_by_decomposition']:>4}/{d['baseline_cpo_cases']:<4}")
    print(f"\nWrote {args.output_dir / 'e4_decomposition.json'}")


if __name__ == "__main__":
    main()
