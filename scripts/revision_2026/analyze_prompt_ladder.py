#!/usr/bin/env python3
"""Estimate the prompt-specification ladder (E5).

The judge prompt used throughout this paper is maximally conflict-favoring: it
enumerates four trigger conditions for CONFLICTING, carries an imperative not to
collapse such a case to a direction, and supplies three demonstrations that all
resolve to CONFLICTING. Every overcommitment rate in the paper is therefore a
claim about what survives that instruction. This script measures how much of the
residual depends on the instruction, by re-running the same fixed sample under
two weaker specifications:

``reduced``  trigger enumeration and demonstrations removed
``neutral``  four label definitions only, no triggers, no imperative, no examples

The primary contrast is paired by claim, per model. As in E4, the reduction in
directional commitment is reported next to what it costs on one-sided evidence,
because a specification that simply produces more CONFLICTING everywhere would
otherwise read as an improvement.

Frozen in 09_PREANALYSIS_ADDENDUM.md §6. No human-audit artifact is read.
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
from collections import Counter
from pathlib import Path
from typing import Any

WORKSPACE = _cpo_workspace()
DIRECTIONAL = {"support", "refute"}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260901
VARIANTS = ("neutral", "reduced")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/probes/runs",
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=WORKSPACE
        / "outputs/revision_2026/current_models_averitec/runs"
        / "averitec-p0-current-models-20260818-v1/raw_results_full.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/probes/analysis",
    )
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


def verdicts_of(rows: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    """Last successful verdict per (model, case); append-only logs may hold retries."""
    out: dict[tuple[str, int], str] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        verdict = (row.get("parsed") or {}).get("verdict_normal")
        if verdict:
            out[(row["model_slot"], int(row["case_id"]))] = verdict
    return out


def paired_bootstrap(pairs: list[tuple[int, int]], resamples: int, seed: int) -> tuple[float, float]:
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


def contrast(
    strong: dict[tuple[str, int], str],
    variant: dict[tuple[str, int], str],
    slot: str,
    cases: list[int],
    positive: str,
) -> dict[str, Any]:
    """Paired rate of `positive` (a predicate name) under strong vs variant."""
    pairs: list[tuple[int, int]] = []
    for case_id in cases:
        a, b = strong.get((slot, case_id)), variant.get((slot, case_id))
        if a is None or b is None:
            continue
        if positive == "directional":
            pairs.append((int(a in DIRECTIONAL), int(b in DIRECTIONAL)))
        else:
            pairs.append((int(a == "conflicting"), int(b == "conflicting")))
    n = len(pairs)
    if not n:
        return {"n": 0}
    before = sum(a for a, _ in pairs) / n
    after = sum(b for _, b in pairs) / n
    lo, hi = paired_bootstrap(pairs, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    return {
        "n": n,
        "strong": before,
        "variant": after,
        "delta": after - before,
        "ci": [lo, hi],
        "strong_wilson": list(wilson(sum(a for a, _ in pairs), n)),
        "variant_wilson": list(wilson(sum(b for _, b in pairs), n)),
    }


def main() -> None:
    args = parse_args()
    verdict_rows = load_jsonl(args.verdicts)
    strong = verdicts_of(verdict_rows)
    gold = {int(r["case_id"]): r["gold_normal"] for r in verdict_rows}
    conflicting = sorted(c for c, g in gold.items() if g == "conflicting")
    pure = sorted(c for c, g in gold.items() if g in DIRECTIONAL)

    report: dict[str, Any] = {
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "case"},
        "n_gold_conflicting": len(conflicting),
        "n_gold_pure": len(pure),
        "variants": {},
    }

    for variant in VARIANTS:
        path = args.runs_dir / f"averitec-probe-e5-{variant}-20260901-v1/raw_results_full.jsonl"
        rows = load_jsonl(path)
        table = verdicts_of(rows)
        entry: dict[str, Any] = {
            "run": str(path),
            "rows": len(rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "models": {},
        }
        for slot in sorted({s for s, _ in table}):
            entry["models"][slot] = {
                "cpo": contrast(strong, table, slot, conflicting, "directional"),
                "false_conflict": contrast(strong, table, slot, pure, "conflicting"),
            }
        report["variants"][variant] = entry

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e5_prompt_ladder.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    for variant in VARIANTS:
        entry = report["variants"][variant]
        print(f"\n=== {variant} specification vs the strong prompt "
              f"({entry['status_counts']}) ===")
        print(f"{'model':<30}{'n':>5}{'CPO strong':>12}{'CPO var':>10}"
              f"{'Delta':>9}{'95% CI':>22}{'FC delta':>10}")
        for slot, d in entry["models"].items():
            c, f = d["cpo"], d["false_conflict"]
            if not c.get("n"):
                continue
            ci = c["ci"]
            print(f"{slot:<30}{c['n']:>5}{c['strong']:>12.3f}{c['variant']:>10.3f}"
                  f"{c['delta']:>+9.3f}{f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':>22}"
                  f"{f['delta']:>+10.3f}")
    print(f"\nWrote {args.output_dir / 'e5_prompt_ladder.json'}")


if __name__ == "__main__":
    main()
