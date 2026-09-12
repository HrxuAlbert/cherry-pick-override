#!/usr/bin/env python3
"""Analyze the E1 recognition probe against the parent verdict run.

Implements the estimands, strata and validity gate frozen in
``09_PREANALYSIS_ADDENDUM.md`` §3. Strata are defined by the parent verdict run's
outputs, never by the probe's own answers, so the probe cannot select its own
denominator.

The primary contrast Delta_spec = R_cco - R_pure_correct compares two *disjoint*
sets of claims, so it is an unpaired difference of proportions. The bootstrap
resamples cases with replacement from the union of the two strata and recomputes
both rates inside each resample, which keeps the case as the resampling unit
without pretending the comparison is paired.

No human-audit artifact is read.
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

# 09_PREANALYSIS_ADDENDUM.md §3.4
G1_MIN_PRESERVED = 0.70
G1_MAX_PURE_CORRECT = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        type=Path,
        default=WORKSPACE
        / "outputs/revision_2026/probes/runs/averitec-probe-e1-recognition-20260901-v1"
        / "raw_results_full.jsonl",
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_ok(rows: list[dict[str, Any]], value: Any) -> dict[tuple[str, int], Any]:
    """Append-only logs may hold retries; keep the last successful row per key."""
    out: dict[tuple[str, int], Any] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        got = value(row)
        if got is not None:
            out[(row["model_slot"], int(row["case_id"]))] = got
    return out


def stratum_of(gold: str, verdict: str) -> str | None:
    if gold == "conflicting":
        return "S_cco" if verdict in DIRECTIONAL else "S_preserved"
    if gold in DIRECTIONAL:
        return "S_pure_correct" if verdict == gold else "S_pure_other"
    if gold == "insufficient":
        return "S_insufficient"
    return None


def bootstrap_unpaired(
    a: list[int], b: list[int], resamples: int, seed: int
) -> tuple[float, float]:
    """95% interval for mean(a) - mean(b), resampling cases from the union."""
    if not a or not b:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    pool = [(1, v) for v in a] + [(0, v) for v in b]
    n = len(pool)
    diffs: list[float] = []
    for _ in range(resamples):
        sa: list[int] = []
        sb: list[int] = []
        for _ in range(n):
            side, value = pool[rng.randrange(n)]
            (sa if side else sb).append(value)
        if sa and sb:
            diffs.append(sum(sa) / len(sa) - sum(sb) / len(sb))
    diffs.sort()
    if not diffs:
        return (float("nan"), float("nan"))
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    return (lo, hi)


def main() -> None:
    args = parse_args()
    probe_rows = load_jsonl(args.probe)
    verdict_rows = load_jsonl(args.verdicts)

    recognition = latest_ok(probe_rows, lambda r: (r.get("parsed") or {}).get("recognition"))
    span_backed = latest_ok(
        probe_rows, lambda r: (r.get("parsed") or {}).get("recognition_span_backed")
    )
    verdicts = latest_ok(
        verdict_rows, lambda r: (r.get("parsed") or {}).get("verdict_normal")
    )
    gold: dict[int, str] = {}
    for row in verdict_rows:
        gold[int(row["case_id"])] = row["gold_normal"]

    models = sorted({slot for slot, _ in verdicts})
    report: dict[str, Any] = {
        "probe_run": str(args.probe),
        "verdict_run": str(args.verdicts),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "gate_g1": {"min_preserved": G1_MIN_PRESERVED, "max_pure_correct": G1_MAX_PURE_CORRECT},
        "models": {},
    }

    for slot in models:
        by_stratum: dict[str, list[int]] = defaultdict(list)
        table = Counter()
        missing = 0
        for case_id, g in gold.items():
            verdict = verdicts.get((slot, case_id))
            answer = recognition.get((slot, case_id))
            if verdict is None or answer is None:
                missing += 1
                continue
            stratum = stratum_of(g, verdict)
            if stratum is None:
                continue
            by_stratum[stratum].append(1 if answer == "yes" else 0)
            if g == "conflicting":
                commit = "directional" if verdict in DIRECTIONAL else "non-directional"
                table[(answer, commit)] += 1

        def rate(name: str) -> dict[str, Any]:
            values = by_stratum.get(name, [])
            k, n = sum(values), len(values)
            lo, hi = wilson(k, n)
            return {"k": k, "n": n, "rate": (k / n) if n else None, "wilson": [lo, hi]}

        rates = {name: rate(name) for name in
                 ("S_cco", "S_preserved", "S_pure_correct", "S_pure_other", "S_insufficient")}
        lo, hi = bootstrap_unpaired(
            by_stratum.get("S_cco", []),
            by_stratum.get("S_pure_correct", []),
            BOOTSTRAP_RESAMPLES,
            BOOTSTRAP_SEED,
        )
        delta = (
            None
            if rates["S_cco"]["rate"] is None or rates["S_pure_correct"]["rate"] is None
            else rates["S_cco"]["rate"] - rates["S_pure_correct"]["rate"]
        )
        checks = {
            "preserved_ge_0.70": (rates["S_preserved"]["rate"] or 0) >= G1_MIN_PRESERVED,
            "pure_correct_lt_0.50": (rates["S_pure_correct"]["rate"] or 1) < G1_MAX_PURE_CORRECT,
            "delta_interval_excludes_zero": bool(lo == lo and (lo > 0 or hi < 0)),
        }
        report["models"][slot] = {
            "missing_cases": missing,
            "rates": rates,
            "delta_spec": delta,
            "delta_spec_ci": [lo, hi],
            "g1_checks": checks,
            "g1_pass": all(checks.values()),
            "recognition_by_commitment_gold_conflicting": {
                f"{a}|{c}": n for (a, c), n in sorted(table.items())
            },
            "span_backed_among_yes_in_cco": sum(
                1
                for case_id, g in gold.items()
                if g == "conflicting"
                and verdicts.get((slot, case_id)) in DIRECTIONAL
                and recognition.get((slot, case_id)) == "yes"
                and span_backed.get((slot, case_id)) == "yes"
            ),
        }

    passing = [s for s, d in report["models"].items() if d["g1_pass"]]
    report["g1_summary"] = {
        "models_passing": passing,
        "models_failing": [s for s in models if s not in passing],
        "e1_withdrawn_entirely": len(passing) <= 1,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e1_recognition.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"{'model':<30}{'R_cco':>16}{'R_preserved':>14}{'R_pure_corr':>14}"
          f"{'Delta_spec':>12}{'95% CI':>22}{'G1':>6}")
    for slot, d in report["models"].items():
        r = d["rates"]
        ci = d["delta_spec_ci"]
        print(f"{slot:<30}"
              f"{r['S_cco']['k']:>4}/{r['S_cco']['n']:<3}={r['S_cco']['rate']:.3f}"
              f"{r['S_preserved']['rate']:>14.3f}{r['S_pure_correct']['rate']:>14.3f}"
              f"{d['delta_spec']:>+12.3f}"
              f"{f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':>22}"
              f"{'PASS' if d['g1_pass'] else 'FAIL':>6}")
    print(f"\nG1: {len(passing)}/{len(models)} models pass; "
          f"E1 withdrawn entirely = {report['g1_summary']['e1_withdrawn_entirely']}")
    print(f"\nWrote {args.output_dir / 'e1_recognition.json'}")


if __name__ == "__main__":
    main()
