#!/usr/bin/env python3
"""Self-enumerated counterevidence in the two-stage (decomposition) condition.

In that condition a judge first lists the evidence items supporting the claim
and those refuting it, and then returns a verdict with its own list in front of
it. For every directional verdict this script asks whether the list already
contained at least one item on the other side.

Two strata are reported per judge:

``cpo``      gold CONFLICTING, stage-2 verdict directional
``control``  gold SUPPORTS or REFUTES, stage-2 verdict equal to gold

The second is the baseline the first has to be read against: how often a judge
lists an opposing item on a one-sided claim it then gets right.

Reads the raw decomposition runs only; no API call, no human-audit artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _workspace(_f=__file__) -> Path:
    here = Path(_f).resolve()
    for parent in here.parents:
        if (parent / "outputs" / "revision_2026").is_dir():
            return parent
    return here.parents[4]


WORKSPACE = _workspace()
DIRECTIONAL = {"support", "refute"}
RUNS = (
    "outputs/revision_2026/probes/runs/averitec-probe-e4-decompose-20260901-v1/raw_results_full.jsonl",
    "outputs/revision_2026/probes/runs_anthropic_resume/averitec-probe-e4-decompose-20260901-v1/raw_results_full.jsonl",
)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(centre - margin) / denom, (centre + margin) / denom]


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def opposing_items(decomposition: dict[str, Any], verdict: str) -> int:
    key, count = ("refuting_items", "n_refuting") if verdict == "support" else ("supporting_items", "n_supporting")
    items = decomposition.get(key)
    return len(items) if isinstance(items, list) else int(decomposition.get(count) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "outputs/revision_2026/probes/analysis")
    args = parser.parse_args()

    stage1: dict[tuple[str, int], dict[str, Any]] = {}
    stage2: dict[tuple[str, int], str] = {}
    gold: dict[tuple[str, int], str] = {}
    for rel in RUNS:
        path = WORKSPACE / rel
        if not path.exists():
            continue
        for row in load(path):                      # append-only logs: the last successful row wins
            if row.get("status") != "ok":
                continue
            key = (row["model_slot"], int(row["case_id"]))
            gold[key] = row["gold_normal"]
            parsed = row.get("parsed") or {}
            if row.get("stage") == "stage1_decompose":
                stage1[key] = parsed
            elif row.get("stage") == "stage2_verdict" and parsed.get("verdict_normal"):
                stage2[key] = parsed["verdict_normal"]

    report: dict[str, Any] = {}
    for slot in sorted({k[0] for k in stage2}):
        def stratum(keep) -> list[int]:
            return [opposing_items(stage1[k], v) for k, v in stage2.items()
                    if k[0] == slot and k in stage1 and keep(gold[k], v)]
        cpo = stratum(lambda g, v: g == "conflicting" and v in DIRECTIONAL)
        control = stratum(lambda g, v: g in DIRECTIONAL and v == g)
        with_items = [x for x in cpo if x > 0]
        k_c, n_c = sum(x > 0 for x in control), len(control)
        report[slot] = {
            "k": len(with_items), "n": len(cpo), "rate": len(with_items) / len(cpo) if cpo else None,
            "wilson": wilson(len(with_items), len(cpo)),
            "median_items": statistics.median(with_items) if with_items else None,
            "control": {"k": k_c, "n": n_c, "rate": k_c / n_c if n_c else None, "wilson": wilson(k_c, n_c)},
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "e4_counterevidence.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'model':<30}{'CPO: opposing listed':>24}{'median':>8}{'control':>18}")
    for slot, d in report.items():
        c = d["control"]
        print(f"{slot:<30}{d['k']:>10}/{d['n']:<3}={d['rate']:.3f}{d['median_items']!s:>12}{c['k']:>10}/{c['n']:<3}={c['rate']:.3f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
