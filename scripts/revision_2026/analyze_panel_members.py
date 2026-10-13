#!/usr/bin/env python3
"""Directional rate on gold-CONFLICTING for the typed panel and for each member.

The historical panel is three judges with majority vote. The cached panel runs
store every member's own verdict, so each member's rate can be set beside the
panel's under both schemas (three-option and typed four-option) without a new
run. The difference panel minus member is paired per claim (bootstrap, 5000
resamples, seed 0, as in the other panel comparisons).

Reads cached predictions only; no API call, no human-audit artifact.
"""

from __future__ import annotations

import argparse
import json
import random
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
N_BOOT, BOOT_SEED = 5000, 0
_AVERITEC = ("m0_commitment/v0.2/outputs/option_a_exp/strengthening/e1_full_4label_utility/raw_results.jsonl",
             "outputs/option_a_exp/strengthening/e1_full_4label_utility/raw_results.jsonl")
_VITAMINC = ("m0_commitment/v0.2/outputs/option_a_exp/strengthening/e4_vitaminc_mixed/raw_results.jsonl",
             "outputs/option_a_exp/strengthening/e4_vitaminc_mixed/raw_results.jsonl")
SCHEMAS = {"three_option": "panel_3judge_3opt", "typed_four_option": "panel_3judge_4opt_strong"}


def first_existing(candidates: tuple[str, ...]) -> Path:
    for rel in candidates:
        if (WORKSPACE / rel).exists():
            return WORKSPACE / rel
    return WORKSPACE / candidates[0]


def paired_ci(pairs: list[tuple[int, int]]) -> list[float]:
    rnd = random.Random(BOOT_SEED)
    n = len(pairs)
    diffs = []
    for _ in range(N_BOOT):
        total = 0
        for _ in range(n):
            a, b = pairs[rnd.randrange(n)]
            total += b - a
        diffs.append(total / n)
    diffs.sort()
    return [diffs[int(0.025 * N_BOOT)], diffs[min(N_BOOT - 1, int(0.975 * N_BOOT))]]


def analyse(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: dict[str, Any] = {}
    for schema, system in SCHEMAS.items():
        sub = [r for r in rows if r.get("system") == system and r.get("gold_normal") == "conflicting"]
        if not sub:
            continue
        panel = [int(r.get("verdict_normal") in DIRECTIONAL) for r in sub]
        entry: dict[str, Any] = {"n": len(sub), "panel": {"k": sum(panel), "rate": sum(panel) / len(sub)}, "members": {}}
        names = [jo.get("model") for jo in sub[0].get("judge_outputs") or []]
        for j, name in enumerate(names):
            member = []
            for r in sub:
                outputs = r.get("judge_outputs") or []
                verdict = ((outputs[j].get("parsed") or {}).get("verdict_normal")) if j < len(outputs) else None
                member.append(int(verdict in DIRECTIONAL))
            pairs = list(zip(member, panel))
            entry["members"][name] = {"k": sum(member), "rate": sum(member) / len(sub),
                                      "panel_minus_member": (sum(panel) - sum(member)) / len(sub),
                                      "ci": paired_ci(pairs)}
        out[schema] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=WORKSPACE / "outputs/revision_2026/audit_independent/panel_members")
    args = parser.parse_args()
    report = {"bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "unit": "case"},
              "averitec": analyse(first_existing(_AVERITEC)), "vitaminc_mixed": analyse(first_existing(_VITAMINC))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "panel_members.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for dataset in ("averitec", "vitaminc_mixed"):
        for schema, e in report[dataset].items():
            print(f"\n{dataset} / {schema} (gold-CONFLICTING n={e['n']}): panel {e['panel']['k']}/{e['n']} = {e['panel']['rate']:.3f}")
            for name, m in e["members"].items():
                print(f"    {name:<22}{m['k']:>4}/{e['n']} = {m['rate']:.3f}   panel - member = {m['panel_minus_member']:+.3f}  [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}]")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
