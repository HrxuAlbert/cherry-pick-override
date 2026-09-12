#!/usr/bin/env python3
"""Inter-reviewer agreement for the blind construct audit.

Reads two reviewer workbooks, matches them by audit ID, and reports the
statistics required by ``02_HUMAN_AUDIT_PROTOCOL_V2.md`` §7: exact five-way
agreement, binary agreement on preserve-conflict versus everything else,
Cohen's kappa for both views, Gwet's AC1 as a prevalence-robust sensitivity
statistic, per-label marginals, and the full disagreement matrix.

It also prints the disagreement rows themselves, because the calibration
round's purpose is to drive a rubric discussion, not to produce a number.

Two guards matter. The script never reads the gold label, the model outputs,
or the ID map, so running it does not break Stage-1 blinding. And per protocol
§4 the calibration round is excluded from the final agreement statistics; the
``--stage`` flag is recorded in the output so a calibration result cannot later
be mistaken for the main-audit one.
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
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl

WORKSPACE = _cpo_workspace()

# The simplified Chinese workbook and the English v2 workbook use different
# headers for the same five-way construct decision.
LABELS = {
    "保留冲突": "preserve_conflict",
    "整体支持": "resolve_support",
    "整体反驳": "resolve_refute",
    "证据不足": "insufficient",
    "无法判断": "unclear",
    "preserve_conflict": "preserve_conflict",
    "resolve_support": "resolve_support",
    "resolve_refute": "resolve_refute",
    "insufficient": "insufficient",
    "unclear": "unclear",
}
ID_HEADERS = ("编号", "audit_id")
DECISION_HEADERS = ("最终判断（必填）", "final_decision")
ORDER = ["preserve_conflict", "resolve_support", "resolve_refute", "insufficient", "unclear"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--stage", choices=("calibration", "main"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_decisions(path: Path) -> dict[str, str]:
    """Return {audit_id: normalized decision}. Unfilled rows are omitted."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for sheet in workbook.sheetnames:
        rows = list(workbook[sheet].iter_rows(values_only=True))
        header_index = next(
            (i for i, row in enumerate(rows) if row and str(row[0]).strip() in ID_HEADERS),
            None,
        )
        if header_index is None:
            continue
        header = [str(c).strip() if c is not None else "" for c in rows[header_index]]
        id_col = next(i for i, h in enumerate(header) if h in ID_HEADERS)
        decision_col = next((i for i, h in enumerate(header) if h in DECISION_HEADERS), None)
        if decision_col is None:
            continue
        out: dict[str, str] = {}
        unknown: set[str] = set()
        for row in rows[header_index + 1:]:
            if not row or row[id_col] in (None, ""):
                continue
            raw = row[decision_col] if decision_col < len(row) else None
            if raw in (None, ""):
                continue
            key = str(raw).strip()
            if key not in LABELS:
                unknown.add(key)
                continue
            out[str(row[id_col]).strip()] = LABELS[key]
        workbook.close()
        if unknown:
            raise SystemExit(f"unrecognized decision values in {path.name}: {sorted(unknown)}")
        return out
    workbook.close()
    raise SystemExit(f"no audit sheet with an id column found in {path}")


def kappa(pairs: list[tuple[str, str]], categories: list[str]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    observed = sum(1 for a, b in pairs if a == b) / n
    marg_a = Counter(a for a, _ in pairs)
    marg_b = Counter(b for _, b in pairs)
    expected = sum((marg_a[c] / n) * (marg_b[c] / n) for c in categories)
    if expected == 1:
        return float("nan")
    return (observed - expected) / (1 - expected)


def gwet_ac1(pairs: list[tuple[str, str]], categories: list[str]) -> float:
    """Prevalence-robust alternative to kappa; kappa collapses when one label dominates."""
    n = len(pairs)
    if not n:
        return float("nan")
    observed = sum(1 for a, b in pairs if a == b) / n
    marg_a = Counter(a for a, _ in pairs)
    marg_b = Counter(b for _, b in pairs)
    pi = {c: (marg_a[c] / n + marg_b[c] / n) / 2 for c in categories}
    k = len(categories)
    expected = sum(pi[c] * (1 - pi[c]) for c in categories) / (k - 1) if k > 1 else 0.0
    if expected == 1:
        return float("nan")
    return (observed - expected) / (1 - expected)


def main() -> None:
    args = parse_args()
    a = read_decisions(args.reviewer_a)
    b = read_decisions(args.reviewer_b)

    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    shared = sorted(set(a) & set(b), key=lambda x: (len(x), x))
    pairs = [(a[i], b[i]) for i in shared]

    five_way = sum(1 for x, y in pairs if x == y) / len(pairs) if pairs else float("nan")
    binary_pairs = [
        ("preserve" if x == "preserve_conflict" else "other",
         "preserve" if y == "preserve_conflict" else "other")
        for x, y in pairs
    ]
    binary = sum(1 for x, y in binary_pairs if x == y) / len(binary_pairs) if binary_pairs else float("nan")

    report: dict[str, Any] = {
        "stage": args.stage,
        "excluded_from_final_statistics": args.stage == "calibration",
        "reviewer_a": {"file": str(args.reviewer_a), "label": args.label_a, "n_filled": len(a)},
        "reviewer_b": {"file": str(args.reviewer_b), "label": args.label_b, "n_filled": len(b)},
        "n_shared_items": len(shared),
        "ids_only_in_a": only_a,
        "ids_only_in_b": only_b,
        "exact_five_way_agreement": five_way,
        "binary_preserve_agreement": binary,
        "kappa_five_way": kappa(pairs, ORDER),
        "kappa_binary": kappa(binary_pairs, ["preserve", "other"]),
        "gwet_ac1_five_way": gwet_ac1(pairs, ORDER),
        "gwet_ac1_binary": gwet_ac1(binary_pairs, ["preserve", "other"]),
        "marginals_a": dict(Counter(x for x, _ in pairs)),
        "marginals_b": dict(Counter(y for _, y in pairs)),
        "disagreements": [
            {"audit_id": i, args.label_a: a[i], args.label_b: b[i]}
            for i in shared
            if a[i] != b[i]
        ],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"stage: {args.stage}"
          + ("  (excluded from final agreement statistics, protocol §4)"
             if args.stage == "calibration" else ""))
    print(f"items: {len(shared)} shared"
          + (f"; only in {args.label_a}: {only_a}" if only_a else "")
          + (f"; only in {args.label_b}: {only_b}" if only_b else ""))
    print()
    print(f"  exact five-way agreement   {five_way:.3f}  ({sum(1 for x,y in pairs if x==y)}/{len(pairs)})")
    print(f"  binary preserve agreement  {binary:.3f}")
    print(f"  Cohen's kappa (five-way)   {report['kappa_five_way']:.3f}")
    print(f"  Cohen's kappa (binary)     {report['kappa_binary']:.3f}")
    print(f"  Gwet's AC1 (five-way)      {report['gwet_ac1_five_way']:.3f}")
    print(f"  Gwet's AC1 (binary)        {report['gwet_ac1_binary']:.3f}")
    print()
    print(f"  {'label':<20}{args.label_a:>6}{args.label_b:>6}")
    for label in ORDER:
        print(f"  {label:<20}{report['marginals_a'].get(label,0):>6}{report['marginals_b'].get(label,0):>6}")
    print()
    if report["disagreements"]:
        print(f"  {len(report['disagreements'])} disagreements to discuss:")
        for d in report["disagreements"]:
            print(f"    id {d['audit_id']:<5} {args.label_a}={d[args.label_a]:<18} {args.label_b}={d[args.label_b]}")
    else:
        print("  no disagreements")
    if args.output:
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
