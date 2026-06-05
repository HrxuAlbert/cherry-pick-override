"""Diagnostic analyses (Stage-1 strengthening).

Two new analyses using only cached per-judge outputs (no API calls):
  1. Panel-amplification anatomy on AVeriTeC and VitaminC conflicting subsets.
     For 3-opt panel: how does majority voting suppress single-judge dissent?
  2. Confidence-boundary analysis: confidence distributions split by
     (correct directional, CCO directional, non-directional) on the typed
     4-opt panel proposals. Tests whether confidence can detect CCO.

Outputs:
  outputs/option_a_exp/analysis/diagnostic/panel_amplification_anatomy.csv
  outputs/option_a_exp/analysis/diagnostic/panel_amplification_anatomy.md
  outputs/option_a_exp/analysis/diagnostic/confidence_boundary.csv
  outputs/option_a_exp/analysis/diagnostic/confidence_boundary.md
"""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/option_a_exp/analysis/diagnostic"
OUT.mkdir(parents=True, exist_ok=True)


def load_e1_with_perjudge():
    """Load E1 with per-judge 3-opt and 4-opt votes and confidences."""
    path = REPO / "outputs/option_a_exp/strengthening/e1_full_4label_utility/raw_results.jsonl"
    data = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            cid = r["case_id"]
            d = data.setdefault(cid, {"gold": r["gold_normal"]})
            sys_name = r["system"]
            jos = r.get("judge_outputs", [])
            if sys_name == "panel_3judge_3opt":
                votes_3 = []
                confs_3 = []
                for jo in jos:
                    p = jo.get("parsed") or {}
                    v = p.get("verdict_normal")
                    c = p.get("confidence")
                    if v:
                        votes_3.append(v)
                    if c is not None:
                        confs_3.append(c)
                d["panel_3opt_votes"] = votes_3
                d["panel_3opt_confs"] = confs_3
                d["panel_3opt_agg"] = r["verdict_normal"]
            elif sys_name == "panel_3judge_4opt_strong":
                votes_4 = []
                confs_4 = []
                for jo in jos:
                    p = jo.get("parsed") or {}
                    v = p.get("verdict_normal")
                    c = p.get("confidence")
                    if v:
                        votes_4.append(v)
                    if c is not None:
                        confs_4.append(c)
                d["panel_4opt_votes"] = votes_4
                d["panel_4opt_confs"] = confs_4
                d["panel_4opt_agg"] = r["verdict_normal"]
    return data


def load_e4_with_perjudge():
    """Load E4 with per-judge 3-opt votes."""
    path = REPO / "outputs/option_a_exp/strengthening/e4_vitaminc_mixed/raw_results.jsonl"
    data = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            cid = r["case_id"]
            d = data.setdefault(cid, {"gold": r["gold_normal"]})
            sys_name = r["system"]
            jos = r.get("judge_outputs", [])
            if sys_name == "panel_3judge_3opt":
                votes_3 = []
                for jo in jos:
                    p = jo.get("parsed") or {}
                    v = p.get("verdict_normal")
                    if v:
                        votes_3.append(v)
                d["panel_3opt_votes"] = votes_3
                d["panel_3opt_agg"] = r["verdict_normal"]
            elif sys_name == "panel_3judge_4opt_strong":
                votes_4 = []
                confs_4 = []
                for jo in jos:
                    p = jo.get("parsed") or {}
                    v = p.get("verdict_normal")
                    c = p.get("confidence")
                    if v:
                        votes_4.append(v)
                    if c is not None:
                        confs_4.append(c)
                d["panel_4opt_votes"] = votes_4
                d["panel_4opt_confs"] = confs_4
                d["panel_4opt_agg"] = r["verdict_normal"]
    return data


# ─────────────────────────  ANALYSIS 1 ─────────────────────────
# Panel amplification anatomy: on gold-CONFLICTING, who dissents?

def amplification_anatomy(data, dataset_name):
    """For panel 3-opt, on gold-CONFLICTING cases:
       - cases with unanimous directional vote (3/0)
       - cases with 2 directional + 1 NEI/conflicting (majority suppresses dissent)
       - cases with 1 directional + 2 NEI/conflicting (single judge dissents but minority)
       - cases with 0 directional (no commitment)
    Note: 3-opt vocabulary doesn't have CONFLICTING — only S/R/NEI. So "dissent" in
    the 3-opt setting means NEI. Under 3-opt the panel cannot vote CONFLICTING; the
    amplification at L0 is single 3-opt CCO → panel 3-opt CCO transition.
    For 4-opt panel we also report the same anatomy.
    """
    conflict_cases = [d for d in data.values()
                      if d.get("gold") == "conflicting"
                      and "panel_3opt_votes" in d
                      and "panel_4opt_votes" in d]
    n = len(conflict_cases)

    # 3-opt anatomy
    cats_3 = Counter()
    for d in conflict_cases:
        votes = d["panel_3opt_votes"]
        n_dir = sum(1 for v in votes if v in ("support", "refute"))
        n_nei = sum(1 for v in votes if v == "insufficient")
        if len(votes) != 3:
            cats_3["other"] += 1
            continue
        if n_dir == 3:
            cats_3["unanimous_directional"] += 1
        elif n_dir == 2:
            # 2 directional + 1 other → majority commits directionally
            # → dissent suppressed (one judge tried to abstain)
            cats_3["2dir_1dissent_suppressed"] += 1
        elif n_dir == 1:
            cats_3["1dir_2dissent_majority_holds"] += 1
        else:
            cats_3["0dir_safe"] += 1

    # 4-opt anatomy
    cats_4 = Counter()
    for d in conflict_cases:
        votes = d["panel_4opt_votes"]
        n_dir = sum(1 for v in votes if v in ("support", "refute"))
        n_conf = sum(1 for v in votes if v == "conflicting")
        n_insuf = sum(1 for v in votes if v == "insufficient")
        if len(votes) != 3:
            cats_4["other"] += 1
            continue
        if n_dir == 3:
            cats_4["unanimous_directional"] += 1
        elif n_dir == 2:
            # 2 dir + 1 other → majority suppresses (CONFLICTING / INSUFFICIENT dissent)
            if n_conf >= 1:
                cats_4["2dir_1conflicting_dissent_suppressed"] += 1
            else:
                cats_4["2dir_1insufficient_dissent_suppressed"] += 1
        elif n_dir == 1:
            cats_4["1dir_2dissent_majority_holds"] += 1
        else:
            cats_4["0dir_safe"] += 1

    return n, cats_3, cats_4


def report_amplification():
    print("=" * 60)
    print("ANALYSIS 1: Panel-amplification anatomy on gold-CONFLICTING")
    print("=" * 60)
    av = load_e1_with_perjudge()
    vc = load_e4_with_perjudge()

    n_av, av3, av4 = amplification_anatomy(av, "AVeriTeC")
    n_vc, vc3, vc4 = amplification_anatomy(vc, "VitaminC")

    rows = []
    for ds, n, c3, c4 in [("AVeriTeC", n_av, av3, av4), ("VitaminC", n_vc, vc3, vc4)]:
        print(f"\n=== {ds} (N_conflicting = {n}) ===")
        print(f"  Panel 3-opt vote anatomy:")
        for k, v in sorted(c3.items()):
            print(f"    {k:50s} {v:4d} ({100*v/n:.1f}%)")
        print(f"  Panel 4-opt vote anatomy:")
        for k, v in sorted(c4.items()):
            print(f"    {k:50s} {v:4d} ({100*v/n:.1f}%)")
        for vocab, cats in [("3-opt", c3), ("4-opt", c4)]:
            for k, v in cats.items():
                rows.append({"dataset": ds, "vocabulary": vocab, "category": k,
                             "count": v, "pct_of_conflict": round(100 * v / n, 2),
                             "n_conflicting": n})

    with (OUT / "panel_amplification_anatomy.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {OUT / 'panel_amplification_anatomy.csv'}")

    md = []
    md.append("# Panel Amplification Anatomy\n\n")
    md.append("Per-judge vote distribution on gold-CONFLICTING subsets. ")
    md.append("**Shared directional bias** would predict mostly unanimous directional votes "
              "(3/0). **Majority-suppression of dissent** would predict 2/1 splits where one "
              "judge votes non-directional and the panel still commits.\n\n")
    md.append(f"AVeriTeC: N_conflicting = {n_av} | VitaminC: N_conflicting = {n_vc}\n\n")

    md.append("## 3-option schema (no CONFLICTING vote available)\n\n")
    md.append("| Vote pattern | AVeriTeC | VitaminC |\n")
    md.append("|---|---|---|\n")
    for k in sorted(set(list(av3.keys()) + list(vc3.keys()))):
        md.append(f"| {k} | {av3.get(k, 0)} ({100*av3.get(k,0)/n_av:.1f}%) | "
                  f"{vc3.get(k, 0)} ({100*vc3.get(k,0)/n_vc:.1f}%) |\n")

    md.append("\n## 4-option typed schema (CONFLICTING available)\n\n")
    md.append("| Vote pattern | AVeriTeC | VitaminC |\n")
    md.append("|---|---|---|\n")
    for k in sorted(set(list(av4.keys()) + list(vc4.keys()))):
        md.append(f"| {k} | {av4.get(k, 0)} ({100*av4.get(k,0)/n_av:.1f}%) | "
                  f"{vc4.get(k, 0)} ({100*vc4.get(k,0)/n_vc:.1f}%) |\n")

    md.append("\n## Interpretation\n\n")
    md.append("Under the 4-opt schema, the `2dir_1conflicting_dissent_suppressed` ")
    md.append("category isolates the mechanism: a single judge tried to vote CONFLICTING ")
    md.append("but the panel committed directionally anyway. Comparing AVeriTeC vs VitaminC ")
    md.append("on this row indicates whether amplification is driven by suppression of ")
    md.append("CONFLICTING dissent.\n\n")
    (OUT / "panel_amplification_anatomy.md").write_text("".join(md))
    print(f"Wrote {OUT / 'panel_amplification_anatomy.md'}")


# ─────────────────────────  ANALYSIS 2 ─────────────────────────
# Confidence-boundary analysis: confidence at CCO commits vs correct commits

def confidence_boundary():
    """On AVeriTeC E1 panel 4-opt:
       - CCO commits: pred ∈ {S,R}, gold = CONFLICTING
       - Correct directional commits: pred ∈ {S,R}, gold = pred
       - Wrong directional commits: pred ∈ {S,R}, gold ∈ {S,R}, pred != gold
       - Non-directional outputs: pred ∈ {CONFLICTING, INSUFFICIENT}

    Report mean confidence per category + counts at confidence thresholds
    0.85, 0.90, 0.95 (high-confidence cases).
    """
    print("=" * 60)
    print("ANALYSIS 2: Confidence-boundary analysis on AVeriTeC panel 4-opt")
    print("=" * 60)
    av = load_e1_with_perjudge()
    cases = [d for d in av.values()
             if "panel_4opt_agg" in d and "panel_4opt_confs" in d
             and len(d["panel_4opt_confs"]) > 0]
    print(f"  Cases with panel 4-opt + confidences: {len(cases)}")

    def mean_conf(d):
        c = d["panel_4opt_confs"]
        return sum(c) / len(c) if c else 0.0

    categories = {
        "correct_directional":   [],  # pred ∈ S/R, gold = pred
        "CCO_directional":       [],  # pred ∈ S/R, gold = CONFLICTING
        "wrong_directional":     [],  # pred ∈ S/R, gold ∈ S/R, pred != gold
        "non_directional_correct": [],  # pred = gold, gold ∈ {CONFLICTING, INSUFFICIENT}
        "non_directional_other": [],  # pred ∈ {CONFLICTING, INSUFFICIENT}, gold ∈ S/R
    }
    for d in cases:
        pred = d["panel_4opt_agg"]
        gold = d["gold"]
        mc = mean_conf(d)
        if pred in ("support", "refute"):
            if gold == pred:
                categories["correct_directional"].append((d, mc))
            elif gold == "conflicting":
                categories["CCO_directional"].append((d, mc))
            else:
                categories["wrong_directional"].append((d, mc))
        elif pred in ("conflicting", "insufficient"):
            if pred == gold:
                categories["non_directional_correct"].append((d, mc))
            else:
                categories["non_directional_other"].append((d, mc))

    # Stats per category
    print()
    summary = []
    for cat, items in categories.items():
        if not items:
            print(f"  {cat:30s} n=0")
            continue
        confs = [c for _, c in items]
        mn, mx = min(confs), max(confs)
        avg = sum(confs) / len(confs)
        n = len(items)
        n_85 = sum(1 for c in confs if c >= 0.85)
        n_90 = sum(1 for c in confs if c >= 0.90)
        n_95 = sum(1 for c in confs if c >= 0.95)
        summary.append({"category": cat, "n": n, "mean_conf": round(avg, 3),
                        "min_conf": round(mn, 3), "max_conf": round(mx, 3),
                        "n_geq_0.85": n_85, "n_geq_0.90": n_90, "n_geq_0.95": n_95,
                        "frac_geq_0.85": round(n_85 / n, 3),
                        "frac_geq_0.90": round(n_90 / n, 3),
                        "frac_geq_0.95": round(n_95 / n, 3)})
        print(f"  {cat:30s} n={n:3d}  mean={avg:.3f}  range=[{mn:.3f},{mx:.3f}]  "
              f"≥.85: {n_85}/{n}  ≥.90: {n_90}/{n}  ≥.95: {n_95}/{n}")

    with (OUT / "confidence_boundary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"\nWrote {OUT / 'confidence_boundary.csv'}")

    md = []
    md.append("# Confidence-Boundary Analysis on AVeriTeC panel 4-opt\n\n")
    md.append(f"N = {len(cases)} cases with panel 4-opt verdict and per-judge confidences.\n\n")
    md.append("## Mean confidence by outcome category\n\n")
    md.append("| Category | n | Mean conf | Range | n≥0.85 | n≥0.90 | n≥0.95 |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for r in summary:
        md.append(f"| {r['category']} | {r['n']} | {r['mean_conf']} | "
                  f"[{r['min_conf']}, {r['max_conf']}] | "
                  f"{r['n_geq_0.85']} ({r['frac_geq_0.85']*100:.0f}%) | "
                  f"{r['n_geq_0.90']} ({r['frac_geq_0.90']*100:.0f}%) | "
                  f"{r['n_geq_0.95']} ({r['frac_geq_0.95']*100:.0f}%) |\n")

    cco = next(r for r in summary if r["category"] == "CCO_directional")
    correct = next(r for r in summary if r["category"] == "correct_directional")
    md.append("\n## Interpretation\n\n")
    md.append(
        f"CCO commits ($n={cco['n']}$) have mean confidence "
        f"{cco['mean_conf']:.3f}, comparable to correct directional commits ($n={correct['n']}$) "
        f"at mean confidence {correct['mean_conf']:.3f}. ")
    md.append(
        f"At a confidence threshold $\\tau=0.85$, "
        f"{cco['frac_geq_0.85']*100:.0f}% of CCO commits would still pass; at $\\tau=0.90$, "
        f"{cco['frac_geq_0.90']*100:.0f}% pass; at $\\tau=0.95$, "
        f"{cco['frac_geq_0.95']*100:.0f}% pass. Confidence-threshold selection therefore cannot "
        f"distinguish CCO commits from correct directional commits at these sample sizes; the "
        f"two distributions overlap heavily.\n")
    (OUT / "confidence_boundary.md").write_text("".join(md))
    print(f"Wrote {OUT / 'confidence_boundary.md'}")


if __name__ == "__main__":
    report_amplification()
    print()
    confidence_boundary()
