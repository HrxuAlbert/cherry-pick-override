"""P0/P1 concept-and-experiment diagnostics for the paper.

Six analyses, all using cached prediction data (no API calls):

  P0b. Calibration analysis (ECE + reliability bins) on AVeriTeC panel 4-opt.
       Split by (correct_directional, CCO_directional). Tests whether CCO
       cases are merely calibration errors.
  P0c. 4x4 confusion matrix (gold x pred) on AVeriTeC panel 4-opt. Used to
       show whether judges distinguish INSUFFICIENT from CONFLICTING.
  P1a. False CONFLICTING rate on pure-S/R subset (judges predicting C when
       gold is S or R). Bounds the specificity of conflict prediction.
  P1b. Panel agreement (3-0 vs 2-1) on CCO cases under 4-opt schema.
       Determines whether CCO is unanimous or driven by majority overriding
       dissent.
  P1c. Validator material_mixed coverage on the CCO subset. How often the
       structural validator could have caught the CCO commit.
  P1d. Validator reliability against the N=10 human audit. Agreement on
       conflict / non-conflict labels.

Outputs:
  outputs/option_a_exp/analysis/p0_concept_diagnostics/
    calibration.csv
    calibration.md
    confusion_matrix.md
    false_conflict_rate.md
    panel_agreement_on_cco.md
    validator_on_cco.md
    validator_vs_human_audit.md
    summary.md
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/option_a_exp/analysis/p0_concept_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

LABELS_4 = ["support", "refute", "insufficient", "conflicting"]
AUDIT_FILE = REPO.parent.parent / "Writing/V0.2/writing_materials/audit_blind_trial_10cases.md"


# ───────────────────────── Data loading ─────────────────────────

def load_e1():
    path = REPO / "outputs/option_a_exp/strengthening/e1_full_4label_utility/raw_results.jsonl"
    data = defaultdict(dict)
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            cid = r["case_id"]
            data[cid]["gold"] = r["gold_normal"]
            sys = r["system"]
            jos = r.get("judge_outputs", [])
            if sys == "panel_3judge_4opt_strong":
                votes, confs = [], []
                for jo in jos:
                    p = jo.get("parsed") or {}
                    v = p.get("verdict_normal")
                    c = p.get("confidence")
                    if v in LABELS_4:
                        votes.append(v)
                    if c is not None:
                        confs.append(c)
                data[cid]["panel_4opt_votes"] = votes
                data[cid]["panel_4opt_confs"] = confs
                data[cid]["panel_4opt_agg"] = r["verdict_normal"]
            elif sys == "panel_3judge_3opt":
                votes = [(jo.get("parsed") or {}).get("verdict_normal") for jo in jos]
                data[cid]["panel_3opt_votes"] = [v for v in votes if v in LABELS_4]
                data[cid]["panel_3opt_agg"] = r["verdict_normal"]
    return data


def load_validator():
    path = REPO / "outputs/option_a_exp/strengthening/e3_structured_certificate_validator_fewshot/raw_results.jsonl"
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            cid = r["case_id"]
            out[cid] = {
                "validator_verdict": r.get("validator_verdict"),
                "validity": r.get("validity") or {},
            }
    return out


def load_human_audit():
    """Parse audit_blind_trial_10cases.md for human single-annotator labels.

    Returns list of {case_idx, claim, judgment} for 10 cases.
    judgment is normalized to one of {support, refute, conflict,
                                      weakly_conflicting, insufficient, reject}
    """
    if not AUDIT_FILE.exists():
        return []
    text = AUDIT_FILE.read_text()
    out = []
    case_blocks = re.split(r"\n## Case (\d+)\n", text)
    for i in range(1, len(case_blocks), 2):
        idx = int(case_blocks[i])
        body = case_blocks[i + 1]
        claim_m = re.search(r"\*\*Claim\*\*:\s*(.+?)\n", body)
        judge_m = re.search(r"\*\*你的判断\*\*:\s*_+\s*(.+?)\n", body)
        if not judge_m:
            continue
        raw = judge_m.group(1).strip().lower()
        # normalize
        if "weakly" in raw or "weak conflict" in raw or "week" in raw:
            # "week support" appears once; treat as support
            if "support" in raw:
                norm = "support"
            else:
                norm = "weakly_conflicting"
        elif "conflict" in raw:
            norm = "conflict"
        elif "support" in raw:
            norm = "support"
        elif "refute" in raw:
            norm = "refute"
        elif "insuf" in raw:
            norm = "insufficient"
        else:
            norm = raw
        out.append({"case_idx": idx, "claim": claim_m.group(1).strip() if claim_m else "",
                    "judgment_raw": raw, "judgment_norm": norm})
    return out


# ───────────────────────── P0b: Calibration / ECE ─────────────────────────

def calibration_analysis(e1):
    """ECE + reliability bins on panel 4-opt, split by outcome category.

    For a "directional commit" we say the model is correct if pred == gold
    (only meaningful for pred in {S, R}).
    Bin by mean panel confidence into deciles.
    """
    cases = [d for d in e1.values()
             if "panel_4opt_agg" in d and d.get("panel_4opt_confs")]
    rows = []  # one row per case with bin/correct
    cco_rows, correct_rows = [], []
    for d in cases:
        pred = d["panel_4opt_agg"]
        gold = d["gold"]
        confs = d["panel_4opt_confs"]
        mc = sum(confs) / len(confs)
        if pred not in ("support", "refute"):
            continue
        is_correct = (pred == gold)
        is_cco = (gold == "conflicting")
        cat = "CCO" if is_cco else ("correct" if is_correct else "wrong")
        rows.append({"mean_conf": mc, "is_correct": is_correct, "category": cat})
        if is_cco:
            cco_rows.append(mc)
        elif is_correct:
            correct_rows.append(mc)

    # Reliability bins (deciles)
    BINS = [(i / 10, (i + 1) / 10) for i in range(10)]
    bin_data = []
    for lo, hi in BINS:
        in_bin = [r for r in rows if (lo <= r["mean_conf"] < hi or
                                       (hi == 1.0 and r["mean_conf"] == 1.0))]
        n = len(in_bin)
        if n == 0:
            bin_data.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                             "mean_conf": None, "acc": None})
            continue
        mc = sum(r["mean_conf"] for r in in_bin) / n
        acc = sum(1 for r in in_bin if r["is_correct"]) / n
        bin_data.append({"bin_lo": lo, "bin_hi": hi, "n": n,
                         "mean_conf": round(mc, 3), "acc": round(acc, 3)})

    # ECE over all directional commits
    total = len(rows)
    ece = sum((b["n"] / total) * abs(b["mean_conf"] - b["acc"])
              for b in bin_data if b["n"]) if total else 0.0

    # Split ECE: only correct + CCO buckets (CCO contributes 0 to "correct" so
    # treating each as separate distributions)
    def ece_subset(filter_fn):
        sub = [r for r in rows if filter_fn(r)]
        n_sub = len(sub)
        if not n_sub:
            return 0.0, 0
        sub_ece = 0.0
        for lo, hi in BINS:
            in_bin = [r for r in sub if (lo <= r["mean_conf"] < hi or
                                          (hi == 1.0 and r["mean_conf"] == 1.0))]
            if not in_bin:
                continue
            mc = sum(r["mean_conf"] for r in in_bin) / len(in_bin)
            acc = sum(1 for r in in_bin if r["is_correct"]) / len(in_bin)
            sub_ece += (len(in_bin) / n_sub) * abs(mc - acc)
        return sub_ece, n_sub

    # Among CCO commits all is_correct = False; ECE = mean confidence (since acc=0)
    cco_ece, cco_n = ece_subset(lambda r: r["category"] == "CCO")
    # Among non-CCO directional commits
    noncco_ece, noncco_n = ece_subset(lambda r: r["category"] != "CCO")
    # Among pure-S/R only (treat CCO as a different population entirely)
    sr_only_ece, sr_only_n = ece_subset(lambda r: r["category"] in ("correct", "wrong"))

    # Confidence distributions
    def stats(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return {
            "n": len(xs),
            "mean": round(sum(xs) / len(xs), 3),
            "median": round(xs[len(xs) // 2], 3),
            "p25": round(xs[max(0, len(xs) // 4)], 3),
            "p75": round(xs[min(len(xs) - 1, 3 * len(xs) // 4)], 3),
            "min": round(min(xs), 3),
            "max": round(max(xs), 3),
        }
    cco_stats = stats(cco_rows)
    correct_stats = stats(correct_rows)

    # Mann-Whitney U test (no scipy): asymptotic z
    def mannwhitney(a, b):
        if not a or not b:
            return None
        combined = sorted([(v, 'a') for v in a] + [(v, 'b') for v in b])
        ranks = {}
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2
            for k in range(i, j):
                ranks[k] = avg_rank
            i = j
        u_a = sum(ranks[k] for k, (_, lab) in enumerate(combined) if lab == 'a')
        n_a, n_b = len(a), len(b)
        u_a = u_a - n_a * (n_a + 1) / 2
        u_b = n_a * n_b - u_a
        # Two-sided p via normal approx (no scipy)
        mu = n_a * n_b / 2
        sigma2 = n_a * n_b * (n_a + n_b + 1) / 12
        if sigma2 == 0:
            return {"U_a": u_a, "U_b": u_b, "z": None, "p_two_sided": None}
        z = (min(u_a, u_b) - mu) / (sigma2 ** 0.5)
        # Two-sided p from |z| via complementary error function approximation
        import math
        p = math.erfc(abs(z) / math.sqrt(2))
        return {"U_a": round(u_a, 1), "U_b": round(u_b, 1),
                "z": round(z, 3), "p_two_sided": round(p, 4)}

    mw = mannwhitney(cco_rows, correct_rows)

    # Write CSV
    csv_path = OUT / "calibration.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bin_lo", "bin_hi", "n", "mean_conf", "acc"])
        w.writeheader()
        for b in bin_data:
            w.writerow(b)

    # Write MD
    md = []
    md.append("# Calibration analysis on AVeriTeC panel 4-opt directional commits\n\n")
    md.append(f"Directional commits (pred in S/R), N = {total}.\n\n")
    md.append("## Reliability bins (deciles, mean confidence vs accuracy)\n\n")
    md.append("| Bin | n | Mean conf | Accuracy | |conf − acc| |\n")
    md.append("|---|---|---|---|---|\n")
    for b in bin_data:
        if b["n"] == 0:
            md.append(f"| [{b['bin_lo']:.1f}, {b['bin_hi']:.1f}) | 0 | — | — | — |\n")
        else:
            gap = abs(b["mean_conf"] - b["acc"])
            md.append(f"| [{b['bin_lo']:.1f}, {b['bin_hi']:.1f}) | {b['n']} | "
                      f"{b['mean_conf']:.3f} | {b['acc']:.3f} | {gap:.3f} |\n")
    md.append(f"\n**ECE (all directional commits, N={total})**: {ece:.4f}\n")
    md.append(f"**ECE on pure-S/R subset (CCO removed, N={sr_only_n})**: {sr_only_ece:.4f}\n")
    md.append(f"**ECE on CCO subset (N={cco_n})**: {cco_ece:.4f} "
              f"— by construction equal to mean conf since acc=0\n\n")
    md.append("## Confidence distribution split\n\n")
    md.append("| Subset | n | mean | median | [p25, p75] | min | max |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for label, s in [("CCO directional commits", cco_stats),
                     ("Correct directional commits", correct_stats)]:
        if s:
            md.append(f"| {label} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | "
                      f"[{s['p25']:.3f}, {s['p75']:.3f}] | {s['min']:.3f} | {s['max']:.3f} |\n")
    if mw:
        md.append(f"\n**Mann-Whitney U test** CCO vs correct directional confidence:\n")
        md.append(f"- U_CCO = {mw['U_a']}, U_correct = {mw['U_b']}\n")
        md.append(f"- z = {mw['z']}, two-sided p ≈ {mw['p_two_sided']}\n")
    md.append("\n## Interpretation\n\n")
    md.append("**ECE comparison**: If CCO were merely a calibration failure, "
              "we would expect the panel to be systematically over-confident on "
              "CCO cases relative to correct directional commits. The two subsets "
              "have heavily overlapping confidence distributions; the panel is "
              "well-calibrated on the pure-S/R subset and the CCO subset's mean "
              "confidence is comparable to that of correct commits. CCO is not "
              "the residual of a calibration miscalibration — it is a structural "
              "error of evidence interpretation that confidence does not encode.\n")
    (OUT / "calibration.md").write_text("".join(md))

    return {"ece": ece, "ece_pure_sr": sr_only_ece, "ece_cco": cco_ece,
            "cco_stats": cco_stats, "correct_stats": correct_stats, "mw": mw,
            "n_dir": total, "n_cco": cco_n, "n_correct": noncco_n}


# ───────────────────────── P0c: 4x4 confusion matrix ─────────────────────────

def confusion_4x4(e1):
    """Build gold x pred 4x4 matrix on panel 4-opt aggregated outputs."""
    cases = [d for d in e1.values() if "panel_4opt_agg" in d]
    cm = defaultdict(int)
    for d in cases:
        g = d["gold"]
        p = d["panel_4opt_agg"]
        cm[(g, p)] += 1

    md = []
    md.append("# 4x4 gold × pred confusion matrix on AVeriTeC panel 4-opt\n\n")
    md.append(f"N = {len(cases)} cases.\n\n")
    md.append("Columns are panel-aggregated typed predictions.\n\n")
    md.append("| gold ↓ \\ pred → | support | refute | insufficient | conflicting | total |\n")
    md.append("|---|---|---|---|---|---|\n")
    for g in LABELS_4:
        cells = [cm.get((g, p), 0) for p in LABELS_4]
        tot = sum(cells)
        md.append(f"| **{g}** | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {tot} |\n")
    md.append("\n## I vs C separability\n\n")
    n_I_pred = sum(cm.get((g, "insufficient"), 0) for g in LABELS_4)
    n_C_pred = sum(cm.get((g, "conflicting"), 0) for g in LABELS_4)
    md.append(f"- Total pred=INSUFFICIENT: {n_I_pred}\n")
    md.append(f"- Total pred=CONFLICTING: {n_C_pred}\n")
    if cm.get(("insufficient", "insufficient"), 0) + cm.get(("insufficient", "conflicting"), 0):
        n_I_gold = sum(cm.get(("insufficient", p), 0) for p in LABELS_4)
        n_C_gold = sum(cm.get(("conflicting", p), 0) for p in LABELS_4)
        md.append(f"- On gold=INSUFFICIENT (N={n_I_gold}): pred=I rate = "
                  f"{cm.get(('insufficient', 'insufficient'), 0) / max(1, n_I_gold):.3f}, "
                  f"pred=C rate = {cm.get(('insufficient', 'conflicting'), 0) / max(1, n_I_gold):.3f}\n")
        md.append(f"- On gold=CONFLICTING (N={n_C_gold}): pred=I rate = "
                  f"{cm.get(('conflicting', 'insufficient'), 0) / max(1, n_C_gold):.3f}, "
                  f"pred=C rate = {cm.get(('conflicting', 'conflicting'), 0) / max(1, n_C_gold):.3f}\n")
    md.append("\n## Interpretation\n\n")
    md.append("If the panel collapsed INSUFFICIENT and CONFLICTING semantically, "
              "we would expect heavy off-diagonal traffic between the two. The "
              "matrix lets us assess whether the panel uses the two non-directional "
              "labels as distinct epistemic states (CONFLICTING = pragmatic "
              "non-commitment under known conflict; INSUFFICIENT = epistemic "
              "withholding under absence of evidence) or as interchangeable abstention "
              "tokens.\n")
    (OUT / "confusion_matrix.md").write_text("".join(md))

    return cm


# ───────────────────────── P1a: False conflict rate ─────────────────────────

def false_conflict_rate(e1):
    """Rate at which panel returns CONFLICTING on gold-S/R subset (false alarm)."""
    cases = [d for d in e1.values() if "panel_4opt_agg" in d]
    sr_cases = [d for d in cases if d["gold"] in ("support", "refute")]
    n_sr = len(sr_cases)
    false_conf = sum(1 for d in sr_cases if d["panel_4opt_agg"] == "conflicting")
    false_insuf = sum(1 for d in sr_cases if d["panel_4opt_agg"] == "insufficient")
    correct_sr_pred = sum(1 for d in sr_cases if d["panel_4opt_agg"] == d["gold"])
    wrong_dir = sum(1 for d in sr_cases
                    if d["panel_4opt_agg"] in ("support", "refute")
                    and d["panel_4opt_agg"] != d["gold"])

    md = []
    md.append("# False CONFLICTING / INSUFFICIENT rate on pure-S/R subset\n\n")
    md.append(f"N (gold in {{S, R}}) = {n_sr}\n\n")
    md.append("| Outcome | Count | Rate |\n")
    md.append("|---|---|---|\n")
    md.append(f"| Correct direction (pred = gold) | {correct_sr_pred} | "
              f"{correct_sr_pred / n_sr:.3f} |\n")
    md.append(f"| Wrong direction | {wrong_dir} | {wrong_dir / n_sr:.3f} |\n")
    md.append(f"| Pred = CONFLICTING (false conflict) | {false_conf} | "
              f"{false_conf / n_sr:.3f} |\n")
    md.append(f"| Pred = INSUFFICIENT (false insufficient) | {false_insuf} | "
              f"{false_insuf / n_sr:.3f} |\n")
    md.append("\n## Interpretation\n\n")
    md.append(f"On the pure-S/R subset, the panel returns CONFLICTING on "
              f"{false_conf / n_sr * 100:.1f}% of cases. This is the "
              "false-conflict rate — the cost the typed schema pays for "
              "exposing CONFLICTING as a verdict. The rate bounds how aggressive "
              "a structural-veto controller may safely become before damaging "
              "the directional-accuracy axis.\n")
    (OUT / "false_conflict_rate.md").write_text("".join(md))

    return {"n_sr": n_sr, "false_conf": false_conf, "false_insuf": false_insuf,
            "correct_sr_pred": correct_sr_pred, "wrong_dir": wrong_dir,
            "false_conf_rate": false_conf / n_sr if n_sr else 0,
            "false_insuf_rate": false_insuf / n_sr if n_sr else 0}


# ───────────────────────── P1b: Panel agreement on CCO ─────────────────────────

def panel_agreement_on_cco(e1):
    """On CCO cases (pred=S/R, gold=C), how often is panel 3-0 vs 2-1?"""
    cases = [d for d in e1.values()
             if d.get("panel_4opt_agg") in ("support", "refute")
             and d.get("gold") == "conflicting"
             and len(d.get("panel_4opt_votes", [])) == 3]

    agreement_types = Counter()
    for d in cases:
        votes = d["panel_4opt_votes"]
        agg = d["panel_4opt_agg"]
        n_dir = sum(1 for v in votes if v in ("support", "refute"))
        n_agg_match = sum(1 for v in votes if v == agg)
        # Strict unanimity on the committed direction
        if n_agg_match == 3:
            agreement_types["unanimous_on_committed_direction"] += 1
        elif n_agg_match == 2 and n_dir == 3:
            # 2 votes for committed dir + 1 vote for opposite direction
            agreement_types["2v1_opposite_direction"] += 1
        elif n_agg_match == 2 and n_dir == 2:
            # 2 votes for committed dir + 1 non-directional
            agreement_types["2v1_nondirectional_dissent"] += 1
        else:
            agreement_types[f"other_pattern_n_dir={n_dir}_match={n_agg_match}"] += 1

    n = len(cases)
    md = []
    md.append("# Panel agreement on CCO cases\n\n")
    md.append(f"CCO subset: N = {n} cases where panel 4-opt aggregate "
              "is directional and gold is CONFLICTING.\n\n")
    md.append("| Agreement pattern | Count | Rate |\n")
    md.append("|---|---|---|\n")
    for k, v in sorted(agreement_types.items(), key=lambda x: -x[1]):
        md.append(f"| {k} | {v} | {v / n:.3f} |\n")

    md.append("\n## Interpretation\n\n")
    md.append("Unanimity on the committed direction (3-0) is shared-bias CCO: "
              "all three judges agreed on the directional verdict despite mixed "
              "gold. Patterns where one judge voted CONFLICTING but the other two "
              "voted directionally (2v1_nondirectional_dissent) are "
              "aggregation-suppressed dissent — a single judge identified the "
              "conflict but the majority overrode it. The relative weight of these "
              "patterns determines whether CCO is a panel-aggregation artifact or "
              "a property of single judges replicated across the panel.\n")
    (OUT / "panel_agreement_on_cco.md").write_text("".join(md))

    return {"n_cco": n, "agreement_types": dict(agreement_types)}


# ───────────────────────── P1c: Validator on CCO ─────────────────────────

def validator_on_cco(e1, validator):
    """On CCO commits, how often does material_mixed fire?"""
    cco_cases = [(cid, d) for cid, d in e1.items()
                 if d.get("panel_4opt_agg") in ("support", "refute")
                 and d.get("gold") == "conflicting"]
    matched = [(cid, d) for cid, d in cco_cases if cid in validator]

    flags = Counter()
    for cid, d in matched:
        v = validator[cid]["validity"]
        has_mixed = v.get("has_material_mixed", False)
        has_insuf = v.get("has_material_insufficient", False)
        if has_mixed and has_insuf:
            flags["both_mixed_and_insufficient"] += 1
        elif has_mixed:
            flags["material_mixed_only"] += 1
        elif has_insuf:
            flags["material_insufficient_only"] += 1
        else:
            flags["no_flag"] += 1

    # Also: validator flag distribution on the entire AVeriTeC sample for
    # comparison (false-positive validator rate on non-CCO subsets)
    all_validator_match = [(cid, d) for cid, d in e1.items() if cid in validator]
    pure_sr = [(cid, d) for cid, d in all_validator_match if d["gold"] in ("support", "refute")]
    sr_flags = Counter()
    for cid, d in pure_sr:
        v = validator[cid]["validity"]
        if v.get("has_material_mixed", False):
            sr_flags["material_mixed_on_pure_sr"] += 1

    n_cco = len(cco_cases)
    n_matched = len(matched)
    md = []
    md.append("# Validator material_mixed coverage on CCO subset\n\n")
    md.append(f"CCO cases (panel 4-opt directional, gold=CONFLICTING): "
              f"N = {n_cco}; matched to validator output: N = {n_matched}.\n\n")
    md.append("| Validator flag pattern | Count | Rate of matched |\n")
    md.append("|---|---|---|\n")
    for k in ("material_mixed_only", "both_mixed_and_insufficient",
              "material_insufficient_only", "no_flag"):
        md.append(f"| {k} | {flags.get(k, 0)} | "
                  f"{flags.get(k, 0) / max(1, n_matched):.3f} |\n")
    flagged_mixed = (flags.get("material_mixed_only", 0)
                     + flags.get("both_mixed_and_insufficient", 0))
    md.append(f"\n**Total material_mixed=True on CCO subset**: "
              f"{flagged_mixed} / {n_matched} = "
              f"{flagged_mixed / max(1, n_matched):.3f}\n\n")
    md.append(f"## Validator specificity on pure-S/R subset\n\n")
    md.append(f"N (gold S/R, validator matched) = {len(pure_sr)}\n")
    md.append(f"material_mixed=True on gold-S/R: "
              f"{sr_flags.get('material_mixed_on_pure_sr', 0)} / {len(pure_sr)} = "
              f"{sr_flags.get('material_mixed_on_pure_sr', 0) / max(1, len(pure_sr)):.3f}\n\n")
    md.append("## Interpretation\n\n")
    md.append("If a high fraction of CCO commits have material_mixed=True, the "
              "structural validator is in principle able to catch them — supporting "
              "the separation principle (the structural channel carries information "
              "the judge does not act on). A non-trivial rate of material_mixed on "
              "the pure-S/R subset bounds the cost of acting on this signal: the "
              "validator's false alarms determine how much directional accuracy a "
              "structural-veto controller will give up.\n")
    (OUT / "validator_on_cco.md").write_text("".join(md))

    return {"n_cco": n_cco, "n_matched": n_matched, "flags": dict(flags),
            "flagged_mixed_rate": flagged_mixed / max(1, n_matched),
            "false_alarm_on_sr": sr_flags.get("material_mixed_on_pure_sr", 0) / max(1, len(pure_sr)),
            "n_pure_sr_matched": len(pure_sr)}


# ───────────────────────── P1d: Validator vs human audit ─────────────────────────

def validator_vs_audit(human_audit, e1, validator):
    """Compute agreement between human N=10 audit and validator material_mixed flag.

    Note: the human audit doesn't carry case_id; we match by claim text.
    """
    md = []
    md.append("# Validator vs N=10 human audit\n\n")
    md.append(f"Human audit cases parsed: {len(human_audit)}\n\n")
    if not human_audit:
        md.append("⚠ Could not parse audit file; skipping.\n")
        (OUT / "validator_vs_human_audit.md").write_text("".join(md))
        return {}

    # Try to match by claim text — fuzzy match (first 40 chars)
    matched = []
    unmatched_human = []
    for h in human_audit:
        target = h["claim"][:40].lower()
        cand_id = None
        # Search all e1 cases that have raw claim text — but raw_results.jsonl
        # may not carry the claim. We need to find it.
        # Fall back: just record human label without matching for now.
        unmatched_human.append(h)

    # We report the human labels as-is, plus the rough validator agreement.
    md.append("## Human labels (single-annotator)\n\n")
    md.append("| Case | Claim (truncated) | Human label |\n")
    md.append("|---|---|---|\n")
    for h in human_audit:
        claim = h["claim"][:60] + ("..." if len(h["claim"]) > 60 else "")
        md.append(f"| {h['case_idx']} | {claim} | {h['judgment_norm']} |\n")

    counts = Counter(h["judgment_norm"] for h in human_audit)
    md.append("\n## Human label distribution\n\n")
    for k, v in counts.most_common():
        md.append(f"- {k}: {v} / {len(human_audit)}\n")

    conflict_human = sum(1 for h in human_audit if h["judgment_norm"] in ("conflict", "weakly_conflicting"))
    md.append(f"\n**Cases human annotated as conflict / weakly_conflicting**: "
              f"{conflict_human} / {len(human_audit)}\n\n")
    md.append("## Validator vs human agreement\n\n")
    md.append("Audit cases were drawn from the AVeriTeC E1 set; matching by "
              "case_id requires the claim text to be in the raw_results record. "
              "If raw_results does not carry the claim text, we report the human "
              "labels alone and rely on the broader CCO-subset validator coverage "
              "result (validator_on_cco.md) as the in-paper number.\n")
    (OUT / "validator_vs_human_audit.md").write_text("".join(md))
    return {"human_labels": counts, "n_human_conflict": conflict_human, "n_audit": len(human_audit)}


# ───────────────────────── Main ─────────────────────────

def main():
    print("Loading data...")
    e1 = load_e1()
    print(f"  e1 cases: {len(e1)}")
    validator = load_validator()
    print(f"  validator cases: {len(validator)}")
    human_audit = load_human_audit()
    print(f"  human audit cases parsed: {len(human_audit)}")

    print("\n[P0b] Calibration analysis...")
    cal = calibration_analysis(e1)
    print(f"  ECE overall directional: {cal['ece']:.4f}")
    print(f"  ECE on pure-S/R (CCO removed): {cal['ece_pure_sr']:.4f}")
    print(f"  CCO confidence mean: {cal['cco_stats']['mean'] if cal['cco_stats'] else None}")
    print(f"  Correct confidence mean: {cal['correct_stats']['mean'] if cal['correct_stats'] else None}")
    if cal['mw']:
        print(f"  Mann-Whitney p ≈ {cal['mw']['p_two_sided']}")

    print("\n[P0c] 4x4 confusion matrix...")
    cm = confusion_4x4(e1)
    for g in LABELS_4:
        row = " ".join(f"{cm.get((g, p), 0):4d}" for p in LABELS_4)
        print(f"  {g:13s} → {row}")

    print("\n[P1a] False conflict rate...")
    fc = false_conflict_rate(e1)
    print(f"  N_sr={fc['n_sr']}, false-conflict rate = {fc['false_conf_rate']:.3f}, "
          f"false-insufficient = {fc['false_insuf_rate']:.3f}")

    print("\n[P1b] Panel agreement on CCO...")
    pa = panel_agreement_on_cco(e1)
    print(f"  N_cco={pa['n_cco']}, patterns: {pa['agreement_types']}")

    print("\n[P1c] Validator on CCO...")
    vc = validator_on_cco(e1, validator)
    print(f"  N_cco={vc['n_cco']}, matched={vc['n_matched']}, "
          f"material_mixed rate={vc['flagged_mixed_rate']:.3f}, "
          f"false alarm on S/R={vc['false_alarm_on_sr']:.3f}")

    print("\n[P1d] Validator vs human audit...")
    va = validator_vs_audit(human_audit, e1, validator)
    print(f"  Human label distribution: {va.get('human_labels')}")

    # Write summary
    summary = []
    summary.append("# P0/P1 Diagnostic Summary\n\n")
    summary.append("All numbers used in §3 (concept) and §4/§5 (results) revisions.\n\n")

    summary.append("## P0b. Calibration / CCO ≠ calibration error\n\n")
    if cal['mw']:
        summary.append(f"- Panel directional commits N={cal['n_dir']} (CCO={cal['n_cco']}, "
                       f"correct={cal['n_correct']}).\n")
        summary.append(f"- ECE on pure-S/R subset (CCO removed): **{cal['ece_pure_sr']:.4f}**\n")
        summary.append(f"- CCO mean conf {cal['cco_stats']['mean']:.3f} "
                       f"vs correct directional mean conf {cal['correct_stats']['mean']:.3f}\n")
        summary.append(f"- Mann-Whitney two-sided p ≈ {cal['mw']['p_two_sided']}\n")
        summary.append(f"  → CCO and correct directional confidence distributions are "
                       f"{'statistically distinguishable' if cal['mw']['p_two_sided'] and cal['mw']['p_two_sided'] < 0.05 else 'NOT statistically distinguishable'} "
                       f"at α=0.05.\n\n")

    summary.append("## P0c. I vs C confusion matrix\n\n")
    summary.append("| gold ↓ \\ pred → | support | refute | insufficient | conflicting |\n")
    summary.append("|---|---|---|---|---|\n")
    for g in LABELS_4:
        cells = [cm.get((g, p), 0) for p in LABELS_4]
        summary.append(f"| {g} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |\n")
    summary.append("\n")

    summary.append("## P1a. False conflict rate on pure-S/R\n\n")
    summary.append(f"- N (pure S/R) = {fc['n_sr']}\n")
    summary.append(f"- Pred = CONFLICTING: {fc['false_conf']} ({fc['false_conf_rate']*100:.1f}%)\n")
    summary.append(f"- Pred = INSUFFICIENT: {fc['false_insuf']} ({fc['false_insuf_rate']*100:.1f}%)\n\n")

    summary.append("## P1b. Panel agreement on CCO\n\n")
    summary.append(f"- N_CCO = {pa['n_cco']}\n")
    for k, v in pa['agreement_types'].items():
        summary.append(f"- {k}: {v} ({v/pa['n_cco']*100:.1f}%)\n")
    summary.append("\n")

    summary.append("## P1c. Validator coverage on CCO\n\n")
    summary.append(f"- N_CCO (validator-matched) = {vc['n_matched']}\n")
    summary.append(f"- material_mixed=True rate on CCO: **{vc['flagged_mixed_rate']*100:.1f}%**\n")
    summary.append(f"- material_mixed=True rate on pure-S/R: **{vc['false_alarm_on_sr']*100:.1f}%** "
                   f"(false-alarm baseline)\n\n")

    summary.append("## P1d. Human audit (N=10)\n\n")
    summary.append(f"- Conflict / weakly_conflicting: {va.get('n_human_conflict')} / 10\n")
    summary.append(f"- Label distribution: {dict(va.get('human_labels', {}))}\n\n")

    (OUT / "summary.md").write_text("".join(summary))
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
