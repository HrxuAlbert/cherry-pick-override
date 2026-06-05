"""Apples-to-apples random Stage-1 control.

The original random-veto control downgraded E's directional commits to
NO-COMMIT, which mechanically cannot change Rec_C (since random veto
removes S/R predictions but does not produce CONFLICTING predictions).
F's Stage-1, by contrast, *promotes* directional commits to CONFLICTING.
The fair comparison is therefore to replace E's k directional commits
with CONFLICTING (not NO-COMMIT), at 2000 random seeds, and compare F to
the resulting distribution.

This control answers: among E's k commits, did F's Stage-1 choose the
\"right\" k to promote---or could a random Stage-1 reach the same
operating point?

No new API calls; uses cached E1 and E4 predictions.

Outputs:
  outputs/option_a_exp/analysis/defense_pack/
    fair_random_stage1_distributions.json
    fair_random_stage1_summary.md
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/option_a_exp/analysis"))

from analyze_selective_typed_controller import (  # noqa: E402
    load_e1, load_e3_validator,
    controller_e_confidence, controller_f_combined,
)
from analyze_vg_rctc_on_e4 import load_e4_panel, load_e4_validator  # noqa: E402

OUT = REPO / "outputs/option_a_exp/analysis/defense_pack"
OUT.mkdir(parents=True, exist_ok=True)

N_SEEDS = 2000
RANDOM_SEED = 0


def metrics_full(cases, preds):
    """Same-denominator metrics."""
    n = len(cases)
    n_sr = sum(1 for c in cases if c["gold"] in ("support", "refute"))
    n_c = sum(1 for c in cases if c["gold"] == "conflicting")

    n_commit = sum(1 for p in preds if p in ("support", "refute"))
    n_correct = sum(1 for c, p in zip(cases, preds)
                    if p in ("support", "refute") and p == c["gold"])
    n_wrong = n_commit - n_correct
    n_cco = sum(1 for c, p in zip(cases, preds)
                if p in ("support", "refute") and c["gold"] == "conflicting")
    n_acc_sr = sum(1 for c, p in zip(cases, preds)
                   if p in ("support", "refute") and c["gold"] == p
                   and c["gold"] in ("support", "refute"))
    n_rec_c = sum(1 for c, p in zip(cases, preds)
                  if c["gold"] == "conflicting" and p == "conflicting")
    return {
        "cov": n_commit / n,
        "se": (n_wrong / n_commit) if n_commit else 0.0,
        "cco_N": n_cco / n,
        "acc_sr": n_acc_sr / n_sr if n_sr else 0.0,
        "rec_c": n_rec_c / n_c if n_c else 0.0,
    }


def pctile_le(val, arr):
    return float(sum(1 for x in arr if x <= val)) / len(arr)


def pctile_ge(val, arr):
    return float(sum(1 for x in arr if x >= val)) / len(arr)


def run_fair_random_stage1(cases, tau_e, tau_f, dataset_name):
    """Generate F vs random-Stage-1 distribution on the given cases.

    F = controller_f_combined at tau_f.
    Random Stage-1 baseline: take E's (tau_e) directional commits; randomly
    pick k = |E_commit \\ F_commit| of them and PROMOTE TO CONFLICTING
    (rather than no_commit as in the original control).
    """
    e_preds = [controller_e_confidence(c, tau_e) for c in cases]
    f_preds = [controller_f_combined(c, c["validity"], tau_f) for c in cases]

    e_metrics = metrics_full(cases, e_preds)
    f_metrics = metrics_full(cases, f_preds)

    e_commit_idx = [i for i, p in enumerate(e_preds) if p in ("support", "refute")]
    f_commit_set = set(i for i, p in enumerate(f_preds) if p in ("support", "refute"))
    k = len(set(e_commit_idx) - f_commit_set)
    print(f"\n=== {dataset_name} ===")
    print(f"  E τ={tau_e}: {e_metrics}")
    print(f"  F τ={tau_f}: {f_metrics}")
    print(f"  k = {k} promotions (E commits not in F commit set)")

    rnd = random.Random(RANDOM_SEED)
    rand = {"se": [], "cco_N": [], "acc_sr": [], "rec_c": []}
    for _ in range(N_SEEDS):
        veto_set = set(rnd.sample(e_commit_idx, k))
        rand_preds = list(e_preds)
        for i in veto_set:
            # APPLES-TO-APPLES: promote to CONFLICTING (not no_commit).
            rand_preds[i] = "conflicting"
        m = metrics_full(cases, rand_preds)
        rand["se"].append(m["se"])
        rand["cco_N"].append(m["cco_N"])
        rand["acc_sr"].append(m["acc_sr"])
        rand["rec_c"].append(m["rec_c"])

    # Percentiles (one-sided in the "F is better than random" direction)
    # SE / CCO_N: lower is better, so empirical p = fraction of random ≤ F
    # Acc_S/R / Rec_C: higher is better, so p = fraction of random ≥ F
    pcts = {
        "se":     pctile_le(f_metrics["se"], rand["se"]),
        "cco_N":  pctile_le(f_metrics["cco_N"], rand["cco_N"]),
        "acc_sr": pctile_ge(f_metrics["acc_sr"], rand["acc_sr"]),
        "rec_c":  pctile_ge(f_metrics["rec_c"], rand["rec_c"]),
    }

    means = {k: sum(v) / len(v) for k, v in rand.items()}
    print(f"  Random Stage-1 distribution means: {means}")
    print(f"  F percentiles vs distribution:")
    print(f"    SE      (F ≤ random?): F={f_metrics['se']:.3f}, mean={means['se']:.3f}, frac{{random ≤ F}}={pcts['se']*100:.1f}%")
    print(f"    CCO_N   (F ≤ random?): F={f_metrics['cco_N']:.3f}, mean={means['cco_N']:.3f}, frac{{random ≤ F}}={pcts['cco_N']*100:.1f}%")
    print(f"    Acc_S/R (F ≥ random?): F={f_metrics['acc_sr']:.3f}, mean={means['acc_sr']:.3f}, frac{{random ≥ F}}={pcts['acc_sr']*100:.1f}%")
    print(f"    Rec_C   (F ≥ random?): F={f_metrics['rec_c']:.3f}, mean={means['rec_c']:.3f}, frac{{random ≥ F}}={pcts['rec_c']*100:.1f}%")
    return {
        "dataset": dataset_name,
        "tau_e": tau_e, "tau_f": tau_f, "k": k,
        "e_metrics": e_metrics, "f_metrics": f_metrics,
        "rand_se": rand["se"], "rand_cco_N": rand["cco_N"],
        "rand_acc_sr": rand["acc_sr"], "rand_rec_c": rand["rec_c"],
        "pcts": pcts, "means": means,
    }


def main():
    # AVeriTeC
    print("Loading AVeriTeC E1...")
    e1 = load_e1()
    val = load_e3_validator()
    av_cases = []
    for cid, d in e1.items():
        if "panel_4opt_agg" not in d:
            continue
        c = dict(d)
        c["case_id"] = cid
        c["validity"] = (val.get(cid) or {}).get("validity") or {}
        av_cases.append(c)
    print(f"  N = {len(av_cases)}")

    av_result = run_fair_random_stage1(av_cases, tau_e=0.90, tau_f=0.85,
                                        dataset_name="AVeriTeC F τ=0.85 vs E τ=0.90")

    # VitaminC
    print("\nLoading VitaminC E4...")
    e4 = load_e4_panel()
    e4_val = load_e4_validator()
    vc_cases = []
    for cid, d in e4.items():
        v = e4_val.get(cid)
        if not v or v["case_key"] != d["case_key"]:
            continue
        c = dict(d)
        c["case_id"] = cid
        c["validity"] = v["validity"]
        vc_cases.append(c)
    print(f"  N = {len(vc_cases)}")

    vc_result = run_fair_random_stage1(vc_cases, tau_e=0.90, tau_f=0.90,
                                        dataset_name="VitaminC F τ=0.90 vs E τ=0.90")

    # Dump JSON
    dump_path = OUT / "fair_random_stage1_distributions.json"
    with dump_path.open("w") as f:
        json.dump({"averitec": av_result, "vitaminc": vc_result}, f, indent=2)
    print(f"\nWrote {dump_path}")

    # Summary MD
    md = []
    md.append("# Fair Random Stage-1 Control (Apples-to-Apples)\n\n")
    md.append("**Date**: 2026-06-03\n")
    md.append("**Brief**: The original random-veto control downgraded E's directional commits to "
              "NO-COMMIT, which mechanically cannot change `Rec_C` (random veto removes S/R "
              "predictions but does not produce CONFLICTING predictions). This fair control "
              "instead **promotes** k of E's directional commits to CONFLICTING (matching what "
              "F's Stage-1 does), so the random null can in principle reach any of F's "
              "operating-point metrics.\n\n")
    for r in [av_result, vc_result]:
        md.append(f"## {r['dataset']}\n\n")
        md.append(f"k = {r['k']} promotions. F's actual percentiles under the fair random-Stage-1 "
                  "null (2000 seeds):\n\n")
        md.append("| Metric | F | Random mean | Empirical p (one-sided in F's direction) |\n")
        md.append("|---|---|---|---|\n")
        md.append(f"| SE ↓ | {r['f_metrics']['se']:.3f} | {r['means']['se']:.3f} | "
                  f"{r['pcts']['se']*100:.1f}% (frac random ≤ F) |\n")
        md.append(f"| CCO_N ↓ | {r['f_metrics']['cco_N']:.3f} | {r['means']['cco_N']:.3f} | "
                  f"{r['pcts']['cco_N']*100:.1f}% (frac random ≤ F) |\n")
        md.append(f"| Acc_S/R ↑ | {r['f_metrics']['acc_sr']:.3f} | {r['means']['acc_sr']:.3f} | "
                  f"{r['pcts']['acc_sr']*100:.1f}% (frac random ≥ F) |\n")
        md.append(f"| Rec_C ↑ | {r['f_metrics']['rec_c']:.3f} | {r['means']['rec_c']:.3f} | "
                  f"{r['pcts']['rec_c']*100:.1f}% (frac random ≥ F) |\n\n")
    md.append("## Interpretation\n\n")
    md.append("Under this fair control, the random null can reach any of F's "
              "metric values (Rec_C is no longer mechanically capped at baseline). "
              "Reading: how surprising is F's actual operating point vs random Stage-1 "
              "promotion of the same number k of E's directional commits?\n")
    (OUT / "fair_random_stage1_summary.md").write_text("".join(md))
    print(f"Wrote {OUT / 'fair_random_stage1_summary.md'}")


if __name__ == "__main__":
    main()
