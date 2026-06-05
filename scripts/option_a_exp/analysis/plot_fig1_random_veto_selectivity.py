"""Generate fig1_random_veto_selectivity.pdf — 2x2 panel APPLES-TO-APPLES version.

The control random-Stage-1 promotes k of E's directional commits to
CONFLICTING (matching what F's Stage-1 does), not to NO-COMMIT as in the
original control. Reading data from fair_random_stage1_distributions.json.

Top row (honest reading): SE (selective error) and CCO under random-veto
distribution. F sits near the random median — consistent with bootstrap
CIs straddling 0 reported in §5.

Bottom row (structural-selectivity claim): Acc_{S/R} and Rec_C. F sits at
the 100th percentile — the veto preserves correct directional commits
and extends conflict recall in ways no random veto budget can.

The two readings together are the complete cross-method story under the
Day-7 analysis-first framing: honest about magnitude (top), robust on
direction-preservation (bottom).

No new API calls. No new experiments. Pure post-hoc analysis of existing
E and F predictions.

Outputs:
  - Writing/V0.2/figures/fig1_random_veto_selectivity.pdf
  - Writing/V0.2/figures/fig1_random_veto_selectivity.png
  - outputs/option_a_exp/analysis/defense_pack/random_veto_raw_distributions.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/option_a_exp/analysis"))

from analyze_selective_typed_controller import (  # noqa: E402
    load_e1, load_e3_validator,
    controller_e_confidence, controller_f_combined,
)

WRITING_DIR = Path("/Users/haoranxu/Desktop/PhD/Claim_Commitment/Writing/V0.2/figures")
DATA_DUMP_DIR = REPO / "outputs/option_a_exp/analysis/defense_pack"

N_SEEDS = 2000
RANDOM_SEED = 0
TAU_E = 0.90
TAU_F = 0.85


def metrics_full(cases, preds):
    """Same-denominator metrics on a full-N prediction list."""
    n = len(cases)
    n_pure_sr = sum(1 for c in cases if c["gold"] in ("support", "refute"))
    n_conflict = sum(1 for c in cases if c["gold"] == "conflicting")

    n_commit = sum(1 for p in preds if p in ("support", "refute"))
    n_correct = sum(1 for c, p in zip(cases, preds)
                    if p in ("support", "refute") and p == c["gold"])
    n_wrong = n_commit - n_correct
    n_cco_full = sum(1 for c, p in zip(cases, preds)
                     if p in ("support", "refute") and c["gold"] == "conflicting")
    n_acc_sr = sum(1 for c, p in zip(cases, preds)
                   if p in ("support", "refute") and c["gold"] == p
                   and c["gold"] in ("support", "refute"))
    n_conf_recall = sum(1 for c, p in zip(cases, preds)
                        if c["gold"] == "conflicting" and p == "conflicting")
    return {
        "cov": n_commit / n,
        "se": (n_wrong / n_commit) if n_commit else 0.0,
        "cco": n_cco_full / n,
        "acc_sr": n_acc_sr / n_pure_sr if n_pure_sr else 0.0,
        "rec_c": n_conf_recall / n_conflict if n_conflict else 0.0,
    }


def main():
    print("Loading AVeriTeC E1 cases...")
    e1 = load_e1()
    val = load_e3_validator()
    cases = []
    for cid, d in e1.items():
        if "panel_4opt_agg" not in d:
            continue
        c = dict(d)
        c["case_id"] = cid
        c["validity"] = (val.get(cid) or {}).get("validity") or {}
        cases.append(c)
    print(f"  N = {len(cases)}")

    print(f"\nComputing E τ={TAU_E} and F τ={TAU_F} predictions...")
    e_preds = [controller_e_confidence(c, TAU_E) for c in cases]
    f_preds = [controller_f_combined(c, c["validity"], TAU_F) for c in cases]
    f_metrics = metrics_full(cases, f_preds)
    e_metrics = metrics_full(cases, e_preds)
    print(f"  E: {e_metrics}")
    print(f"  F: {f_metrics}")

    e_commit_idx = [i for i, p in enumerate(e_preds) if p in ("support", "refute")]
    f_commit_set = set(i for i, p in enumerate(f_preds) if p in ("support", "refute"))
    k = len(set(e_commit_idx) - f_commit_set)
    print(f"  k = {k}")

    print(f"\nRunning APPLES-TO-APPLES random Stage-1 control "
          f"(promote to CONFLICTING) with N={N_SEEDS} seeds...")
    rnd = random.Random(RANDOM_SEED)
    rand_se, rand_cco, rand_acc_sr, rand_rec_c = [], [], [], []
    for _ in range(N_SEEDS):
        veto_set = set(rnd.sample(e_commit_idx, k))
        rand_preds = list(e_preds)
        for i in veto_set:
            # Apples-to-apples with F's Stage-1: promote to CONFLICTING,
            # NOT to no_commit. The original control (no_commit) is mechanically
            # unable to change Rec_C, so this fair control replaces it.
            rand_preds[i] = "conflicting"
        m = metrics_full(cases, rand_preds)
        rand_se.append(m["se"])
        rand_cco.append(m["cco"])
        rand_acc_sr.append(m["acc_sr"])
        rand_rec_c.append(m["rec_c"])

    rand_se = np.array(rand_se)
    rand_cco = np.array(rand_cco)
    rand_acc_sr = np.array(rand_acc_sr)
    rand_rec_c = np.array(rand_rec_c)

    def pctile_le(val, arr):
        return float((arr <= val).sum()) / len(arr)

    pcts = {
        "se":    pctile_le(f_metrics["se"], rand_se),
        "cco":   pctile_le(f_metrics["cco"], rand_cco),
        "acc":   pctile_le(f_metrics["acc_sr"], rand_acc_sr),
        "rec":   pctile_le(f_metrics["rec_c"], rand_rec_c),
    }
    print(f"\n  F percentiles vs random-veto distribution:")
    print(f"    SE        ({'lower better'}): F={f_metrics['se']:.3f}  random_mean={rand_se.mean():.3f}  pctile={pcts['se']*100:.1f}%")
    print(f"    CCO       ({'lower better'}): F={f_metrics['cco']:.3f}  random_mean={rand_cco.mean():.3f}  pctile={pcts['cco']*100:.1f}%")
    print(f"    Acc_S/R   ({'higher better'}): F={f_metrics['acc_sr']:.3f}  random_mean={rand_acc_sr.mean():.3f}  pctile={pcts['acc']*100:.1f}%")
    print(f"    Rec_C     ({'higher better'}): F={f_metrics['rec_c']:.3f}  random_mean={rand_rec_c.mean():.3f}  pctile={pcts['rec']*100:.1f}%")

    DATA_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    dump_path = DATA_DUMP_DIR / "random_veto_raw_distributions.json"
    with dump_path.open("w") as f:
        json.dump({
            "comparison": "AVeriTeC F τ=0.85 vs E τ=0.90",
            "n_seeds": N_SEEDS,
            "k_vetoes": k,
            "f_metrics": f_metrics,
            "e_metrics": e_metrics,
            "rand_se": rand_se.tolist(),
            "rand_cco": rand_cco.tolist(),
            "rand_acc_sr": rand_acc_sr.tolist(),
            "rand_rec_c": rand_rec_c.tolist(),
            "f_pctile_se": pcts["se"],
            "f_pctile_cco": pcts["cco"],
            "f_pctile_acc": pcts["acc"],
            "f_pctile_rec": pcts["rec"],
        }, f, indent=2)
    print(f"\n  Wrote raw distributions → {dump_path}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "cm",
        "mathtext.default": "regular",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 4,
        "axes.titlepad": 8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })

    HIST = "#bcd2e6"
    F_COLOR = "#0b3d91"
    E_COLOR = "#6b6b6b"
    HONEST_BG = "#fcf3ea"
    SELECT_BG = "#ecf3ec"
    HONEST_BANNER = "#c87b3a"
    SELECT_BANNER = "#3a7a4f"

    fig = plt.figure(figsize=(9.6, 4.0))
    gs = fig.add_gridspec(
        2, 2,
        left=0.06, right=0.985, top=0.86, bottom=0.13,
        wspace=0.20, hspace=0.78,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)]
                     for r in range(2)])

    def draw_panel(ax, rand_arr, f_val, e_val, xlabel, panel_label,
                   pct, lower_better, bg_color, percentile_color):
        ax.set_facecolor(bg_color)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        lo = min(rand_arr.min(), f_val, e_val)
        hi = max(rand_arr.max(), f_val, e_val)
        pad = max((hi - lo) * 0.08, 0.005)
        lo -= pad
        hi += pad
        bins = np.linspace(lo, hi, 28)
        ax.hist(rand_arr, bins=bins, color=HIST, edgecolor="white",
                linewidth=0.5, alpha=0.95)
        ax.axvline(e_val, color=E_COLOR, linewidth=1.1, linestyle=(0, (4, 2)))
        ax.axvline(f_val, color=F_COLOR, linewidth=2.0)
        ax.set_xlim(lo, hi)

        ax.set_xlabel(xlabel, fontsize=10.5)
        arrow = "↓" if lower_better else "↑"
        ax.set_title(
            f"{panel_label}  F at {pct*100:.0f}th percentile  ({arrow} better)",
            fontsize=10, color=percentile_color, fontweight="bold", pad=6, loc="left",
        )
        ax.tick_params(labelsize=8.5)
        ax.set_yticks([])

        ymax = ax.get_ylim()[1]
        if lower_better:
            f_ha = "left" if f_val < e_val else "right"
            e_ha = "right" if f_val < e_val else "left"
        else:
            f_ha = "right" if f_val > e_val else "left"
            e_ha = "left" if f_val > e_val else "right"
        sep = (hi - lo)
        offset = sep * 0.006
        sign_f = -1 if f_ha == "right" else 1
        sign_e = -1 if e_ha == "right" else 1
        ax.text(f_val + sign_f * offset, ymax * 0.96, f"F = {f_val:.3f}",
                color=F_COLOR, fontsize=9, fontweight="bold",
                ha=f_ha, va="top")
        if abs(e_val - f_val) > sep * 0.025:
            ax.text(e_val + sign_e * offset, ymax * 0.78, f"E = {e_val:.3f}",
                    color=E_COLOR, fontsize=9, ha=e_ha, va="top")
        else:
            ax.text(e_val + sign_e * offset, ymax * 0.62, f"E = {e_val:.3f}",
                    color=E_COLOR, fontsize=9, ha=e_ha, va="top")

    draw_panel(axes[0, 0], rand_se,
               f_metrics["se"], e_metrics["se"],
               r"$\mathrm{SE}$  (selective error)",
               "(a)", pcts["se"], lower_better=True,
               bg_color=HONEST_BG, percentile_color=HONEST_BANNER)
    draw_panel(axes[0, 1], rand_cco,
               f_metrics["cco"], e_metrics["cco"],
               r"$\mathrm{CCO}$  (Cherry-pick Override rate)",
               "(b)", pcts["cco"], lower_better=True,
               bg_color=HONEST_BG, percentile_color=HONEST_BANNER)
    draw_panel(axes[1, 0], rand_acc_sr,
               f_metrics["acc_sr"], e_metrics["acc_sr"],
               r"$\mathrm{Acc}_{\mathrm{S/R}}$  (pure-S/R accuracy)",
               "(c)", pcts["acc"], lower_better=False,
               bg_color=SELECT_BG, percentile_color=SELECT_BANNER)
    draw_panel(axes[1, 1], rand_rec_c,
               f_metrics["rec_c"], e_metrics["rec_c"],
               r"$\mathrm{Rec}_{\mathrm{C}}$  (conflict recall on gold-$\mathrm{C}$ subset)",
               "(d)", pcts["rec"], lower_better=False,
               bg_color=SELECT_BG, percentile_color=SELECT_BANNER)

    # Row banners at top of each row, ABOVE the panels (uses fig coords).
    # Side ribbons: vertical rectangles on the far-left margin colored to indicate
    # which row tells the magnitude vs structural-selectivity story. The caption
    # explains; the visual cue here is enough to read at a glance.
    fig.patches.extend([
        plt.Rectangle((0.001, 0.50), 0.012, 0.36, transform=fig.transFigure,
                      facecolor=HONEST_BANNER, edgecolor="none", clip_on=False),
        plt.Rectangle((0.001, 0.13), 0.012, 0.37, transform=fig.transFigure,
                      facecolor=SELECT_BANNER, edgecolor="none", clip_on=False),
    ])
    fig.text(0.018, 0.68, "MAGNITUDE",
             rotation=90, va="center", ha="left", fontsize=8.5,
             color=HONEST_BANNER, fontweight="bold")
    fig.text(0.018, 0.315, "SELECTIVITY",
             rotation=90, va="center", ha="left", fontsize=8.5,
             color=SELECT_BANNER, fontweight="bold")

    # Legend at very top.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=HIST,
                      label=f"Random Stage-1 promotion ({N_SEEDS} seeds, k={k})"),
        plt.Line2D([0], [0], color=F_COLOR, linewidth=2.0,
                   label="F = L5 two-channel probe at τ=0.85"),
        plt.Line2D([0], [0], color=E_COLOR, linewidth=1.1, linestyle=(0, (4, 2)),
                   label="E = L3 (confidence-only, no veto) at τ=0.90"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False, fontsize=9, columnspacing=2.2)

    WRITING_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = WRITING_DIR / "fig1_random_veto_selectivity.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    out_png = WRITING_DIR / "fig1_random_veto_selectivity.png"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close(fig)
    print(f"\n  Wrote {out_pdf}")
    print(f"  Wrote {out_png}")


if __name__ == "__main__":
    main()
