"""Generate Figure 1: 4-column x 2-row apples-to-apples random Stage-1 panel.

Layout: rows = datasets (AVeriTeC, VitaminC-Mixed), columns = metrics
(SE, CCO_N, Acc_S/R, Rec_C). Columns 1-2 (magnitude axis) are shaded
warm; columns 3-4 (selectivity axis) are shaded cool.

For each panel: KDE of the apples-to-apples random Stage-1 distribution
(promotion to CONFLICTING, 2000 seeds), with shaded [5%, 95%] band, a
dashed E (baseline) line, and a solid F (controller) line. Annotations
report F's empirical percentile against the null.

Reads pre-computed distributions from
  outputs/option_a_exp/analysis/random_stage1_null/fair_random_stage1_distributions.json
(produced by fair_random_stage1.py). No new random draws here, so the
plot is deterministic given that input file.

Outputs (written under REPO/figures by default; override with FIG_DIR env var):
  - figures/fig1_random_veto_selectivity.pdf
  - figures/fig1_random_veto_selectivity.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parents[3]
FIG_DIR = Path(os.environ.get("FIG_DIR", REPO / "figures"))
DATA = REPO / "outputs/option_a_exp/analysis/random_stage1_null/fair_random_stage1_distributions.json"

METRICS = [
    ("rand_se",     "se",     r"$\mathrm{SE}$",            "selective error",        True,  "magnitude"),
    ("rand_cco_N",  "cco_N",  r"$\mathrm{CCO}_N$",          "same-denom CCO rate",    True,  "magnitude"),
    ("rand_acc_sr", "acc_sr", r"$\mathrm{Acc}_{\mathrm{S/R}}$", "pure-S/R accuracy",  False, "selectivity"),
    ("rand_rec_c",  "rec_c",  r"$\mathrm{Rec}_{\mathrm{C}}$",   "conflict recall",    False, "selectivity"),
]

DATASETS = [
    ("averitec", "AVeriTeC ($N=285$)"),
    ("vitaminc", "VitaminC-Mixed ($N=250$)"),
]

MAG_BG = "#fcf3ea"
SEL_BG = "#ecf3ec"
MAG_BANNER = "#c87b3a"
SEL_BANNER = "#3a7a4f"
KDE_FILL = "#bcd2e6"
KDE_LINE = "#5b87b8"
BAND_FILL = "#9bb9d6"
F_COLOR = "#0b3d91"
E_COLOR = "#6b6b6b"


def draw_one_panel(ax, rand_arr, f_val, e_val, pct, lower_better,
                    bg, banner_color, x_label_top, y_label_left):
    """Draw a single KDE-style panel."""
    ax.set_facecolor(bg)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # x range from the data, padded
    lo = min(rand_arr.min(), f_val, e_val)
    hi = max(rand_arr.max(), f_val, e_val)
    pad = max((hi - lo) * 0.10, 0.005)
    lo -= pad
    hi += pad

    # KDE of the random distribution
    try:
        kde = gaussian_kde(rand_arr, bw_method="silverman")
        xs = np.linspace(lo, hi, 200)
        ys = kde(xs)
    except Exception:
        # Fallback if KDE fails (e.g. zero variance)
        xs = np.linspace(lo, hi, 200)
        ys = np.zeros_like(xs)

    ax.fill_between(xs, 0, ys, color=KDE_FILL, alpha=0.85, linewidth=0)
    ax.plot(xs, ys, color=KDE_LINE, linewidth=0.9)

    # 5-95 percentile band
    p5, p95 = np.percentile(rand_arr, [5, 95])
    mask = (xs >= p5) & (xs <= p95)
    if mask.any():
        ax.fill_between(xs[mask], 0, ys[mask], color=BAND_FILL,
                        alpha=0.55, linewidth=0)

    # Reference lines
    ymax = ys.max() if ys.max() > 0 else 1.0
    ax.axvline(e_val, color=E_COLOR, linewidth=1.0, linestyle=(0, (4, 2)),
               ymin=0, ymax=0.95)
    ax.axvline(f_val, color=F_COLOR, linewidth=1.8, ymin=0, ymax=0.95)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, ymax * 1.32)

    arrow = "↓" if lower_better else "↑"
    pct_str = f"$F$ at {pct*100:.0f}th pctl ({arrow})"
    ax.set_title(pct_str, fontsize=8.5, color=banner_color,
                 fontweight="bold", pad=2, loc="left")
    ax.tick_params(labelsize=7.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2.5, pad=2)

    if x_label_top:
        ax.set_xlabel(x_label_top, fontsize=8.5, labelpad=2)

    # F / E numeric annotations near the lines
    sep = hi - lo
    f_ha = ("left" if lower_better else "right") if f_val < e_val else ("right" if lower_better else "left")
    sign_f = -1 if f_ha == "right" else 1
    sign_e = 1 if sign_f < 0 else -1
    e_ha = "right" if sign_e < 0 else "left"

    ax.text(f_val + sign_f * sep * 0.008, ymax * 1.22, f"$F${f_val:.3f}",
            color=F_COLOR, fontsize=7.5, fontweight="bold",
            ha=f_ha, va="top")
    # Only annotate E if visibly separated
    if abs(e_val - f_val) > sep * 0.04:
        ax.text(e_val + sign_e * sep * 0.008, ymax * 1.05, f"$E${e_val:.3f}",
                color=E_COLOR, fontsize=7.0, ha=e_ha, va="top")


def main():
    if not DATA.exists():
        raise SystemExit(f"Missing {DATA}; run fair_random_stage1.py first.")
    with DATA.open() as f:
        all_data = json.load(f)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "mathtext.fontset": "cm",
        "mathtext.default": "regular",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(11.2, 4.2))
    gs = fig.add_gridspec(
        2, 4,
        left=0.07, right=0.985, top=0.80, bottom=0.13,
        wspace=0.22, hspace=0.55,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)]
                     for r in range(2)])

    k_values = {}
    for row, (ds_key, ds_label) in enumerate(DATASETS):
        if ds_key not in all_data:
            for col in range(4):
                axes[row, col].set_visible(False)
            continue
        d = all_data[ds_key]
        k_values[ds_key] = d["k"]
        for col, (rand_key, met_key, sym, name, lower_better, axis_kind) in enumerate(METRICS):
            rand_arr = np.array(d[rand_key])
            f_val = d["f_metrics"][met_key]
            e_val = d["e_metrics"][met_key]
            pct = d["pcts"][met_key]
            bg = MAG_BG if axis_kind == "magnitude" else SEL_BG
            banner = MAG_BANNER if axis_kind == "magnitude" else SEL_BANNER
            x_label = f"{sym}  ({name})" if row == 1 else None
            y_label = ds_label if col == 0 else None
            draw_one_panel(axes[row, col], rand_arr,
                           f_val, e_val, pct,
                           lower_better=lower_better,
                           bg=bg, banner_color=banner,
                           x_label_top=x_label,
                           y_label_left=y_label)
            if col == 0:
                axes[row, col].set_ylabel(ds_label, fontsize=9, labelpad=4,
                                           fontweight="bold")

    # Column-group headers ("MAGNITUDE" / "SELECTIVITY") across top
    fig.text(0.215, 0.92, "MAGNITUDE  (F near random median; expected)",
             ha="center", va="center", fontsize=9, fontweight="bold",
             color=MAG_BANNER)
    fig.text(0.685, 0.92, "SELECTIVITY  (F at distribution extreme; structural)",
             ha="center", va="center", fontsize=9, fontweight="bold",
             color=SEL_BANNER)

    # Legend at top center
    k_str = f"AVeriTeC k={k_values.get('averitec','?')}; VitaminC k={k_values.get('vitaminc','?')}"
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=KDE_FILL,
                      label=f"Random Stage-1 promotion (2000 seeds; {k_str})"),
        plt.Rectangle((0, 0), 1, 1, color=BAND_FILL,
                      label="[5, 95] percentile band"),
        plt.Line2D([0], [0], color=F_COLOR, linewidth=1.8,
                   label=r"$F$ = L5 two-channel probe"),
        plt.Line2D([0], [0], color=E_COLOR, linewidth=1.0, linestyle=(0, (4, 2)),
                   label=r"$E$ = L3 (confidence-only)"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=4,
               frameon=False, fontsize=8, columnspacing=1.6)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "fig1_random_veto_selectivity.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    out_png = FIG_DIR / "fig1_random_veto_selectivity.png"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
