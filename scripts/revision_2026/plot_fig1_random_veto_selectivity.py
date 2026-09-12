"""Generate fig1_random_veto_selectivity.pdf — 4-col × 2-row apples-to-apples version.

Layout: rows = datasets (AVeriTeC, VitaminC-Mixed), columns = metrics
(SE, CCO_N, Acc_S/R, Rec_C). Columns 1-2 (magnitude axis) are shaded
warm; columns 3-4 (selectivity axis) are shaded cool.

For each panel: KDE of the apples-to-apples random Stage-1 distribution
(promotion to CONFLICTING, 2000 seeds), with shaded [5%, 95%] band, a
dashed E (baseline) line, and a solid F (controller) line. Annotations
report F's empirical percentile against the null.

Reads pre-computed distributions from
  outputs/option_a_exp/analysis/defense_pack/fair_random_stage1_distributions.json
(produced by `fair_random_stage1.py`). No new random draws here, so the
plot is deterministic given that input file.

Outputs:
  - Writing/V0.2/figures/fig1_random_veto_selectivity.pdf
  - Writing/V0.2/figures/fig1_random_veto_selectivity.png
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

import json
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

WORKSPACE = _cpo_workspace()
# The v1 script hardcoded an absolute path into a directory that has since been
# renamed, so it wrote nowhere useful. Resolve from the workspace instead, and
# write straight into the manuscript's figure directory.
# Manuscript figure directory in the author's tree; in the public release the
# figures sit at the repo root. FIG_DIR overrides both, matching the
# convention the v1 release already documents.
def _fig_dir():
    import os
    env = os.environ.get("FIG_DIR")
    if env:
        return pathlib.Path(env)
    for cand in (WORKSPACE / "overleaf-paper/figures", WORKSPACE / "figures"):
        if cand.is_dir():
            return cand
    return WORKSPACE / "figures"
WRITING_DIR = _fig_dir()
# The null distributions live in different places in the author's tree and in
# the public release, where fair_random_stage1.py writes them under
# outputs/option_a_exp/analysis/random_stage1_null/. Try both.
_DATA_CANDIDATES = (
    WORKSPACE / "m0_commitment/v0.2/outputs/option_a_exp/analysis"
    / "defense_pack/fair_random_stage1_distributions.json",
    WORKSPACE / "outputs/option_a_exp/analysis/random_stage1_null"
    / "fair_random_stage1_distributions.json",
    WORKSPACE / "outputs/option_a_exp/analysis/defense_pack"
    / "fair_random_stage1_distributions.json",
)
DATA = next((p for p in _DATA_CANDIDATES if p.exists()), _DATA_CANDIDATES[0])

# The metric's preferred direction belongs on the axis label, where it is
# stated once per column, not in every panel title next to the p-value: a
# reader was left to combine "6th pctl" with an arrow to work out the sign.
# Symbol and direction only. With the spelled-out name beside them the four
# axis labels ran into each other across columns; the names are in the caption,
# and the symbols are the ones used throughout the paper.
METRICS = [
    ("rand_se",     "se",     r"$\mathrm{SE}\downarrow$",              "", True,  "magnitude"),
    ("rand_cco_N",  "cco_N",  r"$\mathrm{CPO}_N\downarrow$",            "", True,  "magnitude"),
    ("rand_acc_sr", "acc_sr", r"$\mathrm{Acc}_{\mathrm{S/R}}\uparrow$", "", False, "selectivity"),
    ("rand_rec_c",  "rec_c",  r"$\mathrm{Rec}_{\mathrm{C}}\uparrow$",   "", False, "selectivity"),
]

# Rotated row labels are bounded by panel HEIGHT, not width. At the paper
# canvas each panel is about 1.0in tall, so the sample sizes moved to the
# caption; "AVeriTeC ($N=285$)" no longer fits and overlapped the row below.
DATASETS = [
    ("averitec", "AVeriTeC"),
    ("vitaminc", "VitaminC-Mixed"),
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
    # Stop the reference lines below the label band. ylim is 1.52x the data
    # height, so the KDE tops out at 0.66 of the axes; lines drawn to 0.95 ran
    # straight through the value block at 0.79-0.97 whenever they fell under it,
    # and no choice of corner could avoid that.
    LINE_TOP = 0.70
    ax.axvline(e_val, color=E_COLOR, linewidth=1.0, linestyle=(0, (4, 2)),
               ymin=0, ymax=LINE_TOP)
    ax.axvline(f_val, color=F_COLOR, linewidth=1.8, ymin=0, ymax=LINE_TOP)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, ymax * 1.52)

    # The stored quantity is frac{random at least as good as F} --- a one-sided
    # empirical p-value, not a percentile of the value. Labelling it "pctl" made
    # the Acc_S/R panel read as an error: F sits far to the right of the null yet
    # was titled "0th pctl". The body text (\S6) already reports these as p.
    # Only an empty tail justifies the 1/(n+1) form. At two decimals a genuine
    # p of 0.0015 printed as ".00", which reads as exactly zero.
    n_draws = len(rand_arr)
    if pct <= 0:
        pct_str = f"$p < 1/{n_draws + 1}$"
    elif pct < 0.01:
        pct_str = f"$p = {pct:.3f}$".replace("0.", ".")
    else:
        pct_str = f"$p = {pct:.2f}$".replace("0.", ".")
    ax.set_title(pct_str, fontsize=8.5, color=banner_color,
                 fontweight="bold", pad=2, loc="left")
    ax.tick_params(labelsize=7.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2.5, pad=2)

    if x_label_top:
        ax.set_xlabel(x_label_top, fontsize=8.5, labelpad=2)

    # F and E values as a two-line block in the emptier top corner, not beside
    # their own lines. In a 1.5in panel the line-adjacent placement collided
    # whenever F and E were close (they differ by 0.007 on SE) and was clipped
    # by the axis when a value sat on the limit (E=0.650 on Acc_S/R). A corner
    # block cannot collide with the reference lines or leave the panel.
    # Pick the corner whose block span is furthest from both reference lines.
    # Testing only "are both lines on the right" left the block on top of F
    # whenever the two lines straddled the panel (Rec_C: E at .77, F at .86).
    sep = hi - lo
    fx, ex = (f_val - lo) / sep, (e_val - lo) / sep
    BLOCK = 0.34
    left_clear = min(abs(fx - BLOCK / 2), abs(ex - BLOCK / 2))
    right_clear = min(abs(fx - (1 - BLOCK / 2)), abs(ex - (1 - BLOCK / 2)))
    x_anchor, ha = ((0.03, "left") if left_clear >= right_clear else (0.97, "right"))
    ax.text(x_anchor, 0.97, f"$F$ {f_val:.3f}", transform=ax.transAxes,
            color=F_COLOR, fontsize=7.5, fontweight="bold", ha=ha, va="top")
    ax.text(x_anchor, 0.79, f"$E$ {e_val:.3f}", transform=ax.transAxes,
            color=E_COLOR, fontsize=7.0, ha=ha, va="top")


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

    # Canvas width must track the width the figure is placed at. The v1 geometry
    # was 11.2in with 7-9pt fonts; at \textwidth (7.0in) that is a 0.59 downscale,
    # so the smallest labels landed at 4.2pt. At 7.6in the scale is ~0.95 and every
    # label renders at 6.6pt or above. Keep these two numbers in step.
    fig = plt.figure(figsize=(7.6, 3.2))
    gs = fig.add_gridspec(
        2, 4,
        left=0.105, right=0.985, top=0.72, bottom=0.16,
        wspace=0.24, hspace=0.62,
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
            x_label = (f"{sym}  ({name})" if name else sym) if row == 1 else None
            y_label = ds_label if col == 0 else None
            draw_one_panel(axes[row, col], rand_arr,
                           f_val, e_val, pct,
                           lower_better=lower_better,
                           bg=bg, banner_color=banner,
                           x_label_top=x_label,
                           y_label_left=y_label)
            if col == 0:
                axes[row, col].set_ylabel(ds_label, fontsize=8, labelpad=3,
                                           fontweight="bold")

    # Column-group headers ("MAGNITUDE" / "SELECTIVITY") across top
    # Descriptive group names, not results. The previous banners asserted
    # "F near random median" over the magnitude columns and "F at distribution
    # extreme" over the others; that holds on the AVeriTeC row only. On
    # VitaminC-Mixed it is false for three of the four cells, and it labelled
    # the same p = .06 as "near median" in one column and "extreme" in another.
    # The per-panel p-values carry the result, correctly, cell by cell.
    fig.text(0.317, 0.855, "MAGNITUDE  (error rates)",
             ha="center", va="center", fontsize=8, fontweight="bold",
             color=MAG_BANNER)
    fig.text(0.762, 0.855, "DIRECTION PRESERVATION  (which commits survive)",
             ha="center", va="center", fontsize=8, fontweight="bold",
             color=SEL_BANNER)

    # Legend at top center
    # The legend drives the saved width: bbox_inches="tight" expands the bounding
    # box to contain it, so a 4-column single row of long labels produced a 10.2in
    # canvas regardless of figsize. Two columns, and the k values moved to the
    # caption, keep the saved width at the figsize.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=KDE_FILL,
                      label="Random Stage-1 promotion (2000 seeds)"),
        plt.Rectangle((0, 0), 1, 1, color=BAND_FILL,
                      label="[5, 95] percentile band"),
        plt.Line2D([0], [0], color=F_COLOR, linewidth=1.8,
                   label=r"$F$ = L5 two-channel probe"),
        plt.Line2D([0], [0], color=E_COLOR, linewidth=1.0, linestyle=(0, (4, 2)),
                   label=r"$E$ = L3 (confidence-only)"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.085), ncol=2,
               frameon=False, fontsize=8, columnspacing=1.6)

    print(f"caption needs: AVeriTeC k={k_values.get('averitec','?')}, "
          f"VitaminC k={k_values.get('vitaminc','?')}")
    WRITING_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = WRITING_DIR / "fig1_random_veto_selectivity.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    out_png = WRITING_DIR / "fig1_random_veto_selectivity.png"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
