#!/usr/bin/env python3
"""Figure 1: recognition versus commitment on the gold-Conflicting subset.

Panel (a) partitions each valid judge's gold-Conflicting cases by what the
fresh-context probe answered and what the judge actually committed to. The dark
red block is the object of the paper: the judge identified material conflict and
returned a direction anyway.

Panel (b) shows the probe's recognition rate per stratum for all four judges with
Wilson intervals, including the pre-specified validity condition on the preserved
stratum, so a reader can see which judges' readouts are admissible and why.

Reads only the analysis artifact; no raw run and no human-audit file.
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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

WORKSPACE = _cpo_workspace()

# House palette. Red family = committed directionally (unauthorized);
# green family = preserved. Dark = the probe recognised the conflict.
C_REC_DIR = "#B64342"
C_NOREC_DIR = "#E9A6A1"
C_REC_PRES = "#8BCF8B"
C_NOREC_PRES = "#DDF3DE"
C_GATE = "#767676"
STRATUM_COLORS = {
    "CPO": C_REC_DIR,
    "preserved": C_REC_PRES,
    "pure correct": "#3775BA",
    "insufficient": "#CFCECE",
}

NAMES = {
    "closed_openai_current": "GPT-5.6 Terra",
    "closed_anthropic_current": "Claude Sonnet 5",
    "openweight_gpt_oss_120b": "gpt-oss-120b",
    "openweight_mistral_small_4": "Mistral Small 4",
}
ORDER = [
    "closed_openai_current",
    "closed_anthropic_current",
    "openweight_gpt_oss_120b",
    "openweight_mistral_small_4",
]
STRATA = [
    ("S_cco", "CPO"),
    ("S_preserved", "preserved"),
    ("S_pure_correct", "pure correct"),
    ("S_insufficient", "insufficient"),
]
G1_MIN_PRESERVED = 0.70


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 15,
            "axes.linewidth": 2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_a(ax, report: dict) -> None:
    valid = [s for s in ORDER if report["models"][s]["g1_pass"]]
    ys = list(range(len(valid)))[::-1]
    height = 0.52
    for y, slot in zip(ys, valid):
        cells = report["models"][slot]["recognition_by_commitment_gold_conflicting"]
        rec_dir = cells.get("yes|directional", 0)
        norec_dir = cells.get("no|directional", 0)
        rec_pres = cells.get("yes|non-directional", 0)
        norec_pres = cells.get("no|non-directional", 0)
        segments = [
            (rec_dir, C_REC_DIR, "white"),
            (norec_dir, C_NOREC_DIR, "#272727"),
            (rec_pres, C_REC_PRES, "#272727"),
            (norec_pres, C_NOREC_PRES, "#272727"),
        ]
        left = 0
        for count, colour, textcolour in segments:
            ax.barh(y, count, height=height, left=left, color=colour,
                    edgecolor="black", linewidth=1.5)
            if count >= 12:
                ax.text(left + count / 2, y, str(count), ha="center", va="center",
                        color=textcolour, fontsize=15, fontweight="bold")
            left += count
        total_dir = rec_dir + norec_dir
        ax.text(total_dir + 2, y + height / 2 + 0.10,
                f"{rec_dir}/{total_dir} of directional commitments recognised",
                ha="left", va="bottom", fontsize=14, color="#272727")

    ax.set_yticks(ys)
    ax.set_yticklabels([NAMES[s] for s in valid], fontsize=15)
    ax.set_xlim(0, 168)
    ax.set_ylim(-0.44, len(valid) - 0.38)
    ax.set_xlabel("gold-Conflicting claims (N = 150)")  # no subscript: mathtext
    # subscripts render at 0.7x, i.e. 5.0pt after the page downscale
    ax.set_title("(a) What the judge saw, and what it committed to", fontsize=15, pad=12)
    ax.legend(
        handles=[
            Patch(facecolor=C_REC_DIR, edgecolor="black", label="recognised, committed directionally"),
            Patch(facecolor=C_NOREC_DIR, edgecolor="black", label="not recognised, committed directionally"),
            Patch(facecolor=C_REC_PRES, edgecolor="black", label="recognised, preserved"),
            Patch(facecolor=C_NOREC_PRES, edgecolor="black", label="not recognised, preserved"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.52), ncol=2,
        frameon=False, fontsize=14, handlelength=1.6, columnspacing=1.4,
    )


def panel_b(ax, report: dict) -> None:
    width = 0.20
    for index, (key, label) in enumerate(STRATA):
        xs, ys, los, his = [], [], [], []
        for position, slot in enumerate(ORDER):
            rate = report["models"][slot]["rates"][key]
            if rate["rate"] is None:
                continue
            lo, hi = rate["wilson"]
            xs.append(position + (index - 1.5) * width)
            ys.append(rate["rate"])
            los.append(rate["rate"] - lo)
            his.append(hi - rate["rate"])
        ax.bar(xs, ys, width=width, color=STRATUM_COLORS[label], edgecolor="black",
               linewidth=1.2, label=label)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="#272727",
                    elinewidth=1.4, capsize=3)

    # The dashed line is the validity condition on the 'preserved' stratum. It is
    # explained in the caption rather than annotated in-plot: at the right-hand end
    # the label crowded the "probe invalid" marker over the Mistral group.
    ax.axhline(G1_MIN_PRESERVED, color=C_GATE, linestyle="--", linewidth=1.8, zorder=0)

    # One marker per contiguous run of invalid groups, not one per group: two
    # identical "probe invalid" labels side by side read as a duplication bug
    # rather than as two models failing the same condition. A shaded span carries
    # which groups are covered, so the label only has to be said once.
    invalid = [i for i, slot in enumerate(ORDER)
               if not report["models"][slot]["g1_pass"]]
    runs = []
    for i in invalid:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    for run in runs:
        x0, x1 = run[0] - 0.46, run[-1] + 0.46
        ax.axvspan(x0, x1, color=C_REC_DIR, alpha=0.07, zorder=-1, linewidth=0)
        ax.text((x0 + x1) / 2, 0.975, "probe invalid", ha="center", va="top",
                fontsize=14, color=C_REC_DIR, style="italic")

    # Explicit two-line labels: automatic wrapping gave the four groups different
    # label heights, which is what the invalid marker collided with.
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(["GPT-5.6\nTerra", "Claude\nSonnet 5",
                        "gpt-oss\n120b", "Mistral\nSmall 4"], fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("recognises material conflict")
    ax.set_title("(b) Recognition rate by stratum, with Wilson intervals", fontsize=15, pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=4, frameon=False,
              fontsize=14, handlelength=1.4, columnspacing=1.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/probes/analysis/e1_recognition.json",
    )
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE / "overleaf-paper/figures/fig1_recognition.pdf"
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))

    style()
    # Wide canvas, large fonts, downscaled on the page: shrinking the canvas without
    # scaling the fonts collapsed the legends into the axis labels. At \textwidth in a
    # two-column layout the scale is 7.0/15.2 = 0.46, so the 15pt base font lands at
    # 6.9pt, the usual figure-text size. Every explicit size here is therefore >= 14pt:
    # anything smaller renders below 6.5pt on the page. Do not add a size under 14.
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 4.5), gridspec_kw={"width_ratios": [1.18, 1.0]})
    panel_a(axes[0], report)
    panel_b(axes[1], report)
    fig.tight_layout(pad=1.6)
    fig.subplots_adjust(bottom=0.34)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
