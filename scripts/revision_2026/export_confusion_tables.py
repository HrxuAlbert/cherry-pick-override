#!/usr/bin/env python3
"""Generate the appendix table of verdicts by gold label (confusion matrices).

One block of four rows per judge (the typed panel and the four contemporary
judges), counts of each returned verdict for each gold label, on AVeriTeC and on
VitaminC-Mixed. The contemporary judges are read from the analysis output
``confusion_matrices.csv`` of each dataset; the panel is read from the cached
panel predictions. No API call, no human-audit artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _workspace(_f=__file__) -> Path:
    here = Path(_f).resolve()
    for parent in here.parents:
        if (parent / "outputs" / "revision_2026").is_dir():
            return parent
    return here.parents[4]


WORKSPACE = _workspace()
GOLDS = (("support", r"\textsc{Supports}"), ("refute", r"\textsc{Refutes}"),
         ("conflicting", r"\textsc{Conflicting}"), ("insufficient", r"\textsc{Insufficient}"))
PREDS = ("support", "refute", "conflicting", "insufficient")
JUDGES = (("closed_openai_current", "GPT-5.6 Terra"), ("closed_anthropic_current", "Claude Sonnet~5"),
          ("openweight_gpt_oss_120b", "gpt-oss-120b"), ("openweight_mistral_small_4", "Mistral Small~4"))
CURRENT = {"averitec": "outputs/revision_2026/current_models_averitec/analysis_dataset_defined/confusion_matrices.csv",
           "vitaminc": "outputs/revision_2026/current_models_vitaminc/analysis_dataset_defined/confusion_matrices.csv"}
PANEL = {"averitec": ("m0_commitment/v0.2/outputs/option_a_exp/strengthening/e1_full_4label_utility/raw_results.jsonl",
                      "outputs/option_a_exp/strengthening/e1_full_4label_utility/raw_results.jsonl"),
         "vitaminc": ("m0_commitment/v0.2/outputs/option_a_exp/strengthening/e4_vitaminc_mixed/raw_results.jsonl",
                      "outputs/option_a_exp/strengthening/e4_vitaminc_mixed/raw_results.jsonl")}
EXPECTED = {"averitec": {"support": 50, "refute": 50, "conflicting": 150, "insufficient": 35},
            "vitaminc": {"support": 50, "refute": 50, "conflicting": 100, "insufficient": 50}}
_OUTPUT_CANDIDATES = (WORKSPACE / "overleaf-paper/sections/appendix_confusion.tex", WORKSPACE / "appendix_confusion.tex")
OUTPUT = next((p for p in _OUTPUT_CANDIDATES if p.parent.is_dir()), _OUTPUT_CANDIDATES[0])


def first_existing(candidates: tuple[str, ...]) -> Path:
    for rel in candidates:
        if (WORKSPACE / rel).exists():
            return WORKSPACE / rel
    return WORKSPACE / candidates[0]


def current_counts(dataset: str) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    with (WORKSPACE / CURRENT[dataset]).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pred = row["prediction"] if row["prediction"] in PREDS else "none"
            out.setdefault(row["model_slot"], Counter())[(row["gold"], pred)] += int(row["count"])
    return out


def panel_counts(dataset: str) -> Counter:
    counts: Counter = Counter()
    path = first_existing(PANEL[dataset])
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("system") != "panel_3judge_4opt_strong":
            continue
        pred = row.get("verdict_normal")
        counts[(row["gold_normal"], pred if pred in PREDS else "none")] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    blocks: list[tuple[str, dict[str, Counter]]] = [("Typed panel", {d: panel_counts(d) for d in CURRENT})]
    current = {d: current_counts(d) for d in CURRENT}
    for slot, name in JUDGES:
        blocks.append((name, {d: current[d][slot] for d in CURRENT}))

    for name, by_dataset in blocks:                      # every row must sum to the fixed stratum size
        for dataset, counts in by_dataset.items():
            for gold, expected in EXPECTED[dataset].items():
                total = sum(v for (g, _), v in counts.items() if g == gold)
                if total != expected:
                    raise SystemExit(f"{name} / {dataset} / gold {gold}: {total} rows, expected {expected}")

    def cells(counts: Counter, gold: str) -> str:
        out = []
        for pred in PREDS + ("none",):
            value = counts.get((gold, pred), 0)
            text = str(value) if value or pred != "none" else "0"
            out.append(r"\textbf{%s}" % text if pred == gold else text)
        return " & ".join(out)

    rows = []
    for index, (name, by_dataset) in enumerate(blocks):
        if index:
            rows.append(r"\addlinespace[3pt]")
        for k, (gold, label) in enumerate(GOLDS):
            rows.append("%s & %s & %s & %s \\\\" % (name if k == 0 else "", label,
                                                   cells(by_dataset["averitec"], gold), cells(by_dataset["vitaminc"], gold)))

    tex = r"""% GENERATED FILE --- do not edit by hand.
% Produced by Writing/V0.2/revision_plan/scripts/export_confusion_tables.py from
% the analysis outputs (contemporary judges) and the cached panel predictions.
\section{Verdicts by Gold Label}
\label{sec:appendix:confusion}

Table~\ref{tab:confusion} gives the full verdict counts behind the rates in
Tables~\ref{tab:current_models} and~\ref{tab:main_results}. Directional
overcommitment is the mass in columns S and R of the \textsc{Conflicting} and
\textsc{Insufficient} rows. The reverse error, a non-directional verdict on
one-sided evidence, is the mass in columns C and I of the \textsc{Supports}
and \textsc{Refutes} rows.

\begin{table*}[t]
\centering
\tblsize
\tblsep{4pt}
\caption{Returned verdict by gold label, in claims: S, R, C and I are
\textsc{Supports}, \textsc{Refutes}, \textsc{Conflicting} and
\textsc{Insufficient}, and ``--'' is a response with no usable verdict.
Correct verdicts are in bold.}
\label{tab:confusion}
\begin{widetabular}{llrrrrrrrrrr}
\toprule
& & \multicolumn{5}{c}{AVeriTeC} & \multicolumn{5}{c}{VitaminC-Mixed} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-12}
Judge & Gold & S & R & C & I & -- & S & R & C & I & -- \\
\midrule
__ROWS__
\bottomrule
\end{widetabular}
\end{table*}
"""
    args.output.write_text(tex.replace("__ROWS__", "\n".join(rows)), encoding="utf-8")
    print(f"wrote {args.output}")
    for name, by_dataset in blocks:
        c = by_dataset["averitec"]
        print(f"  {name:<18} AVeriTeC gold-Conflicting -> S {c[('conflicting','support')]}, R {c[('conflicting','refute')]}, "
              f"C {c[('conflicting','conflicting')]}, I {c[('conflicting','insufficient')]}, none {c[('conflicting','none')]}")


if __name__ == "__main__":
    main()
