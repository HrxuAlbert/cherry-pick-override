"""Two-channel probe cross-dataset evaluation on VitaminC-Mixed.

Joins:
  - VitaminC-Mixed panel 4-opt outputs (votes + confidences)
  - validator outputs on VitaminC-Mixed (validity flags)
by case_id, then evaluates the typed-direct (A), validator-veto (D),
confidence-gate (E), and two-channel (F) controllers.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/option_a_exp/analysis"))
from analyze_selective_typed_controller import (
    compute_metrics, controller_d_validator_veto,
    controller_e_confidence, controller_f_combined, metrics_oneliner,
)

OUT_DIR = REPO / "outputs/option_a_exp/analysis/selective_typed_controller"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_e4_panel():
    """Return dict[case_id] = {gold, panel_4opt_votes, panel_4opt_confs, panel_4opt_agg}."""
    path = REPO / "outputs/option_a_exp/strengthening/e4_vitaminc_mixed/raw_results.jsonl"
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["system"] != "panel_3judge_4opt_strong":
                continue
            cid = r["case_id"]
            votes, confs = [], []
            for jo in r.get("judge_outputs", []):
                p = jo.get("parsed") or {}
                v = p.get("verdict_normal")
                c = p.get("confidence")
                if v in ("support", "refute", "insufficient", "conflicting"):
                    votes.append(v)
                if c is not None:
                    confs.append(c)
            out[cid] = {
                "case_key": r["case_key"],
                "gold": r["gold_normal"],
                "panel_4opt_votes": votes,
                "panel_4opt_confs": confs,
                "panel_4opt_agg": r["verdict_normal"],
            }
    return out


def load_e4_validator():
    """Return dict[case_id] = {validator_verdict, validity, case_key}."""
    path = REPO / "outputs/option_a_exp/strengthening/e3_validator_on_e4_vitaminc_mixed/raw_results.jsonl"
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["case_id"]] = {
                "case_key": r["case_key"],
                "validator_verdict": r.get("validator_verdict"),
                "validity": r.get("validity"),
                "prompt_final": r.get("prompt_final_verdict"),
            }
    return out


def main():
    e4_panel = load_e4_panel()
    e4_val = load_e4_validator()
    print(f"E4 panel cases: {len(e4_panel)}, validator cases: {len(e4_val)}")

    # Join + verify case_key alignment
    cases = []
    mismatches = 0
    for cid, d in e4_panel.items():
        v = e4_val.get(cid)
        if not v:
            mismatches += 1
            continue
        if v["case_key"] != d["case_key"]:
            mismatches += 1
            print(f"  WARN case_id={cid} key mismatch: panel={d['case_key']} validator={v['case_key']}")
            continue
        c = dict(d)
        c["case_id"] = cid
        c["validity"] = v["validity"]
        cases.append(c)
    print(f"Joined cases: {len(cases)} (mismatches: {mismatches})")
    if not cases:
        print("ERROR: no joined cases. Aborting.")
        return

    # ─── A baseline: typed direct ───
    a_scored = [{"gold": c["gold"], "pred": c["panel_4opt_agg"]} for c in cases]
    m_a = compute_metrics(a_scored)
    print("\n[A Panel 4-opt typed direct]", metrics_oneliner(m_a, "A_direct"))

    # ─── D validator-only veto ───
    d_results = []
    for mode in ["any", "conflict_only", "insufficient_only"]:
        scored = [{"gold": c["gold"],
                   "pred": controller_d_validator_veto(c, c["validity"], mode)} for c in cases]
        m = compute_metrics(scored)
        d_results.append({"mode": mode, "metrics": m})
        print(f"[D veto/{mode}]", metrics_oneliner(m, f"D_{mode}"))

    # ─── E confidence selective ───
    e_results = []
    for tau in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]:
        scored = [{"gold": c["gold"], "pred": controller_e_confidence(c, tau)} for c in cases]
        m = compute_metrics(scored)
        e_results.append({"tau": tau, "metrics": m})
    print("\n[E Confidence Selective]")
    for r in e_results:
        print(f"  τ={r['tau']}: " + metrics_oneliner(r["metrics"], f"E_tau={r['tau']}"))

    # ─── F = D conflict_only + E confidence ───
    f_results = []
    for tau in [0.0, 0.7, 0.8, 0.85, 0.9, 0.95]:
        scored = [{"gold": c["gold"],
                   "pred": controller_f_combined(c, c["validity"], tau)} for c in cases]
        m = compute_metrics(scored)
        f_results.append({"tau": tau, "metrics": m})
    print("\n[F two-channel: validator-veto + confidence-gate]")
    for r in f_results:
        print(f"  τ={r['tau']}: " + metrics_oneliner(r["metrics"], f"F_tau={r['tau']}"))

    # ─── F vs E Pareto dominance check (VitaminC) ───
    print("\n=== F vs E Pareto on VitaminC E4 (matched coverage) ===")
    print(f"{'F config':<15} {'F cov':>7} {'F sel_err':>10} {'F CCO':>8} | "
          f"{'E nearest':<10} {'E cov':>7} {'E sel_err':>10} {'E CCO':>8} | "
          f"{'Δ sel_err':>10} {'Δ CCO':>8}")
    for fr in f_results:
        if fr["tau"] in (0.85, 0.9, 0.95):
            fm = fr["metrics"]
            er = min(e_results, key=lambda er: abs(er["metrics"]["coverage"] - fm["coverage"]))
            em = er["metrics"]
            d_err = fm["sel_err"] - em["sel_err"]
            d_cco = fm["cco_full"] - em["cco_full"]
            print(f"F τ={fr['tau']:<10} {fm['coverage']:>7.3f} {fm['sel_err']:>10.3f} "
                  f"{fm['cco_full']:>8.3f} | E τ={er['tau']:<6} {em['coverage']:>7.3f} "
                  f"{em['sel_err']:>10.3f} {em['cco_full']:>8.3f} | "
                  f"{d_err:>+10.3f} {d_cco:>+8.3f}")

    # ─── Save CSV ───
    csv_path = OUT_DIR / "two_channel_probe_on_vitaminc.csv"
    fields = ["controller", "label", "params",
              "coverage", "sel_err", "sel_acc",
              "cco_full", "cco_confl", "acc_sr",
              "conflict_recall", "insufficient_recall",
              "macro_f1", "balanced_acc", "no_commit_n", "n"]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for label, m in [("A_panel_4opt_direct", m_a)]:
            w.writerow(["A", label, "", *[f"{m[k]:.4f}" for k in
                ("coverage","sel_err","sel_acc","cco_full","cco_confl","acc_sr",
                 "conflict_recall","insufficient_recall","macro_f1","balanced_acc")],
                m["no_commit_n"], m["n"]])
        for r in d_results:
            m = r["metrics"]
            w.writerow(["D", f"veto_{r['mode']}", "", *[f"{m[k]:.4f}" for k in
                ("coverage","sel_err","sel_acc","cco_full","cco_confl","acc_sr",
                 "conflict_recall","insufficient_recall","macro_f1","balanced_acc")],
                m["no_commit_n"], m["n"]])
        for r in e_results:
            m = r["metrics"]
            w.writerow(["E", f"conf_tau={r['tau']}", "", *[f"{m[k]:.4f}" for k in
                ("coverage","sel_err","sel_acc","cco_full","cco_confl","acc_sr",
                 "conflict_recall","insufficient_recall","macro_f1","balanced_acc")],
                m["no_commit_n"], m["n"]])
        for r in f_results:
            m = r["metrics"]
            w.writerow(["F", f"validator_veto+conf_tau={r['tau']}", "", *[f"{m[k]:.4f}" for k in
                ("coverage","sel_err","sel_acc","cco_full","cco_confl","acc_sr",
                 "conflict_recall","insufficient_recall","macro_f1","balanced_acc")],
                m["no_commit_n"], m["n"]])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
