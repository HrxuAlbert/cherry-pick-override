"""Selective Typed Commitment Controller (STC / RCTC) — main Solution 2 rescue.

LLM (or typed panel) is treated as a proposal generator. An external rule-based
controller authorizes directional SUPPORT/REFUTE commitments only when the typed
vote distribution is stable, low-conflict, and low-risk. Otherwise it outputs
CONFLICTING, INSUFFICIENT, or NO-COMMIT.

Controllers implemented:
  A: Typed Direct Baseline      = use panel-aggregated typed verdict as-is
  B: Panel-Margin Controller    = sweep (tau_dir, tau_margin, tau_non, tau_conflict, tau_insufficient)
  C: Risk-Calibrated Controller = calibrate on dev split, eval on test
  D: Typed + Validator Veto     = veto S/R when validator flags mixed/insufficient
  E: Confidence Selective Base  = same proposal, commit only when mean conf >= τ

Outputs:
  outputs/option_a_exp/analysis/selective_typed_controller/
    controller_sweep_results.csv
    best_controllers_by_metric.csv
    controller_confusion_matrices.md
    controller_risk_coverage_points.csv
    controller_summary.md
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict, Counter
from itertools import product
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "outputs/option_a_exp/analysis/selective_typed_controller"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_4 = ["support", "refute", "insufficient", "conflicting"]
COMMITMENT_LABELS = LABELS_4 + ["no_commit"]


# ───────────────────────── Data loading ─────────────────────────

def load_e1():
    """Return dict[case_id] = {gold, panel_4opt_votes, panel_4opt_confs,
                               single_4opt, single_4opt_conf,
                               single_3opt, single_3opt_conf}"""
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
                votes = []
                confs = []
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
            elif sys == "single_sonnet_4opt_strong":
                if jos:
                    p = jos[0].get("parsed") or {}
                    data[cid]["single_4opt"] = p.get("verdict_normal")
                    data[cid]["single_4opt_conf"] = p.get("confidence")
            elif sys == "single_haiku_3opt":
                if jos:
                    p = jos[0].get("parsed") or {}
                    data[cid]["single_3opt"] = p.get("verdict_normal")
                    data[cid]["single_3opt_conf"] = p.get("confidence")
            elif sys == "single_haiku_4opt_strong":
                if jos:
                    p = jos[0].get("parsed") or {}
                    data[cid]["haiku_4opt"] = p.get("verdict_normal")
                    data[cid]["haiku_4opt_conf"] = p.get("confidence")
            elif sys == "panel_3judge_3opt":
                votes = []
                for jo in jos:
                    p = jo.get("parsed") or {}
                    v = p.get("verdict_normal")
                    if v in LABELS_4:
                        votes.append(v)
                data[cid]["panel_3opt_votes"] = votes
                data[cid]["panel_3opt_agg"] = r["verdict_normal"]
    return data


def load_e3_validator():
    """Return dict[case_id] = {validator_verdict, validity_flags, prompt_final}."""
    path = REPO / "outputs/option_a_exp/strengthening/e3_structured_certificate_validator_fewshot/raw_results.jsonl"
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            cid = r["case_id"]
            out[cid] = {
                "validator_verdict": r.get("validator_verdict"),
                "validity": r.get("validity") or {},
                "prompt_final": r.get("prompt_final_verdict"),
            }
    return out


def load_e4():
    """Return dict[case_id] = panel_4opt for cross-dataset eval."""
    path = REPO / "outputs/option_a_exp/strengthening/e4_vitaminc_mixed/raw_results.jsonl"
    data = defaultdict(dict)
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            sys = r["system"]
            if sys != "panel_3judge_4opt_strong":
                continue
            cid = r["case_id"]
            jos = r.get("judge_outputs", [])
            votes, confs = [], []
            for jo in jos:
                p = jo.get("parsed") or {}
                v = p.get("verdict_normal")
                c = p.get("confidence")
                if v in LABELS_4:
                    votes.append(v)
                if c is not None:
                    confs.append(c)
            data[cid] = {
                "gold": r["gold_normal"],
                "panel_4opt_votes": votes,
                "panel_4opt_confs": confs,
                "panel_4opt_agg": r["verdict_normal"],
            }
    return data


# ───────────────────────── Vote distribution ─────────────────────────

def vote_distribution(votes):
    """Return dict[label] = fraction (sums to 1.0)."""
    if not votes:
        return {l: 0.0 for l in LABELS_4}
    c = Counter(votes)
    n = len(votes)
    return {l: c.get(l, 0) / n for l in LABELS_4}


# ───────────────────────── Controllers ─────────────────────────

def controller_a_typed_direct(case):
    """Just use panel-aggregated typed verdict."""
    return case.get("panel_4opt_agg")


def controller_b_panel_margin(case, tau_dir, tau_margin, tau_non,
                                tau_conflict, tau_insufficient):
    """Panel-margin rule.

    1. directional check: top_directional >= tau_dir AND margin >= tau_margin
       AND nondirectional_mass <= tau_non
    2. else: conflict if p_C >= tau_conflict
    3. else: insufficient if p_I >= tau_insufficient
    4. else: no_commit
    """
    votes = case.get("panel_4opt_votes", [])
    if not votes:
        return "no_commit"
    d = vote_distribution(votes)
    p_S, p_R, p_I, p_C = d["support"], d["refute"], d["insufficient"], d["conflicting"]
    dir_mass = p_S + p_R
    non_mass = p_I + p_C
    margin = abs(p_S - p_R)
    top_dir = max(p_S, p_R)

    if top_dir >= tau_dir and margin >= tau_margin and non_mass <= tau_non:
        return "support" if p_S > p_R else "refute"
    if p_C >= tau_conflict:
        return "conflicting"
    if p_I >= tau_insufficient:
        return "insufficient"
    return "no_commit"


def controller_d_validator_veto(case, val_flags, val_mode="any"):
    """Take panel 4-opt verdict; veto S/R if validator says mixed/insufficient.

    val_mode = "any":  veto if has_material_mixed OR has_material_insufficient
                       (downgrade to CONFLICTING / INSUFFICIENT respectively)
    val_mode = "conflict_only": veto only if has_material_mixed
    val_mode = "insufficient_only": veto only if has_material_insufficient
    """
    base = case.get("panel_4opt_agg")
    if base not in ("support", "refute"):
        return base
    if not val_flags:
        return base
    has_mixed = val_flags.get("has_material_mixed", False)
    has_insuf = val_flags.get("has_material_insufficient", False)
    if val_mode == "any":
        if has_mixed:
            return "conflicting"
        if has_insuf:
            return "insufficient"
        return base
    if val_mode == "conflict_only":
        return "conflicting" if has_mixed else base
    if val_mode == "insufficient_only":
        return "insufficient" if has_insuf else base
    return base


def controller_e_confidence(case, tau_conf):
    """Selective baseline: same panel proposal but commit only if mean conf >= τ.

    Below τ → downgrade S/R to no_commit (NEI-equivalent for selective).
    """
    base = case.get("panel_4opt_agg")
    confs = case.get("panel_4opt_confs", [])
    if not confs:
        return base
    mean_c = sum(confs) / len(confs)
    if base in ("support", "refute") and mean_c < tau_conf:
        return "no_commit"
    return base


def controller_f_combined(case, val_flags, tau_conf):
    """F: Validator-Guided Risk-Controlled Controller.

    Two-stage external authorization:
      Stage 1 (validator veto): if typed proposal is S/R AND validator flags
                                 material_mixed → downgrade to CONFLICTING.
      Stage 2 (confidence threshold): if surviving proposal is still S/R AND
                                       mean panel confidence < τ → no_commit.

    Uses structural evidence signal (Stage 1) + model self-confidence (Stage 2)
    — two orthogonal authorization channels.
    """
    # Stage 1: validator veto
    pred = controller_d_validator_veto(case, val_flags, val_mode="conflict_only")
    # Stage 2: confidence threshold
    if pred in ("support", "refute"):
        confs = case.get("panel_4opt_confs", [])
        if confs:
            mean_c = sum(confs) / len(confs)
            if mean_c < tau_conf:
                pred = "no_commit"
    return pred


# ───────────────────────── Metrics ─────────────────────────

def compute_metrics(cases, pred_field="pred"):
    """Compute common-denominator metrics. cases = list of dicts with gold + pred.

    Returns:
      coverage          = (S+R commits) / N
      sel_err           = wrong (S/R) commits / commits
      sel_acc           = correct (S/R) commits / commits
      cco_full          = (conflicting gold predicted S/R) / N      [same denom!]
      cco_confl         = (conflicting gold predicted S/R) / N_confl
      acc_sr            = (correct S/R commits) / N_sr
      acc_sr_strict     = correct / N_sr (committed wrong or abstain counted as wrong)
      conflict_recall   = (conflicting gold predicted C) / N_confl
      insufficient_recall = (insufficient gold predicted I) / N_insuf
      no_commit_n       = number of NO-COMMIT outputs
      macro_f1, balanced_acc  (over 4 labels; no_commit excluded from per-class)
    """
    n = len(cases)
    n_confl = sum(1 for c in cases if c["gold"] == "conflicting")
    n_sr = sum(1 for c in cases if c["gold"] in ("support", "refute"))
    n_insuf = sum(1 for c in cases if c["gold"] == "insufficient")

    commits = 0
    correct_commits = 0
    wrong_commits = 0
    cco_mistakes = 0
    sr_correct = 0
    conflict_correct = 0
    insuf_correct = 0
    no_commit = 0

    for c in cases:
        g = c["gold"]
        p = c[pred_field]
        if p == "no_commit":
            no_commit += 1
        if p in ("support", "refute"):
            commits += 1
            if g == p:
                correct_commits += 1
                if g in ("support", "refute"):
                    sr_correct += 1
            else:
                wrong_commits += 1
                if g == "conflicting":
                    cco_mistakes += 1
        if g == "conflicting" and p == "conflicting":
            conflict_correct += 1
        if g == "insufficient" and p == "insufficient":
            insuf_correct += 1

    coverage = commits / n if n else 0
    sel_err = wrong_commits / commits if commits else 0
    sel_acc = correct_commits / commits if commits else 0
    cco_full = cco_mistakes / n if n else 0
    cco_confl = cco_mistakes / n_confl if n_confl else 0
    acc_sr = sr_correct / n_sr if n_sr else 0
    conflict_recall = conflict_correct / n_confl if n_confl else 0
    insuf_recall = insuf_correct / n_insuf if n_insuf else 0

    # Per-class F1
    cm = defaultdict(int)
    for c in cases:
        g = c["gold"]
        p = c[pred_field]
        if g in LABELS_4:
            cm[(g, p)] += 1
    per_class = {}
    f1s, recalls = [], []
    for cl in LABELS_4:
        tp = cm.get((cl, cl), 0)
        fp = sum(cm.get((g, cl), 0) for g in LABELS_4 if g != cl)
        fn = sum(cm.get((cl, p), 0) for p in COMMITMENT_LABELS if p != cl)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        per_class[cl] = {"p": prec, "r": rec, "f1": f1, "n": tp + fn}
        f1s.append(f1); recalls.append(rec)
    macro_f1 = mean(f1s)
    bal_acc = mean(recalls)

    return {
        "n": n, "n_confl": n_confl, "n_sr": n_sr, "n_insuf": n_insuf,
        "coverage": coverage,
        "sel_err": sel_err,
        "sel_acc": sel_acc,
        "cco_full": cco_full,
        "cco_confl": cco_confl,
        "acc_sr": acc_sr,
        "conflict_recall": conflict_recall,
        "insufficient_recall": insuf_recall,
        "no_commit_n": no_commit,
        "macro_f1": macro_f1,
        "balanced_acc": bal_acc,
        "per_class": per_class,
        "cm": dict(cm),
    }


# ───────────────────────── Split utility ─────────────────────────

def stratified_split(cases, seed=0, dev_frac=0.5):
    """Stratify by gold_normal, deterministic split."""
    by_gold = defaultdict(list)
    for i, c in enumerate(cases):
        by_gold[c["gold"]].append(i)
    rnd = random.Random(seed)
    dev_idx, test_idx = set(), set()
    for g, idxs in by_gold.items():
        rnd.shuffle(idxs)
        cut = int(len(idxs) * dev_frac)
        dev_idx.update(idxs[:cut])
        test_idx.update(idxs[cut:])
    return [cases[i] for i in sorted(dev_idx)], [cases[i] for i in sorted(test_idx)]


# ───────────────────────── Sweep + Calibration ─────────────────────────

def sweep_controller_b(cases, *, taus=None):
    """Sweep panel-margin thresholds on cases. Return list of (params, metrics)."""
    if taus is None:
        taus = {
            "tau_dir":         [0.5, 0.67, 0.75, 1.0],
            "tau_margin":      [0.0, 0.33, 0.5, 0.67],
            "tau_non":         [0.0, 0.25, 0.33, 0.5],
            "tau_conflict":    [0.34, 0.5, 0.67],
            "tau_insufficient":[0.34, 0.5, 0.67],
        }
    results = []
    keys = list(taus.keys())
    for combo in product(*[taus[k] for k in keys]):
        params = dict(zip(keys, combo))
        scored = []
        for c in cases:
            pred = controller_b_panel_margin(c, **params)
            scored.append({"gold": c["gold"], "pred": pred})
        m = compute_metrics(scored)
        results.append({"params": params, "metrics": m})
    return results


def calibrate_controller_c(dev_cases, test_cases, alpha_targets):
    """For each α target, pick params on dev that satisfy CCO/sel_err ≤ α
       while maximizing coverage. Then evaluate on test."""
    dev_results = sweep_controller_b(dev_cases)
    out = []
    for alpha in alpha_targets:
        # Candidates: CCO_full <= α on dev
        cands = [r for r in dev_results if r["metrics"]["cco_full"] <= alpha]
        if not cands:
            out.append({"alpha": alpha, "ok": False, "best_params": None,
                        "dev_metrics": None, "test_metrics": None})
            continue
        # Among candidates, maximize coverage; tie-break by macro_f1
        best = max(cands, key=lambda r: (r["metrics"]["coverage"], r["metrics"]["macro_f1"]))
        # Evaluate on test
        test_scored = []
        for c in test_cases:
            pred = controller_b_panel_margin(c, **best["params"])
            test_scored.append({"gold": c["gold"], "pred": pred})
        test_metrics = compute_metrics(test_scored)
        out.append({"alpha": alpha, "ok": True,
                    "best_params": best["params"],
                    "dev_metrics": best["metrics"],
                    "test_metrics": test_metrics})
    return out


# ───────────────────────── Reporting helpers ─────────────────────────

def cm_markdown(cm, title):
    out = [f"### {title}\n"]
    out.append("| gold ↓ \\ pred → | support | refute | insufficient | conflicting | no_commit | total |")
    out.append("|---|---|---|---|---|---|---|")
    for g in LABELS_4:
        cells = [cm.get((g, p), 0) for p in COMMITMENT_LABELS]
        tot = sum(cells)
        out.append(f"| **{g}** | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} | {tot} |")
    return "\n".join(out)


def metrics_oneliner(m, label):
    return (f"**{label}** | cov={m['coverage']:.3f} | sel_err={m['sel_err']:.3f} | "
            f"CCO_full={m['cco_full']:.3f} | CCO_confl={m['cco_confl']:.3f} | "
            f"acc_S/R={m['acc_sr']:.3f} | conf_recall={m['conflict_recall']:.3f} | "
            f"macro_F1={m['macro_f1']:.3f} | bal_acc={m['balanced_acc']:.3f} | "
            f"no_commit={m['no_commit_n']}")


# ───────────────────────── Main ─────────────────────────

def main():
    e1 = load_e1()
    e3_val = load_e3_validator()

    # Build E1 case list with everything joined on case_id
    e1_cases = []
    for cid, d in e1.items():
        if "gold" not in d or "panel_4opt_agg" not in d:
            continue
        c = dict(d)
        c["case_id"] = cid
        v = e3_val.get(cid)
        if v:
            c["validator_verdict"] = v.get("validator_verdict")
            c["validity"] = v.get("validity")
        else:
            c["validator_verdict"] = None
            c["validity"] = None
        e1_cases.append(c)
    e1_cases.sort(key=lambda x: x["case_id"])
    print(f"Loaded E1: {len(e1_cases)} cases")
    gold_dist = Counter(c["gold"] for c in e1_cases)
    print(f"  Gold distribution: {dict(gold_dist)}")
    matched = sum(1 for c in e1_cases if c["validity"])
    print(f"  Validator-matched: {matched}")

    # ─── Variant A: typed direct (= Panel+typed H12 baseline) ───
    a_scored = [{"gold": c["gold"], "pred": controller_a_typed_direct(c)} for c in e1_cases]
    m_a = compute_metrics(a_scored)
    print("\n[A Typed Direct]", metrics_oneliner(m_a, "panel_4opt_direct"))

    # Also: single Sonnet 4-opt typed direct
    sonnet_scored = [{"gold": c["gold"], "pred": c.get("single_4opt") or "no_commit"} for c in e1_cases]
    m_sonnet = compute_metrics(sonnet_scored)
    print("[A Sonnet 4-opt direct]", metrics_oneliner(m_sonnet, "single_sonnet_4opt"))

    # Reference baselines (3-opt for context)
    haiku3_scored = [{"gold": c["gold"], "pred": c.get("single_3opt") or "no_commit"} for c in e1_cases]
    m_haiku3 = compute_metrics(haiku3_scored)
    print("[Baseline Haiku 3-opt]", metrics_oneliner(m_haiku3, "single_haiku_3opt"))

    # ─── Variant B: panel-margin sweep on full E1 ───
    b_results = sweep_controller_b(e1_cases)
    print(f"\n[B Panel-Margin] {len(b_results)} threshold combos swept")

    # ─── Variant C: calibration with dev/test split ───
    dev, test = stratified_split(e1_cases, seed=0, dev_frac=0.5)
    print(f"\n[C Risk-Calibrated] dev N={len(dev)}, test N={len(test)}")
    c_results = calibrate_controller_c(dev, test, [0.10, 0.15, 0.20, 0.25, 0.30])
    for r in c_results:
        if not r["ok"]:
            print(f"  α={r['alpha']:.2f}: NO valid params on dev")
            continue
        tm = r["test_metrics"]
        print(f"  α={r['alpha']:.2f}: test cov={tm['coverage']:.3f} CCO_full={tm['cco_full']:.3f} "
              f"sel_err={tm['sel_err']:.3f} acc_SR={tm['acc_sr']:.3f}")

    # ─── Variant D: validator-as-veto ───
    matched_cases = [c for c in e1_cases if c["validity"]]
    d_modes = ["any", "conflict_only", "insufficient_only"]
    d_results = []
    for mode in d_modes:
        scored = [{"gold": c["gold"], "pred": controller_d_validator_veto(c, c["validity"], mode)}
                  for c in matched_cases]
        m = compute_metrics(scored)
        d_results.append({"mode": mode, "metrics": m})
        print(f"\n[D Validator Veto / {mode}]", metrics_oneliner(m, f"veto_{mode}"))

    # ─── Variant E: confidence-threshold selective on panel 4-opt ───
    e_results = []
    for tau in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        scored = [{"gold": c["gold"], "pred": controller_e_confidence(c, tau)} for c in e1_cases]
        m = compute_metrics(scored)
        e_results.append({"tau": tau, "metrics": m})
    print("\n[E Confidence Selective]")
    for r in e_results:
        print(f"  τ={r['tau']:.2f}: " + metrics_oneliner(r["metrics"], f"conf_tau={r['tau']}"))

    # ─── Variant F: D + E combined (Validator-Guided RCTC) ───
    f_results = []
    for tau in [0.0, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        scored = [{"gold": c["gold"],
                   "pred": controller_f_combined(c, c["validity"], tau)} for c in e1_cases]
        m = compute_metrics(scored)
        f_results.append({"tau": tau, "metrics": m})
    print("\n[F Validator-Guided RCTC = D conflict_only + E confidence]")
    for r in f_results:
        print(f"  τ={r['tau']:.2f}: " + metrics_oneliner(r["metrics"], f"F_tau={r['tau']}"))

    # ─── Find Pareto-best B configs ───
    # Pareto on (sel_err, -coverage). Lower sel_err + higher coverage = better
    def pareto_front(results, x_key, y_key):
        """x_key = sel_err (min), y_key = coverage (max)."""
        sorted_r = sorted(results, key=lambda r: (r["metrics"][x_key], -r["metrics"][y_key]))
        front = []
        best_cov = -1
        for r in sorted_r:
            if r["metrics"][y_key] > best_cov:
                front.append(r)
                best_cov = r["metrics"][y_key]
        return front

    b_pareto = pareto_front(b_results, "sel_err", "coverage")
    print(f"\n[B Pareto front] {len(b_pareto)} non-dominated configs")
    for r in b_pareto:
        m = r["metrics"]
        print(f"  sel_err={m['sel_err']:.3f} cov={m['coverage']:.3f} "
              f"CCO_full={m['cco_full']:.3f} acc_SR={m['acc_sr']:.3f} "
              f"params={r['params']}")

    # ─── Best B configs by criterion ───
    # 1. balanced: maximize macro_f1 with coverage >= 0.35 and acc_sr >= 0.60
    bal_cands = [r for r in b_results
                 if r["metrics"]["coverage"] >= 0.35 and r["metrics"]["acc_sr"] >= 0.60]
    best_balanced = max(bal_cands, key=lambda r: r["metrics"]["macro_f1"]) if bal_cands else None

    # 2. safety: minimize CCO_full with coverage >= 0.25
    safe_cands = [r for r in b_results if r["metrics"]["coverage"] >= 0.25]
    best_safety = min(safe_cands, key=lambda r: r["metrics"]["cco_full"]) if safe_cands else None

    # 3. coverage-max: cov >= 0.50 with min sel_err
    cov_cands = [r for r in b_results if r["metrics"]["coverage"] >= 0.50]
    best_cov = min(cov_cands, key=lambda r: r["metrics"]["sel_err"]) if cov_cands else None

    print("\n[B Best by Criterion]")
    for name, r in [("balanced (cov≥0.35, acc_SR≥0.60, max macro_F1)", best_balanced),
                    ("safety (cov≥0.25, min CCO_full)", best_safety),
                    ("coverage (cov≥0.50, min sel_err)", best_cov)]:
        if r:
            print(f"  {name}:")
            print(f"    {metrics_oneliner(r['metrics'], 'B')}")
            print(f"    params: {r['params']}")
        else:
            print(f"  {name}: NO candidate")

    # ─── Write all CSVs and markdown ───

    # 1. controller_sweep_results.csv (all B sweep + A + D + E)
    sweep_csv = OUT_DIR / "controller_sweep_results.csv"
    fields = ["controller", "label", "params",
              "coverage", "sel_err", "sel_acc",
              "cco_full", "cco_confl", "acc_sr",
              "conflict_recall", "insufficient_recall",
              "macro_f1", "balanced_acc", "no_commit_n", "n"]
    with sweep_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        # A
        for label, m in [("panel_4opt_typed_direct", m_a),
                         ("single_sonnet_4opt_direct", m_sonnet),
                         ("single_haiku_3opt_baseline", m_haiku3)]:
            w.writerow(["A", label, "", *[f"{m[k]:.4f}" for k in
                ("coverage","sel_err","sel_acc","cco_full","cco_confl","acc_sr",
                 "conflict_recall","insufficient_recall","macro_f1","balanced_acc")],
                m["no_commit_n"], m["n"]])
        # B (full sweep)
        for r in b_results:
            m = r["metrics"]
            w.writerow(["B", "panel_margin",
                json.dumps(r["params"]),
                *[f"{m[k]:.4f}" for k in
                ("coverage","sel_err","sel_acc","cco_full","cco_confl","acc_sr",
                 "conflict_recall","insufficient_recall","macro_f1","balanced_acc")],
                m["no_commit_n"], m["n"]])
        # C (calibration test metrics)
        for r in c_results:
            if not r["ok"]:
                continue
            m = r["test_metrics"]
            w.writerow(["C", f"calibrated_alpha={r['alpha']}",
                json.dumps(r["best_params"]),
                *[f"{m[k]:.4f}" for k in
                ("coverage","sel_err","sel_acc","cco_full","cco_confl","acc_sr",
                 "conflict_recall","insufficient_recall","macro_f1","balanced_acc")],
                m["no_commit_n"], m["n"]])
        # D (validator veto)
        for r in d_results:
            m = r["metrics"]
            w.writerow(["D", f"veto_{r['mode']}", "",
                *[f"{m[k]:.4f}" for k in
                ("coverage","sel_err","sel_acc","cco_full","cco_confl","acc_sr",
                 "conflict_recall","insufficient_recall","macro_f1","balanced_acc")],
                m["no_commit_n"], m["n"]])
        # E (confidence selective)
        for r in e_results:
            m = r["metrics"]
            w.writerow(["E", f"conf_tau={r['tau']}", "",
                *[f"{m[k]:.4f}" for k in
                ("coverage","sel_err","sel_acc","cco_full","cco_confl","acc_sr",
                 "conflict_recall","insufficient_recall","macro_f1","balanced_acc")],
                m["no_commit_n"], m["n"]])
        # F (validator + confidence combined)
        for r in f_results:
            m = r["metrics"]
            w.writerow(["F", f"validator_veto+conf_tau={r['tau']}", "",
                *[f"{m[k]:.4f}" for k in
                ("coverage","sel_err","sel_acc","cco_full","cco_confl","acc_sr",
                 "conflict_recall","insufficient_recall","macro_f1","balanced_acc")],
                m["no_commit_n"], m["n"]])
    print(f"\nWrote {sweep_csv}")

    # 2. best_controllers_by_metric.csv
    best_csv = OUT_DIR / "best_controllers_by_metric.csv"
    with best_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "controller", "params"] + fields[3:])
        for name, r in [("balanced", best_balanced),
                        ("safety", best_safety),
                        ("max_coverage", best_cov)]:
            if r:
                m = r["metrics"]
                w.writerow([name, "B_panel_margin", json.dumps(r["params"]),
                    *[f"{m[k]:.4f}" for k in
                    ("coverage","sel_err","sel_acc","cco_full","cco_confl","acc_sr",
                     "conflict_recall","insufficient_recall","macro_f1","balanced_acc")],
                    m["no_commit_n"], m["n"]])
    print(f"Wrote {best_csv}")

    # 3. controller_risk_coverage_points.csv (for plotting)
    rc_csv = OUT_DIR / "controller_risk_coverage_points.csv"
    with rc_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "label", "coverage", "sel_err", "cco_full",
                    "cco_confl", "acc_sr", "macro_f1"])
        for tag, label, m in [
            ("A_baseline", "Panel 4-opt typed (A)", m_a),
            ("A_baseline", "Sonnet 4-opt typed (A-single)", m_sonnet),
            ("A_baseline", "Haiku 3-opt (raw baseline)", m_haiku3),
        ]:
            w.writerow([tag, label, f"{m['coverage']:.4f}", f"{m['sel_err']:.4f}",
                        f"{m['cco_full']:.4f}", f"{m['cco_confl']:.4f}",
                        f"{m['acc_sr']:.4f}", f"{m['macro_f1']:.4f}"])
        for r in b_pareto:
            m = r["metrics"]
            w.writerow(["B_pareto", f"PanelMargin {r['params']}",
                        f"{m['coverage']:.4f}", f"{m['sel_err']:.4f}",
                        f"{m['cco_full']:.4f}", f"{m['cco_confl']:.4f}",
                        f"{m['acc_sr']:.4f}", f"{m['macro_f1']:.4f}"])
        for r in c_results:
            if not r["ok"]: continue
            m = r["test_metrics"]
            w.writerow(["C_calibrated", f"Calibrated α={r['alpha']}",
                        f"{m['coverage']:.4f}", f"{m['sel_err']:.4f}",
                        f"{m['cco_full']:.4f}", f"{m['cco_confl']:.4f}",
                        f"{m['acc_sr']:.4f}", f"{m['macro_f1']:.4f}"])
        for r in d_results:
            m = r["metrics"]
            w.writerow(["D_validator_veto", f"Veto {r['mode']}",
                        f"{m['coverage']:.4f}", f"{m['sel_err']:.4f}",
                        f"{m['cco_full']:.4f}", f"{m['cco_confl']:.4f}",
                        f"{m['acc_sr']:.4f}", f"{m['macro_f1']:.4f}"])
        for r in e_results:
            m = r["metrics"]
            w.writerow(["E_conf_selective", f"Conf τ={r['tau']}",
                        f"{m['coverage']:.4f}", f"{m['sel_err']:.4f}",
                        f"{m['cco_full']:.4f}", f"{m['cco_confl']:.4f}",
                        f"{m['acc_sr']:.4f}", f"{m['macro_f1']:.4f}"])
        for r in f_results:
            m = r["metrics"]
            w.writerow(["F_combined", f"F validator+conf τ={r['tau']}",
                        f"{m['coverage']:.4f}", f"{m['sel_err']:.4f}",
                        f"{m['cco_full']:.4f}", f"{m['cco_confl']:.4f}",
                        f"{m['acc_sr']:.4f}", f"{m['macro_f1']:.4f}"])
    print(f"Wrote {rc_csv}")

    # 4. controller_confusion_matrices.md (best B configs + A + D + E)
    cm_md = OUT_DIR / "controller_confusion_matrices.md"
    sections = ["# Controller Confusion Matrices\n"]
    sections.append("## Baseline references\n")
    sections.append(cm_markdown(m_a["cm"], "A: Panel 4-opt typed direct (N=285)"))
    sections.append(cm_markdown(m_sonnet["cm"], "A: Sonnet 4-opt typed direct (N=285)"))
    sections.append(cm_markdown(m_haiku3["cm"], "Baseline: Haiku 3-opt (N=285)"))
    if best_balanced:
        sections.append(f"\n## B Balanced: params={best_balanced['params']}\n")
        sections.append(cm_markdown(best_balanced["metrics"]["cm"], "B balanced"))
    if best_safety:
        sections.append(f"\n## B Safety: params={best_safety['params']}\n")
        sections.append(cm_markdown(best_safety["metrics"]["cm"], "B safety"))
    if best_cov:
        sections.append(f"\n## B Coverage: params={best_cov['params']}\n")
        sections.append(cm_markdown(best_cov["metrics"]["cm"], "B coverage"))
    for r in c_results:
        if not r["ok"]: continue
        sections.append(f"\n## C Calibrated α={r['alpha']}: params={r['best_params']}\n")
        sections.append(cm_markdown(r["test_metrics"]["cm"],
                                     f"C calibrated α={r['alpha']} on TEST"))
    for r in d_results:
        sections.append(f"\n## D Validator Veto ({r['mode']})\n")
        sections.append(cm_markdown(r["metrics"]["cm"], f"D veto {r['mode']}"))
    for r in e_results:
        if r["tau"] in (0.7, 0.8, 0.9):
            sections.append(f"\n## E Confidence Selective τ={r['tau']}\n")
            sections.append(cm_markdown(r["metrics"]["cm"], f"E conf τ={r['tau']}"))
    for r in f_results:
        if r["tau"] in (0.8, 0.85, 0.9):
            sections.append(f"\n## F Validator-Guided RCTC τ={r['tau']}\n")
            sections.append(cm_markdown(r["metrics"]["cm"], f"F τ={r['tau']}"))
    cm_md.write_text("\n\n".join(sections))
    print(f"Wrote {cm_md}")

    # 5. controller_summary.md (decision memo seed)
    summary = OUT_DIR / "controller_summary.md"
    survival = {
        "balanced":   best_balanced is not None and (
            best_balanced["metrics"]["acc_sr"] >= 0.60 and
            best_balanced["metrics"]["coverage"] >= 0.35 and
            (m_a["sel_err"] - best_balanced["metrics"]["sel_err"]) >= 0.05 and
            best_balanced["metrics"]["cco_full"] < m_a["cco_full"]
        ),
        "pareto":     False,  # filled below by comparison with E
        "safety":     best_safety is not None and best_safety["metrics"]["cco_full"] < 0.05,
    }
    # Pareto criterion: at comparable coverage to E selective, does any B point dominate?
    pareto_wins = 0
    for er in e_results:
        em = er["metrics"]
        for br in b_results:
            bm = br["metrics"]
            if abs(bm["coverage"] - em["coverage"]) <= 0.05 and bm["sel_err"] < em["sel_err"] - 0.02:
                pareto_wins += 1
                break
    survival["pareto"] = pareto_wins >= len(e_results) // 2

    lines = [
        "# Selective Typed Commitment Controller — Decision Memo\n",
        f"E1 N={len(e1_cases)}, validator-matched={matched}\n",
        "## Reference baselines\n",
        f"- A Panel 4-opt typed direct: cov={m_a['coverage']:.3f}, sel_err={m_a['sel_err']:.3f}, "
        f"CCO_full={m_a['cco_full']:.3f}, acc_S/R={m_a['acc_sr']:.3f}",
        f"- A Sonnet 4-opt typed: cov={m_sonnet['coverage']:.3f}, sel_err={m_sonnet['sel_err']:.3f}, "
        f"CCO_full={m_sonnet['cco_full']:.3f}, acc_S/R={m_sonnet['acc_sr']:.3f}",
        f"- Haiku 3-opt baseline: cov={m_haiku3['coverage']:.3f}, sel_err={m_haiku3['sel_err']:.3f}, "
        f"CCO_full={m_haiku3['cco_full']:.3f}, acc_S/R={m_haiku3['acc_sr']:.3f}",
        "\n## Best B (Panel-Margin Controller) configs\n",
    ]
    for name, r in [("Balanced (cov≥0.35, acc_S/R≥0.60, max macro_F1)", best_balanced),
                    ("Safety (cov≥0.25, min CCO_full)", best_safety),
                    ("Coverage (cov≥0.50, min sel_err)", best_cov)]:
        if r:
            m = r["metrics"]
            lines.append(f"### {name}")
            lines.append(f"- params: {r['params']}")
            lines.append(f"- cov={m['coverage']:.3f}, sel_err={m['sel_err']:.3f}, "
                         f"CCO_full={m['cco_full']:.3f}, acc_S/R={m['acc_sr']:.3f}, "
                         f"conflict_recall={m['conflict_recall']:.3f}, macro_F1={m['macro_f1']:.3f}, "
                         f"no_commit={m['no_commit_n']}\n")
        else:
            lines.append(f"### {name}: NO candidate satisfies constraints\n")

    lines.append("## C Risk-Calibrated (dev N=%d, test N=%d)\n" % (len(dev), len(test)))
    for r in c_results:
        if not r["ok"]:
            lines.append(f"- α={r['alpha']}: NO valid params on dev")
            continue
        tm = r["test_metrics"]
        lines.append(f"- α={r['alpha']}: test cov={tm['coverage']:.3f} "
                     f"sel_err={tm['sel_err']:.3f} CCO_full={tm['cco_full']:.3f} "
                     f"acc_S/R={tm['acc_sr']:.3f} macro_F1={tm['macro_f1']:.3f} "
                     f"params={r['best_params']}")
    lines.append("\n## D Validator-as-Veto\n")
    for r in d_results:
        m = r["metrics"]
        lines.append(f"- {r['mode']}: cov={m['coverage']:.3f} sel_err={m['sel_err']:.3f} "
                     f"CCO_full={m['cco_full']:.3f} acc_S/R={m['acc_sr']:.3f} "
                     f"macro_F1={m['macro_f1']:.3f}")
    lines.append("\n## E Confidence Selective Baseline (panel mean conf)\n")
    for r in e_results:
        m = r["metrics"]
        lines.append(f"- τ={r['tau']}: cov={m['coverage']:.3f} sel_err={m['sel_err']:.3f} "
                     f"CCO_full={m['cco_full']:.3f} acc_S/R={m['acc_sr']:.3f}")

    lines.append("\n## F Validator-Guided RCTC (Stage1 veto + Stage2 conf threshold)\n")
    for r in f_results:
        m = r["metrics"]
        lines.append(f"- τ={r['tau']}: cov={m['coverage']:.3f} sel_err={m['sel_err']:.3f} "
                     f"CCO_full={m['cco_full']:.3f} acc_S/R={m['acc_sr']:.3f} "
                     f"macro_F1={m['macro_f1']:.3f}")

    # F vs E Pareto comparison
    lines.append("\n## F vs E — Pareto dominance check\n")
    lines.append("| target cov | E sel_err | F sel_err | Δ | E CCO | F CCO | Δ |")
    lines.append("|---|---|---|---|---|---|---|")
    # Compare F at each τ vs E at nearest coverage
    for fr in f_results:
        if fr["tau"] in (0.85, 0.9, 0.95):
            fm = fr["metrics"]
            # Find nearest E by coverage
            er_near = min(e_results, key=lambda er: abs(er["metrics"]["coverage"] - fm["coverage"]))
            em = er_near["metrics"]
            d_err = fm["sel_err"] - em["sel_err"]
            d_cco = fm["cco_full"] - em["cco_full"]
            lines.append(f"| F τ={fr['tau']} (cov={fm['coverage']:.3f}) | "
                         f"E τ={er_near['tau']} → {em['sel_err']:.3f} | {fm['sel_err']:.3f} | "
                         f"{d_err:+.3f} | {em['cco_full']:.3f} | {fm['cco_full']:.3f} | "
                         f"{d_cco:+.3f} |")

    lines.append("\n## Survival Criteria Check\n")
    lines.append(f"- **Balanced improvement**: {'✅ PASS' if survival['balanced'] else '❌ FAIL'}")
    lines.append(f"- **Risk-coverage dominance over E**: "
                 f"{'✅ PASS' if survival['pareto'] else '❌ FAIL'} "
                 f"(B dominated E at {pareto_wins}/{len(e_results)} τ-points)")
    lines.append(f"- **Safety mode**: {'✅ PASS' if survival['safety'] else '❌ FAIL'} "
                 f"(needs CCO_full < 0.05)")

    summary.write_text("\n".join(lines))
    print(f"Wrote {summary}")

    # ─── Cross-dataset E4 sanity ───
    # NB: E4 has no validator outputs (E3 was AVeriTeC-only). So F variant cannot
    # be evaluated on E4. We evaluate B (Panel-Margin) and E (Confidence) variants.
    e4 = load_e4()
    e4_cases = [{"gold": d["gold"], "panel_4opt_votes": d["panel_4opt_votes"],
                 "panel_4opt_confs": d["panel_4opt_confs"],
                 "panel_4opt_agg": d["panel_4opt_agg"]}
                for d in e4.values()]
    print(f"\n=== Cross-dataset E4 (VitaminC) check, N={len(e4_cases)} ===")
    print("-- A: Panel 4-opt typed direct --")
    scored = [{"gold": c["gold"], "pred": c["panel_4opt_agg"]} for c in e4_cases]
    m4_a = compute_metrics(scored)
    print(f"  cov={m4_a['coverage']:.3f} sel_err={m4_a['sel_err']:.3f} "
          f"CCO_full={m4_a['cco_full']:.3f} acc_S/R={m4_a['acc_sr']:.3f}")
    print("-- E: Confidence-threshold selective --")
    for tau in [0.85, 0.9, 0.95]:
        scored = [{"gold": c["gold"], "pred": controller_e_confidence(c, tau)} for c in e4_cases]
        m = compute_metrics(scored)
        print(f"  τ={tau}: cov={m['coverage']:.3f} sel_err={m['sel_err']:.3f} "
              f"CCO_full={m['cco_full']:.3f} acc_S/R={m['acc_sr']:.3f}")
    print("-- B: Panel-Margin (top configs from E1) --")
    top_configs = [r for r in [best_balanced, best_safety, best_cov] if r is not None]
    for r in top_configs:
        scored = [{"gold": c["gold"], "pred": controller_b_panel_margin(c, **r["params"])}
                  for c in e4_cases]
        m = compute_metrics(scored)
        print(f"  params={r['params']}")
        print(f"  → cov={m['coverage']:.3f} sel_err={m['sel_err']:.3f} CCO_full={m['cco_full']:.3f} "
              f"acc_S/R={m['acc_sr']:.3f} macro_F1={m['macro_f1']:.3f}")
    print("(F cannot be evaluated on E4: no validator outputs for VitaminC.)")


if __name__ == "__main__":
    main()
