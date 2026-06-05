"""Paired-bootstrap CIs for headline differences + conflict-if-any panel baseline.

For each headline claim, compute the paired bootstrap CI (5000 resamples) on
the per-case-outcome paired difference. Also compute the conflict-if-any
panel baseline: "if any judge votes CONFLICTING, withhold the directional
commit".

Headline CI claims:
  H1. Panel 3-opt CCO_C vs single Haiku 3-opt CCO_C on AVeriTeC conflicting
      subset (N=150). Reported: 0.887 vs 0.840 (+4.7 pp panel amplification).
  H2. Typed direct (panel 4-opt) CCO_C vs single Haiku 3-opt CCO_C on AVeriTeC
      conflicting subset. Reported: 0.187 vs 0.840 (L0 -> L1 vocabulary
      reduction).
  H3. Validator-as-classifier Acc_S/R vs typed direct Acc_S/R on AVeriTeC
      pure-S/R subset. Reported: 0.39 vs 0.78 (L4 classifier-mode collapse).
  H4. Panel 3-opt CCO_C vs single 3-opt CCO_C on VitaminC conflicting subset
      (cross-dataset null). Reported: about 0.76 vs 0.72, NOT statistically
      separated; supports scoping the amplification claim to AVeriTeC.

Plus:
  B1. Conflict-if-any panel baseline: if any judge votes CONFLICTING, output
      CONFLICTING. Computed at the typed 4-opt panel level on AVeriTeC.

No API calls; pure post-hoc analysis of cached panel predictions.

Outputs:
  outputs/option_a_exp/analysis/bootstrap_cis.csv
  outputs/option_a_exp/analysis/bootstrap_cis_summary.md
"""
from __future__ import annotations
import csv
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/option_a_exp/analysis"))
from analyze_selective_typed_controller import load_e1  # noqa: E402

N_BOOT = 5000
BOOT_SEED = 0
OUT_DIR = REPO / "outputs/option_a_exp/analysis"

# ─── helpers ────────────────────────────────────────────────────────────

def load_vitaminc_e4():
    """Load VitaminC E4 single Haiku 3-opt and panel 3-opt + 4-opt predictions per case."""
    path = REPO / "outputs/option_a_exp/strengthening/e4_vitaminc_mixed/raw_results.jsonl"
    data = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            cid = r["case_id"]
            d = data.setdefault(cid, {"gold": r["gold_normal"]})
            sys_name = r["system"]
            jos = r.get("judge_outputs", [])
            if sys_name == "single_haiku_3opt" and jos:
                p = jos[0].get("parsed") or {}
                d["single_haiku_3opt"] = p.get("verdict_normal")
            elif sys_name == "panel_3judge_3opt":
                votes = []
                for jo in jos:
                    p = jo.get("parsed") or {}
                    v = p.get("verdict_normal")
                    if v:
                        votes.append(v)
                d["panel_3opt_votes"] = votes
                d["panel_3opt_agg"] = r["verdict_normal"]
            elif sys_name == "panel_3judge_4opt_strong":
                votes = []
                for jo in jos:
                    p = jo.get("parsed") or {}
                    v = p.get("verdict_normal")
                    if v:
                        votes.append(v)
                d["panel_4opt_votes"] = votes
                d["panel_4opt_agg"] = r["verdict_normal"]
    return data


def pctile(vals, q):
    s = sorted(vals)
    return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]


def paired_boot_diff_ci(pairs, fn_a, fn_b, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired bootstrap of (fn_b - fn_a) on per-case `pairs`.

    Each pair is a tuple/dict the two fns can consume; fn returns a per-case
    indicator (1/0) and we average for the rate. Resamples with replacement.
    """
    rnd = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return None, None, None
    a_vals = [fn_a(p) for p in pairs]
    b_vals = [fn_b(p) for p in pairs]
    diffs_observed = sum(b_vals) / n - sum(a_vals) / n
    boot_diffs = []
    for _ in range(n_boot):
        idx = [rnd.randint(0, n - 1) for _ in range(n)]
        a_mean = sum(a_vals[i] for i in idx) / n
        b_mean = sum(b_vals[i] for i in idx) / n
        boot_diffs.append(b_mean - a_mean)
    return diffs_observed, pctile(boot_diffs, 0.025), pctile(boot_diffs, 0.975)


# ─── load + filter datasets ────────────────────────────────────────────

print("Loading AVeriTeC E1...")
e1 = load_e1()
e1_conflict = []
e1_pure_sr = []
for cid, d in e1.items():
    if not all(k in d for k in ("single_3opt", "panel_3opt_agg", "panel_4opt_agg")):
        continue
    if d["gold"] == "conflicting":
        e1_conflict.append(d)
    if d["gold"] in ("support", "refute"):
        e1_pure_sr.append(d)
print(f"  conflict subset: {len(e1_conflict)}; pure-S/R subset: {len(e1_pure_sr)}")

print("Loading validator-as-classifier (E3 fewshot, validator_verdict — deterministic rule)...")
val_path = REPO / "outputs/option_a_exp/strengthening/e3_structured_certificate_validator_fewshot/raw_results.jsonl"
val_pred = {}
with val_path.open() as f:
    for line in f:
        r = json.loads(line)
        # Use validator_verdict (deterministic-rule output), NOT prompt_final_verdict
        # (which is the LLM's verdict after the validator prompt, a different system).
        val_pred[r["case_id"]] = (r.get("validator_verdict") or "").lower().strip()
print(f"  validator predictions: {len(val_pred)}")

print("Loading VitaminC E4...")
vc = load_vitaminc_e4()
vc_conflict = [d for d in vc.values()
               if "single_haiku_3opt" in d and "panel_3opt_agg" in d
               and d["gold"] == "conflicting"]
print(f"  VitaminC conflict subset: {len(vc_conflict)}")

# ─── H1: panel 3-opt vs single 3-opt CCO_C on AVeriTeC conflicting ────────

print("\n[H1] Panel 3-opt vs single 3-opt CCO_C on AVeriTeC conflicting subset")
def f_single3(d): return 1 if d["single_3opt"] in ("support", "refute") else 0
def f_panel3(d): return 1 if d["panel_3opt_agg"] in ("support", "refute") else 0
diff, lo, hi = paired_boot_diff_ci(e1_conflict, f_single3, f_panel3)
single3_rate = sum(f_single3(d) for d in e1_conflict) / len(e1_conflict)
panel3_rate = sum(f_panel3(d) for d in e1_conflict) / len(e1_conflict)
H1 = {"name": "H1_panel_vs_single_3opt_CCO_C_AVeriTeC",
      "single_rate": single3_rate, "panel_rate": panel3_rate,
      "delta_observed": diff, "ci_low": lo, "ci_high": hi, "n": len(e1_conflict)}
print(f"  single_3opt CCO_C={single3_rate:.4f}  panel_3opt CCO_C={panel3_rate:.4f}")
print(f"  Δ(panel-single) = {diff:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

# ─── H2: typed direct (panel 4-opt) vs single 3-opt CCO_C on AVeriTeC ───────

print("\n[H2] Typed direct (panel 4-opt) vs single 3-opt CCO_C on AVeriTeC conflicting")
def f_p4(d): return 1 if d["panel_4opt_agg"] in ("support", "refute") else 0
diff, lo, hi = paired_boot_diff_ci(e1_conflict, f_single3, f_p4)
p4_rate = sum(f_p4(d) for d in e1_conflict) / len(e1_conflict)
H2 = {"name": "H2_typed_panel_vs_single3_CCO_C_AVeriTeC",
      "single3_rate": single3_rate, "typed_panel_rate": p4_rate,
      "delta_observed": diff, "ci_low": lo, "ci_high": hi, "n": len(e1_conflict)}
print(f"  single_3opt CCO_C={single3_rate:.4f}  panel_typed CCO_C={p4_rate:.4f}")
print(f"  Δ(typed-single) = {diff:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

# ─── H3: validator-as-classifier vs typed-direct Acc_S/R on AVeriTeC pure-S/R ─

print("\n[H3] Validator-as-classifier vs typed direct Acc_S/R on AVeriTeC pure-S/R")
# val_pred maps case_id → prompt_final_verdict ∈ {support, refute, conflicting, insufficient}
# typed direct prediction = panel_4opt_agg
e1_pure_sr_with_val = []
for cid, d in e1.items():
    if d.get("gold") not in ("support", "refute"):
        continue
    if "panel_4opt_agg" not in d:
        continue
    if cid not in val_pred:
        continue
    e1_pure_sr_with_val.append({"cid": cid, "gold": d["gold"], "typed": d["panel_4opt_agg"],
                                "val": val_pred[cid]})
print(f"  pure-S/R cases with validator: {len(e1_pure_sr_with_val)}")
def f_typed_correct(p): return 1 if p["typed"] == p["gold"] else 0
def f_val_correct(p):  return 1 if p["val"] == p["gold"] else 0
typed_acc = sum(f_typed_correct(p) for p in e1_pure_sr_with_val) / len(e1_pure_sr_with_val)
val_acc = sum(f_val_correct(p) for p in e1_pure_sr_with_val) / len(e1_pure_sr_with_val)
diff, lo, hi = paired_boot_diff_ci(e1_pure_sr_with_val, f_typed_correct, f_val_correct)
H3 = {"name": "H3_validator_vs_typed_Acc_SR_AVeriTeC",
      "typed_acc": typed_acc, "validator_acc": val_acc,
      "delta_observed": diff, "ci_low": lo, "ci_high": hi, "n": len(e1_pure_sr_with_val)}
print(f"  typed direct Acc_S/R={typed_acc:.4f}  validator-classifier Acc_S/R={val_acc:.4f}")
print(f"  Δ(val-typed) = {diff:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

# ─── H4: VitaminC cross-dataset panel-vs-single 3-opt CCO_C (null check) ─

print("\n[H4] VitaminC panel 3-opt vs single 3-opt CCO_C on conflicting subset (cross-dataset null)")
def f_single3v(d): return 1 if d["single_haiku_3opt"] in ("support", "refute") else 0
def f_panel3v(d): return 1 if d["panel_3opt_agg"] in ("support", "refute") else 0
diff, lo, hi = paired_boot_diff_ci(vc_conflict, f_single3v, f_panel3v)
v_single_rate = sum(f_single3v(d) for d in vc_conflict) / len(vc_conflict)
v_panel_rate = sum(f_panel3v(d) for d in vc_conflict) / len(vc_conflict)
H4 = {"name": "H4_panel_vs_single_3opt_CCO_C_VitaminC",
      "single_rate": v_single_rate, "panel_rate": v_panel_rate,
      "delta_observed": diff, "ci_low": lo, "ci_high": hi, "n": len(vc_conflict)}
print(f"  single_3opt CCO_C={v_single_rate:.4f}  panel_3opt CCO_C={v_panel_rate:.4f}")
print(f"  Δ(panel-single) = {diff:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

# ─── B1: conflict-if-any panel baseline (on typed panel votes) ─────────

print("\n[B1] Conflict-if-any panel baseline on AVeriTeC E1 (typed 4-opt)")
# Rule: if any of the 3 judges votes CONFLICTING in the 4-opt panel, output CONFLICTING.
# Otherwise pass through the typed panel aggregation.
# Need panel_4opt_votes per case in E1.
e1_with_votes = []
for cid, d in e1.items():
    if "panel_4opt_votes" not in d or "panel_4opt_agg" not in d:
        continue
    e1_with_votes.append(d)
print(f"  AVeriTeC E1 cases with panel 4-opt votes: {len(e1_with_votes)}")

# Compute same-denominator metrics for this baseline
N = len(e1_with_votes)
n_sr = sum(1 for d in e1_with_votes if d["gold"] in ("support", "refute"))
n_c  = sum(1 for d in e1_with_votes if d["gold"] == "conflicting")

n_commit = 0
n_wrong  = 0
n_cco    = 0
n_acc_sr = 0
n_rec_c  = 0
for d in e1_with_votes:
    votes = d["panel_4opt_votes"]
    if "conflicting" in votes:
        pred = "conflicting"
    else:
        pred = d["panel_4opt_agg"]  # fall back to majority
    g = d["gold"]
    if pred in ("support", "refute"):
        n_commit += 1
        if pred != g:
            n_wrong += 1
        if g == "conflicting":
            n_cco += 1
        if g in ("support", "refute") and pred == g:
            n_acc_sr += 1
    if g == "conflicting" and pred == "conflicting":
        n_rec_c += 1

B1 = {
    "name": "B1_conflict_if_any_panel_baseline_AVeriTeC",
    "cov": n_commit / N, "se": (n_wrong / n_commit) if n_commit else 0.0,
    "cco_N": n_cco / N, "acc_sr": n_acc_sr / n_sr, "rec_c": n_rec_c / n_c,
    "n": N,
}
print(f"  Cov={B1['cov']:.4f}  SE={B1['se']:.4f}  CCO_N={B1['cco_N']:.4f}  "
      f"Acc_S/R={B1['acc_sr']:.4f}  Rec_C={B1['rec_c']:.4f}")

# ─── Write CSV + summary ────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = OUT_DIR / "bootstrap_cis.csv"
rows = []
for H in [H1, H2, H3, H4]:
    rows.append({"hypothesis": H["name"], "n": H["n"],
                 "rate_a": H.get("single_rate", H.get("single3_rate", H.get("typed_acc"))),
                 "rate_b": H.get("panel_rate", H.get("typed_panel_rate", H.get("validator_acc"))),
                 "delta": H["delta_observed"], "ci_low": H["ci_low"], "ci_high": H["ci_high"]})
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        r2 = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
        w.writerow(r2)
print(f"\nWrote {csv_path}")

md_path = OUT_DIR / "bootstrap_cis_summary.md"
md = f"""# Bootstrap CIs for Headline Differences + Conflict-If-Any Baseline

**Method**: Paired bootstrap, 5000 resamples, seed=0. Per-case paired
differences on the same denominator subset.

## H1. Panel 3-opt vs single 3-opt CCO_C (AVeriTeC conflicting subset, N={H1['n']})
Single Haiku 3-opt: CCO_C = {H1['single_rate']:.3f}; Panel 3-opt: CCO_C = {H1['panel_rate']:.3f}.
**Δ(panel − single) = {H1['delta_observed']:+.3f}; 95% bootstrap CI [{H1['ci_low']:+.3f}, {H1['ci_high']:+.3f}].**

CI {'INCLUDES' if H1['ci_low'] < 0 < H1['ci_high'] else 'DOES NOT include'} 0. Interpretation: panel amplification on AVeriTeC {'is' if not (H1['ci_low'] < 0 < H1['ci_high']) else 'is NOT'} robustly separated from null at our N.

## H2. Typed-panel (L1) vs single 3-opt (L0) CCO_C (AVeriTeC conflicting, N={H2['n']})
Single Haiku 3-opt: CCO_C = {H2['single3_rate']:.3f}; Typed panel: CCO_C = {H2['typed_panel_rate']:.3f}.
**Δ(typed − single) = {H2['delta_observed']:+.3f}; 95% CI [{H2['ci_low']:+.3f}, {H2['ci_high']:+.3f}].**

CI {'INCLUDES' if H2['ci_low'] < 0 < H2['ci_high'] else 'DOES NOT include'} 0. The vocabulary fix (L0→L1) is {'robustly' if not (H2['ci_low'] < 0 < H2['ci_high']) else 'NOT robustly'} separated from null.

## H3. Validator-as-classifier vs typed direct Acc_S/R (AVeriTeC pure-S/R, N={H3['n']})
Typed direct Acc_S/R = {H3['typed_acc']:.3f}; Validator-as-classifier Acc_S/R = {H3['validator_acc']:.3f}.
**Δ(validator − typed) = {H3['delta_observed']:+.3f}; 95% CI [{H3['ci_low']:+.3f}, {H3['ci_high']:+.3f}].**

CI {'INCLUDES' if H3['ci_low'] < 0 < H3['ci_high'] else 'DOES NOT include'} 0. Validator-classifier collapse {'is' if not (H3['ci_low'] < 0 < H3['ci_high']) else 'is NOT'} robustly separated from null.

## H4. VitaminC panel 3-opt vs single 3-opt CCO_C (cross-dataset null, N={H4['n']})
Single Haiku 3-opt: CCO_C = {H4['single_rate']:.3f}; Panel 3-opt: CCO_C = {H4['panel_rate']:.3f}.
**Δ(panel − single) = {H4['delta_observed']:+.3f}; 95% CI [{H4['ci_low']:+.3f}, {H4['ci_high']:+.3f}].**

CI {'INCLUDES' if H4['ci_low'] < 0 < H4['ci_high'] else 'DOES NOT include'} 0. The panel-amplification effect {'does' if H4['ci_low'] < 0 < H4['ci_high'] else 'does NOT'} replicate on VitaminC under this CI.

## B1. Conflict-if-any panel baseline (AVeriTeC E1, N={B1['n']})
Rule: if any of the 3 typed judges votes CONFLICTING, the system outputs CONFLICTING; otherwise the typed-panel majority pass-through.

| Metric | Value |
|---|---|
| Cov | {B1['cov']:.3f} |
| SE | {B1['se']:.3f} |
| CCO_N | {B1['cco_N']:.3f} |
| Acc_S/R | {B1['acc_sr']:.3f} |
| Rec_C | {B1['rec_c']:.3f} |

Compare to L1 typed direct (panel + typed): Cov 0.396, SE 0.310, CCO_N 0.098, Acc_S/R 0.780, Rec_C 0.773.
And to L5 (F τ=0.85): Cov 0.281, SE 0.200, CCO_N 0.046, Acc_S/R 0.640, Rec_C 0.860.

Conflict-if-any reaches CCO_N {B1['cco_N']:.3f} but at Acc_S/R {B1['acc_sr']:.3f} (vs typed direct 0.780).
This is the natural simplest CONFLICTING-aware aggregation rule; it sits inside the ladder space but does not dominate L5.
"""
md_path.write_text(md)
print(f"Wrote {md_path}")
