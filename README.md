# Cherry-pick Override: Code and data release

Code, prompts, and cached prediction data for the paper:

> **Cherry-pick Override: Unsafe Directional Commitment in LLM Judges under Mixed Evidence**
> Haoran Xu (University of Glasgow)

## What this repo contains

Analysis-only release: cached predictions from the panel and validator are
shipped so that all paper tables and figures can be reproduced *without*
re-running the LLM API calls. The original prompts are also included so a
reader can audit what was sent to the models.

```
.
├── scripts/option_a_exp/
│   ├── analysis/                                  # 7 analysis scripts (see table below)
│   └── prompts/judges/                            # 4 prompts used in the paper
└── outputs/option_a_exp/strengthening/            # cached predictions (about 10 MB)
    ├── e1_full_4label_utility/                    # AVeriTeC panel (3-opt + 4-opt + per-judge confidences)
    ├── e3_structured_certificate_validator_fewshot/   # AVeriTeC validator (few-shot certificate prompt)
    ├── e3_validator_on_e4_vitaminc_mixed/         # VitaminC-Mixed validator
    └── e4_vitaminc_mixed/                         # VitaminC-Mixed panel
```

## Reproducing the paper's numbers

Python 3.12 with `numpy`, `matplotlib`, and `scipy`. Install with
`pip install -r requirements.txt`.

| Script | Reproduces |
|---|---|
| `analyze_selective_typed_controller.py` | Table 1 (intervention ladder, AVeriTeC) and Table 4 (channel-orthogonality ablation) |
| `analyze_controllers_on_vitaminc.py`    | The same controllers evaluated on the VitaminC-Mixed substrate (Appendix VitaminC table) |
| `fair_random_stage1.py`                 | Apples-to-apples random Stage-1 null distributions (Figure 1 data, AVeriTeC + VitaminC) |
| `plot_fig1_random_veto_selectivity.py`  | Figure 1 (4-column x 2-row KDE panel) |
| `diagnostic_analyses.py`                | Panel-amplification anatomy (Appendix table) and confidence-boundary numbers in the L3 subsection |
| `concept_diagnostics.py`                | Calibration / ECE; 4x4 gold-by-pred matrix; false-conflict rate on pure-S/R; panel-agreement on CCO; validator coverage on CCO |
| `bootstrap_cis_and_baselines.py`        | Paired-bootstrap CIs (the four numbers cited at the end of the Results section) and the conflict-if-any panel baseline |

Run from the repo root (order matters: `fair_random_stage1.py` produces
the JSON that `plot_fig1_random_veto_selectivity.py` consumes):

```bash
PYTHONHASHSEED=0 python3.12 scripts/option_a_exp/analysis/concept_diagnostics.py
PYTHONHASHSEED=0 python3.12 scripts/option_a_exp/analysis/analyze_selective_typed_controller.py
PYTHONHASHSEED=0 python3.12 scripts/option_a_exp/analysis/analyze_controllers_on_vitaminc.py
PYTHONHASHSEED=0 python3.12 scripts/option_a_exp/analysis/fair_random_stage1.py
PYTHONHASHSEED=0 python3.12 scripts/option_a_exp/analysis/diagnostic_analyses.py
PYTHONHASHSEED=0 python3.12 scripts/option_a_exp/analysis/bootstrap_cis_and_baselines.py
PYTHONHASHSEED=0 python3.12 scripts/option_a_exp/analysis/plot_fig1_random_veto_selectivity.py
```

Outputs are written under `outputs/option_a_exp/analysis/`. The figure
defaults to `figures/` under the repo root; override the location with
the `FIG_DIR` environment variable.

## Data schema

Each `raw_results.jsonl` file is a JSONL of records:

```json
{
  "case_id": "averitec_1234",
  "gold_normal": "conflicting" | "support" | "refute" | "insufficient",
  "system": "panel_3judge_4opt_strong" | "panel_3judge_3opt" | "single_haiku_3opt" | "single_sonnet_4opt_strong" | "single_haiku_4opt_strong",
  "verdict_normal": "support" | "refute" | "insufficient" | "conflicting",
  "judge_outputs": [
    {"parsed": {"verdict_normal": ..., "confidence": 0.0-1.0}, ...},
    ...
  ]
}
```

For the validator (`e3_*`):

```json
{
  "case_id": "averitec_1234",
  "validator_verdict": "support" | "refute" | "insufficient" | "conflicting",
  "validity": {
    "has_material_mixed": true|false,
    "has_material_insufficient": true|false,
    ...
  },
  "prompt_final_verdict": ...
}
```

## Prompts

- `honest.txt` — 3-option judge (S/R/NEI), used for L0 (pre-contract baseline)
- `honest_4opt_strong.txt` — 4-option typed judge (S/R/I/C), used L1 onwards
- `certificate_strict.txt` — zero-shot evidence-state validator (ablation 5 in §5)
- `certificate_strict_fewshot.txt` — few-shot evidence-state validator (used in L4/L5)

## Models

Panel: Anthropic Claude Haiku 4.5, Anthropic Claude Sonnet 4.5, OpenAI
GPT-4o-mini. Validator: Anthropic Claude Haiku 4.5. All sampling is greedy
(temperature 0) and deterministic per judge.

## License

MIT (see `LICENSE`).

## Citation

```
@misc{xu2026cherrypick,
  author = {Haoran Xu},
  title = {Cherry-pick Override: Unsafe Directional Commitment in LLM Judges under Mixed Evidence},
  year = {2026},
  note = {arXiv preprint}
}
```
