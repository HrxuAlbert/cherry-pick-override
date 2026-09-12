# Cherry-pick Override: Code and data release

Code, prompts, and cached prediction data for the paper:

> **Cherry-pick Override: LLM Judges Under-use the Non-Directional Verdicts
> Their Contract Authorizes**
> Haoran Xu (University of Glasgow) --- arXiv v2

The v1 release is preserved under `scripts/option_a_exp/` and
`outputs/option_a_exp/`; the v2 material added for the revision sits under
`scripts/revision_2026/`, `tests/revision_2026/` and
`outputs/revision_2026/`.

## What this repo contains

Analysis-only release: cached predictions from the panel and validator are
shipped so that all paper tables and figures can be reproduced *without*
re-running the LLM API calls. The original prompts are also included so a
reader can audit what was sent to the models.

```
.
├── scripts/
│   ├── option_a_exp/analysis/        # v1: 7 analysis scripts
│   ├── option_a_exp/prompts/judges/  # the 4 prompts, reproduced in the appendix
│   └── revision_2026/                # v2: runners, analysers, plot scripts
├── tests/revision_2026/              # v2: unit tests for the above
├── manifests/                        # v2: frozen model rosters for the replication
└── outputs/
    ├── option_a_exp/strengthening/   # v1 cached predictions (about 10 MB)
    │   ├── e1_full_4label_utility/                  # AVeriTeC panel (3-opt + 4-opt + per-judge confidences)
    │   ├── e3_structured_certificate_validator_fewshot/  # AVeriTeC validator (few-shot certificate prompt)
    │   ├── e3_validator_on_e4_vitaminc_mixed/       # VitaminC-Mixed validator
    │   └── e4_vitaminc_mixed/                       # VitaminC-Mixed panel
    └── revision_2026/                # v2 runs and analysis outputs (about 34 MB)
        ├── probes/                   # E1, E2b, E4, E5 --- raw outputs and analysis
        ├── current_models_averitec/  # four contemporary judges, AVeriTeC
        ├── current_models_vitaminc/  # the same four on VitaminC-Mixed
        └── audit_independent/        # feature signals, error taxonomy, dataset profiles
```

## Reproducing the paper's numbers

Python 3.12 with `numpy`, `matplotlib`, and `scipy`. Install with
`pip install -r requirements.txt`.

| Script | Reproduces |
|---|---|
| `analyze_selective_typed_controller.py` | Table 7 (intervention ladder, AVeriTeC) and Table 9 (channel complementarity) |
| `analyze_controllers_on_vitaminc.py`    | Table 15 (the same controllers on VitaminC-Mixed) |
| `fair_random_stage1.py`                 | Apples-to-apples random Stage-1 null distributions (Figure 1 data, AVeriTeC + VitaminC) |
| `plot_fig1_random_veto_selectivity.py`  | Superseded by the v2 copy under `scripts/revision_2026/` (see below) |
| `diagnostic_analyses.py`                | Table 16 (panel-amplification anatomy) and the confidence-boundary numbers in the L3 subsection |
| `concept_diagnostics.py`                | Calibration / ECE; 4x4 gold-by-pred matrix; false-conflict rate on pure-S/R; panel-agreement on CCO; validator coverage on CCO |
| `bootstrap_cis_and_baselines.py`        | The four paired-bootstrap CIs cited in Section 4 and the conflict-if-any panel baseline (Table 7, L2) |

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

## The v2 revision

The revision adds four probe experiments, a frozen replication on four
contemporary judges, and a set of independent diagnostics. Their analysis
outputs are the direct source of every number in the v2 tables, and the raw
model outputs are shipped so those outputs can be re-derived rather than
merely inspected.

| Script (`scripts/revision_2026/`) | Reproduces |
|---|---|
| `analyze_current_model_results.py`     | Table 6 (four contemporary judges, both substrates) |
| `analyze_commitment_probes.py`         | Table 8 and Figure 2 data (E1 fresh-context recognition probe) |
| `plot_fig1_recognition.py`             | Figure 2 |
| `analyze_prompt_ladder.py`             | Table 11 (E5 prompt-specification ladder) |
| `analyze_decomposition_probe.py`       | Table 12 (E4 decomposition before verdict) |
| `analyze_feature_signals.py`           | Table 10 (out-of-fold screening) |
| `analyze_error_taxonomy.py`            | The error-taxonomy counts in Section 4 |
| `analyze_cached_model_landscape.py`    | The cached-landscape cross-check |
| `plot_fig1_random_veto_selectivity.py` | Figure 1 (v2 copy; the v1 script under `scripts/option_a_exp/` produced the v1 figure and still says `CCO`) |
| `run_current_models.py`, `run_commitment_probes.py` | The runners that made the API calls, including the spend guard and the append-only resume |
| `export_prompt_appendix.py`            | The prompt appendix, generated from the files in `scripts/option_a_exp/prompts/judges/` and checked against the SHA-256 in each run manifest |

The analysis scripts read from `outputs/revision_2026/` and need no API access.
Run them from the repo root:

```bash
python3.12 scripts/revision_2026/analyze_commitment_probes.py
python3.12 scripts/revision_2026/analyze_prompt_ladder.py
python3.12 scripts/revision_2026/analyze_decomposition_probe.py
python3.12 scripts/revision_2026/plot_fig1_recognition.py
```

Figure 1 needs the null distributions first, which the v1 script produces:

```bash
python3.12 scripts/option_a_exp/analysis/fair_random_stage1.py
python3.12 scripts/revision_2026/plot_fig1_random_veto_selectivity.py
```

`CPO_WORKSPACE` overrides the root the scripts resolve; `FIG_DIR` overrides
where figures are written. `tests/revision_2026/` holds the unit tests for the
runners and analysers.

Several analysers refuse to write into a non-empty output directory; pass
their `--force` flag or point `--output-root` at a fresh path. That is a guard
against silently overwriting a recorded run, not a failure.

Two scripts cannot run from this release, and it is worth saying why rather
than leaving them to fail:

- `reconstruct_e1_source_map.py` and `reconstruct_vitaminc_source_map.py`
  rebuild the case-to-source maps from the AVeriTeC and VitaminC development
  sets. Those are third-party datasets and are not redistributed here; obtain
  them from the original releases and pass `--source`.
- `build_probe_manifests.py` also reads prompt templates from a path in the
  author's working tree. The manifests it would produce are already shipped in
  each run directory as `executed_manifest.json`, which is what the analysers
  read, so nothing downstream depends on re-running it.

### What is not here, and why

The paper describes a two-reviewer human audit of the AVeriTeC sample that was
not completed, and reports only its calibration round (Section 7). **None of
the audit material is in this release**: not the reviewers' annotations, not
the blinding map, not the workbook builders. Two people labelled those
twenty claims, and publishing per-reviewer judgements would identify them.
`analyze_audit_agreement.py` is included so the agreement statistics in
Section 7 can be read off the code that computed them, but it has no data to
run against here.

Also omitted: runs superseded by a later configuration (the E2b 512-token
pass, replaced at 1024 tokens after truncation bias was found, and an E1 smoke
run at 192 tokens), which the paper describes but does not use.

## Third-party data

The claims and evidence inside the cached model inputs come from two public
benchmarks and are redistributed here only as the exact strings sent to the
models:

- **AVeriTeC** --- Schlichtkrull, Guo and Vlachos, *AVeriTeC: A Dataset for
  Real-world Claim Verification with Evidence from the Web*, NeurIPS 2023
  Datasets and Benchmarks.
- **VitaminC** --- Schuster, Fisch and Barzilay, *Get Your Vitamin C! Robust
  Fact Verification with Contrastive Evidence*, NAACL 2021.

The MIT license below covers this repository's own code and derived analysis
outputs, not the benchmark text.

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

The historical panel (v1, still the subject of Table 7): Anthropic Claude
Haiku 4.5, Anthropic Claude Sonnet 4.5, OpenAI GPT-4o-mini, with Claude
Haiku 4.5 as the validator.

The frozen replication added in v2 (Table 6, and the probes in Sections 5
and 6) uses four judges scored individually, two proprietary and two
open-weight: GPT-5.6 Terra, Claude Sonnet 5, gpt-oss-120b and Mistral
Small 4. Model identifiers, providers and the resolved roster for each run
are recorded in that run's `executed_manifest.json`.

All sampling is greedy (temperature 0) and deterministic per judge. The one
exception is the E2b logprob probe, which needed a provider that returns
logprobs; the substitution is recorded in the run manifest and the probe is
reported in the paper as an uninformative negative result.

## License

MIT (see `LICENSE`).

## Citation

```
@misc{xu2026cherrypick,
  author = {Haoran Xu},
  title = {Cherry-pick Override: LLM Judges Under-use the Non-Directional Verdicts Their Contract Authorizes},
  year = {2026},
  note = {arXiv preprint}
}
```
