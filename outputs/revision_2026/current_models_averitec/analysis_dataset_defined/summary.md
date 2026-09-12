# Current-model AVeriTeC results (dataset-defined)

> This report is audit-independent. Human-confirmed CCO is computed separately after the blinded labels are frozen.

| Model slot | Dataset CCO | 95% Wilson | Conflict recall | Pure S/R accuracy | False conflict | Insufficient directional | Parse failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| `closed_openai_current` | 44/150 (0.293) | [0.226, 0.371] | 0.613 | 0.800 | 0.090 | 0.086 | 0.000 |
| `closed_anthropic_current` | 45/150 (0.300) | [0.232, 0.378] | 0.633 | 0.870 | 0.050 | 0.200 | 0.007 |
| `openweight_gpt_oss_120b` | 72/150 (0.480) | [0.402, 0.559] | 0.373 | 0.790 | 0.040 | 0.286 | 0.000 |
| `openweight_mistral_small_4` | 32/150 (0.213) | [0.155, 0.286] | 0.720 | 0.680 | 0.180 | 0.200 | 0.000 |
