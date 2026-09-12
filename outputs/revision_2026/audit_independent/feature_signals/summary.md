# Confidence versus evidence-structure diagnostic

> Provisional scope: the target is the AVeriTeC dataset conflict label, not the pending human audit. These numbers diagnose the existing cache and cannot serve as the final human-confirmed E4 result.

Eligible directional outputs: 344 from 3 models and 161 cases. Five-fold cross-validation is grouped by case. Model fixed effects are included in every specification.

| Feature specification | AUPRC | AUROC | Recall @ top 10% | Recall @ top 20% | Recall @ top 30% |
|---|---:|---:|---:|---:|---:|
| `model_only` | 0.335 | 0.523 | 0.110 | 0.220 | 0.339 |
| `confidence_only` | 0.464 | 0.694 | 0.156 | 0.312 | 0.486 |
| `structure_only` | 0.523 | 0.718 | 0.239 | 0.376 | 0.459 |
| `confidence_plus_structure` | 0.589 | 0.780 | 0.220 | 0.422 | 0.569 |

## Paired case-cluster bootstrap

Combined minus confidence-only AUPRC: +0.125, 95% interval [+0.018, +0.254] (2000 resamples).

**Provisional Finding.** Evidence-structure features are evaluated as incremental screening signals under grouped out-of-fold prediction. The finding must be recomputed with frozen human labels before it can support the paper's construct-level claim.
