# Audit-independent cached-model landscape

> Scope: all CCO values below use dataset labels, not the pending human audit.

## Model/system metrics

| Dataset | Model/system | Accuracy | Macro-F1 | Dataset CCO | 95% Wilson | Conflict recall | Pure S/R acc. | False conflict | Insufficient directional |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AVeriTeC | `single_haiku_3opt` | 37.9 | 40.1 | 126/150 (84.0) | [77.3, 89.0] | 0.0 | 84.0 | 0.0 | 31.4 |
| AVeriTeC | `single_haiku_4opt_strong` | 68.1 | 66.3 | 43/150 (28.7) | [22.0, 36.4] | 65.3 | 73.0 | 15.0 | 22.9 |
| AVeriTeC | `single_sonnet_4opt_strong` | 74.7 | 72.8 | 30/150 (20.0) | [14.4, 27.1] | 75.3 | 78.0 | 17.0 | 20.0 |
| AVeriTeC | `panel_3judge_3opt` | 41.4 | 44.2 | 133/150 (88.7) | [82.6, 92.8] | 0.0 | 95.0 | 0.0 | 34.3 |
| AVeriTeC | `panel_3judge_4opt_strong` | 76.5 | 74.6 | 28/150 (18.7) | [13.2, 25.7] | 77.3 | 78.0 | 17.0 | 17.1 |
| VitaminC-Mixed | `single_haiku_3opt` | 39.2 | 36.6 | 77/100 (77.0) | [67.8, 84.2] | 0.0 | 75.0 | 0.0 | 54.0 |
| VitaminC-Mixed | `single_haiku_4opt_strong` | 62.8 | 61.9 | 44/100 (44.0) | [34.7, 53.8] | 55.0 | 77.0 | 3.0 | 44.0 |
| VitaminC-Mixed | `single_sonnet_4opt_strong` | 64.8 | 62.4 | 32/100 (32.0) | [23.7, 41.7] | 67.0 | 73.0 | 3.0 | 52.0 |
| VitaminC-Mixed | `panel_3judge_3opt` | 41.2 | 38.4 | 77/100 (77.0) | [67.8, 84.2] | 0.0 | 79.0 | 0.0 | 52.0 |
| VitaminC-Mixed | `panel_3judge_4opt_strong` | 66.8 | 64.4 | 30/100 (30.0) | [21.9, 39.6] | 70.0 | 74.0 | 5.0 | 50.0 |

## Paired contrasts

| Dataset | Contrast | Metric | A | B | Delta B-A | 95% paired bootstrap | n |
|---|---|---|---:|---:|---:|---:|---:|
| AVeriTeC | `panel_vs_single_3opt` | dataset_defined_cco | 84.0 | 88.7 | +4.7 pp | [+1.3, +8.0] pp | 150 |
| AVeriTeC | `panel_vs_single_3opt` | pure_sr_accuracy | 84.0 | 95.0 | +11.0 pp | [+5.0, +17.0] pp | 100 |
| AVeriTeC | `haiku_3opt_to_4opt` | dataset_defined_cco | 84.0 | 28.7 | -55.3 pp | [-63.3, -47.3] pp | 150 |
| AVeriTeC | `haiku_3opt_to_4opt` | pure_sr_accuracy | 84.0 | 73.0 | -11.0 pp | [-19.0, -3.0] pp | 100 |
| AVeriTeC | `panel_3opt_to_4opt` | dataset_defined_cco | 88.7 | 18.7 | -70.0 pp | [-77.3, -62.7] pp | 150 |
| AVeriTeC | `panel_3opt_to_4opt` | pure_sr_accuracy | 95.0 | 78.0 | -17.0 pp | [-25.0, -9.0] pp | 100 |
| AVeriTeC | `typed_panel_vs_typed_single` | dataset_defined_cco | 28.7 | 18.7 | -10.0 pp | [-15.3, -4.7] pp | 150 |
| AVeriTeC | `typed_panel_vs_typed_single` | pure_sr_accuracy | 73.0 | 78.0 | +5.0 pp | [-1.0, +11.0] pp | 100 |
| VitaminC-Mixed | `panel_vs_single_3opt` | dataset_defined_cco | 77.0 | 77.0 | +0.0 pp | [-6.0, +5.0] pp | 100 |
| VitaminC-Mixed | `panel_vs_single_3opt` | pure_sr_accuracy | 75.0 | 79.0 | +4.0 pp | [+1.0, +8.0] pp | 100 |
| VitaminC-Mixed | `haiku_3opt_to_4opt` | dataset_defined_cco | 77.0 | 44.0 | -33.0 pp | [-42.0, -24.0] pp | 100 |
| VitaminC-Mixed | `haiku_3opt_to_4opt` | pure_sr_accuracy | 75.0 | 77.0 | +2.0 pp | [-3.0, +8.0] pp | 100 |
| VitaminC-Mixed | `panel_3opt_to_4opt` | dataset_defined_cco | 77.0 | 30.0 | -47.0 pp | [-57.0, -37.0] pp | 100 |
| VitaminC-Mixed | `panel_3opt_to_4opt` | pure_sr_accuracy | 79.0 | 74.0 | -5.0 pp | [-12.0, +1.0] pp | 100 |
| VitaminC-Mixed | `typed_panel_vs_typed_single` | dataset_defined_cco | 44.0 | 30.0 | -14.0 pp | [-22.0, -7.0] pp | 100 |
| VitaminC-Mixed | `typed_panel_vs_typed_single` | pure_sr_accuracy | 77.0 | 74.0 | -3.0 pp | [-8.0, +1.0] pp | 100 |

## Data-supported provisional findings

**Finding 1 (dataset-defined continuity only).** Adding the explicit four-way contract reduces panel directional commitments on dataset-labelled conflicts from 88.7% to 18.7% on AVeriTeC and from 77.0% to 30.0% on VitaminC-Mixed; it does not establish the human-confirmed rate.

**Finding 2 (error trade-off).** The same table reports pure-directional accuracy, false-conflict errors, and insufficient-case directional errors beside CCO, preventing a lower commitment rate from being interpreted as an unconditional improvement.

**Finding 3 (cross-dataset boundary).** Panel and prompt effects are reported separately for the two substrates; a pattern present on one dataset is not generalized to the other without a paired contrast and an audited construct check.
