# Frozen experiment-dataset profiles

These statistics are generated from the exact materialized inputs used by the current-model runner and do not use human annotations.

| Dataset | Label | n | Claim chars (median [IQR]) | Evidence chars (median [IQR]) | Evidence units (median [IQR]) | Prompt chars (median [IQR]) |
|---|---|---:|---:|---:|---:|---:|
| AVeriTeC | all | 285 | 92 [64, 130] | 727 [380, 1295] | 3 [2, 4] | 2808 [2440, 3388] |
| AVeriTeC | support | 50 | 86 [66, 113] | 342 [224, 606] | 2 [1, 3] | 2412 [2291, 2685] |
| AVeriTeC | refute | 50 | 116 [68, 159] | 624 [330, 1078] | 2 [1, 3] | 2763 [2381, 3140] |
| AVeriTeC | insufficient | 35 | 95 [63, 128] | 461 [293, 788] | 2 [2, 3] | 2514 [2364, 2899] |
| AVeriTeC | conflicting | 150 | 90 [62, 130] | 982 [627, 1608] | 3 [2, 4] | 3062 [2715, 3708] |
| VitaminC-Mixed | all | 250 | 62 [48, 84] | 175 [114, 278] | 1 [1, 2] | 2217 [2156, 2317] |
| VitaminC-Mixed | support | 50 | 58 [41, 76] | 135 [108, 171] | 1 [1, 1] | 2174 [2132, 2206] |
| VitaminC-Mixed | refute | 50 | 52 [43, 62] | 136 [100, 190] | 1 [1, 1] | 2170 [2134, 2217] |
| VitaminC-Mixed | insufficient | 50 | 85 [60, 110] | 112 [91, 154] | 1 [1, 1] | 2178 [2146, 2234] |
| VitaminC-Mixed | conflicting | 100 | 64 [49, 81] | 296 [223, 363] | 2 [2, 2] | 2326 [2254, 2419] |
