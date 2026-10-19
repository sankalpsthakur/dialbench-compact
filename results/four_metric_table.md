# Four-metric table (DialBench RPM-10K, official split, 5-seed mean of the pre-registered primary; self-evaluated)

| metric (paper Table 4) | ours, 5-seed mean | MRLM | best published | rank |
|---|---|---|---|---|
| Acc_eps (%, higher is better) | 68.56 | 62.4 | 62.4 (MRLM) | first: above all 18 published models |
| Acc_theta (%, higher is better) | 74.88 | 70.9 | 70.9 (MRLM) | first: above all 18 published models |
| Ref (lower is better) | 0.0759 | 0.063 | 0.038 (LLaVA-v1.6-Vicuna-7B) | behind: 10 of the 18 published models are lower |
| Rel (lower is better) | 0.392 | 0.535 | 0.215 (LLaVA-v1.6-Vicuna-7B) | behind: 5 of the 18 published models are lower |

Per seed (official scorer, n = 2,000):

| seed | Acc_eps | Acc_theta | Ref | Rel |
|---|---|---|---|---|
| 0 | 68.45 % | 74.66 % | 0.0814 | 0.3780 |
| 1 | 69.10 % | 74.40 % | 0.0712 | 0.3447 |
| 2 | 68.80 % | 75.18 % | 0.0816 | 0.4294 |
| 3 | 68.10 % | 74.87 % | 0.0654 | 0.3700 |
| 4 | 68.35 % | 75.29 % | 0.0798 | 0.4392 |

Published values: arXiv 2511.21982 Table 4 (HTML version), read 2026-10-19; 18 models. The counts of published models with a better Ref / Rel were made by hand on that table. Ours: src/dialbench_metrics.py (port of benchmark/benchmark.py of Event-AHU/DialBench).
