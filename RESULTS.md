# Results summary

All numbers are self-evaluated on DialBench RPM-10K with the ported official scorer (`src/dialbench_metrics.py`). Every test evaluation, including audits, crashed attempts and int8 looks, is in `TEST-LOG.md`; the raw kernel outputs behind the tables are in `docs/`. Times are UTC and come from the private repository's git log.

## 1. The model and the recipe

- **Model (the pre-registered primary `a2-r18-224`):** a ResNet-18 backbone initialised from torchvision ImageNet weights, a 1x1 convolution to 64 channels, flatten, a 512-unit hidden layer, and two outputs: 505 histogram bins over the reading as a fraction of the dial range (-0.01 to 1.25 in steps of 0.0025) and a range class (3, 6, 10 and the other classes with at least 20 training rows, plus a rare class). 13,079,234 parameters (13,078,208 for the split-v3 runs, whose range head has fewer classes).
- **Reading:** the window decoder (the 9-bin window with the largest probability mass; its half-width of 4 bins equals the metric's tolerance of 1 % of the range; the centre is refined by mass-weighting) gives the fraction; the reading is `fraction * predicted range`. The range of the label file is never an input; it is used by the scorer for the tolerance only.
- **Training:** 224 px input (the official resize, aspect ignored), 80 epochs, batch 64, AdamW (learning rate 6e-4, 3 times that for the head, weight decay 0.02), 2 warm-up epochs then cosine decay, EMA 0.995, fp16 autocast, soft labels (sigma 0.004) with an L1 term on the expectation and a range-class cross entropy (weight 0.3), geometric and photometric augmentation. Source of truth: `kernels/gpu-final/configs.json`, `src/rpm_train_torch.py`.
- **Test rule:** EMA weights at the last epoch, window decoder, no test-time augmentation, no checkpoint selection; each model is scored once on its own test part.

## 2. Stage A: selection on validation only (`FROZEN-4.md`, 2026-10-10; no test image scored)

| candidate | validation Acc_eps (EMA, final epoch, 984 scored rows) | parameters | run time |
|---|---|---|---|
| `a1-mnv3s224-f100` (run 1's configuration, 100 epochs) | 64.74 % | 3.45 M | 42.6 min |
| **`a2-r18-224`** (frozen as the primary) | **68.09 %** | 13.08 M | 27.2 min |
| `a4-mnv3s320` | 66.36 % | 5.12 M | 32.7 min |
| `a3`, `a5`, `a6`, `a7` | not run: the pre-registered time guard skipped them after the measured speeds came out 2.1 to 3.8 times slower than estimated | | |

One standard error on a validation score is about 1.5 points. The selection rule is mechanical and does not claim the winner is truly best.

## 3. Stage B: the test looks (`FROZEN-4.md`, 2026-10-17; 2 x Tesla T4, one kernel, 6,657 s wall)

| run | split | seed | validation | test Acc_eps | Wilson 95 % | wrapper seconds |
|---|---|---|---|---|---|---|
| `f-r18-224-off-s0` | official, n = 2,000 | 0 | 68.50 % | **68.45 %** | [66.38, 70.45] | 1,729 |
| `f-r18-224-off-s1` | official | 1 | 69.72 % | **69.10 %** | [67.04, 71.09] | 1,728 |
| `f-r18-224-off-s2` | official | 2 | 68.70 % | **68.80 %** | [66.74, 70.79] | 1,668 |
| `f-r18-224-off-s3` | official | 3 | 68.50 % | **68.10 %** | [66.02, 70.11] | 1,697 |
| `f-r18-224-off-s4` | official | 4 | 69.11 % | **68.35 %** | [66.28, 70.35] | 1,690 |
| `f-r18-224-v3-s0` | split v3, n = 1,853 scored | 0 | 52.86 % | **54.61 %** | [52.34, 56.87] | 1,453 |
| `f-r18-224-v3-s1` | split v3 | 1 | 54.56 % | **54.07 %** | [51.80, 56.33] | 1,621 |
| `f-r18-224-v3-s2` | split v3 | 2 | 54.03 % | **54.51 %** | [52.23, 56.76] | 1,555 |

All eight runs are valid by the criteria fixed in advance (CUDA device, the pre-flight check of the initial weights, finite losses, result files written); none was dropped or replaced. Worker 0 skipped `f-r18-224-v3-s2` under its time budget and worker 1 ran it. Validation versus test is descriptive: official mean validation 68.90 % against mean test 68.56 %; v3 mean validation 53.81 % against mean test 54.40 %.

**Seed-mean statistic** (the pre-registered claim statistic, from the saved per-row hits):

| split | seeds | H (seed mean) | Wilson 95 % | frame-block bootstrap 95 % | no-learning copy floor | H minus floor |
|---|---|---|---|---|---|---|
| official | 5 | **68.56 %** | [66.49, 70.56] | 71 blocks of 160 ids: [64.90, 71.90] | 41.75 % | +26.81 [+23.34, +30.10] |
| split v3 (organiser baseline, no published number) | 3 | **54.40 %** | [52.12, 56.65] | 14 blocks: [46.72, 61.57] | 34.11 % | +20.29 [+17.33, +23.40] |

The verdict rule of `FROZEN-4.md` section 0 (Wilson and block lower bounds both above 62.4 %) allows its first wording row for the pre-registered primary only. The scope of the wording (four metrics, with Ref and Rel behind) is in `README.md` and `results/four_metric_table.md`. `docs/rescore-check.txt` shows that the public prediction files re-score to these numbers.

## 4. The four metrics (official split, 5-seed mean)

| metric | ours | MRLM | best published (paper Table 4) | rank |
|---|---|---|---|---|
| Acc_eps | 68.56 % | 62.4 % | 62.4 % (MRLM) | first of 19 entries |
| Acc_theta | 74.88 % | 70.9 % | 70.9 % (MRLM) | first of 19 entries |
| Ref (lower is better) | 0.0759 | 0.063 | 0.038 (LLaVA-v1.6-Vicuna-7B) | 10 published models are lower |
| Rel (lower is better) | 0.392 | 0.535 | 0.215 (LLaVA-v1.6-Vicuna-7B) | 5 published models are lower |

## 5. Edge numbers (`FROZEN-3.md` rule, seed-0 weights; kernel `gpu-int8`)

| model | parameters | ONNX fp32 | static int8 (`ref_minmax_all_u8`) | int8 test Acc_eps (Wilson 95 %) against the trainer's fp32 of the same weights | Kaggle Xeon 2.2 GHz, 1 thread, batch 1 (median / p95) | Apple M4, 1 thread, batch 1 |
|---|---|---|---|---|---|---|
| official split | 13,079,234 | 52.31 MB | 13.17 MB | 68.05 % [65.97, 70.06] against 68.45 % (n = 2,000) | fp32 41.08 / 44.33 ms; int8 21.51 / 22.74 ms | int8 median 9.47 ms (min 9.21, p95 9.87; 200 runs, ONNX Runtime 1.24.4, 1-minute load average 3.7); fp32 not measured (the file was over the 50 MB fetch cap) |
| split v3 | 13,078,208 | 52.31 MB | 13.17 MB | 54.45 % [52.18, 56.71] against 54.61 % (n = 1,853) | fp32 40.43 / 41.87 ms; int8 21.55 / 22.73 ms | same graph as above, not measured separately |

The int8 file was selected on validation by the rule fixed in `FROZEN-3.md` (the smallest file within 1.0 point of fp32 validation: 68.19 % against 68.50 % on the official split) and scored once per model on test. Unlike the MobileNetV3-small models below, whose fully static int8 exports lost 8 to 28 points, ResNet-18 keeps its accuracy under static int8; ten of the eleven recipes of the sweep stay within 1.0 point on the official validation (`TEST-LOG.md`).

## 6. Earlier runs on CPU (2026-10-07, MobileNetV3-small, 3,453,346 parameters, one seed each)

| run | split | test Acc_eps | interval | floor | note |
|---|---|---|---|---|---|
| 1 (`FROZEN-1.md`), 30 epochs | official | 58.25 % | Wilson [56.07, 60.39] | 41.75 % | below the published 62.4 %; reported as a negative result at the time, the target was not lowered |
| 2 (`FROZEN-2.md`), same configuration | split v3 (1,853 scored rows) | 50.13 % | Wilson [47.86, 52.41]; post hoc 14-block bootstrap [42.20, 57.96] | 34.11 % | +16.0 points over the floor on v3 and +16.5 on the official split: removing near-duplicates lowers the floor and the model by about the same 8 points |
| int8 `dyn_gemm_s8` (`FROZEN-3.md`), integer weights on the fully connected layers only, 8.23 MB | official / v3 | 58.30 % / 50.19 % | Wilson [56.13, 60.44] / [47.91, 52.46] | | fully static int8 collapsed (28.6 % and 30.95 % on the official test) |

## 7. Compute

Stage A was booked at 7,800 billed GPU-seconds and Stage B at 13,314 (2 GPUs x 6,657 s), together 21,114 s = 5.86 GPU-hours of the 8 allowed to this track, on the free Kaggle tier (2 x Tesla T4; the quota is billed per GPU held). The account counter moved by at most about 1.3 times the wall time for the two-GPU session, below the 2 times assumed when the budget was fixed, so the booking is conservative. The follow-ups (split, copy floor, claim statistic, int8 sweep, paired comparison) ran as CPU kernels at no GPU cost.

## 8. Not done or not measured

- The published 62.4 % (MRLM) was not reproduced; its weights are on Baidu Netdisk and its recipe is not public.
- fp32 latency on the Apple M4 for the ResNet-18 was not measured; the int8 latency was taken once, at a load average of 3.7.
- No ablation separates the effect of training length from the effect of the backbone.
- No claim on MC1296 or on any other dataset.
