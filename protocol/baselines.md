# Baselines (organizer baselines on the corrected protocol; official scorer; Acc_eps = share with |pred - gt| <= 0.01 * range)

No-learning floors (audit kernels, test labels read for scoring only; logged in `TEST-LOG.md`):

| Baseline | Official `test.json` (n = 2,000 unless noted) | v2 test (n = 1,964 with known range) |
|---|---|---|
| predict 0 | 6.1 % | not run |
| median(train reading/range) x range | 1.2 % | not run |
| 0.5 x range | 1.55 % | not run |
| nearest-neighbour label copy, gray 28 px | 41.75 % | not run |
| nearest-neighbour label copy, gray 56 px and rgb 32 px (min cosine) | 41.77 % (n = 1,999) | 40.22 % |
| same, official-test subset with no train neighbour >= 0.95 / 0.98 / 0.99 | 41.49 % (n=1,405) / 43.42 % (1,709) / 42.90 % (1,823) | |

Published, for scale (paper Table 4, fine-tuned unless closed-source): MRLM 62.4 %, LLaVA-v1.6-Vicuna-7B 56.2 %, LLaMA3.2-Vision-11B 53.6 %, LLaVA-1.5-7B 49.2 %, Qwen2-VL-7B 45.8 %, Gemini2.5-pro 15.8 %, GPT-5 7.8 %.

Learned baselines (one compact CNN, MobileNetV3-small, 3.45 M parameters, ImageNet-initialised, 30 epochs, one CPU seed; self-evaluated; `TEST-LOG.md`):

| Split (train part -> test part) | Acc_eps | 95 % intervals | no-learning copy floor on the same rows | model minus floor (paired, post hoc, block bootstrap) |
|---|---|---|---|---|
| official: train.json -> test.json (n = 2,000) | 58.25 % | Wilson [56.07, 60.39] | 41.75 % | +16.50 [+13.63, +19.32] |
| v3: v3 train -> v3 test (n = 1,853 of 1,896) | 50.13 % | Wilson [47.86, 52.41]; block bootstrap over 14 blocks [42.20, 57.96] | 34.11 % | +16.03 [+11.86, +20.52] |

Same two models with int8 weights on the fully connected layers only (`dyn_gemm_s8`, 8.23 MB): 58.30 % (official) and 50.19 % (v3). Fully static int8 variants lose 8 to 28 points on validation (`TEST-LOG.md`, `FROZEN-3.md`).

Learned baselines, GPU plan of `FROZEN-4.md` (Stage B, 2026-10-17): ResNet-18 at 224 px, 13.08 M parameters, ImageNet initialisation, 80 epochs on 2 x T4, one frozen configuration (chosen on validation only), seed mean of the pre-registered primary; self-evaluated; `TEST-LOG.md`, `docs/stageB-output.txt`:

| Split (train part -> test part) | Seeds | Per-seed Acc_eps | Seed mean H | Wilson 95 % | Block bootstrap 95 % | No-learning copy floor | H minus floor (block bootstrap) |
|---|---|---|---|---|---|---|---|
| official: train.json -> test.json (n = 2,000) | 5 | 68.45, 69.10, 68.80, 68.10, 68.35 | **68.56 %** | [66.49, 70.56] | 71 blocks of 160 ids [64.90, 71.90] | 41.75 % | +26.81 [+23.34, +30.10] |
| v3: v3 train -> v3 test (n = 1,853 of 1,896) | 3 | 54.61, 54.07, 54.51 | **54.40 %** | [52.12, 56.65] | 14 blocks [46.72, 61.57] | 34.11 % | +20.29 [+17.33, +23.40] |

Same models, seed 0, static int8 (`ref_minmax_all_u8`, 13.17 MB against 52.31 MB fp32): 68.05 % (official) and 54.45 % (v3). The official-split row carries the claim of `SUBMISSION.md` section 0.1 (with its caveats); the v3 row is an organizer baseline with no published number.

## Split v3 (twin-purged id-block split, `protocol/splits_v3.json`, sha256 c16e51f6b9bd25cfa9925f19412da2510f9eaad9944dcddfc4d3a261307fc2e3)

No-learning floors computed by kernel `bench-gauges-rpm-split3` (official metric; rows with no determinable range excluded; log in `docs/split3-output.txt`, run before any model was scored on v3):

| Baseline | v3 test | v3 val | official test (for scale) |
|---|---|---|---|
| nearest-neighbour label copy (min of gray 56 px and rgb 32 px cosine, nearest image of the split's train part) | 34.16 % (n = 1,853) | 24.26 % (n = 944) | 41.77 % (n = 1,999) |
| predict 0 | 4.10 % | not run | 6.1 % |

v3 sizes: train 7,176 / val 945 / test 1,896 images (712 purged). Test composition by range 3 / 6 / 10 / other: 661 / 576 / 411 / 248 (official test 661 / 665 / 531 / 142), so v3 test has a larger share of the hard "other" meters. The manifest's `description` string says "leak-free": read it as twin-purged (see FROZEN-2.md); the file is kept byte for byte because its hash is the pre-registered artifact.
