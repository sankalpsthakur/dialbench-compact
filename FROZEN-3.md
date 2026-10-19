# Pre-registered int8 recipe sweep (frozen before the kernel is pushed; first committed 2026-10-07 10:29 UTC as `09fe0f7`, amended 11:04 UTC as `d9b6e16` (calibration pool and script hash); times corrected from `git log`)

Why: run 1's exported int8 models collapsed (test 28.6 % all-ops, 30.95 % conv-only; validation 28.35 % and 33.43 % against 61.28 % for fp32; `TEST-LOG.md`). The edge brief asks for a working quantized variant. This sweep looks for a post-training int8 recipe that keeps accuracy. **Disclosure: the grid below was written after seeing run 1's int8 failure (validation and test numbers). The choice among recipes is made on validation only, by the rule below.**

Models (own weights, no new training): the run-1 EMA weights (official split; validation = the 1,002 held-out `train.json` rows of run 1) and the run-2 EMA weights (split v3; validation = the v3 val part, 945 images). Weights come from the finished kernels `bench-gauges-cpu-mnv3s-b` (v2) and `bench-gauges-cpu-v3`, mounted as kernel sources.

Kernel `sankalpsthakur/bench-gauges-int8-sweep` (private, CPU, internet on for `pip install onnxruntime`; sources rpm-prep, rpm-split3, cpu-mnv3s-b, cpu-v3). Code `src/rpm_int8_sweep.py` bundled in `kernels/int8-sweep/script.py` (sha256 `a33c8a526325b4cc32c7cc6e0355422ca274d9e83b937b02c8c9c038ca79ad9c`). Smoke-tested locally on the synthetic fixture (plumbing only: all 11 recipes build and run on a tiny random model).

Procedure per model: export fp32 ONNX from the weights (opset 17, normalisation inside the graph, as in the run's export); build each recipe below from that file; calibrate on 256 random images of the model's own TRAIN part (seed 0); score every recipe on the model's VALIDATION part only; measure file size and 1-thread batch-1 latency on the Kaggle CPU.

Recipes (all ONNX Runtime post-training quantization, QDQ format, per-channel int8 weights, bias not quantized unless noted; "u8"/"s8" = activation type):
1. `ref_minmax_all_u8`: min-max calibration, all supported ops (the run-1 recipe, reference)
2. `pct9999_all_u8`: percentile 99.99 calibration, all ops
3. `pct999_all_u8`: percentile 99.9, all ops
4. `entropy_all_u8`: entropy calibration, all ops
5. `pct9999_conv_gemm_u8`: percentile 99.99, Conv/Gemm/MatMul only
6. `pct9999_no_dw_u8`: as 5, depthwise convolutions kept in fp32
7. `pct9999_no_dw_no_se_u8`: as 6, squeeze-excite nodes also kept in fp32
8. `pct9999_all_s8sym`: percentile 99.99, signed symmetric activations, all ops
9. `dyn_gemm_s8`: dynamic quantization of Gemm/MatMul only (int8 weights, activation scale computed per run)
10. `pre_pct9999_all_u8`: ONNX Runtime pre-processing (shape inference + optimisation), then recipe 2
11. `pre_pct9999_no_dw_no_se_u8`: pre-processing, then recipe 7

Selection rule (applied per model, by the kernel, on validation numbers only): among recipes whose validation Acc_eps is within **1.0 percentage point** of the fp32 ONNX validation Acc_eps, take the smallest file; ties go to the higher validation accuracy, then to the earlier recipe in the list. If no recipe qualifies, nothing is scored on test and the result is reported as "no int8 recipe in this grid keeps accuracy" (with the best validation numbers for information).

Test looks: at most ONE per model (the selected recipe, once, on that model's own test part: `labels/test.json` for run 1, the v3 test part for run 2), at most two in total; both are logged in `TEST-LOG.md` as they are printed. No other recipe is scored on any test image.

Reported: every recipe's validation accuracy, file size and latency; the selected recipe's test Acc_eps with Wilson 95 % interval beside the fp32 ONNX test number of the same run; M4 latency of the selected file only under the Mac exception rules. Not claimed: any edge superiority over other models; int8 is presented only if it qualifies under the rule.

Not done on purpose: no quantization-aware training, no recipe added after seeing results, no tuning of the tolerance.

Note on the reference recipe: calibration images come from the model's own training part (run 1: the 7,727 training rows, not the 1,002 validation rows; run 1's own export drew its 256 calibration images from all 8,729 `train.json` rows, so `ref_minmax_all_u8` here will not be bit-identical to run 1's logged int8 numbers).
