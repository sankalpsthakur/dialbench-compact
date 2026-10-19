# Pre-registered run 1 (frozen before launch; written 2026-10-07 08:05 UTC+4)

Purpose: first complete DialBench RPM-10K result with a compact CNN on free CPU compute (GPU quota is 0 until 2026-10-10, TPU probe still queued).

Kernel: `bench-gauges-cpu-mnv3s-b` (private, CPU, internet on for the ImageNet weights and `pip install onnxruntime`).
Pipeline in one kernel: leakage/re-split audit prelude -> train -> test once -> ONNX export -> int8 (QDQ) export -> test of each variant.

Frozen config (trainer `src/rpm_train_torch.py`, defaults as in the file at the commit that contains this document):
```
arch mobilenetv3_small_100 (timm, ImageNet weights), input 224x224 squash (official resize), head: 1x1 conv 64 -> flatten -> 512 -> (505 bins + span classes)
epochs 30, batch 32, AdamW lr 7e-4 (head x3), wd 0.02, 1 warm-up epoch + cosine, EMA 0.995, seed 0
targets: y = reading / span (last number of the label string), soft labels sigma 0.004 over 505 bins, + L1 on expectation, span-class CE x0.3
decoder: window (half-width 4 bins = +-0.01 of span), reading = y_hat * span_hat
train data: labels/train.json minus the 1 image that is also in test.json, minus a stratified random validation split of 1,000 (seed 0)
test: labels/test.json scored once with the official scorer; EMA weights at the last epoch (no best-checkpoint selection)
int8 recipe: ONNX opset 17, static QDQ, per-channel int8 weights, uint8 activations, QuantizeBias off, 256 random TRAIN images for calibration
```
Pre-registered test evaluations (all logged in TEST-LOG.md whatever the outcome): PyTorch fp32 (inside the trainer), PyTorch fp32 (export check), ONNX fp32, ONNX int8.
Claim rule: only a Wilson 95 % lower bound above 62.4 % counts as clearing the published best; otherwise report as a negative or within-noise result.
This run is single-seed on CPU; the 5-seed protocol needs the TPU or the GPU after 2026-10-10.

## Amendment A (written 2026-10-07 08:20, BEFORE any run-1 output exists)
The first document above was committed before the code was rebuilt. What actually runs is Kaggle kernel `sankalpsthakur/bench-gauges-cpu-mnv3s-b` **version 2**, pushed 08:12 (version 1 crashed at 08:11 after its audit prelude on a config-literal bug, so it trained nothing and evaluated no model). Fingerprint of the pushed script: `kernels/cpu-mnv3s-b/script.py`, sha256 `9ce9e33cb875b2c5129c31431c810ce9cadcd4911886efdb98532bf8b686e3dd`, built from commit `7e47734` sources minus the later `init_from` option (not in this script).
Deviations from the text above (none changes the frozen hyperparameters, decoder, data or test rule):
1. Augmentation was rewritten after the first document: blur, noise, glare and random erase are applied to a random subset of each batch (same probabilities and strengths as before) instead of computing all and masking. The blur sigma is drawn from Python's `random`, which `run()` does not seed in this version, so the augmentation stream is not bit-reproducible. (Future runs seed `random` in `run()`.)
2. A fifth pre-registered test evaluation exists: ONNX int8 conv-only (QDQ on Conv/Gemm/MatMul only), next to PyTorch fp32 (trainer), PyTorch fp32 (export check), ONNX fp32 and ONNX int8 (all ops). All five are logged.
3. The kernel starts with a non-fatal JAX self-test prelude (tiny CPU training + JAX-vs-torch weight-mapping check); it touches only train.json images and no test label.
Logged separately: version 1's audit prelude (leakage/re-split) did read test labels and is in TEST-LOG.md.

## Run-2 decision rule (fixed now, from the 5-epoch validation curve of run 1; validation = EMA weights, window decoder, 1,000 held-out train images)
- Final validation >= 64 % and the validation Wilson interval is plausibly above 62.4: run 2 = same frozen config, seed 1, retrained from scratch (two seeds support a claim better than one tuned run). Change nothing.
- Final validation 55-64 % and still rising from epoch 25 to 30: run 2 = continue from run 1's weights (`init_from`) on the same validation split with a new cosine cycle; write FROZEN-2 before pushing; its test result is a second logged look and is disclosed as such.
- Final validation < 55 %: stop chasing the claim on CPU; report run 1 as an honest negative; spend the remaining time on the per-similarity-bucket analysis of run 1's test output, the GPU runbook and the edge numbers.

## Amendment B (written 2026-10-07 08:27, before any run-1 output exists): run 2 is now fixed
The coordinator directed at 08:24 that every model must be reported on both the official split and a leak-free split, with the leak-free split (group rule + SHA256 manifest) pre-registered before any model is scored on it, and with the nearest-neighbour copy baseline (41.8 % official) shown beside the model scores on both splits. Therefore:
- Run 2 = this same frozen run-1 configuration, retrained from scratch on the leak-free split's train part and scored once on that split's test part (own FROZEN-2 file with the split rule, the manifest hash and the code hash, written before run 2 is pushed). The 3-branch rule in Amendment A is demoted to "what to do with time left after run 2".
- Run 1's twin-free analysis (accuracy by nearest-train similarity bucket, `by_train_similarity` in its result JSON) is reported next to it as a no-retrain leak-reduced view; it is not a substitute for the retrained leak-free row.
- A model trained on one split is only ever scored on that split's test part. Official-trained models are never scored on leak-free test images (they may have trained on them).
