# Pre-registered run 2: the frozen run-1 configuration on split v3 (written 2026-10-07 about 10:21 UTC, commit `20fa331`, BEFORE any model is scored on v3; times in this file were corrected from `git log` at 11:25 UTC)

Purpose (coordinator directive, FROZEN-1 amendment B): report every model on the official split AND on a split with the near-duplicate train/test twins removed, with a hash-pinned split rule, and with the no-learning nearest-neighbour copy baseline beside each score.
Nothing below was chosen by looking at a model score on v3. No model has been scored on any v3 part when this file is committed. The pre-flight kernel described at the end is pushed after this commit.

## 1. Split v3 (built by kernel `sankalpsthakur/bench-gauges-rpm-split3` v1, private, CPU, internet off, 66 s of compute)

Name it what it is: a **twin-purged id-block split, leak-reduced, not leak-free**. (The manifest's own `description` string says "leak-free"; it is kept byte for byte because the file hash below is the pre-registered artifact. Residual same-meter similarity below the twin threshold remains; see the residual line below.)

Pool: all 10,729 distinct images referenced by `labels/train.json` + `labels/test.json` (the zip's 10,730th file is referenced by no row). Features: min(cosine of 56x56 gray thumbnail, cosine of 32x32 rgb thumbnail) on the 224x224 official-resize images; a "twin" pair is >= 0.98 (as in the v2 audit).

Rule (code: `kernels/rpm-split3/rpm_split3.py`, sha256 `9781164e122d3bf45fc84d24ea5d053443bc4810f6fc2e569fa88d38341376ea`):
1. Image ids (the N in `N_img.jpg`, range 1..11332) are ordered; consecutive ids form blocks of B ids. B is set by a rule on label-free image statistics: the smallest id distance d >= 40 whose mean pair similarity exceeds random pairs by at most 10 % of the distance-1 excess; at least 100, at most 320. Measured: random pairs 0.1120; id distance 1: 0.3294 (5.2 % of pairs >= 0.95); 5: 0.2685; 10: 0.2452; 40: 0.1831; 80: 0.1539 (excess 0.042, above the 0.0217 cut); 160: 0.1305 (excess 0.0185, below); 320: 0.1162. **B = 160** (71 blocks of 131 to 160 images; the id structure is consistent with ids following capture order, as the paper's "frames extracted from videos" suggests, but this is inferred from image similarity, not stated by the authors).
2. Whole blocks are assigned to test (about 2,000 images), then val (about 1,000), rest train, in a seeded random block order. The seed is the first of 2026, 2027, ... that makes each of the four range classes (3, 6, 10, other) at least 8 % of test and at least 60 % in train (this check reads label ranges, never readings or model scores). **Seed 2029** (2026-2028 failed the check). Before the purge: train 7,592 / val 1,052 / test 2,085.
3. Symmetric twin purge: every cross-part pair (train/val/test) with similarity >= 0.98 loses BOTH members. Purged: test 189, val 107, train 416 (712 images). **Final: train 7,176 / val 945 / test 1,896**; 50 / 7 / 14 blocks; the three parts and the purged set are disjoint and cover 10,729 images (checked).
4. Manifest `protocol/splits_v3.json` (image names, block ids, purged lists; no pixels or labels), 387,301 bytes, **sha256 `c16e51f6b9bd25cfa9925f19412da2510f9eaad9944dcddfc4d3a261307fc2e3`** (printed by the kernel and re-computed locally: equal). Kernel log: `docs/split3-output.txt`.

Facts printed by the kernel (no model involved):
- Residual: nearest-train similarity of the final test images, percentiles 50 / 90 / 99 / 100 = 0.799 / 0.961 / 0.978 / 0.980 (the official split has median 0.869 and 95th percentile 0.996 for test -> train).
- Of the 2,000 official test images: 18.2 % land in v3 test, 9.0 % in v3 val, 65.8 % in v3 train, 7.0 % are purged. So v3 train contains most of the official test images and their labels; a v3-trained model is only ever scored on v3 test, and an official-trained model only on the official test.
- Composition (range 3 / 6 / 10 / other): v3 test 661 / 576 / 411 / 248 (official test 661 / 665 / 531 / 142); v3 train 2,556 / 2,428 / 1,486 / 706. v3 test has a larger share of the hard "other" meters (13 % versus 7 %), so v3 and official numbers are not a like-for-like ablation; per-class numbers are reported.
- No-learning floors on v3 (official metric, rows with a determinable range): nearest-neighbour label copy 34.16 % on v3 test (n = 1,853), 24.26 % on v3 val (n = 944); predict 0: 4.10 %. Official split copy floor: 41.77 % (n = 1,999).

## 2. Run 2 = the run-1 configuration, retrained from scratch on the v3 train part, scored once on the v3 test part

Kernel `sankalpsthakur/bench-gauges-cpu-v3` (private, CPU, internet on for the timm ImageNet weights and `pip install onnxruntime`; sources: rpm-prep, model-port, rpm-split3). Script `kernels/cpu-v3/script.py`, sha256 `f66d552aa3e85c71529dbf57df3e1d52c63ad8d17db8f6ab4acd81afa268fa76` (see amendment 1 in section 5), config below, trainer config hash printed in the log.

Configuration: identical to FROZEN-1 in every hyperparameter (checked key by key against run 1's logged config: the only differences are `tag`, `split_file`, `split_sha256` and the unused `init_from=None`; `n_val`/`val_mode` stay at their defaults and are ignored when a split file is given). MobileNetV3-small (timm, ImageNet weights), 224x224 squash resize, 30 epochs, batch 32, AdamW lr 7e-4 (head x3), wd 0.02, 1 warm-up epoch + cosine, EMA 0.995, seed 0, soft labels sigma 0.004 over 505 bins + L1 + span-class CE x0.3, window decoder (half-width 4 bins), augmentations as before. Test = EMA weights at epoch 30; no checkpoint selection, no early stopping; the v3 val part is monitored every 5 epochs and used for nothing else.

Data rules for the `split_file` mode (code `prepare()` in `src/rpm_train_torch.py`, covered by `tests/test_split_rows.py`, `tests/check_split_prepare.py`, and a synthetic-fixture smoke test of the whole train, test and export path):
- Rows of both label files are pooled by image name. `7443_img.jpg` is in both files; its test.json row is used (as in the split kernel's baselines).
- A row's full-scale value ("span", used for the target y = reading / span and the span-class head) is: the numeric `range` (test.json rows), else the "0-3"-style string (train.json), else the range stated in the chain-of-thought text when its low end is 0 (669 train rows), else unknown. Unknown-span training rows contribute no loss. Span classes (values with >= 20 training rows plus a rare class) are built from the v3 train part only.
- The scorer range of a test row is the same value. Test rows with no determinable range are excluded from scoring and counted (expected 43 of 1,896: the kernel baselines score 1,853 of the 1,896 test images). The range is never an input to the model.
- The official scorer is used unchanged (`Acc_eps` with the last-number rule); Acc_theta, Ref, Rel are reported as well.

Pre-registered test evaluations (all go in `TEST-LOG.md`, whatever the outcome), the same list as run 1: (1) PyTorch fp32 inside the trainer (also prints argmax and expectation decoders; the window decoder is the frozen one), (2) PyTorch fp32 in the export step, (3) ONNX fp32, (4) ONNX int8 static QDQ all ops (256 random v3-train images for calibration), (5) ONNX int8 conv-only. Reported with each: Wilson 95 % interval, grouped bootstrap (union-find thumbnail groups at 0.98), per meter type (test.json rows only), per range class, per source file (train.json rows versus test.json rows), per range source, accuracy by nearest-train similarity, and the nearest-neighbour copy baseline recomputed in the same kernel on the same scored rows.

Claim rule: there is no published number on v3, so run 2 is an **organizer baseline on a corrected protocol**, never a "best published" claim; the official-split claim stays with run 1 (a negative: 58.25 %, upper bound 60.39 % under the published 62.4 %). The v3 result is read against its own copy floor (34.2 %) and with the composition caveat above. If v3 accuracy is within the interval of run 1, say so; if it is lower, say how much of the drop the composition and the twin removal can each explain only as far as the per-class and per-bucket tables show, not more.

## 3. What differs from the run-1 code (besides the data path)

`random` is now seeded in `run()` (run 1 did not: the blur-sigma stream was not reproducible); `Targets.span_class` no longer crashes when no span value reaches 20 rows (tiny fixtures only); the export step takes its data from the same `prepare()`; result JSONs carry the split info and the copy baseline. The JAX self-test prelude is NOT in this kernel (run 1's crashed on a config bug, now fixed; it runs in the pre-flight below). Everything the model sees and the loss are unchanged.

## 4. After run 2 (decided now)

- Optional run 2b: the same configuration with seed 1 on the same split, pushed only if it can finish before the 14:30 UTC time box; if run, both seeds are reported and the v3 headline is their mean with both values and both intervals shown.
- An int8 recipe sweep (calibration method, op types, pre-processing) is selected on v3 VALIDATION only, in a separate kernel, and gets its own pre-registration (FROZEN-3) before any test image is scored with a new recipe. Run 2's own int8 results use the run-1 recipes as listed above.
- Latency on the M4 (1 thread, batch 1) is measured only under the Mac exception rules.

## 5. Amendments

**Amendment 1 (about 10:23 UTC, commit `f667cc9`; before any push, before any model is scored on v3): no pre-flight kernel.** The pre-flight described in the first version of this file (`bench-gauges-cpu-v3-pre`, validation only) is dropped and its kernel directory deleted: all four program CPU slots were taken by other tracks when it was due, it would not have exercised the test stage, and the first minutes of the run-2 kernel itself show the same path and data errors. Instead: (a) the v3-only reporting block of the test stage is wrapped in try/except so that the core test metrics and the saved weights survive any bug in it; (b) the kernel was rebuilt, so the run-2 script hash above changed from `137835a6...` to `f66d552a...`. Nothing the model sees, no hyperparameter, no data rule and no scoring rule changed. The fixed JAX self-test prelude is not part of run 2; it will run in its own tiny kernel if the optional TPU probe goes ahead.
