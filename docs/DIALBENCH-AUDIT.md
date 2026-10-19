# DialBench / RPM-10K audit (2026-10-07)

Everything below was computed inside private Kaggle CPU kernels (`bench-gauges-rpm-prep`, `bench-gauges-rpm-eda2`) on the Dropbox zip
(3,185,292,936 bytes, sha256 `97e09a89381336316f3893709cee626f51de09892ff7a80d328767b3cac9fe6f`). Only aggregates are stored in this repo.
Sources: arXiv 2511.21982 (HTML v1), github.com/Event-AHU/DialBench at `main` (pushed 2026-08-27), README, `benchmark/benchmark.py`.

## Release structure
- `RPM-10K/images/*.jpg` (10,730 files, 6,496 distinct sizes, most common 768x1080, phone photos up to 3072x4096), `RPM-10K/labels/train.json` (8,730 rows), `RPM-10K/labels/test.json` (2,000 rows).
- Download channels in the README: Baidu Netdisk (2026-04-08; GitHub issue #3 reports only 1,371 images were obtainable then) and Dropbox (2026-08-27; complete). Weights: Baidu only.
- README license block: "Code: BSD 3-Clause License, Dataset: TBD, Model weights: TBD". The repo `LICENSE` file is MIT.

## Split naming (the inconsistency)
| Where | Wording |
|---|---|
| Paper Sec. 3.1 "Train/Test Split" | two non-overlapping subsets, 81 % / 19 %, "no meter instance appears in more than one split", "training and testing" |
| Paper Sec. 3.2 | "8,730 for training and 2,000 for validation" (the only use of "validation" in the paper) |
| Paper Sec. 3.1 usage protocol | the test split is "reserved solely for final performance reporting" |
| Release | `labels/train.json` and `labels/test.json`; the scorer takes any JSON path, no split name |
| Table 4 | models "retrained" on the dataset; scored on the 2,000-image split |
So the 2,000 images are "test" in the release and in the protocol text, and "validation" in one sentence. There is no third split, so if checkpoints or hyperparameters were chosen on it, the published numbers carry selection bias that cannot be checked. Our protocol: validation is carved from `train.json`, `test.json` is evaluated once per frozen config and logged in `TEST-LOG.md`.

## Data defects found
- `7443_img.jpg` is in both `train.json` and `test.json` (its reading and range are not reproduced here): 1 of 2,000 test rows is also a train row.
- 8,730 + 2,000 rows reference 10,729 distinct image names; the zip holds 10,730 images, so exactly one file is referenced by no row (we did not list its name).
- Field mismatch: train rows have `reading`, `detail` (chain-of-thought), `range` as a string ("0-3", "0-6", "0-10") plus `graduation value` and `key scale`, null for 820 rows. Test rows have `reading`, numeric `range` (30 distinct values), `meter_type` (biao1-biao6, others), `environment_conditions`. The 820 null-range train rows are the "others" meters; 691 of them state a range in the CoT text.
- Readings that are not one number: train 285 rows with two numbers (a main dial and a sub-dial, or an upper and a lower dial, in one text; the example strings are not reproduced here), 54 rows with no number at all ("pointer is not clear", "none", "no pointer"); test 61 two-number rows, 7 with no number (one is Chinese text), 1 negative. The scorer takes the LAST number, and no number scores as 0.0, so an unreadable image is "correct" if a model outputs 0.
- Test labels with reading/range > 1 exist (max 8.62), i.e. label noise.
- Paper statistics versus data: "nearly 2,000 web-collected images" versus 1,014 "others"-type rows in total (820 train + 194 test); "more than 300 dial types" versus six named types plus others.

## Scorer facts (from `benchmark/benchmark.py`)
- Acc_eps = mean(|pred - gt| <= 0.01 * range) over all rows (float `<=`); Acc_theta = share of nonzero-gt rows with |error|/gt < 0.05; Ref = mean(min(|error|/range, 10)); Rel = mean(min(|error|/gt, 100)).
- Inputs are resized to 224x224 (Lanczos, aspect ignored). `range` comes from the label file and is used only for epsilon.
- The model output is parsed by `extract_last_number`: any trailing number in a VLM answer (for example a unit range) becomes the prediction.
- The paper's prose calls Acc_theta an "angular error" and mixes the Ref/Rel descriptions; the code is the authority and is what we ported.

## Leakage evidence (benchmark audit, logged in TEST-LOG.md)
Computed in `bench-gauges-rpm-eda2` (thumbnail cosine, gray 28/56 px and rgb 32 px) and `bench-gauges-rpm-resplit` (a pair counts as "same scene" only if BOTH gray 56x56 and rgb 32x32 cosines clear the threshold). Raw output: `docs/resplit-output.txt`.

1. Test images are not farther from train than train images are from each other. Share of images with a neighbour at cosine >= tau in the other pool (8,729 train images, 1,999 test images, the duplicated image excluded):

| tau | test with a train neighbour | train with another train neighbour |
|---|---|---|
| 0.90 | 43.2 % | 42.7 % |
| 0.95 | 29.7 % | 27.4 % |
| 0.98 | 14.5 % | 13.9 % |
| 0.99 | 8.8 % | 8.5 % |
| 0.995 | 6.05 % | 5.1 % |
Percentiles (5/25/50/75/95/99) of the nearest-neighbour cosine: test -> train 0.538 / 0.724 / 0.869 / 0.961 / 0.996 / 0.999, train -> train 0.537 / 0.722 / 0.865 / 0.957 / 0.995 / 0.999. This is what a random frame-level split of one pool looks like. It does not show that the same physical meter is on both sides (the same dial type also scores high), but it gives no support to "no meter instance appears in more than one split". At tau 0.98 there are 3,877 similar pairs among the 10,729 images and 160 groups contain both train and test images (347 at 0.95, 107 at 0.99).
2. Correction to an earlier reading of ours: nearest-neighbour label copy is NOT a leak detector. It scores 41.8 % on the official test (n=1,999), 40.2 % on the group-disjoint v2 test, and 41.5 % / 43.4 % / 42.9 % on the official-test subsets with no train neighbour above 0.95 / 0.98 / 0.99. The score is dominated by the label prior (readings cluster, and a same-family neighbour often shares the reading band), so it is a floor for the metric, not evidence of overlap. For scale: constants score 0.012 to 0.061, GPT-5 7.8 %, LLaVA-v1.6-7B 56.2 %, MRLM 62.4 % (paper Table 4).
3. How much overlap helps a trained model is measured by the per-similarity-bucket accuracy of the frozen model (`by_train_similarity` in each result JSON).
4. v2 re-split (`protocol/splits_v2.json`): union-find on pairs with both cosines >= 0.98 over all 10,729 distinct images, whole groups assigned, seed 2026: 7,707 train / 1,022 val / 2,000 test; span mix 3/6/10/other in v2 test = 745/698/410/111 versus 661/665/531/142 in the official test (composition differs because assignment is by image groups, not stratified).

5. Image ids follow capture order (kernel `bench-gauges-rpm-split3`, `docs/split3-output.txt`; image similarity only, no labels). The N in `N_img.jpg` runs 1..11,332. Mean thumbnail similarity (min of gray 56 px and rgb 32 px cosine) of two images whose ids differ by about d: 0.329 at d = 1 (5.2 % of pairs at 0.95 or more), 0.269 at 5, 0.245 at 10, 0.183 at 40, 0.154 at 80, 0.131 at 160, 0.116 at 320, 0.108 at 1,280, against 0.112 for random pairs. So near-identical frames sit at neighbouring ids, consistent with the paper's "frames extracted from videos at regular intervals", and a split by id range keeps videos together while a random split (the release) does not. This is inferred from the images; the authors have not published instance or video ids (question 3 of the email).
6. Split v3 (`protocol/splits_v3.json`, rule and hash in `FROZEN-2.md`): 160-id blocks assigned whole to train/val/test, then every cross-part pair at similarity >= 0.98 is purged on both sides: train 7,176 / val 945 / test 1,896 images (712 purged). Residual nearest-train similarity of the v3 test: median 0.80, 90th percentile 0.96, maximum 0.98 (official: median 0.87, 95th percentile 0.996). The no-learning nearest-neighbour label copy drops from 41.8 % (official test) to 34.2 % (v3 test), so part of the official floor is overlap, and the rest is the label prior of the metric.

## How the split inconsistency affects comparisons
- Table 4 reports 18 models retrained on the training split and scored on the 2,000-image split that Sec. 3.2 calls "validation" and Sec. 3.1 calls "testing". If any model or checkpoint was chosen on that split, the ranking in Table 4 has selection bias that cannot be checked from the release. Our numbers avoid it: validation comes out of `train.json` and `test.json` is scored once per frozen configuration.
- Because test images sit as close to train as train images sit to each other (table above), the official split measures interpolation within recorded scenes. Claims about unseen meters need the group-disjoint v2 split (`protocol/splits_v2.json`); both numbers are reported.
- The scorer parses the last number of any output string and divides by a test-side `range`; models that print extra numbers (units, ranges, a second dial) are scored on the wrong one. Compare models only through the same numeric extraction.
