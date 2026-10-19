# RPM-10K / DialBench corrected-protocol package (organizer baselines)

Contents
- `score.py`: the authors' scorer ported verbatim (`src/dialbench_metrics.py`, oracle in `tests/official_reference.py`), plus Wilson and grouped-bootstrap intervals.
- `splits_v2.json` (produced by kernel `bench-gauges-rpm-resplit`; image names and group ids only, no pixels): a group-disjoint re-split of all 10,729 distinct images (union-find on min(cosine of 56x56 gray, cosine of 32x32 rgb) >= 0.98, groups assigned whole, seed 2026), about 7.7k train / 1k val / 2k test.
- `splits_v3.json` (kernel `bench-gauges-rpm-split3`; names, block ids, purged lists; sha256 c16e51f6...fc2e3): the same pool re-split by 160-id blocks (ids follow capture order) with every cross-part twin pair (similarity >= 0.98) purged on both sides: 7,176 / 945 / 1,896. Leak-reduced, not leak-free (its `description` string says leak-free: ignore it, the file is kept byte for byte for its hash). Rule: `FROZEN-2.md`; code: `kernels/rpm-split3/rpm_split3.py`. Label rows of both files are pooled by image name (the one image in both files keeps its test.json row).
- `rescore_seeds.py` (added 2026-10-19): several per-seed prediction files at once (`predictions/`), on the official split or, with `--v3`, on the test part of split v3: per-seed metrics, the means of the four metrics, H, the Wilson interval and the pre-registered frame-block bootstrap of `FROZEN-4.md` section 0 (the code of `kernels/gpu-seedmean`; `tests/test_rescore_seeds.py` compares the two).
- `check_release.py` (added 2026-10-19): compares the release you downloaded (row counts, file sizes, zip hash, the order of the test names against the public predictions) with the one the experiments used.
- `score_v3.py`: the same scorer for one part of split v3 (rows of both label files pooled by image name, scorer range per row as specified in `FROZEN-2.md`, rows with no determinable range excluded and counted), with Wilson and block-bootstrap intervals. Covered by `tests/test_score_v3.py`.
- `baselines.md`: constant predictors, nearest-neighbour label copy, and the compact CNN on the official split, on similarity-bucketed subsets of the official test, on `splits_v2` and on split v3 (organizer baselines: official split 58.25 % against a 41.75 % copy floor; v3 50.13 % against 34.11 %).
- Write-up of the defects: `docs/DIALBENCH-AUDIT.md`.

Why
1. The official `test.json` is as close to `train.json` as train is to itself (kNN-copy 41.8 % vs 42.1 % leave-one-out), so "no meter instance appears in more than one split" is not borne out by the release. The group-disjoint split removes train/test twins.
2. The paper calls the 2,000-image split both test and validation; there is no third split. The protocol here: validation is carved from train, the test split is scored once per frozen config.
3. One image is in both files; one file in the zip is unreferenced; 61 test and 54+7 rows have no clean single number. The scorer takes the last number and scores "no number" as 0.0. We keep the scorer as is for comparability and report the affected rows separately.

How to use
```
python protocol/score.py RPM-10K/labels/test.json predictions.json [--groups groups.json]
python protocol/rescore_seeds.py RPM-10K/labels/test.json predictions/official_seed*.json
python protocol/score_v3.py protocol/splits_v3.json RPM-10K/labels/train.json RPM-10K/labels/test.json predictions_v3.json --part test
```
predictions.json maps image file name to the predicted reading (number or text; the last number is taken, as in the official script).
