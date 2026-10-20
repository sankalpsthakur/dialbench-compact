# dialbench-compact

Code, protocol, pre-registrations, test log and per-seed predictions behind a compact pointer-meter reader for DialBench RPM-10K: a 13.08 M-parameter ResNet-18 (224 px input; 13.2 MB as static int8 ONNX) trained on the official training split. Self-evaluated, not an official ranking.

**This repository contains no RPM-10K images, no labels and no model weights.** The dataset licence is "TBD" in the README of the DialBench repository, so the data and the weights stay with the authors' terms. The kit lets you recompute every reported number from the authors' own release.

## Result (scoped to four metrics)

- **Setting:** DialBench RPM-10K official split (`labels/train.json` 8,730 rows, `labels/test.json` 2,000 rows), official scorer (`benchmark/benchmark.py` of [Event-AHU/DialBench](https://github.com/Event-AHU/DialBench), ported and equivalence-tested against a verbatim copy of its evaluation logic), 5 seeds, self-evaluated. Runs of 2026-10-17; the table below was recomputed from the saved predictions on 2026-10-19.
- **Wording:** best published **Acc_eps and Acc_theta** in Table 4 of the DialBench paper (arXiv 2511.21982), and **behind on Ref and Rel**. It is never "best published result" without that scope.

| metric (paper Table 4) | ours, 5-seed mean | MRLM | best published | rank |
|---|---|---|---|---|
| Acc_eps, % (higher is better) | **68.56** | 62.4 | 62.4 (MRLM) | first: above all 18 published models |
| Acc_theta, % (higher is better) | **74.88** | 70.9 | 70.9 (MRLM) | first: above all 18 published models |
| Ref (lower is better) | 0.0759 | 0.063 | 0.038 (LLaVA-v1.6-Vicuna-7B) | **behind: 10 of the 18 published models are lower** |
| Rel (lower is better) | 0.392 | 0.535 | 0.215 (LLaVA-v1.6-Vicuna-7B) | **behind: 5 of the 18 published models are lower** |

- **Acc_eps in detail:** per seed 68.45, 69.10, 68.80, 68.10, 68.35 %; Wilson 95 % interval of the seed mean [66.49, 70.56]; frame-block bootstrap (71 blocks of 160 consecutive image ids, 5,000 draws) [64.90, 71.90]. Both lower bounds are above MRLM's 62.4 %. The Wilson interval treats images as independent; frames of the same video are not, which is why the block interval is the one that fixes the wording. MRLM has 8.7 B parameters; this model has 0.15 % of that. Acc_theta, Ref and Rel have no intervals; Acc_eps is the pre-registered statistic.
- **Edge numbers (seed 0 weights):** ONNX fp32 52.31 MB; static int8 13.17 MB with test Acc_eps 68.05 % (Wilson [65.97, 70.06]) against 68.45 % for fp32; one CPU thread, batch 1: 41.1 ms fp32 and 21.5 ms int8 on a Kaggle Xeon 2.2 GHz, 9.5 ms int8 on one Apple M4 thread.
- **Per-seed values, intervals and every test look:** `RESULTS.md`, `results/four_metric_table.md`, `TEST-LOG.md`.

### What the result does and does not show

- **Not reproduced.** The published 62.4 % was not reproduced: the MRLM weights are on Baidu Netdisk, and the paper's split naming (Section 3.1 describes an 81 % / 19 % train/test split, Section 3.2 gives 8,730 images for training and 2,000 for "validation"), checkpoint selection and training recipe are not public. "Best published" rests on Table 4 of the paper (re-read on 2026-10-19). A literature check on 2026-10-17 (arXiv search, the DialBench repository, web search) found no official-split RPM-10K result above 62.4 %; the 77.0 % of arXiv 2609.31998 is on its own 1,382-image set, not on RPM-10K. Repeat that check before relying on the wording.
- **The official split probably overlaps.** By thumbnail similarity, test images are as close to train images as train images are to each other (`docs/DIALBENCH-AUDIT.md`), which is what a random split of video frames looks like; the authors have not published video or instance ids, so this is inferred from the images, not stated. The number then measures the official protocol, with near-duplicate frames on both sides, not generalisation to unseen videos. On split v3, a twin-purged id-block re-split of ours (stricter, not proven leak-free, no published number, so no "best published" wording attaches), the same recipe scores **54.40 %** (3 seeds: 54.61, 54.07, 54.51; Wilson [52.12, 56.65]; 14-block bootstrap [46.72, 61.57]) against a no-learning nearest-neighbour label-copy floor of **34.11 %**: +20.29 points (block bootstrap of the difference [+17.33, +23.40]). On the official split the floor is 41.75 % and the model is 26.81 points above it ([+23.34, +30.10]).
- **One frozen configuration.** It was chosen on validation only (Stage A: ResNet-18 at 224 px, validation 68.09 %; the two other candidates that ran scored 66.36 and 64.74 %; four more were skipped by the pre-registered time guard), trained from scratch once per seed, and each model was scored once on the official test (EMA weights at the last epoch, window decoder, no test-time augmentation, no checkpoint selection). The eight trainer test looks and the two int8 looks are all in `TEST-LOG.md`; no run was dropped or replaced. Only the pre-registered primary carries the claim. GPU runs are not bit-reproducible: seeds fix initialisation, data order and augmentation draws, not cuDNN.
- **Two ingredients, not separated.** An earlier MobileNetV3-small (30 CPU epochs, one seed) scored 58.25 %. The same configuration trained for 100 GPU epochs validated at 64.74 % against 61.28 %, and the ResNet-18 reached 68.09 %: the gain comes from training length and backbone together, with no ablation between them.
- **Self-evaluated, narrow.** No board exists and nothing here was submitted anywhere. Not claimed: "#1" or "SOTA"; any claim on split v3; any claim on MC1296 (a second gauge dataset; its test labels are not public).
- **The predictions are model outputs, not labels,** but they carry information about the labels: they fall within the scorer's tolerance of the label on 68.56 % of the official test rows. They contain no label, range or hit mask (see `predictions/README.md`).

## Protocol

- **Benchmark and target:** DialBench RPM-10K official split; the target was Acc_eps above MRLM's 62.4 %, fixed before any GPU run and never lowered.
- **Claim rule, fixed before the runs (`FROZEN-4.md` section 0):** the statistic is the seed mean H of the pre-registered primary configuration; "clears the target" needs the Wilson 95 % lower bound above 62.4 % (H of at least 64.523 %), and the frame-block bootstrap is printed beside it and fixes the wording. Result: H = 68.56 %, both lower bounds above 62.4 %.
- **Selection on validation only:** a stratified validation part of about 1,000 `train.json` images (1,002 rows after rounding by stratum) is held out; `test.json` is scored once per frozen configuration and every look is logged.
- **Pre-registration:** `FROZEN-1.md` (run 1), `FROZEN-2.md` (run 2 on split v3), `FROZEN-3.md` (int8 recipe sweep) and `FROZEN-4.md` (the GPU plan with its amendments and result record).
- **Scorer:** `src/dialbench_metrics.py`, numpy only, equivalence-tested against `tests/official_reference.py` (the authors' logic, verbatim). The scorer takes the last number in a text and counts a missing number as 0.0, as the authors' script does; the audit lists where that matters.
- **Audit of the release:** `docs/DIALBENCH-AUDIT.md` (split naming, one image in both label files, 10,730 image files against 10,729 named, rows with two or no numbers, near-duplicate frames across train and test).

## Open questions to the authors

The audit raised three questions, which were put to the DialBench authors by e-mail on 2026-10-07 (the licence question was repeated on 2026-10-17); no answer was known on 2026-10-19. On 2026-10-19 (UTC) the licence question was also posted as [Event-AHU/DialBench#5](https://github.com/Event-AHU/DialBench/issues/5):

1. **Licence:** which licence applies to the RPM-10K data, may weights trained on it be released, and may a scorer port, a split file of image names and baselines be published?
2. **Split naming:** Section 3.1 of the paper describes an 81 % / 19 % train/test split and Section 3.2 calls the 2,000 held-out images "validation", while the release names that file `test.json`; which is intended, and was any checkpoint or hyperparameter chosen on it?
3. **Instance overlap:** Section 3.1 says no meter instance appears in more than one split, Section 3.2 says most images are video frames, and by thumbnail similarity test images sit as close to train images as train images do to each other; is the split by video or by meter, and can instance ids be shared?

## Verify the tooling (no data needed)

```bash
pip install numpy pytest
python -m pytest -q tests                 # all pass; the one test that needs torch is skipped without it
sha256sum --check --quiet MANIFEST.txt    # on macOS: shasum -a 256 --check --quiet MANIFEST.txt
bash tests/test_gpu_push_wait.sh          # the push gate against a fake `kaggle`; needs BSD `date -v` (macOS), so it is not part of CI
```

The tests cover: the scorer against the authors' logic; the label, span and split logic; the re-scorer against the pre-registered claim-statistic kernel; the kernel bundles (embedded modules, hashes); the format of the prediction files; the logic of the GPU wrapper (forked stub workers, fake `nvidia-smi`), the Stage B planner and the push gate with stubs. CI runs the Python tests and the manifest check on Python 3.12 and 3.13.

## Reproduce

Everything that needs the data is in [REPRODUCE.md](REPRODUCE.md). In short, with the authors' `labels/test.json` and `labels/train.json` you can

```bash
python protocol/check_release.py RPM-10K/labels --pred predictions/official_seed0.json     # is it the release the numbers were computed on?
python protocol/rescore_seeds.py RPM-10K/labels/test.json predictions/official_seed*.json  # the five seeds: four-metric means, Wilson and block intervals
python protocol/score.py RPM-10K/labels/test.json predictions/official_seed0.json          # one seed, with the ported official scorer
```

and re-train on Kaggle with the kernels in `kernels/` (private kernels `bench-gauges-*`; the training needs the data, which stays inside the kernel).

## Repository layout

| path | what |
|---|---|
| `src/` | the ported scorer (`dialbench_metrics.py`), label / target / decoder logic (`rpm_common.py`), the PyTorch and JAX trainers, ONNX and int8 export, the int8 sweep |
| `protocol/` | scorers (`score.py`, `score_v3.py`, `rescore_seeds.py`), `check_release.py`, the two re-splits `splits_v2.json` and `splits_v3.json` (image names only), `baselines.md` |
| `predictions/` | per-seed predictions: image file name to predicted reading, nothing else |
| `results/`, `RESULTS.md`, `TEST-LOG.md` | the four-metric table, the results summary, and every test evaluation in order, including audits and int8 looks |
| `FROZEN-1.md` ... `FROZEN-4.md` | the pre-registrations and their amendments |
| `docs/` | the audit of the release, the raw outputs of the audit and Stage A / B kernels, `rescore-check.txt` |
| `kernels/` | the Kaggle kernel sources as run, with metadata; `script.py` is generated from `src/` and `main.py` by `tools/build_bundle.py` |
| `tools/` | kernel generator and checker, the Stage B planner, push and wait helpers, latency tools, summarizers, and the tools that made `predictions/` and `results/` (they need private files) |
| `tests/` | unit tests, the verbatim oracle of the authors' scorer, synthetic fixtures for local smoke runs |
| `REDACTIONS.md`, `MANIFEST.txt` | the 19 files that differ from the private repository's version (with both hashes), and the sha256 of every file here |
| `LICENSE`, `LICENSES/`, `NOTICE` | Apache-2.0 for the code; the MIT text of the DialBench LICENSE file; what is derived from DialBench |

## What differs from the private working repository

- **Fresh history.** The pre-registrations were committed in the author's private working repository before the runs they govern. The commit hashes cited in `FROZEN-*.md` and `TEST-LOG.md` (for example `170a997`, `40ec7e1`) refer to that repository. This repository is a fresh export, so its history does not prove that order.
- **Not included:** the working notes (`STATUS.md`, `SUBMISSION.md`, the drafts of e-mails and issues), whose results are in this README, `RESULTS.md` and `TEST-LOG.md`; the program-level files that the logs and helper scripts refer to (`PLAN.md`, `RESOURCES.md`, `bin/guard.sh`, the `locks/` directories); the raw Kaggle kernel outputs (`logs/`), the weights and the data; the material on a second gauge dataset (MC1296, no claim); two diagnostic kernels that produced no result (`env-probe`, `cpu-bench`).
- **Redactions** (each marked in the text it touches; `REDACTIONS.md` lists the 19 files that differ from the private repository's version, with both sha256 values and where the private value is quoted. The hashes quoted in `FROZEN-*.md`, `TEST-LOG.md` and `docs/` are those of the private versions, so they do not match these 19 files):
  - names of unrelated Kaggle kernels of the author's account are replaced by `[names redacted]`, three times in `FROZEN-4.md`;
  - the absolute path of the build machine in 72 comment lines of the 13 generated bundles `kernels/*/script.py` is replaced by `<repo>/`. The embedded module sources and `main.py` are unchanged, so the hashes in the comment lines still match. `tests/test_kernel_bundles.py` checks that each bundle embeds exactly the `src/` modules it declares;
  - the Dropbox share link of the authors' data in `kernels/rpm-prep/rpm_prep.py` is replaced by a placeholder (take the link from the DialBench README);
  - in three Kaggle helper scripts (`tools/gpu_push_wait.sh`, `slot_push_wait.sh`, `wait_kernel.sh`) the default path of the author's benchmarks directory is replaced by a required environment variable `BM`;
  - in `docs/DIALBENCH-AUDIT.md` two parentheticals are replaced by "not reproduced here": the reading and range of a single test image, and one example label string.
- **Added after the runs** (2026-10-19; they change no result): `predictions/`, `protocol/rescore_seeds.py`, `protocol/check_release.py`, `tools/reduce_predictions.py`, `tools/four_metric_table.py`, `tools/rescore_check.py`, `results/`, `docs/rescore-check.txt`, and the tests `test_kernel_bundles.py`, `test_predictions_reduced.py`, `test_rescore_seeds.py`, `test_check_release.py`. `tests/test_export_mapping.py` now skips when torch is not installed, and `tests/test_dialbench_metrics.py` and `tests/test_rpm_common.py` use synthetic two-number strings where they used to quote label text.
- **Terms used in the logs:** *owner* is the author; *lead* or *coordinator* is the coordinating session of the AI-assisted workflow below, and a *builder* is a session working on this track; *tracks* are the other benchmark projects that shared the same compute account; `locks/` holds the lock directories that kept those tracks from using the shared Kaggle account's GPU sessions at the same time.

## How this was made

The work was run with AI coding agents (Claude Code) under the author's direction. Agents wrote the code, built and monitored the Kaggle kernels, kept the pre-registration and the test log, analysed the results and drafted the documents.

## Licences

- The code is licensed Apache-2.0 (`LICENSE`), except the files derived from DialBench code, which keep the notice of their source: the README of the DialBench repository says "Code: BSD 3-Clause License", while its LICENSE file is the MIT License ("Copyright (c) 2025 Event-AHU", reproduced in `LICENSES/MIT-DialBench.txt`). See `NOTICE`.
- **RPM-10K:** the README of the DialBench repository lists the dataset licence as "TBD". Nothing from the dataset is in this repository except image file names and aggregate statistics.
- **Weights:** none are released. They stay private until the authors state the terms for models trained on RPM-10K.
- The authors were asked by e-mail on 2026-10-07 and 2026-10-17 which licence they intend, and were told the result on 2026-10-18; no answer was known on 2026-10-19.
- This is independent work, not affiliated with or endorsed by the DialBench authors.

## Citation

Please cite the DialBench paper; the authors have not stated a preferred citation for RPM-10K.

```bibtex
@misc{wang2025dialbench,
  title         = {DialBench: Towards Accurate Reading Recognition of Pointer Meter using Large Foundation Models},
  author        = {Wang, Futian and Weng, Chaoliu and Wang, Xiao and Chen, Zhen and Zhao, Zhicheng and Tang, Jin},
  year          = {2025},
  eprint        = {2511.21982},
  archivePrefix = {arXiv}
}
```
