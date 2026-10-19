# Reproducing the numbers

Two levels: **A. re-score** the published predictions with the authors' data (minutes, numpy only), and **B. re-train** on Kaggle (hours of GPU). Everything that touches RPM-10K pixels or labels runs on your machine (A) or inside your own private Kaggle kernels (B); nothing of the dataset is in this repository.

## 0. Get the data from the authors

Download RPM-10K from the links in the README of [github.com/Event-AHU/DialBench](https://github.com/Event-AHU/DialBench) (as of 2026-10-19 it lists a Dropbox and a Baidu Netdisk link). Its licence is "TBD" there: check the README for the current terms before you use the data. Unpack so that you have `RPM-10K/labels/train.json`, `RPM-10K/labels/test.json` and `RPM-10K/images/`.

## 1. Check that you have the same release

```bash
python protocol/check_release.py RPM-10K/labels --pred predictions/official_seed0.json --zip RPM-10K.zip --images RPM-10K/images
```

Expected: every line `ok`. The values were recorded from the release the experiments used (downloaded on 2026-10-07 by `kernels/rpm-prep`, whose log prints the same zip hash):

| what | value |
|---|---|
| zip | 3,185,292,936 bytes, sha256 `97e09a89381336316f3893709cee626f51de09892ff7a80d328767b3cac9fe6f` |
| `labels/test.json` | 2,000 rows, 491,106 bytes |
| `labels/train.json` | 8,730 rows, 6,253,664 bytes |
| images | 10,730 files; the two label files name 10,729 distinct images; `7443_img.jpg` is in both files; one file is named by no row |
| order | the image names of `test.json`, in file order, are the keys of every `predictions/official_seed*.json` |

The sha256 of the two label files was not recorded; the script prints it. If something differs, the release has changed since 2026-10-07: the predictions then belong to the old files, and we would like to know which part changed.

## 2. Re-score the predictions

```bash
python protocol/score.py        RPM-10K/labels/test.json predictions/official_seed0.json     # one seed: Acc_eps, Wilson interval, Acc_theta, Ref, Rel
python protocol/rescore_seeds.py RPM-10K/labels/test.json predictions/official_seed*.json     # the five seeds: means of the four metrics, H, Wilson, frame-block bootstrap
python protocol/rescore_seeds.py RPM-10K/labels/test.json predictions/v3_seed*.json \
       --v3 protocol/splits_v3.json --train-json RPM-10K/labels/train.json                    # split v3, 3 seeds, test part
python protocol/score_v3.py protocol/splits_v3.json RPM-10K/labels/train.json RPM-10K/labels/test.json predictions/v3_seed0.json --part test   # one v3 seed
```

Expected (the same numbers as `results/four_metric_table.md`, `TEST-LOG.md` and `docs/rescore-check.txt`):

| command | n | per-seed Acc_eps | mean of the four metrics / H | Wilson 95 % | block bootstrap 95 % |
|---|---|---|---|---|---|
| `rescore_seeds.py` official, 5 seeds | 2,000 | 0.6845, 0.6910, 0.6880, 0.6810, 0.6835 | Acc_eps 0.6856, Acc_theta 0.7488, Ref 0.0759, Rel 0.3923 | [0.6649, 0.7056] | 71 blocks of 160 ids: [0.6490, 0.7190] |
| `rescore_seeds.py --v3`, 3 seeds | 1,853 scored, 43 without a determinable range excluded | 0.5461, 0.5407, 0.5451 | Acc_eps 0.5440 | [0.5212, 0.5665] | 14 blocks: [0.4672, 0.6157] |

The scorer is the authors' (`src/dialbench_metrics.py`, a port of `benchmark/benchmark.py`; `tests/official_reference.py` is their logic verbatim and `tests/test_dialbench_metrics.py` compares the two). It reads `range` from the label file only to set the tolerance (1 % of the range); a model never sees it. A prediction text is parsed like the authors' script does: the last number in it, 0.0 if there is none; the prediction files hold plain decimal strings (never scientific notation) for that reason.

## 3. What the scorer does with the labels (audit)

`docs/DIALBENCH-AUDIT.md` lists what we found in the release: the split naming, the image that is in both label files, the unreferenced file, rows whose reading holds two numbers or none (the scorer takes the last number), and the near-duplicate frames across train and test (thumbnail similarity) that motivated split v3. The raw kernel outputs behind its tables are `docs/resplit-output.txt` and `docs/split3-output.txt`.

## 4. Re-train (optional; Kaggle, private kernels)

The pipeline ran as private Kaggle kernels named `bench-gauges-*` under the author's account. To repeat it, create your own kernels from the directories in `kernels/` and change the ids in each `kernel-metadata.json` (`id` and `kernel_sources`) to your account. Open points you have to handle yourself: the data link (the placeholder `URL` in `kernels/rpm-prep/rpm_prep.py`, taken from the DialBench README; the kernel logs whether the byte count matches and prints the sha256 that step 1 lists) and the push/wait helpers in `tools/`, which refer to program-level files that are not included (see README.md).

| order | kernel | does | notes |
|---|---|---|---|
| 1 | `model-port` | converts torchvision ImageNet ResNet-18 / ResNet-34 weights to `.npz` | internet on; the trainer checks `PREFLIGHT ... missing []` |
| 2 | `rpm-prep` | downloads the zip inside the kernel, hashes it, builds the 224 px and 320 px uint8 caches and label copies | internet on; the data never leaves the kernel |
| 3 | `rpm-split3` | builds `protocol/splits_v3.json` | expect sha256 `c16e51f6b9bd25cfa9925f19412da2510f9eaad9944dcddfc4d3a261307fc2e3` |
| 4 | `copy-hits` | the no-learning nearest-neighbour label-copy baseline with per-row hits | CPU |
| 5 | `gpu-sweep1` | Stage A of `FROZEN-4.md`: seven candidates, validation only, no test image | 2 x Tesla T4, about 1 h wall |
| 6 | `gpu-final` | Stage B: eight runs of the frozen primary (official seeds 0 to 4, v3 seeds 0 to 2), one test look each | 2 x Tesla T4, 6,657 s wall for the run reported here |
| 7 | `gpu-seedmean` | the pre-registered claim statistic from the saved per-row hits | CPU, numpy only |
| 8 | `gpu-int8` | the int8 recipe sweep of `FROZEN-3.md` on the seed-0 weights, one test look for the selected recipe | CPU |

Earlier runs and audits: `rpm-eda2`, `rpm-resplit` (the audit and the group-disjoint split v2), `cpu-mnv3s-b` (run 1), `cpu-v3` (run 2 on split v3), `int8-sweep` (`FROZEN-3.md` on the run-1 and run-2 weights), `paired`, `jax-selftest`, `model-port`. `kernels/gpu-final4` and `gpu-final6` are the smaller quota tiers of the same kernel (not run); `gpu-smoke` is a CPU twin of the GPU wrapper (orchestration test, no test image); `cpu-mnv3s` is a 6-epoch validation-only pilot; `tpu-probe`, `tpu-probe-nb` and `tpu-sweep1` are stale TPU builds (the two probes never left the queue, `FROZEN-4.md` section 9) and `jax-cputest` an early JAX CPU build. None of these produced a reported number.

GPU training is not bit-reproducible: seeds fix initialisation, data order and augmentation draws, not cuDNN. Expect Acc_eps within a few tenths of a point per seed (the five seeds span 68.10 to 69.10 %), not identical values.

## 5. About the file hashes

`FROZEN-*.md` and `TEST-LOG.md` quote sha256 values of kernel scripts and tools as they were pushed or used. Nineteen files of this kit carry marked redactions and therefore have another hash than the one quoted; `REDACTIONS.md` lists each of them with the hash in the private repository, the hash here and where the private hash is quoted. For the 13 generated bundles `kernels/*/script.py` the only change is in 72 comment lines that named the build machine's absolute path and now read `<repo>/...`; everything else in them is unchanged. `tests/test_kernel_bundles.py` decodes each bundle (nothing is executed) and checks that the declared hash of every embedded module matches its text, that the Stage A / B kernels embed exactly the `src/` files, and that each bundle ends with its `main.py`. `MANIFEST.txt` lists the sha256 of every file as it is in this repository.
