# Per-seed predictions (model outputs only)

One JSON object per trained model: `{"<image file name>": "<predicted reading>"}`, one entry per line, in the row order of the split.

| file | model | rows | order |
|---|---|---|---|
| `official_seed0.json` ... `official_seed4.json` | the five Stage B ResNet-18 models (FROZEN-4 primary `a2-r18-224`, seeds 0 to 4), trained on the official `labels/train.json` minus a 1,002-image validation part | 2,000 | the order of the official `labels/test.json` |
| `v3_seed0.json` ... `v3_seed2.json` | the three Stage B models trained on the train part of split v3 (`protocol/splits_v3.json`) | 1,896 | the `test` list of `protocol/splits_v3.json` |

The value is the prediction in the dial's own units (`y_hat * span_hat`, the model's output; FROZEN-4 section 1), written as the shortest decimal string that parses back to the same float and never in scientific notation: the official scorer takes the last number found in the text of a prediction, and `str(4.5e-05)` would be read as -5 (`tests/test_predictions_reduced.py`).

**What was dropped.** The full prediction files that the Kaggle kernel wrote (private, never published) hold seven arrays. Only `pred` is kept. `last` (the reading of the test label) and `rng` (the dial range of the test label) are ground truth; `hits` and `known` are derived from them; `span_hat` (the model's own range class) is not needed to score; `img_idx` is an internal cache index and is replaced by the image file name. No label, no range, no hit mask and no image is in these files. `tools/reduce_predictions.py` documents the reduction and checks it (it needs the private files).

**What they are.** Model outputs, not labels. They do carry information about the labels: on the official test they fall within the scorer's tolerance (1 % of the dial range) of the label on 68.56 % of the rows on average. They are published so that every reported score can be recomputed, not as data.

**Re-scoring** (you need the official `labels/test.json` and `labels/train.json` from the DialBench authors; this repository does not contain them):

```
python protocol/score.py LABELS/test.json predictions/official_seed0.json
python protocol/rescore_seeds.py LABELS/test.json predictions/official_seed*.json
python protocol/rescore_seeds.py LABELS/test.json predictions/v3_seed*.json --v3 protocol/splits_v3.json --train-json LABELS/train.json
```

`docs/rescore-check.txt` records that these commands, run against stand-in label files rebuilt from the private full prediction files, give the numbers recorded on 2026-10-17 (per-seed and seed-mean Acc_eps, Wilson and block-bootstrap intervals, and all four metrics).
