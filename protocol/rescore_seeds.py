#!/usr/bin/env python3
"""Re-score several per-seed prediction files with the ported official DialBench scorer and print the seed-level statistic of FROZEN-4 section 0.
usage (official split):  rescore_seeds.py TEST_JSON PRED_JSON [PRED_JSON ...] [--target 0.624]
usage (split v3 test):   rescore_seeds.py TEST_JSON PRED_JSON [PRED_JSON ...] --v3 SPLITS_V3_JSON --train-json TRAIN_JSON
  TEST_JSON / TRAIN_JSON  the official labels/test.json and labels/train.json; you download them from the DialBench authors, they are not part of this repository
  PRED_JSON               {image_name: predicted_reading} per seed, for example predictions/official_seed0.json ... official_seed4.json (the last number in the text is taken and a
                          missing prediction counts as 0.0, exactly as in the official script)
Per seed: Acc_eps, Acc_theta, Ref, Rel (src/dialbench_metrics.py, equivalence-tested against the authors' code in tests/).
Over the seeds: the mean of every metric (the four-metric table of the README), H = mean Acc_eps, its Wilson 95 % interval at n scored rows, and the frame-block bootstrap of
FROZEN-4 section 0: the seed-averaged per-image hit is resampled over blocks, 5,000 draws, numpy RandomState(0), 95 % percentile interval. Official split: blocks of 160
consecutive image ids ((id - 1) // 160, id = the N of N_img.jpg; 71 blocks). Split v3: the manifest's blocks (14 in the test part); rows are pooled by image name from both label
files and rows with no determinable range are excluded and counted, as in protocol/score_v3.py. The Wilson and bootstrap code is the one of kernels/gpu-seedmean/script.py
(tests/test_rescore_seeds.py compares the two). The scorer reads `range` from the label file only to set the tolerance; a model must never see it."""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dialbench_metrics import dialbench_metrics, extract_last_number
from rpm_common import pool_rows, row_span

BLOCK = 160
DRAWS = 5000
KEYS = ("acc_eps", "acc_theta", "ref", "rel")


def wilson(p, n, z=1.959964):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def boot(per_image, blk, draws=DRAWS):
    ub = np.unique(blk)
    s = np.array([per_image[blk == u].sum() for u in ub], float)
    k = np.array([(blk == u).sum() for u in ub], float)
    gi = np.random.RandomState(0).randint(0, len(ub), size=(draws, len(ub)))
    m = s[gi].sum(1) / k[gi].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(ub)


def score_seeds(names, gt, rg, blk, preds):
    """names: image names of the scored rows; gt, rg: ground-truth readings and ranges (float arrays); blk: block id per row; preds: list of {image: prediction}."""
    per_seed, hits, missing = [], [], []
    for pr_map in preds:
        pr = np.array([extract_last_number(str(pr_map.get(n, ""))) for n in names])
        m = dialbench_metrics(pr, gt, rg)
        per_seed.append({k: m[k] for k in KEYS})
        hits.append(np.abs(pr - gt) <= 0.01 * rg)
        missing.append(int(sum(1 for n in names if n not in pr_map)))
    hits = np.array(hits, float)
    n = hits.shape[1]
    H = float(hits.mean(1).mean())
    wl, wh = wilson(H, n)
    bl, bh, nb = boot(hits.mean(0), blk)
    return {"n": n, "seeds": len(preds), "missing_predictions": missing, "per_seed": per_seed,
            "mean": {k: float(np.mean([s[k] for s in per_seed])) for k in KEYS},
            "H": H, "wilson95": [wl, wh], "block_bootstrap95": [bl, bh], "n_blocks": nb, "draws": DRAWS}


def rescore(test_rows, preds):
    """Official split. test_rows: the rows of labels/test.json in file order."""
    names = [it["image"] for it in test_rows]
    gt = np.array([extract_last_number(str(it["reading"])) for it in test_rows])
    rg = np.array([float(it["range"]) for it in test_rows])
    ids = np.array([int(n.split("_")[0]) for n in names])
    out = score_seeds(names, gt, rg, (ids - 1) // BLOCK, preds)
    out["split"] = "official"
    out["block"] = BLOCK
    return out


def rescore_v3(man, train_rows, test_rows, preds, part="test"):
    """Split v3. man: protocol/splits_v3.json (names per part, block id per name); rows of both label files pooled by image name."""
    by = pool_rows(train_rows, test_rows)
    names = [n for n in man[part] if n in by]
    gt = np.array([extract_last_number(str(by[n]["reading"])) for n in names])
    rg = np.array([row_span(by[n])[0] for n in names], dtype=np.float64)
    known = ~np.isnan(rg)
    blk = np.array([man["block_of"][n] for n in names])
    kn = [n for n, k in zip(names, known) if k]
    out = score_seeds(kn, gt[known], rg[known], blk[known], preds)
    out.update({"split": "v3 " + part, "n_names": len(man[part]), "n_rows": len(names), "n_excluded_unknown_range": int((~known).sum())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_json"); ap.add_argument("pred_json", nargs="+")
    ap.add_argument("--target", type=float, default=0.624, help="published best Acc_eps to compare the lower bounds with (official split only)")
    ap.add_argument("--v3", default=None, help="score the test part of split v3 (needs --train-json)")
    ap.add_argument("--train-json", default=None)
    a = ap.parse_args()
    preds = [json.load(open(p)) for p in a.pred_json]
    test_rows = json.load(open(a.test_json))
    if a.v3:
        assert a.train_json, "--v3 needs --train-json"
        out = rescore_v3(json.load(open(a.v3)), json.load(open(a.train_json)), test_rows, preds)
    else:
        out = rescore(test_rows, preds)
        out["target"] = a.target
        out["wilson_lower_bound_above_target"] = out["wilson95"][0] > a.target
        out["block_lower_bound_above_target"] = out["block_bootstrap95"][0] > a.target
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
