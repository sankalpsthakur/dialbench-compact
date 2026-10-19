#!/usr/bin/env python3
"""Score predictions on one part (test or val) of split v3 with the official DialBench metric (ported verbatim in src/dialbench_metrics.py).
usage: score_v3.py SPLITS_V3_JSON TRAIN_JSON TEST_JSON PREDICTIONS_JSON [--part test|val]
  SPLITS_V3_JSON    protocol/splits_v3.json (image names per part, 160-id block per image)
  TRAIN_JSON / TEST_JSON   the official labels/train.json and labels/test.json (rows of both files are pooled by image name; the one image in both files keeps its test.json row)
  PREDICTIONS_JSON  {image_name: predicted_reading_as_number_or_text}; the last number is taken, a missing prediction counts as 0.0, as in the official script
Scorer range of a row = its numeric `range` (test.json rows), else the "0-3"-style string (train.json rows), else the range stated in the chain-of-thought text when its low end is 0;
rows with no determinable range are excluded and counted (43 of the 1,896 v3 test rows). The range is used only for the tolerance, never as a model input.
Prints Acc_eps (the headline), Acc_theta, Ref, Rel, the Wilson 95 % interval and a block bootstrap 95 % interval (the sampling unit of v3 is the 160-id block)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dialbench_metrics import dialbench_metrics, extract_last_number, wilson_interval
from rpm_common import pool_rows, row_span

ap = argparse.ArgumentParser()
ap.add_argument("splits"); ap.add_argument("train_json"); ap.add_argument("test_json"); ap.add_argument("pred_json")
ap.add_argument("--part", default="test", choices=["test", "val"])
a = ap.parse_args()
man = json.load(open(a.splits))
by = pool_rows(json.load(open(a.train_json)), json.load(open(a.test_json)))
pred = json.load(open(a.pred_json))
names = [n for n in man[a.part] if n in by]
gt = np.array([extract_last_number(str(by[n]["reading"])) for n in names])
rng = np.array([row_span(by[n])[0] for n in names], dtype=np.float64)
known = ~np.isnan(rng)
pr = np.array([extract_last_number(str(pred.get(n, ""))) for n in names])
m = dialbench_metrics(pr[known], gt[known], rng[known])
hits = np.abs(pr[known] - gt[known]) <= 0.01 * rng[known]
lo, hi = wilson_interval(int(hits.sum()), len(hits))
blk = np.array([man["block_of"][n] for n in names])[known]
ub = np.unique(blk)
sums = np.array([hits[blk == u].sum() for u in ub], float); cnt = np.array([(blk == u).sum() for u in ub], float)
gi = np.random.RandomState(0).randint(0, len(ub), size=(5000, len(ub)))
means = sums[gi].sum(1) / cnt[gi].sum(1)
out = {"part": a.part, "n_names": len(man[a.part]), "n_rows": len(names), "n_scored": m["n"], "n_excluded_unknown_range": int((~known).sum()),
       "acc_eps": m["acc_eps"], "acc_eps_wilson95": [lo, hi], "acc_eps_block_bootstrap95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "n_blocks": int(len(ub)),
       "acc_theta": m["acc_theta"], "ref": m["ref"], "rel": m["rel"],
       "missing_predictions": int(sum(1 for n in names if n not in pred))}
print(json.dumps(out, indent=1))
