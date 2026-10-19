#!/usr/bin/env python3
"""Official DialBench scorer (ported verbatim from Event-AHU/DialBench benchmark/benchmark.py) plus uncertainty.
usage: score.py TEST_JSON PREDICTIONS_JSON [--groups GROUPS_JSON]
  TEST_JSON         the official labels/test.json (list of {image, reading, range, ...})
  PREDICTIONS_JSON  {image_name: predicted_reading_as_number_or_text}
Prints Acc_eps, Acc_theta, Ref, Rel, the Wilson 95% CI, and (with --groups, {image: group_id}) a grouped-bootstrap 95% CI.
The scorer reads `range` from the label file only to set epsilon; a model must never see it."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dialbench_metrics import dialbench_metrics, extract_last_number, wilson_interval, bootstrap_acc_ci

ap = argparse.ArgumentParser()
ap.add_argument("test_json"); ap.add_argument("pred_json"); ap.add_argument("--groups", default=None)
a = ap.parse_args()
test = json.load(open(a.test_json)); pred = json.load(open(a.pred_json))
gt = np.array([extract_last_number(str(it["reading"])) for it in test])
rg = np.array([float(it["range"]) for it in test])
pr = np.array([extract_last_number(str(pred.get(it["image"], ""))) for it in test])   # missing prediction -> 0.0, as in the official script
m = dialbench_metrics(pr, gt, rg)
hits = np.abs(pr - gt) <= 0.01 * rg
lo, hi = wilson_interval(int(hits.sum()), len(hits))
out = {"n": m["n"], "acc_eps": m["acc_eps"], "acc_eps_wilson95": [lo, hi], "acc_theta": m["acc_theta"], "ref": m["ref"], "rel": m["rel"],
       "missing_predictions": int(sum(1 for it in test if it["image"] not in pred))}
if a.groups:
    g = json.load(open(a.groups)); gid = np.array([g[it["image"]] for it in test])
    out["acc_eps_grouped_bootstrap95"] = list(bootstrap_acc_ci(hits, 2000, 0, gid))
print(json.dumps(out, indent=1))
