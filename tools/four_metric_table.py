#!/usr/bin/env python3
"""The four-metric scope of the DialBench claim, recomputed from the five Stage B official-split prediction files with the ported official scorer (src/dialbench_metrics.py).

usage: four_metric_table.py [--npz-dir DIR] [--out DIR]
Reads the PRIVATE full files `f-r18-224-off-s<k>_test_pred.npz` (pred, last, rng, ...; they carry the test labels' readings and ranges and are never published) and writes
results/four_metric_table.json and results/four_metric_table.md: per-seed Acc_eps, Acc_theta, Ref, Rel, their seed means, and the comparison with the published Table 4 of
arXiv 2511.21982 (Wang et al., DialBench). Anyone with the official labels/test.json gets the same numbers from the public files with
`python protocol/rescore_seeds.py labels/test.json predictions/official_seed*.json`.

The published values below were read from the HTML version of the paper on 2026-10-19 (Table 4, 18 models; Acc_eps and Acc_theta in %, Ref and Rel lower is better); the two counts
(published models with a lower Ref / Rel than ours) were counted by hand on that table. Only the rows needed for the comparison are stored here."""
import argparse, glob, json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dialbench_metrics import dialbench_metrics   # noqa: E402

PUBLISHED = {
    "source": "arXiv 2511.21982 Table 4 (HTML version), read 2026-10-19; 18 models",
    "MRLM": {"acc_eps": 62.4, "acc_theta": 70.9, "ref": 0.063, "rel": 0.535},
    "best": {"acc_eps": {"model": "MRLM", "value": 62.4}, "acc_theta": {"model": "MRLM", "value": 70.9},
             "ref": {"model": "LLaVA-v1.6-Vicuna-7B", "value": 0.038}, "rel": {"model": "LLaVA-v1.6-Vicuna-7B", "value": 0.215}},
    "n_published_models": 18,
    "published_models_better_than_ours": {"acc_eps": 0, "acc_theta": 0, "ref": 10, "rel": 5},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=os.path.join(ROOT, "logs", "kaggle-output", "gpu-final"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.npz_dir, "f-r18-224-off-s*_test_pred.npz")), key=lambda p: int(re.search(r"-s(\d+)_test_pred", p).group(1)))
    assert len(files) == 5, files
    per_seed = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        assert z["known"].all() and len(z["pred"]) == 2000
        m = dialbench_metrics(z["pred"], z["last"], z["rng"])
        per_seed.append({"seed": int(re.search(r"-s(\d+)_test_pred", f).group(1)), **{k: m[k] for k in ("acc_eps", "acc_theta", "ref", "rel")}, "n": m["n"], "n_nonzero_gt": m["n_nonzero_gt"]})
    mean = {k: float(np.mean([s[k] for s in per_seed])) for k in ("acc_eps", "acc_theta", "ref", "rel")}
    # consistency with the published table and the counts above
    assert mean["acc_eps"] * 100 > PUBLISHED["best"]["acc_eps"]["value"] and mean["acc_theta"] * 100 > PUBLISHED["best"]["acc_theta"]["value"]
    assert mean["ref"] > PUBLISHED["best"]["ref"]["value"] and mean["rel"] > PUBLISHED["best"]["rel"]["value"]
    out = {"model": "ResNet-18 224 px, 13,079,234 parameters (FROZEN-4 primary a2-r18-224)", "split": "DialBench RPM-10K official split (labels/train.json -> labels/test.json, n = 2,000)",
           "scorer": "src/dialbench_metrics.py (port of benchmark/benchmark.py of Event-AHU/DialBench)", "per_seed": per_seed, "seed_mean": mean, "published": PUBLISHED}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "four_metric_table.json"), "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    P = PUBLISHED
    rows = [
        ("Acc_eps (%, higher is better)", "%.2f" % (mean["acc_eps"] * 100), "%.1f" % P["MRLM"]["acc_eps"], "%.1f (%s)" % (P["best"]["acc_eps"]["value"], P["best"]["acc_eps"]["model"]), "first: above all 18 published models"),
        ("Acc_theta (%, higher is better)", "%.2f" % (mean["acc_theta"] * 100), "%.1f" % P["MRLM"]["acc_theta"], "%.1f (%s)" % (P["best"]["acc_theta"]["value"], P["best"]["acc_theta"]["model"]), "first: above all 18 published models"),
        ("Ref (lower is better)", "%.4f" % mean["ref"], "%.3f" % P["MRLM"]["ref"], "%.3f (%s)" % (P["best"]["ref"]["value"], P["best"]["ref"]["model"]),
         "behind: %d of the 18 published models are lower" % P["published_models_better_than_ours"]["ref"]),
        ("Rel (lower is better)", "%.3f" % mean["rel"], "%.3f" % P["MRLM"]["rel"], "%.3f (%s)" % (P["best"]["rel"]["value"], P["best"]["rel"]["model"]),
         "behind: %d of the 18 published models are lower" % P["published_models_better_than_ours"]["rel"]),
    ]
    md = ["# Four-metric table (DialBench RPM-10K, official split, 5-seed mean of the pre-registered primary; self-evaluated)", "",
          "| metric (paper Table 4) | ours, 5-seed mean | MRLM | best published | rank |", "|---|---|---|---|---|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "Per seed (official scorer, n = 2,000):", "", "| seed | Acc_eps | Acc_theta | Ref | Rel |", "|---|---|---|---|---|"]
    md += ["| %d | %.2f %% | %.2f %% | %.4f | %.4f |" % (s["seed"], s["acc_eps"] * 100, s["acc_theta"] * 100, s["ref"], s["rel"]) for s in per_seed]
    md += ["", "Published values: " + P["source"] + ". The counts of published models with a better Ref / Rel were made by hand on that table. Ours: " + out["scorer"] + "."]
    with open(os.path.join(a.out, "four_metric_table.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
