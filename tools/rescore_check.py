#!/usr/bin/env python3
"""End-to-end check of the PUBLIC prediction files against the private full ones: rebuild a stand-in for labels/test.json (and for the pooled v3 label rows) from the arrays
`last` and `rng` of the private full prediction files, score predictions/*.json with protocol/rescore_seeds.py, protocol/score.py and protocol/score_v3.py, and compare
with the numbers recorded on 2026-10-17 (docs/stageB-output.txt, TEST-LOG.md). Prints the report that is stored as docs/rescore-check.txt.
The stand-in label files exist only in a temporary directory and are deleted; nothing of them is written to this repository. Needs the private files, so it cannot run on the
public kit alone.
usage: rescore_check.py [--npz-dir DIR] [--copy-hits FILE] [--splits FILE] [--pred-dir DIR]"""
import argparse, glob, json, os, re, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "protocol"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import rescore_seeds as R   # noqa: E402

# recorded values: kernel bench-gauges-gpu-seedmean (docs/stageB-output.txt) and TEST-LOG.md rows
RECORDED = {
    "official": {"per_seed_acc_eps": [0.6845, 0.6910, 0.6880, 0.6810, 0.6835], "H": 0.6856, "wilson95": [0.6649, 0.7056], "block95": [0.6490, 0.7190], "n_blocks": 71, "n": 2000},
    "v3": {"per_seed_acc_eps": [0.5461, 0.5407, 0.5451], "H": 0.5440, "wilson95": [0.5212, 0.5665], "block95": [0.4672, 0.6157], "n_blocks": 14, "n": 1853},
}
lines, fails = [], []


def say(s=""):
    print(s)
    lines.append(s)


def check(label, got, want):
    ok = got == want
    say("  %s %s: got %s, recorded %s" % ("ok  " if ok else "FAIL", label, got, want))
    if not ok:
        fails.append(label)


def f4(x):
    return round(x + 0.0, 4)


def load(path):
    with open(path) as fh:
        return json.load(fh)


def dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=os.path.join(ROOT, "logs", "kaggle-output", "gpu-final"))
    ap.add_argument("--copy-hits", default=os.path.join(ROOT, "logs", "kaggle-output", "copy-hits", "copy_hits_official.npz"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "protocol", "splits_v3.json"))
    ap.add_argument("--pred-dir", default=os.path.join(ROOT, "predictions"))
    a = ap.parse_args()
    co = np.load(a.copy_hits, allow_pickle=False)
    names_off = [str(x) for x in co["names"]]
    z0 = np.load(os.path.join(a.npz_dir, "f-r18-224-off-s0_test_pred.npz"), allow_pickle=False)
    off_rows = [{"image": n, "reading": repr(float(g)), "range": float(r)} for n, g, r in zip(names_off, z0["last"], z0["rng"])]
    man = load(a.splits)
    v0 = np.load(os.path.join(a.npz_dir, "f-r18-224-v3-s0_test_pred.npz"), allow_pickle=False)
    v3_rows = []
    for n, g, r, k in zip(man["test"], v0["last"], v0["rng"], v0["known"]):
        v3_rows.append({"image": n, "reading": repr(float(g)), "range": float(r)} if k else {"image": n, "reading": repr(float(g)), "range": None, "detail": "no pointer"})
    off_preds = [load(p) for p in sorted(glob.glob(os.path.join(a.pred_dir, "official_seed*.json")))]
    v3_preds = [load(p) for p in sorted(glob.glob(os.path.join(a.pred_dir, "v3_seed*.json")))]
    say("Re-scoring of the public prediction files (predictions/*.json) against stand-in label files rebuilt from the private full prediction files; compared with the numbers")
    say("recorded on 2026-10-17 (docs/stageB-output.txt, TEST-LOG.md). The stand-in labels are temporary and are not part of this repository.")
    say()
    say("official split: protocol/rescore_seeds.py, %d seeds, n = %d" % (len(off_preds), len(off_rows)))
    o = R.rescore(off_rows, off_preds)
    r = RECORDED["official"]
    check("per-seed Acc_eps", [f4(s["acc_eps"]) for s in o["per_seed"]], r["per_seed_acc_eps"])
    check("H (seed mean)", f4(o["H"]), r["H"])
    check("Wilson 95 %", [f4(x) for x in o["wilson95"]], r["wilson95"])
    check("block bootstrap 95 %% (%d blocks)" % o["n_blocks"], [f4(x) for x in o["block_bootstrap95"]], r["block95"])
    check("number of blocks", o["n_blocks"], r["n_blocks"])
    say("  four-metric means: Acc_eps %.4f  Acc_theta %.4f  Ref %.4f  Rel %.4f" % tuple(o["mean"][k] for k in R.KEYS))
    t = load(os.path.join(ROOT, "results", "four_metric_table.json"))
    check("four-metric means equal results/four_metric_table.json", [o["mean"][k] for k in R.KEYS], [t["seed_mean"][k] for k in R.KEYS])
    check("per-seed four metrics equal results/four_metric_table.json", [[s[k] for k in R.KEYS] for s in o["per_seed"]], [[s[k] for k in R.KEYS] for s in t["per_seed"]])
    say()
    say("official split: protocol/score.py, seed 0")
    with tempfile.TemporaryDirectory() as d:
        dump(off_rows, os.path.join(d, "test.json"))
        sc = json.loads(subprocess.run([sys.executable, os.path.join(ROOT, "protocol", "score.py"), os.path.join(d, "test.json"), os.path.join(a.pred_dir, "official_seed0.json")],
                                       capture_output=True, text=True, check=True).stdout)
    check("Acc_eps", f4(sc["acc_eps"]), 0.6845)
    check("Wilson 95 %", [f4(x) for x in sc["acc_eps_wilson95"]], [0.6638, 0.7045])
    check("missing predictions", sc["missing_predictions"], 0)
    say()
    say("split v3 (test part): protocol/rescore_seeds.py --v3, %d seeds" % len(v3_preds))
    v = R.rescore_v3(man, [], v3_rows, v3_preds)
    r = RECORDED["v3"]
    check("scored rows / excluded rows", [v["n"], v["n_excluded_unknown_range"]], [1853, 43])
    check("per-seed Acc_eps", [f4(s["acc_eps"]) for s in v["per_seed"]], r["per_seed_acc_eps"])
    check("H (seed mean)", f4(v["H"]), r["H"])
    check("Wilson 95 %", [f4(x) for x in v["wilson95"]], r["wilson95"])
    check("block bootstrap 95 %% (%d blocks)" % v["n_blocks"], [f4(x) for x in v["block_bootstrap95"]], r["block95"])
    say()
    say("split v3 (test part): protocol/score_v3.py, per seed")
    with tempfile.TemporaryDirectory() as d:
        dump(v3_rows, os.path.join(d, "test.json"))
        dump([], os.path.join(d, "train.json"))
        for s in range(len(v3_preds)):
            sc = json.loads(subprocess.run([sys.executable, os.path.join(ROOT, "protocol", "score_v3.py"), a.splits, os.path.join(d, "train.json"), os.path.join(d, "test.json"),
                                            os.path.join(a.pred_dir, "v3_seed%d.json" % s)], capture_output=True, text=True, check=True).stdout)
            check("seed %d Acc_eps (n_scored %d)" % (s, sc["n_scored"]), f4(sc["acc_eps"]), RECORDED["v3"]["per_seed_acc_eps"][s])
            check("seed %d per-seed four metrics equal rescore_seeds" % s, [sc[k] for k in R.KEYS], [v["per_seed"][s][k] for k in R.KEYS])
    say()
    say("RESULT " + ("all checks passed" if not fails else "FAILED: " + "; ".join(fails)))
    with open(os.path.join(ROOT, "docs", "rescore-check.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
