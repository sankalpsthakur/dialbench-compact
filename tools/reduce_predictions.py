#!/usr/bin/env python3
"""Reduce the Stage B per-seed prediction files to the public form: image file name -> predicted reading, nothing else.

Input (private, gitignored, fetched from the Kaggle kernel `bench-gauges-gpu-final`): `f-r18-224-off-s<k>_test_pred.npz` (official split, 2,000 rows) and
`f-r18-224-v3-s<k>_test_pred.npz` (split v3, 1,896 rows) with the arrays pred, hits, span_hat, known, rng, last, img_idx.
  kept     pred      the model's predicted reading in the dial's units (y_hat * span_hat, FROZEN-4 section 1); a model output, not a label
  dropped  last      the reading of the DialBench test label (ground truth)
  dropped  rng       the dial range of the DialBench test label (ground truth; the scorer uses it only for the tolerance)
  dropped  hits      derived from the two above: |pred - last| <= 0.01 * rng
  dropped  known     derived from the label: whether the row has a determinable range
  dropped  span_hat  the model's own range class; not needed to score
  dropped  img_idx   an internal index into the private image cache; replaced by the image file name
Row order: official rows follow labels/test.json (names read from the label-free `names` array of copy_hits_official.npz, whose img_idx, last and rng equal those of the
prediction files); v3 rows follow the `test` list of protocol/splits_v3.json (checked against the official names for the 365 v3 test images that also belong to the official test).
Numbers are written as shortest round-trip DECIMAL strings (never scientific notation): the official scorer takes the last number found in the text of a prediction, and
`str(4.5e-05)` would be read as -5. Every written value is parsed back and the hits recomputed from the strings must equal the saved hits.

usage: reduce_predictions.py [--npz-dir DIR] [--copy-hits FILE] [--splits FILE] [--out DIR]
This tool needs the private files above and cannot run on the public kit alone; it is kept there to document how predictions/*.json were made."""
import argparse, glob, hashlib, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from dialbench_metrics import dialbench_metrics, extract_last_number   # noqa: E402

DECIMAL = re.compile(r"^-?\d+(\.\d+)?$")


def fmt(x):
    """Shortest decimal string that parses back to the same float64, in positional notation."""
    x = float(x)
    assert np.isfinite(x), x
    s = np.format_float_positional(x, unique=True, trim="-")
    assert DECIMAL.match(s) and float(s) == x, (x, s)
    return s


def check_and_write(path, names, z, label):
    pred, last, rng, hits_saved = z["pred"], z["last"], z["rng"], z["hits"]
    assert len(names) == len(pred) == len(last) == len(rng) == len(hits_saved), (label, len(names), len(pred))
    assert len(set(names)) == len(names), label + ": duplicate image names"
    known = z["known"].astype(bool)
    # alignment guard (uses the private label arrays, nothing of them is stored): hits recomputed from pred / last / rng equal the saved hits
    h0 = np.abs(pred[known] - last[known]) <= 0.01 * rng[known]
    assert np.array_equal(h0, hits_saved[known]), label + ": saved hits differ from pred/last/rng"
    strings = [fmt(v) for v in pred]
    back = np.array([extract_last_number(s) for s in strings])   # what the official scorer reads from the text
    assert np.array_equal(back, pred), label + ": a written string does not read back to the prediction"
    h1 = np.abs(back[known] - last[known]) <= 0.01 * rng[known]
    assert np.array_equal(h1, hits_saved[known]), label + ": hits from the written strings differ from the saved hits"
    m = dialbench_metrics(back[known], last[known], rng[known])
    with open(path, "w") as fh:
        fh.write("{\n" + ",\n".join(json.dumps(n) + ": " + json.dumps(s) for n, s in zip(names, strings)) + "\n}\n")
    return m, hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=os.path.join(ROOT, "logs", "kaggle-output", "gpu-final"))
    ap.add_argument("--copy-hits", default=os.path.join(ROOT, "logs", "kaggle-output", "copy-hits", "copy_hits_official.npz"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "protocol", "splits_v3.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "predictions"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    co = np.load(a.copy_hits, allow_pickle=False)
    off_names = [str(x) for x in co["names"]]
    off_idx = co["img_idx"]
    idx2name = dict(zip((int(i) for i in off_idx), off_names))
    man = json.load(open(a.splits))
    v3_names = list(man["test"])
    summary = []
    for split, names in (("off", off_names), ("v3", v3_names)):
        files = sorted(glob.glob(os.path.join(a.npz_dir, "f-r18-224-%s-s*_test_pred.npz" % split)), key=lambda p: int(re.search(r"-s(\d+)_test_pred", p).group(1)))
        assert files, "no prediction files for " + split
        first_idx = None
        for f in files:
            seed = int(re.search(r"-s(\d+)_test_pred", f).group(1))
            z = np.load(f, allow_pickle=False)
            if first_idx is None:
                first_idx = z["img_idx"]
                if split == "off":
                    assert np.array_equal(first_idx, off_idx), "official img_idx differ from the names file"
                    assert np.array_equal(z["last"], co["last"]) and np.array_equal(z["rng"], co["rng"]), "official last/rng differ from the names file"
                else:
                    ov = [(i, idx2name[int(ix)]) for i, ix in enumerate(first_idx) if int(ix) in idx2name]
                    assert ov and all(v3_names[i] == nm for i, nm in ov), "v3 manifest order does not match the official names"
                    print("v3: %d of %d test rows also belong to the official test; their names agree with the manifest order" % (len(ov), len(first_idx)))
            else:
                assert np.array_equal(first_idx, z["img_idx"]), "row order differs between seeds"
            tag = "official" if split == "off" else "v3"
            path = os.path.join(a.out, "%s_seed%d.json" % (tag, seed))
            m, sha = check_and_write(path, names, z, "%s seed %d" % (tag, seed))
            summary.append((os.path.relpath(path, ROOT) if path.startswith(ROOT) else path, len(names), m["acc_eps"], m["acc_theta"], m["ref"], m["rel"], sha))
    for s in summary:
        print("%s  rows %d  Acc_eps %.4f  Acc_theta %.4f  Ref %.4f  Rel %.4f  sha256 %s" % s)


if __name__ == "__main__":
    main()
