"""prepare() on the synthetic fixture (imports torch, builds no model). usage: check_split_prepare.py FIXTURE_DIR"""
import sys, os, types, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
m = types.ModuleType("jax_resnet"); sys.modules["jax_resnet"] = m
import numpy as np
from rpm_train_torch import prepare, get_cfg
fx = sys.argv[1]
base = dict(input_dir=fx, max_train=None, n_val=30)
# official mode: the image in both files is dropped from train; test rows all known
D0 = prepare(get_cfg(base))
assert D0.mode == "official" and len(D0.last) == 100 and D0.test.known.all() and len(D0.test.last) == 40
assert len(D0.tr_idx) + len(D0.va_idx) == 100 and len(np.intersect1d(D0.tr_idx, D0.va_idx)) == 0
# split_file mode
sp = fx + "/splits_v3.json"
D = prepare(get_cfg(dict(base, split_file=sp, split_sha256=hashlib.sha256(open(sp, "rb").read()).hexdigest())))
assert D.mode == "split_file"
assert D.info["n_rows"] == {"train": 84, "val": 15, "test": 40} and D.info["names_without_label_row"]["train"] == 1
assert len(D.tr_idx) == 84 and len(D.va_idx) == 15 and D.va_idx.min() == 84 and len(D.last) == 99
assert D.test.known.sum() == 39 and (~D.test.known).sum() == 1 and D.info["n_test_unknown_range"] == 1
# image 105 (in both files) is a v3 train row and must carry its test.json row (numeric range 10-ish, not the train row's reading 9.99)
i105 = [i for i in range(len(D.img_idx)) if D.img_idx[i] == 105]
assert len(i105) == 1 and D.last[i105[0]] != 9.99 and D.how[i105[0]] == "range_numeric", (i105, D.last[i105[0]], D.how[i105[0]])
# test part = image 37 (unknown) + official-train rows 50..59 (string/None ranges) + test.json rows 116..139
assert set(D.test.src) == {"train.json", "test.json"} and int((D.test.src == "test.json").sum()) == 24
assert D.test.img_idx[0] == 37 and not D.test.known[0]
# span classes come from the manifest's train part only
assert D.T.span_vals.tolist() == [3.0, 6.0, 10.0]
# a wrong pre-registered hash must refuse to run
try:
    prepare(get_cfg(dict(base, split_file=sp, split_sha256="0" * 64))); raise SystemExit("hash check did not fire")
except RuntimeError as e:
    print("hash mismatch refused:", str(e)[:60])
print("PREPARE OK", D.info["n_rows"], "span classes", D.T.span_vals.tolist(), "how", {h: int((D.how == h).sum()) for h in np.unique(D.how)})
