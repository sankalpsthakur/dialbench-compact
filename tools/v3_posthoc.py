#!/usr/bin/env python3
"""POST HOC analysis of a split-v3 run from its saved arrays (numpy only; nothing is re-scored, no model is loaded).
usage: v3_posthoc.py TEST_PRED_NPZ RESULT_JSON SPLITS_V3_JSON
  1. alignment check: npz rows are in manifest order (the trainer keeps manifest order and no test name lacks a label row)
  2. block (cluster) bootstrap of the test accuracy: the sampling unit of split v3 is the 160-id block (14 test blocks), not the image
  3. cost of the pre-registered span-class rule: test rows whose range is not one of the model's span classes
  4. accuracy per range class re-weighted to the official test composition (descriptive standardisation)"""
import json, sys
import numpy as np
npz, res, man = sys.argv[1:4]
z = np.load(npz); r = json.load(open(res)); m = json.load(open(man))
hits, known, rng, last = z["hits"].astype(bool), z["known"].astype(bool), z["rng"], z["last"]
names = m["test"]
assert len(names) == len(hits) == len(known), (len(names), len(hits), len(known))
t = r["test"]; sf = t.get("split_file", {})
assert int(known.sum()) == t["metrics"]["n"], (int(known.sum()), t["metrics"]["n"])
if sf:
    assert int(known.sum()) == sf["n_scored"] and len(known) == sf["n_test_rows"]
blk = np.array([m["block_of"][n] for n in names])
ids = np.array([int(n.split("_")[0]) for n in names])
assert (((ids - 1) // m["B"]) == blk).all(), "block ids do not follow the name ids"
print(f"alignment ok: {len(names)} rows, {int(known.sum())} scored, {len(np.unique(blk))} blocks")
h = hits[known]; b = blk[known]
acc = h.mean()
ub = np.unique(b)
sums = np.array([h[b == u].sum() for u in ub], float); cnt = np.array([(b == u).sum() for u in ub], float)
rs = np.random.RandomState(0)
gi = rs.randint(0, len(ub), size=(5000, len(ub)))
means = sums[gi].sum(1) / cnt[gi].sum(1)
lo, hi = np.percentile(means, [2.5, 97.5])
print(f"test Acc_eps {acc:.4f}; block bootstrap 95% [{lo:.4f}, {hi:.4f}] over {len(ub)} blocks (5000 resamples); trainer Wilson {t['wilson95']}, trainer grouped bootstrap {t['grouped_bootstrap95']}")
print("per block (id block, rows, Acc_eps):", [(int(u), int(c), round(float(s / c), 3)) for u, s, c in zip(ub, sums, cnt)])
# leave-one-block-out spread
loo = [(sums.sum() - s) / (cnt.sum() - c) for s, c in zip(sums, cnt)]
print(f"leave-one-block-out Acc_eps range [{min(loo):.4f}, {max(loo):.4f}]")
# span-class rule cost
sv = np.array(r["span_vals"], float)
inc = np.isclose(rng[known][:, None], sv[None, :]).any(1)
print(f"model span classes {sv.tolist()} (rare -> {r['rare_span']}): scored rows with range in a class: {int(inc.sum())} ({inc.mean():.3f}), Acc_eps {h[inc].mean():.4f}; not in a class: {int((~inc).sum())}, Acc_eps {h[~inc].mean() if (~inc).any() else float('nan'):.4f}")
for v in (1.0, 2.0):
    mk = np.isclose(rng[known], v)
    print(f"  rows with range {v:g}: {int(mk.sum())}, Acc_eps {h[mk].mean() if mk.any() else float('nan'):.4f}")
vals, cnts = np.unique(np.round(rng[known][~inc], 4), return_counts=True)
print("  ranges outside the classes (value: rows):", {float(v): int(c) for v, c in zip(vals, cnts)})
# standardisation to the official test composition (661 / 665 / 531 / 142 of 1,999 known rows)
cl = {"range 3": 661, "range 6": 665, "range 10": 531, "other": 142}
if sf:
    tab = sf["by_range_class"]
    tot = sum(cl.values())
    std = sum(cl[k] / tot * tab[k]["acc_eps"] for k in cl if tab[k]["acc_eps"] is not None)
    print("by_range_class:", {k: (v["n"], None if v["acc_eps"] is None else round(v["acc_eps"], 4)) for k, v in tab.items()})
    print(f"descriptive: accuracy re-weighted to the official test composition (3/6/10/other = 661/665/531/142): {std:.4f} (raw v3 {acc:.4f}); no-learning copy on the same v3 rows {sf['knn_copy_baseline']['acc_eps']:.4f} [{sf['knn_copy_baseline']['wilson95'][0]:.4f}, {sf['knn_copy_baseline']['wilson95'][1]:.4f}]")
