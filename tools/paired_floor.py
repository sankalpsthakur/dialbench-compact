#!/usr/bin/env python3
"""POST HOC paired comparison of a trained model with the no-learning nearest-neighbour copy floor, row by row, from saved arrays (numpy only).
usage: paired_floor.py MODEL_TEST_PRED_NPZ COPY_HITS_NPZ LABEL [SPLITS_V3_JSON]
  official: rows are test.json rows; blocks are 160-id blocks from the image names (the same rule as split v3)
  v3: rows are the manifest's test rows; blocks are the manifest's blocks
Prints model, copy, difference, the 2x2 agreement table, and image-level and block-level bootstrap 95 % intervals of the difference."""
import json, sys
import numpy as np
mp, cp, label = sys.argv[1:4]
man = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
m = np.load(mp); c = np.load(cp)
mh, ch = m["hits"].astype(bool), c["hits"].astype(bool)
known = c["known"].astype(bool)
assert len(mh) == len(ch) == len(known)
if "known" in m.files:
    assert (m["known"].astype(bool) == known).all()
if man is not None:
    names = man["test"]; blk = np.array([man["block_of"][n] for n in names])
else:
    names = [str(x) for x in c["names"]]
    blk = (np.array([int(n.split("_")[0]) for n in names]) - 1) // 160
assert len(names) == len(mh)
mh, ch, blk = mh[known], ch[known], blk[known]
n = len(mh)
d = mh.astype(float) - ch.astype(float)
both, mo, co, nei = int((mh & ch).sum()), int((mh & ~ch).sum()), int((~mh & ch).sum()), int((~mh & ~ch).sum())
print(f"[{label}] n={n}  model {mh.mean():.4f}  copy {ch.mean():.4f}  difference {d.mean():+.4f}")
print(f"  agreement: both hit {both}, model only {mo}, copy only {co}, neither {nei}")
rs = np.random.RandomState(0)
idx = rs.randint(0, n, size=(5000, n)); dm = d[idx].mean(1)
print(f"  image-level bootstrap 95% of the difference: [{np.percentile(dm, 2.5):+.4f}, {np.percentile(dm, 97.5):+.4f}]")
ub = np.unique(blk)
s = np.array([d[blk == u].sum() for u in ub]); k = np.array([(blk == u).sum() for u in ub], float)
gi = rs.randint(0, len(ub), size=(5000, len(ub)))
dm = s[gi].sum(1) / k[gi].sum(1)
print(f"  block-level bootstrap 95% of the difference over {len(ub)} blocks: [{np.percentile(dm, 2.5):+.4f}, {np.percentile(dm, 97.5):+.4f}]")
for nm, h in (("model", mh), ("copy", ch)):
    sb = np.array([h[blk == u].sum() for u in ub], float)
    dm2 = sb[gi].sum(1) / k[gi].sum(1)
    print(f"  block-level bootstrap 95% of {nm} Acc_eps: [{np.percentile(dm2, 2.5):.4f}, {np.percentile(dm2, 97.5):.4f}]")
pb = [(int(u), int(kk), round(float(ss / kk), 3)) for u, ss, kk in zip(ub, s, k)]
print("  per block difference (block, rows, model minus copy):", pb if len(pb) <= 20 else f"{len(pb)} blocks, min {min(p[2] for p in pb)}, max {max(p[2] for p in pb)}")
