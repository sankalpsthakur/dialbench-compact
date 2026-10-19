# bench-gauges-paired: POST HOC paired comparison of the two trained models with the no-learning nearest-neighbour copy floor, row by row (numpy only; nothing is
# trained or scored here, it reads saved per-row hits). Same logic as tools/paired_floor.py + tools/v3_posthoc.py, run in a Kaggle kernel because the Mac was under its disk floor.
import glob, json, sys
import numpy as np


def find(name):
    hits = sorted(glob.glob("/kaggle/input/**/" + name, recursive=True))
    assert hits, name
    return hits[0]


out_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out_lines.append(s)


man = json.load(open(find("splits_v3.json")))


def paired(label, model_npz, copy_npz, blocks_from):
    m = np.load(find(model_npz)); c = np.load(find(copy_npz))
    mh, ch = m["hits"].astype(bool), c["hits"].astype(bool)
    known = c["known"].astype(bool)
    assert len(mh) == len(ch) == len(known)
    if "known" in m.files:
        assert (m["known"].astype(bool) == known).all()
    if blocks_from == "manifest":
        names = man["test"]
        blk = np.array([man["block_of"][n] for n in names])
    else:
        names = [str(x) for x in c["names"]]
        blk = (np.array([int(n.split("_")[0]) for n in names]) - 1) // 160
    assert len(names) == len(mh)
    mh, ch, blk = mh[known], ch[known], blk[known]
    n = len(mh)
    d = mh.astype(float) - ch.astype(float)
    both, mo, co, nei = int((mh & ch).sum()), int((mh & ~ch).sum()), int((~mh & ch).sum()), int((~mh & ~ch).sum())
    say(f"[{label}] n={n}  model {mh.mean():.4f}  copy {ch.mean():.4f}  difference {d.mean():+.4f}")
    say(f"  agreement: both hit {both}, model only {mo}, copy only {co}, neither {nei}")
    rs = np.random.RandomState(0)
    idx = rs.randint(0, n, size=(5000, n)); dm = d[idx].mean(1)
    say(f"  image-level bootstrap 95% of the difference: [{np.percentile(dm, 2.5):+.4f}, {np.percentile(dm, 97.5):+.4f}]")
    ub = np.unique(blk)
    s = np.array([d[blk == u].sum() for u in ub]); k = np.array([(blk == u).sum() for u in ub], float)
    gi = rs.randint(0, len(ub), size=(5000, len(ub)))
    dm = s[gi].sum(1) / k[gi].sum(1)
    say(f"  block-level (160-id blocks) bootstrap 95% of the difference over {len(ub)} blocks: [{np.percentile(dm, 2.5):+.4f}, {np.percentile(dm, 97.5):+.4f}]")
    for nm, h in (("model", mh), ("copy", ch)):
        sb = np.array([h[blk == u].sum() for u in ub], float)
        dm2 = sb[gi].sum(1) / k[gi].sum(1)
        say(f"  block-level bootstrap 95% of {nm} Acc_eps: [{np.percentile(dm2, 2.5):.4f}, {np.percentile(dm2, 97.5):.4f}]")
    pb = [(int(u), int(kk), round(float(ss / kk), 3)) for u, ss, kk in zip(ub, s, k)]
    say("  per block difference (block, rows, model minus copy):", pb if len(pb) <= 20 else f"{len(pb)} blocks, min {min(p[2] for p in pb)}, max {max(p[2] for p in pb)}")
    return mh, ch, blk


paired("official split, run 1", "cpu-mnv3s-b_test_pred.npz", "copy_hits_official.npz", "names")
mh, ch, blk = paired("split v3, run 2", "cpu-v3_test_pred.npz", "copy_hits_v3.npz", "manifest")

# the v3 post-hoc block analysis of the model alone (as tools/v3_posthoc.py): block bootstrap of Acc_eps and leave-one-block-out
ub = np.unique(blk)
sums = np.array([mh[blk == u].sum() for u in ub], float); cnt = np.array([(blk == u).sum() for u in ub], float)
rs = np.random.RandomState(0)
gi = rs.randint(0, len(ub), size=(5000, len(ub)))
means = sums[gi].sum(1) / cnt[gi].sum(1)
say(f"[split v3, run 2] block bootstrap 95% of Acc_eps {mh.mean():.4f}: [{np.percentile(means, 2.5):.4f}, {np.percentile(means, 97.5):.4f}] (check against tools/v3_posthoc.py: 0.4220, 0.5796 expected)")
open("/kaggle/working/paired_floor.txt", "w").write("\n".join(out_lines) + "\n")
say("PAIRED DONE")
