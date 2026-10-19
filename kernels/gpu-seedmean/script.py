# bench-gauges-gpu-seedmean (FROZEN-4 section 0): the pre-registered claim statistic for the GPU Stage B runs, from saved per-row hits. Numpy only; nothing is trained or scored here.
# For every config group f-<name>-<off|v3>-s<seed>_test_pred.npz found in the mounted kernel outputs: per-seed Acc_eps, the seed mean H, the Wilson 95 % interval of H at n = scored rows,
# the block bootstrap of the seed-averaged per-image hit (160-id blocks on the official split, the manifest's blocks on v3; 5,000 draws, RandomState(0)), the paired difference
# to the no-learning copy floor, and the FROZEN-4 section 0 verdict for the official groups. Env IN_ROOT / OUT_ROOT exist only for the local test.
import glob, json, math, os, re
import numpy as np

IN_ROOT, OUT_ROOT = os.environ.get("IN_ROOT", "/kaggle/input"), os.environ.get("OUT_ROOT", "/kaggle/working")
TARGET = 0.624
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def find_all(pattern):
    return sorted(glob.glob(IN_ROOT + "/**/" + pattern, recursive=True))


def find(name):
    hits = find_all(name)
    assert hits, name
    return hits[0]


def wilson(p, n, z=1.959964):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def groups_for(split):
    g = {}
    for f in find_all("f-*-%s-s*_test_pred.npz" % split):
        m = re.match(r"f-(.+)-%s-s(\d+)_test_pred\.npz$" % split, os.path.basename(f))
        if m:
            g.setdefault(m.group(1), {})[int(m.group(2))] = f
    return g


def boot(per_image, blk, draws=5000):
    ub = np.unique(blk)
    s = np.array([per_image[blk == u].sum() for u in ub], float)
    k = np.array([(blk == u).sum() for u in ub], float)
    gi = np.random.RandomState(0).randint(0, len(ub), size=(draws, len(ub)))
    m = s[gi].sum(1) / k[gi].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(ub), s, k, gi


def analyse(split, label, names, blk_all, copy_npz):
    g = groups_for(split)
    if not g:
        say("[%s] no f-*-%s-s*_test_pred.npz found" % (label, split))
        return
    copy = np.load(find(copy_npz))
    for name, seeds in sorted(g.items()):
        hs, known = [], None
        for sd in sorted(seeds):
            z = np.load(seeds[sd])
            kn = z["known"].astype(bool)
            assert known is None or (known == kn).all(), "known masks differ between seeds"
            known = kn
            hs.append(z["hits"].astype(float))
        hs = np.array(hs)
        assert hs.shape[1] == len(blk_all) == len(copy["hits"]), (hs.shape, len(blk_all))
        assert (copy["known"].astype(bool) == known).all(), "copy-floor known mask differs"
        hk, blk = hs[:, known], blk_all[known]
        n = hk.shape[1]
        per_seed = hk.mean(1)
        H = float(per_seed.mean())
        wl, wh = wilson(H, n)
        mean_img = hk.mean(0)
        bl, bh, nb, s, k, gi = boot(mean_img, blk)
        cp = copy["hits"].astype(float)[known]
        db = boot(mean_img - cp, blk)
        say("[%s] config %s: seeds %s, n = %d scored rows" % (label, name, sorted(seeds), n))
        say("  per-seed Acc_eps: %s  (min %.4f, max %.4f, sd %.4f)" % (", ".join("%.4f" % x for x in per_seed), per_seed.min(), per_seed.max(), per_seed.std(ddof=1) if len(per_seed) > 1 else float("nan")))
        say("  H (seed mean) = %.4f   Wilson 95%% at n=%d: [%.4f, %.4f]   block bootstrap 95%% over %d blocks: [%.4f, %.4f]" % (H, n, wl, wh, nb, bl, bh))
        say("  copy floor %.4f   H minus floor %+.4f   block bootstrap 95%% of the difference: [%+.4f, %+.4f]" % (cp.mean(), H - cp.mean(), db[0], db[1]))
        if split == "off":
            w_ok, b_ok = wl > TARGET, bl > TARGET
            verdict = ("Wilson LB > 62.4 and block LB > 62.4: full wording of FROZEN-4 section 0 allowed (primary config only)" if (w_ok and b_ok) else
                       "Wilson LB > 62.4, block LB <= 62.4: 'above 62.4 on the point estimate with a Wilson margin; block interval includes 62.4', no best-published wording" if w_ok else
                       "Wilson LB <= 62.4: negative result (target not met)")
            say("  VERDICT (applies to the pre-registered primary config only): %s" % verdict)


man = json.load(open(find("splits_v3.json"))) if find_all("splits_v3.json") else None
c_off = find_all("copy_hits_official.npz")
if c_off:
    cz = np.load(c_off[0])
    names = [str(x) for x in cz["names"]]
    blk_off = (np.array([int(n.split("_")[0]) for n in names]) - 1) // 160
    analyse("off", "official split", names, blk_off, "copy_hits_official.npz")
if man is not None and find_all("copy_hits_v3.npz"):
    names = man["test"]
    blk_v3 = np.array([man["block_of"][n] for n in names])
    analyse("v3", "split v3", names, blk_v3, "copy_hits_v3.npz")
os.makedirs(OUT_ROOT, exist_ok=True)
with open(os.path.join(OUT_ROOT, "seedmean.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
say("SEEDMEAN DONE")
