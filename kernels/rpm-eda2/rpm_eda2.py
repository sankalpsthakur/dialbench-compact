# bench-gauges-rpm-eda2: CPU kernel (private). Deeper profile of RPM-10K labels + near-duplicate / leakage diagnostics.
# Input: output of private kernel bench-gauges-rpm-prep (caches + label copies). No internet needed.
import os, sys, json, re, time, glob, collections, math, traceback
import numpy as np

OUT = "/kaggle/working"
T0 = time.time()


class Tee:
    def __init__(self, p):
        self.f = open(p, "w")

    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)

    def flush(self):
        sys.__stdout__.flush(); self.f.flush()


sys.stdout = Tee(os.path.join(OUT, "eda2.txt"))


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def section(n):
    print("\n" + "=" * 8, n, "=" * 8, flush=True)


section("inputs")
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        log(os.path.join(root, f), os.path.getsize(os.path.join(root, f)))
base = os.path.dirname(glob.glob("/kaggle/input/**/images_index.json", recursive=True)[0])
log("base", base)
train = json.load(open(glob.glob(base + "/meta__*train*.json")[0]))
test = json.load(open(glob.glob(base + "/meta__*test*.json")[0]))
idx = json.load(open(base + "/images_index.json"))
log("train", len(train), "test", len(test), "images", len(idx["image_rel"]))

NUM = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def last_num(t):
    m = NUM.findall(str(t))
    return float(m[-1]) if m else 0.0


def nums(t):
    return NUM.findall(str(t))


# --------------------------------------------------------------- reading string formats
section("reading string formats")
for name, items in (("train", train), ("test", test)):
    tm = collections.Counter(re.sub(r"[-+]?\d*\.\d+|[-+]?\d+", "#", str(it["reading"]))[:80] for it in items)
    log(name, "reading templates (digits masked):", tm.most_common(12))
    no_num = [it["reading"] for it in items if not nums(it["reading"])]
    log(name, "readings with NO number:", len(no_num), no_num[:8])
    multi = [it["reading"] for it in items if len(nums(it["reading"])) > 1]
    log(name, "readings with >1 number:", len(multi), [str(m)[:80] for m in multi[:5]])
    dec = collections.Counter((len(str(it["reading"]).split(".")[-1]) if "." in str(it["reading"]) else 0) for it in items if len(nums(it["reading"])) == 1)
    log(name, "decimals (single-number readings)", dict(sorted(dec.items())))
    neg = sum(1 for it in items if last_num(it["reading"]) < 0)
    log(name, "negative last-number readings", neg)

# --------------------------------------------------------------- test breakdown
section("test: meter_type x range, reading/range")
tr = np.array([last_num(it["reading"]) for it in test]); trg = np.array([float(it["range"]) for it in test])
mt = np.array([it["meter_type"] for it in test])
ratio = tr / trg
for t in sorted(set(mt)):
    m = mt == t
    rr = collections.Counter(trg[m].tolist())
    log(f"{t}: n={m.sum()} ranges={dict(rr.most_common(8))} ratio min/med/max {ratio[m].min():.3f}/{np.median(ratio[m]):.3f}/{ratio[m].max():.3f}; frac ratio<0 {np.mean(ratio[m] < 0):.3f} >1 {np.mean(ratio[m] > 1):.3f} ==0 {np.mean(tr[m] == 0):.3f}")
log("test overall ratio pct", np.round(np.percentile(ratio, [0, 1, 5, 25, 50, 75, 95, 99, 100]), 3).tolist())
log("test range value counts", dict(collections.Counter(trg.tolist()).most_common(40)))
cond = collections.Counter(c for it in test for c in it["environment_conditions"].split(","))
log("test environment tags", dict(cond.most_common()))
oth = [it for it in test if it["meter_type"] == "others"]
log("others (test) sample readings:", [(str(it["reading"])[:60], it["range"]) for it in oth[:10]])
log("others (test) with >1 number:", sum(1 for it in oth if len(nums(it["reading"])) > 1), "of", len(oth))

# --------------------------------------------------------------- train breakdown
section("train: families, null-range items, detail text")


def fam(it):
    rg = it.get("range")
    return (rg, it.get("graduation value"), json.dumps(it.get("key scale")))


fams = collections.Counter(fam(it) for it in train)
for k, v in fams.most_common():
    log("family", k, v)
trd = np.array([last_num(it["reading"]) for it in train])


def rng_of(it):
    rg = it.get("range")
    if isinstance(rg, str) and "-" in rg:
        try:
            return float(rg.split("-")[-1])
        except Exception:
            return None
    return None


rgs = np.array([rng_of(it) if rng_of(it) is not None else np.nan for it in train])
okm = ~np.isnan(rgs)
rat = trd[okm] / rgs[okm]
log("train family-known items", okm.sum(), "ratio pct", np.round(np.percentile(rat, [0, 1, 5, 25, 50, 75, 95, 99, 100]), 3).tolist(), "frac>1", np.mean(rat > 1), "frac<0", np.mean(rat < 0))
nullm = ~okm
log("train null-range items", nullm.sum())
# try to parse a range out of the CoT detail text for null items
pat = re.compile(r"(?:range|ranges|scale)[^0-9\-]{0,40}(?:from\s*)?(-?\d+(?:\.\d+)?)\s*(?:to|-|~)\s*(-?\d+(?:\.\d+)?)", re.I)
parsed = 0
vals = []
for it, isnull in zip(train, nullm):
    if not isnull:
        continue
    m = pat.search(str(it.get("detail", "")))
    if m:
        parsed += 1
        vals.append((float(m.group(1)), float(m.group(2))))
log("null-range items with a parseable (lo,hi) in detail:", parsed, "of", int(nullm.sum()))
log("parsed (lo,hi) top:", collections.Counter(vals).most_common(15))
ex = [it for it, isnull in zip(train, nullm) if isnull][:3]
for it in ex:
    log("null example:", it["image"], "| reading", it["reading"], "|", str(it["detail"])[:300].replace("\n", " "))
# does detail's answer equal reading?
ans_mis = 0
for it in train:
    m = re.search(r"<answer>(.*?)</answer>", str(it.get("detail", "")), re.S)
    if m and last_num(m.group(1)) != last_num(it["reading"]):
        ans_mis += 1
log("train items whose <answer> in detail disagrees with reading:", ans_mis)

# --------------------------------------------------------------- the one image in both files
section("train/test overlap")
tri = {it["image"]: it for it in train}
for it in test:
    if it["image"] in tri:
        log("IMAGE IN BOTH:", it["image"], "| test:", {k: str(v)[:60] for k, v in it.items() if k != "detail"}, "| train:", {k: str(v)[:60] for k, v in tri[it["image"]].items() if k != "detail"})
log("total annotation rows 8730+2000 =", len(train) + len(test), "unique image names", len(set(tri) | {it['image'] for it in test}), "files in zip 10730")
imgs_in_ann = set(tri) | {it["image"] for it in test}
rel_all = set(idx["image_rel"])
log("images on disk not referenced by any annotation row:", len(set(os.path.basename(r) for r in rel_all) ^ imgs_in_ann))

# --------------------------------------------------------------- near-duplicate / copy baselines
section("near-duplicate diagnostics (x224 cache)")
try:
    X = np.load(base + "/cache/x224.npy", mmap_mode="r")
    N = X.shape[0]
    rel = idx["image_rel"]
    pos = {r: i for i, r in enumerate(rel)}

    def feats(kind):
        out = np.zeros((N, 0), np.float32)
        parts = []
        for s in range(0, N, 1000):
            a = np.asarray(X[s:s + 1000]).astype(np.float32)
            if kind == "gray28":
                g = a.mean(-1).reshape(len(a), 28, 8, 28, 8).mean((2, 4)).reshape(len(a), -1)
            elif kind == "rgb32":
                g = a.reshape(len(a), 32, 7, 32, 7, 3).mean((2, 4)).reshape(len(a), -1)
            elif kind == "gray56":
                g = a.mean(-1).reshape(len(a), 56, 4, 56, 4).mean((2, 4)).reshape(len(a), -1)
            g = g - g.mean(1, keepdims=True)
            g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)
            parts.append(g.astype(np.float32))
        return np.concatenate(parts, 0)

    tr_idx = np.array([pos[it["image"]] for it in train]); te_idx = np.array([pos[it["image"]] for it in test])
    rd_tr = trd
    rd_te = tr
    rg_te = trg
    for kind in ("gray28", "rgb32", "gray56"):
        F = feats(kind)
        Ftr, Fte = F[tr_idx], F[te_idx]
        S = Fte @ Ftr.T
        nn = S.argmax(1); sim = S.max(1)
        pred = rd_tr[nn]
        # for non-others use ratio copy to be scale-aware? plain label copy first (what a leak would give)
        err = np.abs(pred - rd_te)
        hit = err <= 0.01 * rg_te
        log(f"[{kind}] kNN-1 label-copy test Acc_eps = {hit.mean():.4f}  (n={len(hit)})")
        for lo, hi in ((0.0, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 0.99), (0.99, 0.995), (0.995, 1.01)):
            m = (sim >= lo) & (sim < hi)
            if m.sum():
                log(f"   sim in [{lo},{hi}): n={m.sum():4d} copy-hit={hit[m].mean():.3f}")
        # train->train leave-one-out copy for comparison
        S2 = Ftr @ Ftr.T
        np.fill_diagonal(S2, -2)
        nn2 = S2.argmax(1); sim2 = S2.max(1)
        okr = ~np.isnan(rgs)
        hit2 = np.abs(rd_tr[nn2] - rd_tr)[okr] <= 0.01 * rgs[okr]
        log(f"[{kind}] kNN-1 leave-one-out copy within train (family-known items) Acc_eps = {hit2.mean():.4f}")
        log(f"[{kind}] sim pct test->train {np.round(np.percentile(sim, [5, 25, 50, 75, 95]), 4).tolist()} vs train->train {np.round(np.percentile(sim2, [5, 25, 50, 75, 95]), 4).tolist()}")
        if kind == "gray28":
            np.save(os.path.join(OUT, "test_nn_sim.npy"), sim.astype(np.float32))
            np.save(os.path.join(OUT, "test_nn_idx.npy"), nn.astype(np.int32))
except Exception:
    traceback.print_exc(file=sys.stdout)

# --------------------------------------------------------------- trivial official-metric baselines
section("trivial baselines with the official metric")
try:
    def metrics(pred, gt, rng):
        err = np.abs(pred - gt); nz = gt != 0; th = err[nz] / gt[nz]
        return dict(acc_eps=round(float((err <= 0.01 * rng).mean()), 4), acc_theta=round(float((th < 0.05).mean()), 4),
                    ref=round(float(np.clip(err / rng, None, 10).mean()), 4), rel=round(float(np.clip(th, None, 100).mean()), 4))
    med = float(np.median(rat))
    log("predict median(train ratio)*range:", metrics(med * trg, tr, trg))
    log("predict 0.5*range:", metrics(0.5 * trg, tr, trg))
    log("predict 0:", metrics(np.zeros_like(tr), tr, trg))
except Exception:
    traceback.print_exc(file=sys.stdout)
log("done")
