# bench-gauges-rpm-resplit: CPU kernel (private). Leakage audit + group-disjoint re-split ("v2") of RPM-10K, images only (no labels used for grouping).
# Input: outputs of bench-gauges-rpm-prep. Output: splits_v2.json (image NAMES + group ids), audit text. No pixel data leaves the kernel.
import os, sys, json, re, time, glob, collections, traceback
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


sys.stdout = Tee(os.path.join(OUT, "resplit.txt"))


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


base = os.path.dirname(glob.glob("/kaggle/input/**/images_index.json", recursive=True)[0])
train = json.load(open(glob.glob(base + "/meta__*train*.json")[0]))
test = json.load(open(glob.glob(base + "/meta__*test*.json")[0]))
idx = json.load(open(base + "/images_index.json"))
rel = idx["image_rel"]
pos = {r: i for i, r in enumerate(rel)}
X = np.load(base + "/cache/x224.npy", mmap_mode="r")
N = X.shape[0]
log("images", N, "train rows", len(train), "test rows", len(test))

NUM = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
last_num = lambda t: (float(NUM.findall(str(t))[-1]) if NUM.findall(str(t)) else 0.0)

# --- features: 56x56 gray and 32x32 rgb, mean-centred, unit norm
def feats(kind):
    parts = []
    for s in range(0, N, 500):
        a = np.asarray(X[s:s + 500]).astype(np.float32)
        if kind == "gray56":
            g = a.mean(-1).reshape(len(a), 56, 4, 56, 4).mean((2, 4)).reshape(len(a), -1)
        else:
            g = a.reshape(len(a), 32, 7, 32, 7, 3).mean((2, 4)).reshape(len(a), -1)
        g = g - g.mean(1, keepdims=True)
        parts.append((g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)).astype(np.float32))
    return np.concatenate(parts)

F1, F2 = feats("gray56"), feats("rgb32")
S1 = F1 @ F1.T
S2 = F2 @ F2.T
S = np.minimum(S1, S2)          # a pair is "same scene" only if BOTH features agree
del S1, S2
np.fill_diagonal(S, -1.0)
log("similarity matrix", S.shape, S.dtype)

tr_names = {it["image"]: i for i, it in enumerate(train)}
te_names = {it["image"]: i for i, it in enumerate(test)}
img_split = np.zeros(N, np.int8)   # 1 train, 2 test, 3 both
for r in rel:
    k = pos[r]
    img_split[k] = (1 if r in tr_names else 0) + (2 if r in te_names else 0)
tr_img = np.where(img_split == 1)[0]; te_img = np.where(img_split == 2)[0]
log("train-only images", len(tr_img), "test-only images", len(te_img), "in both", int((img_split == 3).sum()))

# --- official split: how many test images have a train twin, by threshold
section = lambda n: print("\n" + "=" * 8, n, "=" * 8, flush=True)
section("official split leakage by threshold (min of gray56, rgb32 cosine)")
nn_te = S[np.ix_(te_img, tr_img)].max(1)
S_tt = S[np.ix_(tr_img, tr_img)]
nn_tr = S_tt.max(1)
for tau in (0.90, 0.95, 0.98, 0.99, 0.995):
    log(f"tau {tau}: test with a train neighbour >= tau: {(nn_te >= tau).mean():.4f}   train with another train neighbour >= tau: {(nn_tr >= tau).mean():.4f}")
log("pct test->train", np.round(np.percentile(nn_te, [5, 25, 50, 75, 95, 99]), 4).tolist(), "train->train", np.round(np.percentile(nn_tr, [5, 25, 50, 75, 95, 99]), 4).tolist())
del S_tt

# --- groups over ALL images (train+test combined), union-find
def groups_at(tau):
    parent = np.arange(N)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    ii, jj = np.nonzero(np.triu(S >= tau, 1))
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(x) for x in range(N)])
    return np.unique(roots, return_inverse=True)[1], len(ii)

section("group structure")
G = {}
for tau in (0.95, 0.98, 0.99):
    g, npairs = groups_at(tau)
    sz = np.bincount(g)
    G[tau] = g
    cross = collections.Counter()
    for gid in np.unique(g[(img_split == 1) | (img_split == 2)]):
        pass
    mixed = sum(1 for gid in range(len(sz)) if len(set(img_split[g == gid])) > 1 and sz[gid] > 1) if len(sz) < 20000 else -1
    log(f"tau {tau}: pairs {npairs}, groups {len(sz)}, largest {sz.max()}, singletons {(sz == 1).sum()}, groups containing both train and test images {mixed}")

# --- v2 split from groups at TAU2 (images only)
TAU2 = 0.98
g = G[TAU2]
rs = np.random.RandomState(2026)
order = rs.permutation(g.max() + 1)
sz = np.bincount(g)
assign = np.zeros(g.max() + 1, np.int8)  # 0 train, 1 val, 2 test
n_te = n_va = 0
for gid in order:
    if n_te < 2000 and n_te + sz[gid] <= 2100:
        assign[gid] = 2; n_te += sz[gid]
    elif n_va < 1000 and n_va + sz[gid] <= 1100:
        assign[gid] = 1; n_va += sz[gid]
split_v2 = assign[g]
log("v2 sizes train/val/test", [(split_v2 == k).sum() for k in (0, 1, 2)], "from tau", TAU2)

# --- baselines on v2 and on the official split (kNN-1 copy, official metric); span for scoring: official range for test rows, parsed train span for train rows
def span_of_train(it):
    rg = it.get("range")
    if isinstance(rg, str) and "-" in rg:
        try:
            return float(rg.split("-")[-1])
        except Exception:
            pass
    m = re.search(r"(?:range|ranges|scale)[^0-9\-]{0,40}(?:from\s*)?(-?\d+(?:\.\d+)?)\s*(?:to|-|~)\s*(-?\d+(?:\.\d+)?)", str(it.get("detail", "")), re.I)
    if m and float(m.group(1)) == 0 and float(m.group(2)) > 0:
        return float(m.group(2))
    return float("nan")

lab = np.full(N, np.nan); rng_ = np.full(N, np.nan)
for it in train:
    k = pos[it["image"]]
    lab[k] = last_num(it["reading"]); rng_[k] = span_of_train(it)
for it in test:
    k = pos[it["image"]]
    lab[k] = last_num(it["reading"]); rng_[k] = float(it["range"])   # scorer-side range, used only to score
section("kNN-1 label-copy baseline (official metric) on splits")
def knn_copy(src, dst, tag):
    ok = ~np.isnan(rng_[dst])
    d = dst[ok]
    nn = src[S[np.ix_(d, src)].argmax(1)]
    hit = np.abs(lab[nn] - lab[d]) <= 0.01 * rng_[d]
    log(f"{tag}: n={len(d)} Acc_eps(copy)={hit.mean():.4f}")
    return hit.mean()
knn_copy(tr_img, te_img, "official train -> official test")
knn_copy(np.where(split_v2 == 0)[0], np.where(split_v2 == 2)[0], "v2 train -> v2 test")
knn_copy(np.where(split_v2 == 0)[0], np.where(split_v2 == 1)[0], "v2 train -> v2 val")
# leak-free subsets of the OFFICIAL test: no train neighbour above tau
for tau in (0.95, 0.98, 0.99):
    keep = te_img[nn_te < tau]
    ok = ~np.isnan(rng_[keep])
    d = keep[ok]
    nn = tr_img[S[np.ix_(d, tr_img)].argmax(1)]
    hit = np.abs(lab[nn] - lab[d]) <= 0.01 * rng_[d]
    log(f"official test subset with NO train neighbour >= {tau}: n={len(d)} kNN-copy Acc_eps={hit.mean():.4f}")

# --- composition of v2 vs official (label-derived, report only)
section("composition check (labels, report only)")
def comp(idxs, tag):
    r = rng_[idxs]
    r = r[~np.isnan(r)]
    c = collections.Counter(np.where(np.isin(r, [3, 6, 10]), r, -1).tolist())
    log(tag, "n", len(idxs), "span counts (3/6/10/other=-1)", dict(c))
comp(te_img, "official test"); comp(np.where(split_v2 == 2)[0], "v2 test"); comp(tr_img, "official train"); comp(np.where(split_v2 == 0)[0], "v2 train")

out = {"description": "group-disjoint re-split of RPM-10K: union-find on min(cosine of 56x56 gray, cosine of 32x32 rgb) >= tau over ALL 10,729 distinct images; groups assigned whole to a split. Image names only.",
       "tau": TAU2, "seed": 2026, "feature": "min(gray56 cosine, rgb32 cosine) on the 224x224 official-resize images",
       "train": [rel[i] for i in np.where(split_v2 == 0)[0]], "val": [rel[i] for i in np.where(split_v2 == 1)[0]], "test": [rel[i] for i in np.where(split_v2 == 2)[0]],
       "group_id": {rel[i]: int(g[i]) for i in range(N)}}
json.dump(out, open(os.path.join(OUT, "splits_v2.json"), "w"))
log("wrote splits_v2.json", len(out["train"]), len(out["val"]), len(out["test"]))
