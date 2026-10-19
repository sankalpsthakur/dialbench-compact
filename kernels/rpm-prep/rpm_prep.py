# bench-gauges-rpm-prep  (CPU kernel, private)
# Downloads RPM-10K (DialBench) INSIDE the kernel, profiles it, and writes small metadata plus
# resized uint8 caches to /kaggle/working for later private bench-gauges-* kernels.
# The raw zip / extracted files stay in /tmp and are never written to the kernel output.
# Data license is "TBD" (README of Event-AHU/DialBench, 2026-10-07): outputs of this kernel stay private.
import os, sys, io, json, time, re, zipfile, hashlib, subprocess, collections, math, traceback
import numpy as np
from PIL import Image
from multiprocessing import Pool

OUT = "/kaggle/working"
TMP = "/tmp/rpm"
ZIP = "/tmp/rpm.zip"
URL = "<the RPM-10K.zip link (dl=1) from the README of github.com/Event-AHU/DialBench; not copied into this kit [link redacted]>"
EXPECT_BYTES = 3185292936
T0 = time.time()


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


sys.stdout = Tee(os.path.join(OUT, "eda.txt"))


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def section(name):
    print("\n" + "=" * 8, name, "=" * 8, flush=True)


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()


# ---------------------------------------------------------------- download
section("download")
os.makedirs(TMP, exist_ok=True)
for attempt in range(4):
    sz = os.path.getsize(ZIP) if os.path.exists(ZIP) else 0
    if sz == EXPECT_BYTES:
        break
    log("attempt", attempt, "have", sz)
    sh(f"curl -sL --retry 3 --retry-delay 5 -C - -o {ZIP} '{URL}'", timeout=1800)
sz = os.path.getsize(ZIP)
log("zip bytes", sz, "expected", EXPECT_BYTES, "OK" if sz == EXPECT_BYTES else "MISMATCH")
h = hashlib.sha256()
with open(ZIP, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
log("sha256", h.hexdigest())
json.dump({"bytes": sz, "sha256": h.hexdigest(), "url_host": "www.dropbox.com (RPM-10K.zip share link from the Event-AHU/DialBench README)"},
          open(os.path.join(OUT, "zip_provenance.json"), "w"))

# ---------------------------------------------------------------- zip listing + extraction (data files only)
section("zip listing")
z = zipfile.ZipFile(ZIP)
infos = z.infolist()
ext_count = collections.Counter(os.path.splitext(i.filename)[1].lower() for i in infos if not i.is_dir())
log("entries", len(infos), "extensions", dict(ext_count))
top = collections.Counter(i.filename.split("/")[0] for i in infos)
log("top-level", dict(top.most_common(20)))
sub = collections.Counter("/".join(i.filename.split("/")[:2]) for i in infos if not i.is_dir())
log("first two path parts (top 25)", sub.most_common(25))
log("sample names", [i.filename for i in infos[:12]])
ALLOWED = {".json", ".jpg", ".jpeg", ".png", ".bmp", ".txt", ".csv", ".jsonl", ".md"}
n_x = 0
for i in infos:
    if i.is_dir() or i.filename.startswith("__MACOSX") or "/._" in i.filename:
        continue
    if os.path.splitext(i.filename)[1].lower() in ALLOWED:
        z.extract(i, TMP)
        n_x += 1
log("extracted data files", n_x)
log("time", sh(f"du -sh {TMP}"))

# ---------------------------------------------------------------- annotation files
section("annotation files")
json_files = []
for root, _, files in os.walk(TMP):
    for fn in files:
        if fn.lower().endswith((".json", ".jsonl", ".txt", ".csv", ".md")):
            p = os.path.join(root, fn)
            json_files.append((p, os.path.getsize(p)))
for p, s in sorted(json_files):
    log("file", p.replace(TMP, ""), s)

ann = {}  # name -> list of items
for p, s in json_files:
    if not p.lower().endswith(".json") or s > 400e6:
        continue
    try:
        d = json.load(open(p))
    except Exception as e:
        log("cannot parse", p, e)
        continue
    if isinstance(d, dict):
        log("dict json", p.replace(TMP, ""), "keys", list(d.keys())[:20])
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                d = v
                log("  using list under key", k)
                break
    if isinstance(d, list) and d and isinstance(d[0], dict):
        name = os.path.relpath(p, TMP)
        ann[name] = d
        log("annotation list", name, "items", len(d), "keys", list(d[0].keys()))

for name, items in ann.items():
    section(f"profile {name}")
    print("first 4 items:")
    for it in items[:4]:
        print("  ", json.dumps(it, ensure_ascii=False)[:600])
    keys = collections.Counter(k for it in items for k in it.keys())
    log("key counts", dict(keys))
    for k in keys:
        vals = [it.get(k) for it in items]
        try:
            uniq = collections.Counter(json.dumps(v, ensure_ascii=False) if not isinstance(v, (str, int, float)) else v for v in vals)
        except Exception:
            continue
        log(f"key {k!r}: unique {len(uniq)}; top:", [(str(a)[:90], c) for a, c in uniq.most_common(12)])
    # numeric view of reading / range
    try:
        rd = np.array([float(re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(it["reading"]))[-1]) for it in items])
        rg = np.array([float(it["range"]) for it in items])
        log("reading: min %.4g max %.4g mean %.4g | range: unique %s" % (rd.min(), rd.max(), rd.mean(), sorted(set(rg.tolist()))[:40]))
        ratio = rd / rg
        log("reading/range: min %.4f max %.4f mean %.4f; frac<0 %.4f; frac>1 %.4f; frac==0 %.4f" % (
            ratio.min(), ratio.max(), ratio.mean(), (ratio < 0).mean(), (ratio > 1).mean(), (rd == 0).mean()))
        log("reading/range deciles", np.round(np.percentile(ratio, [0, 5, 10, 25, 50, 75, 90, 95, 100]), 4).tolist())
        # decimals used in the reading strings
        dec = collections.Counter(len(str(it["reading"]).split(".")[-1]) if "." in str(it["reading"]) else 0 for it in items)
        log("decimals in reading strings", dict(dec))
        log("reading type", collections.Counter(type(it["reading"]).__name__ for it in items))
        for it in items[:0]:
            pass
    except Exception as e:
        log("numeric profile failed", e)
    # queries
    try:
        qs = collections.Counter(it["query"] for it in items)
        log("unique queries", len(qs))
        for q, c in qs.most_common(15):
            print(f"   {c:6d}  {q[:300]!r}")
        # normalise digits to see templates
        tmpl = collections.Counter(re.sub(r"[-+]?\d*\.\d+|[-+]?\d+", "#", it["query"]) for it in items)
        log("unique query templates (digits masked)", len(tmpl))
        for q, c in tmpl.most_common(15):
            print(f"   {c:6d}  {q[:300]!r}")
    except Exception as e:
        log("query profile failed", e)
    # images per item
    try:
        imgs = collections.Counter(it["image"] for it in items)
        mult = collections.Counter(imgs.values())
        log("unique images", len(imgs), "items per image histogram", dict(sorted(mult.items())))
        log("image name samples", list(imgs.keys())[:8])
        pat = collections.Counter(re.sub(r"\d+", "#", os.path.basename(k)) for k in imgs)
        log("filename patterns", pat.most_common(8))
        log("directory parts", collections.Counter(os.path.dirname(k) for k in imgs).most_common(8))
    except Exception as e:
        log("image profile failed", e)

# overlap of images between annotation files
names = list(ann.keys())
section("cross-file overlap")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a = {it.get("image") for it in ann[names[i]]}
        b = {it.get("image") for it in ann[names[j]]}
        log(f"{names[i]} vs {names[j]}: shared image paths {len(a & b)} (|a|={len(a)}, |b|={len(b)})")
        qa = {(it.get("image"), it.get("query")) for it in ann[names[i]]}
        qb = {(it.get("image"), it.get("query")) for it in ann[names[j]]}
        log(f"   shared (image, query) pairs {len(qa & qb)}")

# ---------------------------------------------------------------- resolve images and cache
section("image resolution")
index = collections.defaultdict(list)
for root, _, files in os.walk(TMP):
    for fn in files:
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            index[fn].append(os.path.join(root, fn))
log("image files on disk", sum(len(v) for v in index.values()), "unique basenames", len(index))
dup_names = [k for k, v in index.items() if len(v) > 1]
log("basenames that occur more than once", len(dup_names), dup_names[:5])


def resolve(rel):
    cands = [os.path.join(TMP, rel)]
    base = os.path.basename(rel)
    for c in cands:
        if os.path.exists(c):
            return c
    hits = index.get(base, [])
    if len(hits) == 1:
        return hits[0]
    # disambiguate by suffix match
    suf = [h for h in hits if h.endswith(rel)]
    return suf[0] if suf else (hits[0] if hits else None)


unique_rel = []
seen = {}
for name, items in ann.items():
    for it in items:
        r = it.get("image")
        if r is not None and r not in seen:
            seen[r] = len(unique_rel)
            unique_rel.append(r)
paths = [resolve(r) for r in unique_rel]
missing = [r for r, p in zip(unique_rel, paths) if p is None]
log("unique image refs", len(unique_rel), "unresolved", len(missing), missing[:5])


def load(i):
    p = paths[i]
    try:
        im = Image.open(p)
        info = (im.size[0], im.size[1], im.mode, os.path.getsize(p), im.format)
        im = im.convert("RGB")
        a224 = np.asarray(im.resize((224, 224), Image.Resampling.LANCZOS), dtype=np.uint8)  # official protocol resize
        a320 = np.asarray(im.resize((320, 320), Image.Resampling.LANCZOS), dtype=np.uint8)
        return i, a224, a320, info
    except Exception as e:
        return i, None, None, ("ERR", str(e))


section("decode + cache")
N = len(unique_rel)
X224 = np.zeros((N, 224, 224, 3), np.uint8)
X320 = np.zeros((N, 320, 320, 3), np.uint8)
infos_img = [None] * N
with Pool(4) as pool:
    for k, (i, a, b, info) in enumerate(pool.imap_unordered(load, range(N), chunksize=32)):
        if a is not None:
            X224[i] = a
            X320[i] = b
        infos_img[i] = info
        if k % 2000 == 0:
            log("decoded", k, "/", N)
bad = [i for i, f in enumerate(infos_img) if f is None or f[0] == "ERR"]
log("decode failures", len(bad), [unique_rel[i] for i in bad[:5]])
sizes = collections.Counter((f[0], f[1]) for f in infos_img if f and f[0] != "ERR")
log("distinct (w,h)", len(sizes), "top:", sizes.most_common(12))
w = np.array([f[0] for f in infos_img if f and f[0] != "ERR"])
hh = np.array([f[1] for f in infos_img if f and f[0] != "ERR"])
log("width pct", np.percentile(w, [0, 5, 25, 50, 75, 95, 100]).tolist(), "height pct", np.percentile(hh, [0, 5, 25, 50, 75, 95, 100]).tolist())
log("aspect w/h pct", np.round(np.percentile(w / hh, [0, 5, 25, 50, 75, 95, 100]), 3).tolist())
log("modes", collections.Counter(f[2] for f in infos_img if f and f[0] != "ERR"), "formats", collections.Counter(f[4] for f in infos_img if f and f[0] != "ERR"))

os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
np.save(os.path.join(OUT, "cache", "x224.npy"), X224)
np.save(os.path.join(OUT, "cache", "x320.npy"), X320)
json.dump({"image_rel": unique_rel, "orig_wh": [[f[0], f[1]] if f and f[0] != "ERR" else None for f in infos_img]},
          open(os.path.join(OUT, "images_index.json"), "w"))
for name, items in ann.items():
    meta = []
    for it in items:
        m = dict(it)
        m["img_idx"] = seen[it["image"]]
        meta.append(m)
    safe = name.replace("/", "__")
    json.dump(meta, open(os.path.join(OUT, f"meta__{safe}"), "w"), ensure_ascii=False)
log("wrote caches", X224.shape, X320.shape)

# ---------------------------------------------------------------- near-duplicate audit (thumbnail cosine)
section("near-duplicate audit")
try:
    # 28x28 gray thumbnails from the 224 cache
    g = X224.astype(np.float32).mean(-1).reshape(N, 28, 8, 28, 8).mean((2, 4)).reshape(N, -1)
    g = g - g.mean(1, keepdims=True)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)
    split_of = {}
    for name, items in ann.items():
        low = name.lower()
        sp = "train" if "train" in low else ("test" if "test" in low else ("val" if "val" in low else name))
        for it in items:
            split_of.setdefault(seen[it["image"]], set()).add(sp)
    both = [i for i, s in split_of.items() if len(s) > 1]
    log("images that appear in more than one split-file by path:", len(both))
    sp_idx = collections.defaultdict(list)
    for i, s in split_of.items():
        if len(s) == 1:
            sp_idx[next(iter(s))].append(i)
    for k, v in sp_idx.items():
        log("split", k, "unique images", len(v))
    pc = [1, 5, 25, 50, 75, 95, 99, 100]
    if "train" in sp_idx:
        tr = np.array(sorted(sp_idx["train"]))
        S_tt = g[tr] @ g[tr].T
        np.fill_diagonal(S_tt, -1)
        nn_tt = S_tt.max(1)
        log("train->train(other) max cosine pct", dict(zip(pc, np.round(np.percentile(nn_tt, pc), 4).tolist())))
        log("train->train frac >=0.99 / 0.98 / 0.95:", [(nn_tt >= t).mean().round(4) for t in (0.99, 0.98, 0.95)])
        for other in [k for k in sp_idx if k != "train"]:
            te = np.array(sorted(sp_idx[other]))
            S = g[te] @ g[tr].T
            nn = S.max(1)
            log(f"{other}->train max cosine pct", dict(zip(pc, np.round(np.percentile(nn, pc), 4).tolist())))
            log(f"{other}->train frac >=0.99 / 0.98 / 0.95:", [(nn >= t).mean().round(4) for t in (0.99, 0.98, 0.95)])
        # pseudo-instance groups inside train (union-find over high-similarity pairs), several thresholds
        groups = {}
        for tau in (0.95, 0.97, 0.98, 0.99):
            parent = np.arange(len(tr))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            ii, jj = np.nonzero(np.triu(S_tt >= tau, 1))
            for a_, b_ in zip(ii, jj):
                ra, rb = find(a_), find(b_)
                if ra != rb:
                    parent[ra] = rb
            roots = np.array([find(x) for x in range(len(tr))])
            _, gid = np.unique(roots, return_inverse=True)
            sizes_g = np.bincount(gid)
            log(f"tau {tau}: groups {len(sizes_g)}, largest {sizes_g.max()}, median {np.median(sizes_g)}, singletons {(sizes_g == 1).sum()}")
            groups[str(tau)] = {"train_img_idx": tr.tolist(), "group_id": gid.tolist()}
        json.dump(groups, open(os.path.join(OUT, "train_groups.json"), "w"))
except Exception:
    traceback.print_exc()

# ---------------------------------------------------------------- official metric + trivial baselines
section("official metric on trivial baselines")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    NUM = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

    def last_num(t):
        m = NUM.findall(str(t))
        return float(m[-1]) if m else 0.0

    def metrics(pred, gt, rng):
        err = np.abs(pred - gt)
        nz = gt != 0
        th = err[nz] / gt[nz]
        return dict(acc_eps=float((err <= 0.01 * rng).mean()), acc_theta=float((th < 0.05).mean()),
                    ref=float(np.clip(err / rng, None, 10).mean()), rel=float(np.clip(th, None, 100).mean()))

    trn = [n for n in ann if "train" in n.lower()]
    tst = [n for n in ann if "test" in n.lower() or "val" in n.lower()]
    if trn and tst:
        A = ann[trn[0]]
        gt_tr = np.array([last_num(it["reading"]) for it in A]); rg_tr = np.array([float(it["range"]) for it in A])
        for tn in tst:
            B = ann[tn]
            gt = np.array([last_num(it["reading"]) for it in B]); rg = np.array([float(it["range"]) for it in B])
            r_med = np.median(gt_tr / rg_tr)
            log(tn, "constant 0.5*range:", metrics(0.5 * rg, gt, rg))
            log(tn, "constant median(train ratio)*range:", metrics(r_med * rg, gt, rg))
            log(tn, "predict 0:", metrics(np.zeros_like(gt), gt, rg))
except Exception:
    traceback.print_exc()

log("ALL DONE; output files:", sh(f"ls -la {OUT} {OUT}/cache | head -40"))
