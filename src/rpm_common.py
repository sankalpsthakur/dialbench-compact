# Framework-free data logic for RPM-10K / DialBench: label parsing, targets, bins, decoders, splits, scoring.
# Design statistics come from labels/train.json only. labels/test.json `range` and `meter_type` are scorer/report-only.
import json
import re
import numpy as np

NUM_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
DETAIL_RANGE_RE = re.compile(
    r"(?:range|ranges|scale)[^0-9\-]{0,40}(?:from\s*)?(-?\d+(?:\.\d+)?)\s*(?:to|-|~)\s*(-?\d+(?:\.\d+)?)", re.I)


def last_num(t):
    m = NUM_RE.findall(str(t))
    return float(m[-1]) if m else 0.0  # identical to the official extract_last_number (0.0 if no number)


def span_from_train_row(it):
    """Span (full-scale value, lo==0) for a train.json row, or nan if unknown.
    1) `range` string "0-3"/"0-6"/"0-10"  2) fall back to the (lo,hi) found in the CoT `detail` text when lo==0."""
    rg = it.get("range")
    if isinstance(rg, str) and "-" in rg:
        try:
            lo, hi = rg.split("-")[0], rg.split("-")[-1]
            if float(lo) == 0 and float(hi) > 0:
                return float(hi), "range_field"
        except Exception:
            pass
    m = DETAIL_RANGE_RE.search(str(it.get("detail", "")))
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo == 0 and hi > 0:
            return hi, "detail_parse"
    return float("nan"), "unknown"


def family_of_train_row(it):
    """Coarse dial family from train.json fields (train has no meter_type)."""
    rg = it.get("range")
    if isinstance(rg, str):
        ks = json.dumps(it.get("key scale"))
        return f"{rg}|{it.get('graduation value')}|{ks}"
    return "others"


class Targets:
    """Bin grid + span classes (built from train spans only)."""

    def __init__(self, train_spans, y_min=-0.01, y_max=1.25, bin_w=0.0025, min_class_count=20):
        self.y_min, self.y_max, self.bin_w = y_min, y_max, bin_w
        self.nb = int(round((y_max - y_min) / bin_w)) + 1
        self.centers = y_min + bin_w * np.arange(self.nb)
        sp = np.asarray(train_spans, dtype=np.float64)
        sp = sp[~np.isnan(sp)]
        vals, cnt = np.unique(np.round(sp, 4), return_counts=True)
        self.span_vals = vals[cnt >= min_class_count]
        self.rare_span = float(np.median(sp[~np.isin(np.round(sp, 4), self.span_vals)])) if (~np.isin(np.round(sp, 4), self.span_vals)).any() else 100.0
        self.ns = len(self.span_vals) + 1  # last class = rare

    def span_class(self, span):
        span = np.asarray(span, dtype=np.float64)
        out = np.full(span.shape, -1, np.int64)
        ok = ~np.isnan(span)
        d = np.abs(np.round(span, 4)[..., None] - self.span_vals[None, :]) if len(self.span_vals) else None
        hit = (d < 1e-6) if d is not None else np.zeros(span.shape + (0,), bool)
        cls = np.where(hit.any(-1), hit.argmax(-1) if hit.shape[-1] else 0, len(self.span_vals))
        out[ok] = cls[ok]
        return out

    def class_to_span(self, cls):
        cls = np.asarray(cls)
        vals = np.concatenate([self.span_vals, [self.rare_span]])
        return vals[cls]

    def soft_labels(self, y, sigma=0.004):
        """y: [N] -> [N, nb] gaussian soft labels over bin centers (rows of nan -> zeros)."""
        y = np.asarray(y, dtype=np.float64)
        lab = np.exp(-0.5 * ((self.centers[None, :] - y[:, None]) / sigma) ** 2)
        s = lab.sum(1, keepdims=True)
        lab = np.where(s > 0, lab / np.maximum(s, 1e-12), 0)
        return lab.astype(np.float32)


# ------------------------------------------------------------------ decoders (numpy, on probabilities over y bins)
def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def decode_argmax(p, centers):
    return centers[p.argmax(-1)]


def decode_expect(p, centers):
    return (p * centers[None, :]).sum(-1)


def decode_window(p, centers, half_width_bins=4, refine=True):
    """Centre of the window (2*half+1 bins) holding the most probability mass; optional mass-weighted refinement.
    The window width equals the metric tolerance (+-0.01 of span) when bin_w=0.0025 and half=4."""
    k = 2 * half_width_bins + 1
    csum = np.cumsum(np.pad(p, ((0, 0), (1, 0))), axis=1)
    mass = csum[:, k:] - csum[:, :-k]  # window starting at i covers bins i..i+k-1
    start = mass.argmax(1)
    cidx = start + half_width_bins
    if not refine:
        return centers[cidx]
    out = np.empty(len(p))
    for n in range(len(p)):
        lo, hi = start[n], start[n] + k
        w = p[n, lo:hi]
        out[n] = (w * centers[lo:hi]).sum() / max(w.sum(), 1e-12)
    return out


def official_hits(pred, gt, rng):
    """Boolean hits exactly as the authors' loop (error <= 0.01 * range)."""
    err = np.abs(np.asarray(pred, np.float64) - np.asarray(gt, np.float64))
    return err <= 0.01 * np.asarray(rng, np.float64)


# ------------------------------------------------------------------ splits
def make_val_split(n, strata, n_val, seed=0):
    """Random stratified validation split (test is a random frame split of the same pool, see eda2): returns (train_idx, val_idx)."""
    rs = np.random.RandomState(seed)
    strata = np.asarray(strata)
    val = []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        k = int(round(n_val * len(idx) / n))
        val.extend(rs.choice(idx, size=min(k, len(idx)), replace=False).tolist())
    val = np.array(sorted(val))
    tr = np.setdiff1d(np.arange(n), val)
    return tr, val


def make_group_val_split(group_ids, n_val, seed=0):
    """Hold out whole groups (near-duplicate clusters) until >= n_val items are held out."""
    rs = np.random.RandomState(seed)
    gid = np.asarray(group_ids)
    ug, cnt = np.unique(gid, return_counts=True)
    order = rs.permutation(len(ug))
    held, tot = [], 0
    for o in order:
        if tot >= n_val:
            break
        held.append(ug[o])
        tot += cnt[o]
    val = np.where(np.isin(gid, held))[0]
    return np.setdiff1d(np.arange(len(gid)), val), val


# ------------------------------------------------------------------ pooled rows (split v3: rows come from BOTH label files)
def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def row_span(it):
    """(full-scale value or nan, source) for a label row from EITHER file: a numeric `range` (test.json rows), else span_from_train_row
    (train.json rows: the "0-3" string, else the range stated in the chain of thought). nan = no determinable range."""
    rg = it.get("range")
    if is_num(rg) and rg > 0:
        return float(rg), "range_numeric"
    return span_from_train_row(it)


def row_family(it):
    """Coarse family: train.json rows as family_of_train_row; test.json rows (numeric `range`) as 't:<range>'. Reporting only."""
    rg = it.get("range")
    if is_num(rg):
        return "t:%g" % rg
    return family_of_train_row(it)


def pool_rows(train_rows, test_rows):
    """image name -> label row over both files. The one image that is in both files (7443_img.jpg) keeps its test.json row, as in rpm_split3.py."""
    by = {}
    for it in train_rows:
        by[it["image"]] = it
    for it in test_rows:
        by[it["image"]] = it
    return by


def rows_from_manifest(man, by):
    """({'train': rows, 'val': rows, 'test': rows}, {'train': n, ...}): label rows named by the manifest, in manifest order, and the number of
    names in each part that have no label row (the one image of the zip that no row references)."""
    parts, missing = {}, {}
    for k in ("train", "val", "test"):
        parts[k] = [by[n] for n in man[k] if n in by]
        missing[k] = sum(1 for n in man[k] if n not in by)
    return parts, missing


def thumb_feats(X, idxs, kind):
    """Mean-centred, unit-norm thumbnails of the 224x224 cache images X[idxs]: 'gray56' (56x56 gray) or 'rgb32' (32x32 rgb), as in
    rpm_split3.py / rpm_resplit.py (the pair similarity of a "twin" is the min of the two cosines)."""
    parts = []
    for s in range(0, len(idxs), 500):
        a = np.stack([X[j] for j in idxs[s:s + 500]]).astype(np.float32)
        if kind == "gray56":
            g = a.mean(-1).reshape(len(a), 56, 4, 56, 4).mean((2, 4)).reshape(len(a), -1)
        else:
            g = a.reshape(len(a), 32, 7, 32, 7, 3).mean((2, 4)).reshape(len(a), -1)
        g = g - g.mean(1, keepdims=True)
        parts.append((g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)).astype(np.float32))
    return np.concatenate(parts)


def knn_copy_pred(X, src_idx, src_last, dst_idx, chunk=256):
    """No-learning baseline: every dst image copies the (last-number) reading of its nearest src image, nearest = argmax of min(cos gray56, cos rgb32)."""
    src_last = np.asarray(src_last, dtype=np.float64)
    A1, A2 = thumb_feats(X, src_idx, "gray56"), thumb_feats(X, src_idx, "rgb32")
    B1, B2 = thumb_feats(X, dst_idx, "gray56"), thumb_feats(X, dst_idx, "rgb32")
    out = np.empty(len(dst_idx))
    for s in range(0, len(dst_idx), chunk):
        sim = np.minimum(B1[s:s + chunk] @ A1.T, B2[s:s + chunk] @ A2.T)
        out[s:s + chunk] = src_last[sim.argmax(1)]
    return out


# ------------------------------------------------------------------ leakage-sensitivity helper
def gray28_feats(X, idxs):
    """28x28 gray thumbnails (from the 224x224 cache), mean-centred, unit norm. X: array-like [N,224,224,3] uint8."""
    out = []
    for s in range(0, len(idxs), 500):
        a = np.stack([X[j] for j in idxs[s:s + 500]]).astype(np.float32)
        g = a.mean(-1).reshape(len(a), 28, 8, 28, 8).mean((2, 4)).reshape(len(a), -1)
        g = g - g.mean(1, keepdims=True)
        out.append(g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6))
    return np.concatenate(out)


def nn_similarity(X, train_img_idx, test_img_idx):
    """max cosine similarity (gray28) of each test image to any train image."""
    Ft, Fe = gray28_feats(X, train_img_idx), gray28_feats(X, test_img_idx)
    return (Fe @ Ft.T).max(1)


SIM_BUCKETS = ((0.0, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 0.99), (0.99, 1.01))


def accuracy_by_similarity(hits, sim):
    out = {}
    for lo, hi in SIM_BUCKETS:
        m = (sim >= lo) & (sim < hi)
        out[f"{lo:.2f}-{min(hi, 1.0):.2f}"] = {"n": int(m.sum()), "acc_eps": float(hits[m].mean()) if m.sum() else None}
    return out
