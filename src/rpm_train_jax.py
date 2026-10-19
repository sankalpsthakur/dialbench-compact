# JAX/Flax trainer for RPM-10K (DialBench) for Kaggle TPU (v5e-8) or any JAX backend. Same data logic, targets, decoders and
# scoring as rpm_train_torch.py (those come from rpm_common.py / dialbench_metrics.py). Kernels are built by concatenation:
#   dialbench_metrics.py, rpm_common.py, jax_resnet.py, rpm_train_jax.py, config part, runner tail.
# Data-parallel over all local devices with jit + NamedSharding (batch sharded, params/data replicated).
import os, sys, json, time, math, glob, hashlib, copy, traceback
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from dialbench_metrics import dialbench_metrics, wilson_interval, bootstrap_acc_ci
from rpm_common import *  # noqa: F401,F403
from jax_resnet import ResNetBackbone, convert_tv_to_flax

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


DEFAULTS = dict(
    arch="resnet18", pretrained="npz:resnet18", res=224, head_ch=64, hidden=512, drop=0.2,
    epochs=60, bs=256, lr=1e-3, head_lr_mult=3.0, wd=0.02, warmup_epochs=2, ema=0.995,
    sigma=0.004, w_l1=1.0, w_span=0.3, seed=0, label_smooth_span=0.0,
    rot=12.0, scale=0.15, aspect=0.07, shift=0.06, shear=0.06, p_blur=0.3, p_noise=0.4, p_glare=0.2, p_erase=0.15,
    n_val=1000, val_mode="split", eval_every=5, run_test=False, decoder="window", half_width_bins=4,
    out_dir="/kaggle/working", tag="run", max_train=None, save_weights=True, input_dir=None, width_mult=1.0,
    bf16=True,
)


def get_cfg(cfg):
    c = dict(DEFAULTS)
    c.update(cfg)
    return c


def cfg_hash(c):
    keep = {k: v for k, v in c.items() if k not in ("out_dir", "input_dir")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ data (shared with the torch trainer)
def load_data(c):
    base = c.get("input_dir")
    if base is None:
        base = os.path.dirname(glob.glob("/kaggle/input/**/images_index.json", recursive=True)[0])
    train = json.load(open(glob.glob(base + "/meta__*train*.json")[0]))
    test = json.load(open(glob.glob(base + "/meta__*test*.json")[0]))
    if c.get("drop_test_dups", True):   # the official train.json contains one image that is also in test.json; never train on it
        test_names = {it["image"] for it in test}
        train = [it for it in train if it["image"] not in test_names]
    key = "x224" if c["res"] <= 256 else "x320"
    X = np.load(f"{base}/cache/{key}.npy", mmap_mode="r")
    return base, train, test, X


def build_train_arrays(train):
    last = np.array([last_num(it["reading"]) for it in train])
    span, how = zip(*[span_from_train_row(it) for it in train])
    span = np.array(span)
    fam = np.array([family_of_train_row(it) for it in train])
    img_idx = np.array([it["img_idx"] for it in train])
    return last, span, np.array(how), fam, img_idx


def thumb_groups(Xsub, tau=0.98):
    n = len(Xsub)
    g = np.asarray(Xsub).astype(np.float32).mean(-1).reshape(n, 28, 8, 28, 8).mean((2, 4)).reshape(n, -1)
    g = g - g.mean(1, keepdims=True)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)
    S = g @ g.T
    parent = np.arange(n)

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
    roots = np.array([find(x) for x in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def resize_np(x, res):
    """uint8 [n,H,W,3] numpy -> uint8 [n,res,res,3] using jax.image (antialiased bilinear), chunked."""
    if x.shape[1] == res:
        return x
    out = []
    for i in range(0, len(x), 256):
        b = jnp.asarray(x[i:i + 256]).astype(jnp.float32)
        b = jax.image.resize(b, (b.shape[0], res, res, 3), method="linear", antialias=True)
        out.append(np.asarray(jnp.clip(jnp.round(b), 0, 255).astype(jnp.uint8)))
    return np.concatenate(out)


# ------------------------------------------------------------------ model
class RPMHead(nn.Module):
    nb: int
    ns: int
    head_ch: int = 64
    hidden: int = 512
    drop: float = 0.2
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, f, train: bool):
        x = nn.Conv(self.head_ch, (1, 1), use_bias=False, dtype=self.dtype, param_dtype=jnp.float32, name="reduce")(f)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=self.dtype, param_dtype=jnp.float32, name="reduce_bn")(x)
        x = nn.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(nn.Dense(self.hidden, dtype=self.dtype, param_dtype=jnp.float32, name="fc1")(x))
        x = nn.Dropout(self.drop, deterministic=not train)(x)
        x = nn.Dense(self.nb + self.ns, dtype=jnp.float32, param_dtype=jnp.float32, name="fc2")(x.astype(jnp.float32))
        return x[:, :self.nb], x[:, self.nb:]


class RPMModel(nn.Module):
    arch: str
    nb: int
    ns: int
    head_ch: int = 64
    hidden: int = 512
    drop: float = 0.2
    width_mult: float = 1.0
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool):
        f = ResNetBackbone(arch=self.arch, width_mult=self.width_mult, dtype=self.dtype, name="backbone")(x, train)
        return RPMHead(self.nb, self.ns, self.head_ch, self.hidden, self.drop, self.dtype, name="head")(f, train)


MEAN = jnp.array([0.485, 0.456, 0.406], jnp.float32).reshape(1, 1, 1, 3)
STD = jnp.array([0.229, 0.224, 0.225], jnp.float32).reshape(1, 1, 1, 3)


# ------------------------------------------------------------------ on-device augmentation
def _reflect(t, L):
    period = 2.0 * (L - 1)
    m = jnp.mod(t, period)
    return jnp.where(m > (L - 1), period - m, m)


def _bilinear(img, ys, xs):
    H, W = img.shape[0], img.shape[1]
    ys = _reflect(ys, H)
    xs = _reflect(xs, W)
    y0f, x0f = jnp.floor(ys), jnp.floor(xs)
    wy, wx = (ys - y0f)[..., None], (xs - x0f)[..., None]
    y0 = jnp.clip(y0f.astype(jnp.int32), 0, H - 1); x0 = jnp.clip(x0f.astype(jnp.int32), 0, W - 1)
    y1 = jnp.clip(y0 + 1, 0, H - 1); x1 = jnp.clip(x0 + 1, 0, W - 1)
    return (img[y0, x0] * (1 - wy) * (1 - wx) + img[y0, x1] * (1 - wy) * wx + img[y1, x0] * wy * (1 - wx) + img[y1, x1] * wy * wx)


def aug_one(img, key, c):
    """img float32 [H,W,3] in [0,1]; same distribution as the torch trainer's `augment`."""
    H, W = img.shape[0], img.shape[1]
    ks = jax.random.split(key, 28)
    u = lambda i, lo=0.0, hi=1.0: jax.random.uniform(ks[i], (), minval=lo, maxval=hi)
    on = (u(0) < 0.85).astype(jnp.float32)
    ang = u(1, -1, 1) * math.radians(c["rot"]) * on
    sc = jnp.exp(u(2, -1, 1) * math.log(1 + c["scale"]) * on)
    asp = jnp.exp(u(3, -1, 1) * c["aspect"] * on)
    tx = u(4, -1, 1) * c["shift"] * on
    ty = u(5, -1, 1) * c["shift"] * on
    sh = u(6, -1, 1) * c["shear"] * on
    cs, sn = jnp.cos(ang), jnp.sin(ang)
    a11, a12, a21, a22 = cs / sc * asp, -sn / sc + sh, sn / sc, cs / sc / asp
    # output pixel centres in normalised coords (align_corners=False) -> input coords
    yy, xx = jnp.meshgrid((2 * jnp.arange(H) + 1) / H - 1, (2 * jnp.arange(W) + 1) / W - 1, indexing="ij")
    xi = a11 * xx + a12 * yy + tx
    yi = a21 * xx + a22 * yy + ty
    xs = ((xi + 1) * W - 1) / 2
    ys = ((yi + 1) * H - 1) / 2
    x = _bilinear(img, ys, xs)
    # photometric
    m = x.mean()
    x = (x - m) * (0.7 + 0.6 * u(7)) + m
    x = x * (0.6 + 0.8 * u(8))
    x = jnp.clip(x, 1e-4, 1.0) ** jnp.exp(u(9, -1, 1) * 0.35)
    gray = x.mean(-1, keepdims=True)
    x = gray + (x - gray) * (0.7 + 0.6 * u(10))
    x = x * (0.92 + 0.16 * jax.random.uniform(ks[11], (1, 1, 3)))
    # blur (5-tap separable gaussian, per-sample sigma), applied with prob p_blur
    sigma = 0.5 + 1.5 * u(12)
    tap = jnp.exp(-0.5 * ((jnp.arange(5) - 2.0) / sigma) ** 2)
    tap = tap / tap.sum()
    xp = jnp.pad(x, ((2, 2), (0, 0), (0, 0)), mode="reflect")
    xb = sum(tap[k] * xp[k:k + H] for k in range(5))
    xp = jnp.pad(xb, ((0, 0), (2, 2), (0, 0)), mode="reflect")
    xb = sum(tap[k] * xp[:, k:k + W] for k in range(5))
    x = jnp.where(u(13) < c["p_blur"], xb, x)
    # noise
    nz = jax.random.normal(ks[14], x.shape) * 0.04 * u(15)
    x = x + nz * (u(16) < c["p_noise"])
    # glare
    cy_, cx_ = u(17), u(18)
    rad = 0.08 + 0.2 * u(19)
    gy, gx = jnp.meshgrid(jnp.linspace(0, 1, H), jnp.linspace(0, 1, W), indexing="ij")
    bump = jnp.exp(-((gx - cx_) ** 2 + (gy - cy_) ** 2) / (2 * rad ** 2))[..., None]
    x = x + bump * (0.2 + 0.4 * u(20)) * (u(21) < c["p_glare"])
    # random erasing
    area = (0.02 + 0.06 * u(22)) * H * W
    ar = jnp.exp(u(23, -1, 1) * 0.8)
    eh = jnp.clip(jnp.sqrt(area * ar), 4, H - 1); ew = jnp.clip(jnp.sqrt(area / ar), 4, W - 1)
    k2 = ks[24:28]
    y0 = jax.random.uniform(k2[0]) * (H - eh); x0 = jax.random.uniform(k2[1]) * (W - ew)
    fill = jax.random.uniform(k2[2])
    iy, ix = jnp.arange(H)[:, None], jnp.arange(W)[None, :]
    box = ((iy >= y0) & (iy < y0 + eh) & (ix >= x0) & (ix < x0 + ew))[..., None]
    x = jnp.where(box & (jax.random.uniform(k2[3]) < c["p_erase"]), fill, x)
    return jnp.clip(x, 0.0, 1.0)


# ------------------------------------------------------------------ training
def run(cfg):
    c = get_cfg(cfg)
    devs = jax.devices()
    nd = len(devs)
    mesh = Mesh(np.array(devs), ("data",))
    sh_b = NamedSharding(mesh, P("data"))
    repl = NamedSharding(mesh, P())
    log("config", json.dumps(c, default=str), "hash", cfg_hash(c), "devices", nd, devs[0].platform)
    base, train, test, X = load_data(c)
    last, span, how, fam, img_idx = build_train_arrays(train)
    T = Targets(span)
    log("span classes", T.span_vals.tolist(), "rare->", T.rare_span, "bins", T.nb, "span sources", {h: int((how == h).sum()) for h in np.unique(how)})
    span_cls = T.span_class(span)
    y = last / span
    valid_y = (~np.isnan(span)) & (y >= 0) & (y <= 1.25)
    y_t = np.where(valid_y, y, np.nan)
    N = len(train)
    if c["val_mode"] == "split":
        tr_idx, va_idx = make_val_split(N, fam, c["n_val"], seed=0)
    else:
        tr_idx, va_idx = np.arange(N), np.array([], dtype=int)
    if c["max_train"]:
        tr_idx = tr_idx[: c["max_train"]]
    log("train rows", len(tr_idx), "val rows", len(va_idx))
    Xtr_np = resize_np(np.stack([X[img_idx[i]] for i in tr_idx]), c["res"])
    soft_np = T.soft_labels(y_t[tr_idx], c["sigma"])
    vy_np = valid_y[tr_idx].astype(np.float32)
    sc_np = span_cls[tr_idx].astype(np.int32)
    ytt_np = np.nan_to_num(y_t[tr_idx]).astype(np.float32)
    Xtr = jax.device_put(Xtr_np, repl)
    soft = jax.device_put(soft_np, repl); vy = jax.device_put(vy_np, repl)
    scl = jax.device_put(sc_np, repl); ytt = jax.device_put(ytt_np, repl)
    centers = jax.device_put(T.centers.astype(np.float32), repl)
    log("device data", Xtr.shape, Xtr.dtype)

    dtype = jnp.bfloat16 if c["bf16"] else jnp.float32
    base_arch, _, mult = c["arch"].partition("x")          # "resnet18x0.5" -> resnet18 with width multiplier 0.5
    wm = float(mult) if mult else c["width_mult"]
    c["_base_arch"], c["_width_mult"] = base_arch, wm
    model = RPMModel(base_arch, T.nb, T.ns, c["head_ch"], c["hidden"], c["drop"], wm, dtype)
    rng = jax.random.PRNGKey(c["seed"])
    init_vars = model.init({"params": rng, "dropout": rng}, jnp.zeros((2, c["res"], c["res"], 3), jnp.float32), True)
    params, stats = init_vars["params"], init_vars["batch_stats"]
    if str(c["pretrained"]).startswith("npz:"):
        f = glob.glob(f"/kaggle/input/**/{c['pretrained'][4:]}_imagenet_tv.npz", recursive=True)
        sd = dict(np.load(f[0]))
        bp, bs = convert_tv_to_flax(sd, base_arch)
        params = dict(params); stats = dict(stats)
        params["backbone"] = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), bp)
        stats["backbone"] = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), bs)
        log("loaded pretrained", f[0])
    nparams = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))
    log("params (M)", nparams / 1e6)
    steps_per_epoch = len(tr_idx) // c["bs"]
    total = steps_per_epoch * c["epochs"]
    warm = steps_per_epoch * c["warmup_epochs"]

    def sched(base_lr):
        return optax.warmup_cosine_decay_schedule(0.0, base_lr, warm, total, end_value=base_lr * 1e-3)

    labels = {k: ("backbone" if k == "backbone" else "head") for k in params.keys()}
    tx = optax.multi_transform({"backbone": optax.adamw(sched(c["lr"]), weight_decay=c["wd"]),
                                "head": optax.adamw(sched(c["lr"] * c["head_lr_mult"]), weight_decay=c["wd"])}, labels)
    opt_state = tx.init(params)
    ema_p, ema_s = params, stats
    params, stats, opt_state, ema_p, ema_s = jax.device_put((params, stats, opt_state, ema_p, ema_s), repl)

    def loss_fn(p, s, xb, soft_b, v_b, sc_b, y_b, key):
        out, upd = model.apply({"params": p, "batch_stats": s}, xb, True, rngs={"dropout": key}, mutable=["batch_stats"])
        zb, zs = out
        logp = jax.nn.log_softmax(zb, -1)
        l_bins = -(soft_b * logp).sum(-1)
        exp_y = (jnp.exp(logp) * centers).sum(-1)
        l_l1 = jnp.abs(exp_y - y_b)
        sc_safe = jnp.maximum(sc_b, 0)
        l_span = optax.softmax_cross_entropy_with_integer_labels(zs, sc_safe) * (sc_b >= 0)
        loss = ((l_bins + c["w_l1"] * l_l1) * v_b).sum() / jnp.maximum(v_b.sum(), 1.0) + c["w_span"] * l_span.mean()
        return loss, upd["batch_stats"]

    @jax.jit
    def train_step(params, stats, opt_state, ema_p, ema_s, Xtr, soft, vy, scl, ytt, idx, key):
        k_aug, k_do = jax.random.split(key)
        xb = Xtr[idx].astype(jnp.float32) / 255.0
        keys = jax.random.split(k_aug, xb.shape[0])
        xb = jax.vmap(lambda im, k: aug_one(im, k, c))(xb, keys)
        xb = (xb - MEAN) / STD
        (loss, new_stats), g = jax.value_and_grad(loss_fn, has_aux=True)(params, stats, xb, soft[idx], vy[idx], scl[idx], ytt[idx], k_do)
        updates, opt_state = tx.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        d = c["ema"]
        ema_p = jax.tree_util.tree_map(lambda e, p_: d * e + (1 - d) * p_, ema_p, params)
        ema_s = jax.tree_util.tree_map(lambda e, s_: d * e + (1 - d) * s_, ema_s, new_stats)
        return params, new_stats, opt_state, ema_p, ema_s, loss

    @jax.jit
    def fwd(p, s, xb_u8):
        xb = (xb_u8.astype(jnp.float32) / 255.0 - MEAN) / STD
        return model.apply({"params": p, "batch_stats": s}, xb, False)

    def predict(p, s, imgs_u8_np, bs=256):
        zb, zs = [], []
        n = len(imgs_u8_np)
        pad = (-n) % (bs)
        if pad:
            imgs_u8_np = np.concatenate([imgs_u8_np, np.repeat(imgs_u8_np[-1:], pad, 0)])
        for i in range(0, len(imgs_u8_np), bs):
            a, b = fwd(p, s, jax.device_put(imgs_u8_np[i:i + bs], sh_b))
            zb.append(np.asarray(a)); zs.append(np.asarray(b))
        return np.concatenate(zb)[:n], np.concatenate(zs)[:n]

    if len(va_idx):
        Xva = resize_np(np.stack([X[img_idx[i]] for i in va_idx]), c["res"])

    def val_eval(p, s):
        zb, zs = predict(p, s, Xva)
        pb = softmax(zb)
        dec = dict(argmax=decode_argmax(pb, T.centers), expect=decode_expect(pb, T.centers), window=decode_window(pb, T.centers, c["half_width_bins"]))
        cls = zs.argmax(-1)
        span_hat = T.class_to_span(cls)
        known = ~np.isnan(span[va_idx])
        out = {}
        for name, yh in dec.items():
            hits = official_hits(yh * span_hat, last[va_idx], np.where(known, span[va_idx], 1.0))
            out[name] = float(hits[known].mean())
        out["span_cls_acc"] = float((cls[known] == span_cls[va_idx][known]).mean())
        hw = official_hits(dec["window"] * span_hat, last[va_idx], np.where(known, span[va_idx], 1.0))
        out["by_family"] = {str(f)[:40]: [int(((fam[va_idx] == f) & known).sum()), float(hw[(fam[va_idx] == f) & known].mean()) if ((fam[va_idx] == f) & known).any() else None] for f in np.unique(fam[va_idx])}
        hits_o = official_hits(dec["window"] * np.nan_to_num(span[va_idx], nan=1.0), last[va_idx], np.where(known, span[va_idx], 1.0))
        out["window_oracle_span"] = float(hits_o[known].mean())
        return out

    hist, best, step = [], {"acc": -1, "epoch": -1}, 0
    key0 = jax.random.PRNGKey(c["seed"] + 1)
    nprs = np.random.RandomState(c["seed"])
    t_train = time.time()
    for ep in range(c["epochs"]):
        perm = nprs.permutation(len(tr_idx)).astype(np.int32)
        t_ep = time.time(); losses = []
        for bi in range(steps_per_epoch):
            idx = jax.device_put(perm[bi * c["bs"]:(bi + 1) * c["bs"]], sh_b)
            params, stats, opt_state, ema_p, ema_s, loss = train_step(params, stats, opt_state, ema_p, ema_s, Xtr, soft, vy, scl, ytt, idx,
                                                                      jax.random.fold_in(key0, step))
            losses.append(loss); step += 1
        loss_v = float(jnp.mean(jnp.stack(losses)))
        rec = {"epoch": ep + 1, "loss": loss_v, "secs": round(time.time() - t_ep, 2)}
        if len(va_idx) and ((ep + 1) % c["eval_every"] == 0 or ep + 1 == c["epochs"]):
            rec["val_ema"] = val_eval(ema_p, ema_s)
            rec["val_raw"] = val_eval(params, stats)
            a = rec["val_ema"][c["decoder"]]
            if a > best["acc"]:
                best = {"acc": a, "epoch": ep + 1}
        hist.append(rec)
        if (ep + 1) % max(1, c["eval_every"]) == 0 or ep < 2:
            log(json.dumps(rec))
    train_secs = time.time() - t_train
    res = {"config": c, "config_hash": cfg_hash(c), "params": nparams, "history": hist, "best_val": best, "train_secs": train_secs,
           "span_vals": T.span_vals.tolist(), "rare_span": T.rare_span, "devices": nd}
    if c["save_weights"]:
        flat = {}
        for pre, tree in (("p", ema_p), ("s", ema_s)):
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
                flat[pre + "/" + "/".join(str(getattr(k, "key", k)) for k in path)] = np.asarray(leaf)
        np.savez(os.path.join(c["out_dir"], f"{c['tag']}_ema.npz"), **flat)
    if c["run_test"]:
        log("RUNNING TEST (frozen config)")
        tlast = np.array([last_num(it["reading"]) for it in test])
        trng = np.array([float(it["range"]) for it in test])          # scorer-only
        tmt = np.array([it["meter_type"] for it in test])               # report-only
        timg = np.array([it["img_idx"] for it in test])
        Xte = resize_np(np.stack([X[j] for j in timg]), c["res"])
        zb, zs = predict(ema_p, ema_s, Xte)
        pb = softmax(zb)
        dec = dict(argmax=decode_argmax(pb, T.centers), expect=decode_expect(pb, T.centers), window=decode_window(pb, T.centers, c["half_width_bins"]))
        cls = zs.argmax(-1)
        span_hat = T.class_to_span(cls)
        yh = dec[c["decoder"]]
        pred = yh * span_hat
        hits = official_hits(pred, tlast, trng)
        m = dialbench_metrics(pred, tlast, trng)
        lo, hi = wilson_interval(int(hits.sum()), len(hits))
        groups = thumb_groups(np.stack([X[j] for j in timg]), 0.98)
        glo, ghi = bootstrap_acc_ci(hits, 2000, 0, groups)
        per_type = {t: {"n": int((tmt == t).sum()), "acc_eps": float(hits[tmt == t].mean())} for t in np.unique(tmt)}
        res["test"] = {"metrics": m, "wilson95": [lo, hi], "grouped_bootstrap95": [glo, ghi], "n_test_groups": int(groups.max() + 1),
                       "per_meter_type": per_type, "decoders": {k: float(official_hits(v * span_hat, tlast, trng).mean()) for k, v in dec.items()},
                       "span_cls_hist": np.bincount(cls, minlength=T.ns).tolist(),
                       "by_train_similarity": accuracy_by_similarity(hits, nn_similarity(X, img_idx, timg))}
        np.savez(os.path.join(c["out_dir"], f"{c['tag']}_test_pred.npz"), pred=pred, hits=hits, span_hat=span_hat)
        log("TEST", json.dumps(res["test"]))
    json.dump(res, open(os.path.join(c["out_dir"], f"{c['tag']}_result.json"), "w"), indent=1, default=str)
    return res
