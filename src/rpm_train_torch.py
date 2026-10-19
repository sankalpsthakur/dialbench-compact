# PyTorch trainer/evaluator for RPM-10K (DialBench). Runs inside private bench-gauges-* Kaggle kernels (CPU now, GPU later).
# Kernels embed the src modules (tools/build_kernel.py); locally put src/ on sys.path.
import os, sys, json, time, math, glob, hashlib, copy, traceback
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dialbench_metrics import dialbench_metrics, wilson_interval, bootstrap_acc_ci
from rpm_common import *  # noqa: F401,F403  (label parsing, Targets, decoders, splits, similarity helpers)

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


DEFAULTS = dict(
    arch="resnet18", pretrained="npz:resnet18", res=224, head_ch=64, hidden=512, drop=0.2,
    epochs=10, bs=64, lr=5e-4, head_lr_mult=3.0, wd=0.02, warmup_epochs=1, ema=0.99,
    sigma=0.004, w_l1=1.0, w_span=0.3, seed=0, threads=4, amp=False,
    rot=12.0, scale=0.15, aspect=0.07, shift=0.06, shear=0.06, p_blur=0.3, p_noise=0.4, p_glare=0.2, p_erase=0.15,
    n_val=1000, val_mode="split",   # "split": train on train minus val; "full": train on all 8730, no val
    eval_every=1, run_test=False, decoder="window", half_width_bins=4,
    out_dir="/kaggle/working", tag="run", max_train=None, save_weights=True, input_dir=None, subset_mode=None, init_from=None,
    split_file=None, split_sha256=None,   # split_file: manifest of image names {"train","val","test"} (rpm_split3.py); overrides val_mode/n_val
)


def get_cfg(cfg):
    c = dict(DEFAULTS)
    c.update(cfg)
    return c


def cfg_hash(c):
    keep = {k: v for k, v in c.items() if k not in ("out_dir", "input_dir")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ data
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


def _resolve_split_file(path):
    if os.path.exists(path):
        return path
    hits = sorted(glob.glob("/kaggle/input/**/" + os.path.basename(path), recursive=True))
    if not hits:
        raise FileNotFoundError(path)
    return hits[0]


def prepare(c):
    """The single place that turns a config into the arrays run() and export_and_eval() use.
    default (split_file None): train.json rows (minus the image that is also in test.json), a stratified validation split carved from them,
      test.json rows scored on their own numeric `range`. Identical to the logic that used to be inline in run().
    split_file: rows of BOTH label files pooled by image name and assigned to train/val/test by the manifest (names). Span classes (Targets) are built from the
      manifest's train part only. A test row keeps its own scorer range: its numeric `range` (test.json rows), the "0-3" style string or the range stated in
      its chain of thought (train.json rows); rows with no determinable range are excluded from scoring and counted."""
    if not c.get("split_file"):
        base, train, test, X = load_data(c)
        last, span, how, fam, img_idx = build_train_arrays(train)
        T = Targets(span)
        N = len(train)
        if c["val_mode"] == "split":
            tr_idx, va_idx = make_val_split(N, fam, c["n_val"], seed=0)
        else:
            tr_idx, va_idx = np.arange(N), np.array([], dtype=int)
        if c["max_train"]:
            tr_idx = tr_idx[: c["max_train"]]
        t = SimpleNamespace(last=np.array([last_num(it["reading"]) for it in test]), rng=np.array([float(it["range"]) for it in test]),
                            known=np.ones(len(test), bool), img_idx=np.array([it["img_idx"] for it in test]),
                            meter_type=np.array([it["meter_type"] for it in test]), env=np.array([it["environment_conditions"] for it in test]),
                            src=np.array(["test.json"] * len(test)), span_src=np.array(["range_numeric"] * len(test)))
        return SimpleNamespace(mode="official", base=base, X=X, last=last, span=span, how=how, fam=fam, img_idx=img_idx, T=T, tr_idx=tr_idx, va_idx=va_idx,
                               test=t, sim_ref=img_idx, calib_pool=np.arange(N), info={})
    base, train_raw, test_raw, X = load_data(dict(c, drop_test_dups=False))
    path = _resolve_split_file(c["split_file"])
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if c.get("split_sha256") and sha != c["split_sha256"]:
        raise RuntimeError(f"split file sha256 {sha} differs from the pre-registered {c['split_sha256']}")
    man = json.loads(raw)
    parts, missing = rows_from_manifest(man, pool_rows(train_raw, test_raw))
    rows = parts["train"] + parts["val"]
    ntr, nva = len(parts["train"]), len(parts["val"])
    last = np.array([last_num(it["reading"]) for it in rows])
    sp = [row_span(it) for it in rows]
    span, how = np.array([x[0] for x in sp]), np.array([x[1] for x in sp])
    fam = np.array([row_family(it) for it in rows])
    img_idx = np.array([it["img_idx"] for it in rows])
    T = Targets(span[:ntr])
    tr_idx, va_idx = np.arange(ntr), ntr + np.arange(nva)
    if c["max_train"]:
        tr_idx = tr_idx[: c["max_train"]]
    trows = parts["test"]
    tsp = [row_span(it) for it in trows]
    trng = np.array([x[0] for x in tsp], dtype=np.float64)
    t = SimpleNamespace(last=np.array([last_num(it["reading"]) for it in trows]), rng=trng, known=~np.isnan(trng), img_idx=np.array([it["img_idx"] for it in trows]),
                        meter_type=np.array([it.get("meter_type", "train.json row") for it in trows]), env=np.array([it.get("environment_conditions", "") for it in trows]),
                        src=np.array(["test.json" if "meter_type" in it else "train.json" for it in trows]), span_src=np.array([x[1] for x in tsp]))
    info = dict(split_file=os.path.basename(path), split_sha256=sha, n_names={k: len(man[k]) for k in ("train", "val", "test")}, n_rows={k: len(v) for k, v in parts.items()},
                names_without_label_row=missing, n_test_known_range=int(t.known.sum()), n_test_unknown_range=int((~t.known).sum()),
                n_test_from_test_json=int((t.src == "test.json").sum()), B=man.get("B"), seed=man.get("seed"), tau_twin=man.get("tau_twin"),
                manifest_baselines=man.get("baselines"))
    return SimpleNamespace(mode="split_file", base=base, X=X, last=last, span=span, how=how, fam=fam, img_idx=img_idx, T=T, tr_idx=tr_idx, va_idx=va_idx,
                           test=t, sim_ref=img_idx[tr_idx], calib_pool=tr_idx, info=info)


def thumb_groups(Xsub, tau=0.98):
    """union-find groups on 28x28 gray thumbnails (cosine >= tau); for grouped bootstrap only."""
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


# ------------------------------------------------------------------ model
RESNET_LAYERS = {"resnet10": [1, 1, 1, 1], "resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3]}


def make_backbone(arch, pretrained=False):
    """timm backbone returning the final feature map. 'resnet18', 'resnet10', 'resnet18x0.5' (width multiplier) are built with
    timm's ResNet class (same parameter names as the Flax port in jax_resnet.py); anything else goes through timm.create_model."""
    import timm
    base, _, mult = arch.partition("x")
    if base in RESNET_LAYERS and (mult or base == "resnet10"):
        from timm.models.resnet import ResNet, BasicBlock
        m = float(mult) if mult else 1.0
        ch = [int(round(w * m)) for w in (64, 128, 256, 512)]
        return ResNet(BasicBlock, RESNET_LAYERS[base], channels=ch, stem_width=ch[0], num_classes=0, global_pool="")
    return timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="")


class RPMNet(nn.Module):
    def __init__(self, c, nb, ns):
        super().__init__()
        self.c = c
        self.backbone = make_backbone(c["arch"], pretrained=(c["pretrained"] == "timm"))
        if str(c["pretrained"]).startswith("npz:"):
            f = glob.glob(f"/kaggle/input/**/{c['pretrained'][4:]}_imagenet_tv.npz", recursive=True)
            sd = {k: torch.from_numpy(v) for k, v in np.load(f[0]).items()}
            miss = self.backbone.load_state_dict(sd, strict=False)
            log("loaded", f[0], "missing", miss.missing_keys, "unexpected", miss.unexpected_keys)
        with torch.no_grad():
            self.backbone.eval()
            o = self.backbone(torch.zeros(1, 3, c["res"], c["res"]))
        ch, h, w = o.shape[1:]
        self.reduce = nn.Sequential(nn.Conv2d(ch, c["head_ch"], 1, bias=False), nn.BatchNorm2d(c["head_ch"]), nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c["head_ch"] * h * w, c["hidden"]), nn.ReLU(inplace=True), nn.Dropout(c["drop"]),
                                nn.Linear(c["hidden"], nb + ns))
        self.nb, self.ns = nb, ns

    def forward(self, x):
        z = self.fc(self.reduce(self.backbone(x)))
        return z[:, :self.nb], z[:, self.nb:]


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ------------------------------------------------------------------ augmentation (batched, torch)
def gauss_kernel(sigma, k=5):
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    return g / g.sum()


def augment(x, c):
    """Batched, device-aware augmentation (x float [B,3,H,W] in [0,1]). Blur/noise/glare/erase are applied only to the selected subset."""
    import random
    dev = x.device
    B, _, H, W = x.shape
    r = lambda *s: torch.rand(*s, device=dev)
    on = (r(B) < 0.85).float()
    ang = (r(B) * 2 - 1) * math.radians(c["rot"]) * on
    sc = torch.exp((r(B) * 2 - 1) * math.log(1 + c["scale"]) * on)
    asp = torch.exp((r(B) * 2 - 1) * c["aspect"] * on)
    tx = (r(B) * 2 - 1) * c["shift"] * on
    ty = (r(B) * 2 - 1) * c["shift"] * on
    sh = (r(B) * 2 - 1) * c["shear"] * on
    cos, sin = ang.cos(), ang.sin()
    theta = torch.stack([torch.stack([cos / sc * asp, -sin / sc + sh, tx], 1), torch.stack([sin / sc, cos / sc / asp, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # photometric (whole batch)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * (0.7 + 0.6 * r(B, 1, 1, 1)) + m
    x = x * (0.6 + 0.8 * r(B, 1, 1, 1))
    x = x.clamp(1e-4, 1) ** torch.exp((r(B, 1, 1, 1) * 2 - 1) * 0.35)
    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * (0.7 + 0.6 * r(B, 1, 1, 1))
    x = x * (0.92 + 0.16 * r(B, 3, 1, 1))
    pick = lambda p: (r(B) < p).nonzero().flatten()
    # blur on a subset
    if c["p_blur"] > 0:
        idx = pick(c["p_blur"])
        if len(idx):
            k = gauss_kernel(0.5 + 1.5 * random.random()).to(dev)
            xb = x[idx]
            xb = F.conv2d(F.pad(xb, (2, 2, 0, 0), mode="reflect"), k.view(1, 1, 1, 5).repeat(3, 1, 1, 1), groups=3)
            xb = F.conv2d(F.pad(xb, (0, 0, 2, 2), mode="reflect"), k.view(1, 1, 5, 1).repeat(3, 1, 1, 1), groups=3)
            x[idx] = xb
    # noise on a subset
    if c["p_noise"] > 0:
        idx = pick(c["p_noise"])
        if len(idx):
            x[idx] = x[idx] + torch.randn(len(idx), 3, H, W, device=dev) * 0.04 * r(len(idx), 1, 1, 1)
    # glare on a subset
    if c["p_glare"] > 0:
        idx = pick(c["p_glare"])
        if len(idx):
            yy, xx = torch.meshgrid(torch.linspace(0, 1, H, device=dev), torch.linspace(0, 1, W, device=dev), indexing="ij")
            n = len(idx)
            cx, cy, rad = r(n, 1, 1), r(n, 1, 1), 0.08 + 0.2 * r(n, 1, 1)
            bump = torch.exp(-((xx[None] - cx) ** 2 + (yy[None] - cy) ** 2) / (2 * rad ** 2))
            x[idx] = x[idx] + (bump * (0.2 + 0.4 * r(n, 1, 1))).unsqueeze(1)
    # random erasing on a subset (small; may hide the needle, so kept rare)
    if c["p_erase"] > 0:
        idx = pick(c["p_erase"])
        if len(idx):
            n = len(idx)
            area = (0.02 + 0.06 * r(n)) * H * W
            ar = torch.exp((r(n) * 2 - 1) * 0.8)
            h = torch.sqrt(area * ar).clamp(4, H - 1); w = torch.sqrt(area / ar).clamp(4, W - 1)
            y0 = r(n) * (H - h); x0 = r(n) * (W - w)
            iy = torch.arange(H, device=dev).view(1, H, 1).float(); ix = torch.arange(W, device=dev).view(1, 1, W).float()
            box = ((iy >= y0.view(n, 1, 1)) & (iy < (y0 + h).view(n, 1, 1)) & (ix >= x0.view(n, 1, 1)) & (ix < (x0 + w).view(n, 1, 1))).unsqueeze(1)
            x[idx] = torch.where(box, r(n, 1, 1, 1).expand(n, 3, H, W), x[idx])
    return x.clamp(0, 1)


# ------------------------------------------------------------------ eval helpers
def resize_u8(x, res):
    """uint8 [n,H,W,3] torch -> uint8 [n,res,res,3] (antialiased bilinear); no-op if already res."""
    if x.shape[1] == res:
        return x
    out = []
    for i in range(0, len(x), 256):
        b = x[i:i + 256].permute(0, 3, 1, 2).float()
        b = F.interpolate(b, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
        out.append(b.round().clamp(0, 255).byte().permute(0, 2, 3, 1))
    return torch.cat(out)


@torch.no_grad()
def predict_logits(model, X, idx, c, bs=128):
    model.eval()
    dev = next(model.parameters()).device
    zb, zs = [], []
    for i in range(0, len(idx), bs):
        xb = resize_u8(torch.from_numpy(np.stack([X[j] for j in idx[i:i + bs]])), c["res"])
        xb = xb.permute(0, 3, 1, 2).float() / 255.0
        xb = ((xb - MEAN) / STD).contiguous(memory_format=torch.channels_last).to(dev)
        a, b = model(xb)
        zb.append(a.float().cpu().numpy()); zs.append(b.float().cpu().numpy())
    return np.concatenate(zb), np.concatenate(zs)


def decode_all(zb, zs, T, c):
    p = softmax(zb)
    y_arg = decode_argmax(p, T.centers)
    y_exp = decode_expect(p, T.centers)
    y_win = decode_window(p, T.centers, c["half_width_bins"])
    cls = zs.argmax(-1)
    return dict(argmax=y_arg, expect=y_exp, window=y_win), T.class_to_span(cls), cls


def score(y_hat, span_hat, last, scorer_range, nonlinear_eps=0.01):
    pred = y_hat * span_hat
    hits = official_hits(pred, last, scorer_range)
    return pred, hits


# ------------------------------------------------------------------ main
def run(cfg):
    c = get_cfg(cfg)
    torch.set_num_threads(c["threads"])
    import random
    random.seed(c["seed"]); torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
    os.makedirs(c["out_dir"], exist_ok=True)
    log("config", json.dumps({k: v for k, v in c.items()}, default=str), "hash", cfg_hash(c))
    D = prepare(c)
    X, last, span, how, fam, img_idx, T, tr_idx, va_idx = D.X, D.last, D.span, D.how, D.fam, D.img_idx, D.T, D.tr_idx, D.va_idx
    log("data mode", D.mode, json.dumps(D.info, default=str))
    log("span classes", T.span_vals.tolist(), "rare->", T.rare_span, "bins", T.nb, "span source counts", {h: int((how == h).sum()) for h in np.unique(how)})
    span_cls = T.span_class(span)
    y = last / span
    valid_y = (~np.isnan(span)) & (y >= 0) & (y <= 1.25)
    y_t = np.where(valid_y, y, np.nan)
    log("train rows", len(tr_idx), "val rows", len(va_idx), "valid_y frac in train", float(valid_y[tr_idx].mean()))
    # tensors
    Xtr = resize_u8(torch.from_numpy(np.stack([X[img_idx[i]] for i in tr_idx])), c["res"])  # uint8 [n,res,res,3]
    soft = torch.from_numpy(T.soft_labels(y_t[tr_idx], c["sigma"]))
    vy = torch.from_numpy(valid_y[tr_idx])
    sc_t = torch.from_numpy(span_cls[tr_idx])
    y_tt = torch.from_numpy(np.nan_to_num(y_t[tr_idx]).astype(np.float32))
    centers_t = torch.from_numpy(T.centers.astype(np.float32))
    log("loaded train tensor", tuple(Xtr.shape), Xtr.dtype)

    model = RPMNet(c, T.nb, T.ns).to(memory_format=torch.channels_last)
    if c.get("init_from"):   # continue from a previous run's EMA weights (state dict of RPMNet)
        miss = model.load_state_dict(torch.load(c["init_from"], map_location="cpu"), strict=False)
        log("init_from", c["init_from"], "missing", miss.missing_keys, "unexpected", miss.unexpected_keys)
    nparams = sum(p.numel() for p in model.parameters())
    log("params (M)", nparams / 1e6)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": bb, "lr": c["lr"]}, {"params": hd, "lr": c["lr"] * c["head_lr_mult"]}], weight_decay=c["wd"])
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = len(tr_idx) // c["bs"]
    total = steps_per_epoch * c["epochs"]
    warm = steps_per_epoch * c["warmup_epochs"]
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(c["ema"]), use_buffers=True)
    hist = []
    best = {"acc": -1, "epoch": -1}
    step = 0
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if c["amp"] and torch.cuda.is_available() else (lambda: torch.autocast("cpu", enabled=False))
    dev = "cuda" if (c["amp"] and torch.cuda.is_available()) else "cpu"
    if dev == "cuda":
        model = model.cuda(); ema = ema.cuda(); centers_t = centers_t.cuda(); MEAN_, STD_ = MEAN.cuda(), STD.cuda()
    else:
        MEAN_, STD_ = MEAN, STD
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    def val_eval(m, tag):
        zb, zs = predict_logits(m, X, img_idx[va_idx], c)
        dec, span_hat, cls = decode_all(zb, zs, T, c)
        known = ~np.isnan(span[va_idx])
        out = {}
        for name, yh in dec.items():
            _, hits = score(yh, span_hat, last[va_idx], np.where(known, span[va_idx], 1.0))
            out[name] = float(hits[known].mean())
        out["span_cls_acc"] = float((cls[known] == span_cls[va_idx][known]).mean())
        hw = official_hits(dec["window"] * span_hat, last[va_idx], np.where(known, span[va_idx], 1.0))
        out["by_family"] = {str(f)[:40]: [int(((fam[va_idx] == f) & known).sum()), float(hw[(fam[va_idx] == f) & known].mean()) if ((fam[va_idx] == f) & known).any() else None] for f in np.unique(fam[va_idx])}
        # oracle span (diagnostic): isolates the y head
        _, hits_o = score(dec["window"], span[va_idx], last[va_idx], np.where(known, span[va_idx], 1.0))
        out["window_oracle_span"] = float(np.nan_to_num(hits_o[known].mean()))
        return out

    for ep in range(c["epochs"]):
        model.train()
        perm = torch.randperm(len(tr_idx))
        t_ep = time.time(); run_loss = 0.0
        for bi in range(steps_per_epoch):
            idx = perm[bi * c["bs"]:(bi + 1) * c["bs"]]
            xb = Xtr[idx].permute(0, 3, 1, 2).float() / 255.0
            if dev == "cuda":
                xb = xb.cuda(non_blocking=True)
            xb = augment(xb, c)
            xb = ((xb - MEAN_) / STD_).contiguous(memory_format=torch.channels_last)
            lr_scale = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * max(lr_scale, 1e-3)
            with amp_ctx():
                zb, zs = model(xb)
            zb, zs = zb.float(), zs.float()
            sl = soft[idx].to(zb.device); v = vy[idx].to(zb.device)
            logp = F.log_softmax(zb, -1)
            l_bins = -(sl * logp).sum(-1)
            exp_y = (logp.exp() * centers_t).sum(-1)
            l_l1 = (exp_y - y_tt[idx].to(zb.device)).abs()
            sc = sc_t[idx].to(zb.device)
            l_span = F.cross_entropy(zs, sc.clamp(min=0), reduction="none") * (sc >= 0).float()
            loss = ((l_bins + c["w_l1"] * l_l1) * v.float()).sum() / v.float().sum().clamp(min=1) + c["w_span"] * l_span.mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ema.update_parameters(model)
            run_loss += float(loss); step += 1
        rec = {"epoch": ep + 1, "loss": run_loss / steps_per_epoch, "secs": round(time.time() - t_ep, 1)}
        if len(va_idx) and ((ep + 1) % c["eval_every"] == 0 or ep + 1 == c["epochs"]):
            rec["val_ema"] = val_eval(ema.module, "ema")
            rec["val_raw"] = val_eval(model, "raw")
            a = rec["val_ema"][c["decoder"]]
            if a > best["acc"]:
                best = {"acc": a, "epoch": ep + 1}
        hist.append(rec)
        log(json.dumps(rec))
    res = {"config": c, "config_hash": cfg_hash(c), "params": nparams, "history": hist, "best_val": best,
           "span_vals": T.span_vals.tolist(), "rare_span": T.rare_span}
    if c["save_weights"]:
        torch.save({k: v.cpu() for k, v in ema.module.state_dict().items()}, os.path.join(c["out_dir"], f"{c['tag']}_ema.pt"))
    if c["run_test"]:
        log("RUNNING TEST (frozen config)")
        tt = D.test
        k = tt.known
        zb, zs = predict_logits(ema.module, X, tt.img_idx, c)
        dec, span_hat, cls = decode_all(zb, zs, T, c)
        yh = dec[c["decoder"]]
        rng_eff = np.where(k, tt.rng, 1.0)   # rows without a determinable range are excluded below (only possible in split_file mode)
        pred, hits = score(yh, span_hat, tt.last, rng_eff)
        m = dialbench_metrics(pred[k], tt.last[k], tt.rng[k])
        lo, hi = wilson_interval(int(hits[k].sum()), int(k.sum()))
        groups = thumb_groups(np.stack([X[j] for j in tt.img_idx[k]]), 0.98)
        glo, ghi = bootstrap_acc_ci(hits[k], 2000, 0, groups)
        per_type = {t: {"n": int(((tt.meter_type == t) & k).sum()), "acc_eps": float(hits[(tt.meter_type == t) & k].mean())} for t in np.unique(tt.meter_type[k])}
        res["test"] = {"metrics": m, "wilson95": [lo, hi], "grouped_bootstrap95": [glo, ghi], "n_test_groups": int(groups.max() + 1),
                       "per_meter_type": per_type, "decoders": {kk: float(score(v, span_hat, tt.last, rng_eff)[1][k].mean()) for kk, v in dec.items()},
                       "span_cls_hist": np.bincount(cls, minlength=T.ns).tolist(),
                       "by_train_similarity": accuracy_by_similarity(hits[k], nn_similarity(X, D.sim_ref, tt.img_idx[k]))}
        if D.mode == "split_file":
            try:   # extra v3 reporting; a failure here must not lose the core test result above
                def acc(mask):
                    mm = mask & k
                    return {"n": int(mm.sum()), "acc_eps": float(hits[mm].mean()) if mm.any() else None}
                base_pred = knn_copy_pred(X, img_idx[tr_idx], last[tr_idx], tt.img_idx[k])
                bh = official_hits(base_pred, tt.last[k], tt.rng[k])
                blo, bhi = wilson_interval(int(bh.sum()), len(bh))
                res["test"]["split_file"] = {
                    "info": D.info,
                    "n_test_rows": int(len(k)), "n_scored": int(k.sum()), "n_excluded_unknown_range": int((~k).sum()),
                    "by_source_file": {s_: acc(tt.src == s_) for s_ in np.unique(tt.src)},
                    "by_range_source": {s_: acc(tt.span_src == s_) for s_ in np.unique(tt.span_src)},
                    "by_range_class": {nm: acc(np.isclose(tt.rng, v) if v > 0 else ~np.isin(np.round(np.nan_to_num(tt.rng), 4), [3.0, 6.0, 10.0])) for nm, v in (("range 3", 3.0), ("range 6", 6.0), ("range 10", 10.0), ("other", -1.0))},
                    "knn_copy_baseline": {"n": int(len(bh)), "acc_eps": float(bh.mean()), "wilson95": [blo, bhi], "feature": "min(cos gray56, cos rgb32), nearest image of the split's train part"},
                }
            except Exception:
                traceback.print_exc(); sys.stdout.flush()
                res["test"]["split_file_error"] = traceback.format_exc()[-400:]
        np.savez(os.path.join(c["out_dir"], f"{c['tag']}_test_pred.npz"), pred=pred, hits=hits, span_hat=span_hat, known=k, rng=tt.rng, last=tt.last, img_idx=tt.img_idx)
        log("TEST", json.dumps(res["test"], default=str))
    json.dump(res, open(os.path.join(c["out_dir"], f"{c['tag']}_result.json"), "w"), indent=1, default=str)
    return res
