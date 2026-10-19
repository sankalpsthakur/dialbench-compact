# Edge export + int8 evaluation for RPM-10K models, run INSIDE a private bench-gauges-* CPU kernel (data never leaves Kaggle).
# Needs: dialbench_metrics.py, rpm_common.py, rpm_train_torch.py (for RPMNet/load_data/helpers) concatenated before this part.
# Input: the trained weights of a frozen config (torch .pt state dict or JAX .npz from rpm_train_jax.py).
import os, sys, json, time, glob, subprocess
import numpy as np
import torch
from dialbench_metrics import wilson_interval
from rpm_common import *  # noqa: F401,F403
from jax_resnet import flax_to_tv_state_dict
from rpm_train_torch import RPMNet, load_data, build_train_arrays, get_cfg, resize_u8, predict_logits, prepare, MEAN, STD


def flax_fc1_to_torch(k1, h, w, hc):
    """Flax Dense kernel [h*w*hc, hidden] (flatten of NHWC, order h,w,c) -> torch Linear weight [hidden, hc*h*w] (flatten of NCHW, order c,h,w)."""
    k = np.asarray(k1).reshape(h, w, hc, -1).transpose(2, 0, 1, 3).reshape(hc * h * w, -1)
    return np.ascontiguousarray(k.T)


def jax_npz_to_torch_state(npz_path, c, h, w):
    """Map the Flax RPMModel EMA weights (flattened by rpm_train_jax) onto the torch RPMNet state dict (timm resnet naming)."""
    z = dict(np.load(npz_path))
    P = {k[2:]: v for k, v in z.items() if k.startswith("p/")}
    S = {k[2:]: v for k, v in z.items() if k.startswith("s/")}
    # backbone: rebuild nested dicts and use the tv converter's inverse
    def nest(flat, prefix):
        out = {}
        for k, v in flat.items():
            if not k.startswith(prefix + "/"):
                continue
            parts = k[len(prefix) + 1:].split("/")
            d = out
            for q in parts[:-1]:
                d = d.setdefault(q, {})
            d[parts[-1]] = v
        return out
    bp, bs = nest(P, "backbone"), nest(S, "backbone")
    sd = {"backbone." + k: torch.from_numpy(np.asarray(v)) for k, v in flax_to_tv_state_dict(bp, bs, c["arch"].partition("x")[0]).items()}
    hp = P
    sd["reduce.0.weight"] = torch.from_numpy(np.asarray(hp["head/reduce/kernel"]).transpose(3, 2, 0, 1).copy())
    sd["reduce.1.weight"] = torch.from_numpy(np.asarray(hp["head/reduce_bn/scale"]))
    sd["reduce.1.bias"] = torch.from_numpy(np.asarray(hp["head/reduce_bn/bias"]))
    sd["reduce.1.running_mean"] = torch.from_numpy(np.asarray(S["head/reduce_bn/mean"]))
    sd["reduce.1.running_var"] = torch.from_numpy(np.asarray(S["head/reduce_bn/var"]))
    sd["reduce.1.num_batches_tracked"] = torch.tensor(0)
    hc = c["head_ch"]
    sd["fc.1.weight"] = torch.from_numpy(flax_fc1_to_torch(np.asarray(hp["head/fc1/kernel"]), h, w, hc)); sd["fc.1.bias"] = torch.from_numpy(np.asarray(hp["head/fc1/bias"]))
    sd["fc.4.weight"] = torch.from_numpy(np.asarray(hp["head/fc2/kernel"]).T.copy()); sd["fc.4.bias"] = torch.from_numpy(np.asarray(hp["head/fc2/bias"]))
    return sd


def export_and_eval(c, weights_path, kind, run_test=False, out_dir="/kaggle/working", n_calib=256):
    """kind: 'torch' (.pt) or 'jax' (.npz). Produces fp32 + int8 (QDQ) ONNX, evaluates val (and test if run_test) with ORT."""
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
    c = get_cfg(c)
    D = prepare(c)   # same data logic as rpm_train_torch.run (official split, or a split_file manifest)
    X, last, span, fam, img_idx, T = D.X, D.last, D.span, D.fam, D.img_idx, D.T
    net = RPMNet(dict(c, pretrained="none"), T.nb, T.ns).eval()
    with torch.no_grad():
        o = net.backbone(torch.zeros(1, 3, c["res"], c["res"]))
    h, w = o.shape[2], o.shape[3]
    sd = torch.load(weights_path, map_location="cpu") if kind == "torch" else jax_npz_to_torch_state(weights_path, c, h, w)
    miss = net.load_state_dict(sd, strict=False)
    print("load", "missing", miss.missing_keys, "unexpected", miss.unexpected_keys, flush=True)
    net.eval()
    nparams = sum(p.numel() for p in net.parameters())

    class Wrap(torch.nn.Module):  # normalisation inside the graph: input = uint8-range float image in [0,255], NCHW
        def __init__(s, m):
            super().__init__(); s.m = m
            s.register_buffer("mean", MEAN * 255.0); s.register_buffer("std", STD * 255.0)

        def forward(s, x):
            return torch.cat(s.m((x - s.mean) / s.std), dim=1)

    wrap = Wrap(net).eval()
    d = out_dir
    p32, p8 = os.path.join(d, f"{c['tag']}_fp32.onnx"), os.path.join(d, f"{c['tag']}_int8_qdq.onnx")
    torch.onnx.export(wrap, torch.zeros(1, 3, c["res"], c["res"]), p32, input_names=["x"], output_names=["y"], opset_version=17, dynamo=False,
                      dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}})
    rs = np.random.RandomState(0)
    calib_rows = rs.choice(D.calib_pool, min(n_calib, len(D.calib_pool)), replace=False)       # TRAIN images only (the split's train part in split_file mode)

    class Rd(CalibrationDataReader):
        def __init__(s):
            s.it = iter([{"x": resize_u8(torch.from_numpy(np.stack([X[img_idx[i]]])), c["res"]).permute(0, 3, 1, 2).float().numpy()} for i in calib_rows])

        def get_next(s):
            return next(s.it, None)

    quantize_static(p32, p8, Rd(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, extra_options={"QuantizeBias": False})
    p8b = os.path.join(d, f"{c['tag']}_int8_convonly.onnx")
    try:
        quantize_static(p32, p8b, Rd(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=True, op_types_to_quantize=["Conv", "Gemm", "MatMul"], extra_options={"QuantizeBias": False})
    except Exception as e:
        print("int8_convonly failed", repr(e)[:200], flush=True); p8b = None
    res = {"tag": c["tag"], "params": nparams, "fp32_onnx_MB": os.path.getsize(p32) / 1e6, "int8_onnx_MB": os.path.getsize(p8) / 1e6}
    if p8b:
        res["int8_convonly_onnx_MB"] = os.path.getsize(p8b) / 1e6

    def lat(path, n=120):   # secondary hardware data point: this Kaggle x86 CPU, 1 thread, batch 1
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        x = np.random.rand(1, 3, c["res"], c["res"]).astype(np.float32) * 255.0
        for _ in range(15):
            sess.run(None, {"x": x})
        ts = []
        for _ in range(n):
            t = time.perf_counter(); sess.run(None, {"x": x}); ts.append((time.perf_counter() - t) * 1e3)
        ts.sort()
        return {"median_ms": ts[len(ts) // 2], "p95_ms": ts[int(0.95 * len(ts)) - 1], "min_ms": ts[0]}
    res["kaggle_xeon_1t_fp32"] = lat(p32); res["kaggle_xeon_1t_int8"] = lat(p8)
    if p8b:
        res["kaggle_xeon_1t_int8_convonly"] = lat(p8b)
    print("latency (Kaggle Xeon 2.2GHz, 1 thread, batch 1)", json.dumps({k: res[k] for k in ("kaggle_xeon_1t_fp32", "kaggle_xeon_1t_int8")}), flush=True)

    def ort_logits(path, imgs_idx):
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        outs = []
        for i in range(0, len(imgs_idx), 64):
            xb = resize_u8(torch.from_numpy(np.stack([X[j] for j in imgs_idx[i:i + 64]])), c["res"]).permute(0, 3, 1, 2).float().numpy()
            outs.append(s.run(None, {"x": xb})[0])
        z = np.concatenate(outs)
        return z[:, :T.nb], z[:, T.nb:]

    def torch_logits(imgs_idx):
        return predict_logits(net, X, imgs_idx, c)

    def decode_score(zb, zs, last_, rng_, mask=None):
        pb = softmax(zb)
        y = decode_window(pb, T.centers, c["half_width_bins"])
        sp = T.class_to_span(zs.argmax(-1))
        hits = official_hits(y * sp, last_, rng_)
        return hits

    # validation (train.json held-out split, same as training) ------------------------------------------------
    if len(D.va_idx):
        va_idx = D.va_idx
        known = ~np.isnan(span[va_idx])
        variants_val = [("torch_fp32", lambda: torch_logits(img_idx[va_idx])), ("onnx_fp32", lambda: ort_logits(p32, img_idx[va_idx])),
                        ("onnx_int8", lambda: ort_logits(p8, img_idx[va_idx]))]
        if p8b:
            variants_val.append(("onnx_int8_convonly", lambda: ort_logits(p8b, img_idx[va_idx])))
        for name, fn in variants_val:
            zb, zs = fn()
            hits = decode_score(zb, zs, last[va_idx], np.where(known, span[va_idx], 1.0))
            res[f"val_{name}"] = float(hits[known].mean())
            print("val", name, res[f"val_{name}"], flush=True)
    if run_test:
        tt = D.test; kk = tt.known; timg = tt.img_idx
        tlast, trng = tt.last, np.where(kk, tt.rng, 1.0)   # rows without a determinable range are excluded (split_file mode only)
        variants_test = [("torch_fp32", lambda: torch_logits(timg)), ("onnx_fp32", lambda: ort_logits(p32, timg)), ("onnx_int8", lambda: ort_logits(p8, timg))]
        if p8b:
            variants_test.append(("onnx_int8_convonly", lambda: ort_logits(p8b, timg)))
        for name, fn in variants_test:
            zb, zs = fn()
            hits = decode_score(zb, zs, tlast, trng)[kk]
            lo, hi = wilson_interval(int(hits.sum()), len(hits))
            res[f"test_{name}"] = {"acc_eps": float(hits.mean()), "wilson95": [lo, hi], "n": int(len(hits))}
            print("TEST", name, res[f"test_{name}"], flush=True)
    res["data_mode"] = D.mode; res["split_info"] = D.info
    json.dump(res, open(os.path.join(d, f"{c['tag']}_export_result.json"), "w"), indent=1, default=str)
    return res
