# Post-training int8 recipe sweep for trained RPM models, run INSIDE a private bench-gauges-* CPU kernel (FROZEN-3).
# For each model: export fp32 ONNX from the saved EMA weights, build every recipe of RECIPES, score each on the model's VALIDATION part only,
# apply the pre-registered selection rule, and score ONLY the selected recipe once on the test part.
#   selection rule: among recipes whose validation Acc_eps is within TOL_POINTS of the fp32 ONNX validation Acc_eps, take the smallest file;
#   ties -> higher validation accuracy -> earlier in RECIPES. If no recipe qualifies, nothing is scored on test (the best validation numbers are reported for information).
# Needs dialbench_metrics.py, rpm_common.py, rpm_train_torch.py before this part.
import os, sys, json, time, glob, traceback
import numpy as np
import torch
from dialbench_metrics import wilson_interval
from rpm_common import *  # noqa: F401,F403
from rpm_train_torch import RPMNet, get_cfg, resize_u8, prepare, MEAN, STD

TOL_POINTS = 1.0   # accepted loss in validation Acc_eps, in percentage points, versus the fp32 ONNX

# name, kind, calibration, op types (None = everything the quantizer supports), activation type, extra options, exclusions, pre-process first
RECIPES = [
    dict(name="ref_minmax_all_u8", kind="static", calib="minmax", ops=None, act="QUInt8"),                                       # the run-1 recipe
    dict(name="pct9999_all_u8", kind="static", calib="percentile", pct=99.99, ops=None, act="QUInt8"),
    dict(name="pct999_all_u8", kind="static", calib="percentile", pct=99.9, ops=None, act="QUInt8"),
    dict(name="entropy_all_u8", kind="static", calib="entropy", ops=None, act="QUInt8"),
    dict(name="pct9999_conv_gemm_u8", kind="static", calib="percentile", pct=99.99, ops=["Conv", "Gemm", "MatMul"], act="QUInt8"),
    dict(name="pct9999_no_dw_u8", kind="static", calib="percentile", pct=99.99, ops=["Conv", "Gemm", "MatMul"], act="QUInt8", exclude=["depthwise"]),
    dict(name="pct9999_no_dw_no_se_u8", kind="static", calib="percentile", pct=99.99, ops=["Conv", "Gemm", "MatMul"], act="QUInt8", exclude=["depthwise", "se"]),
    dict(name="pct9999_all_s8sym", kind="static", calib="percentile", pct=99.99, ops=None, act="QInt8", extra={"ActivationSymmetric": True}),
    dict(name="dyn_gemm_s8", kind="dynamic", ops=["Gemm", "MatMul"]),                                                           # weight int8 + per-run activation scale, FC layers only
    dict(name="pre_pct9999_all_u8", kind="static", calib="percentile", pct=99.99, ops=None, act="QUInt8", preprocess=True),
    dict(name="pre_pct9999_no_dw_no_se_u8", kind="static", calib="percentile", pct=99.99, ops=["Conv", "Gemm", "MatMul"], act="QUInt8", exclude=["depthwise", "se"], preprocess=True),
]


def export_fp32(net, c, path):
    class Wrap(torch.nn.Module):  # identical to rpm_export.export_and_eval: normalisation inside the graph, input = float image in [0,255], NCHW
        def __init__(s, m):
            super().__init__(); s.m = m
            s.register_buffer("mean", MEAN * 255.0); s.register_buffer("std", STD * 255.0)

        def forward(s, x):
            return torch.cat(s.m((x - s.mean) / s.std), dim=1)
    torch.onnx.export(Wrap(net).eval(), torch.zeros(1, 3, c["res"], c["res"]), path, input_names=["x"], output_names=["y"], opset_version=17, dynamo=False,
                      dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}})


def node_groups(path):
    """names of depthwise Conv nodes (group > 1) and of squeeze-excite nodes (name contains '/se/' or 'se.')"""
    import onnx
    m = onnx.load(path)
    dw, se = [], []
    for n in m.graph.node:
        if n.op_type == "Conv":
            g = next((a.i for a in n.attribute if a.name == "group"), 1)
            if g > 1:
                dw.append(n.name)
        if n.name and ("/se/" in n.name or ".se." in n.name or "/se." in n.name):
            se.append(n.name)
    return dw, se


def build_recipe(fp32_path, out_path, rec, calib_reader_factory, groups, work):
    from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantFormat, QuantType, CalibrationMethod
    src = fp32_path
    if rec.get("preprocess"):
        from onnxruntime.quantization.shape_inference import quant_pre_process
        src = os.path.join(work, "pre_" + os.path.basename(fp32_path))
        quant_pre_process(fp32_path, src)
        dw, se = node_groups(src)    # node names can change in pre-processing
        groups = (dw, se)
    if rec["kind"] == "dynamic":
        quantize_dynamic(src, out_path, weight_type=QuantType.QInt8, op_types_to_quantize=rec["ops"])
        return
    extra = {"QuantizeBias": False}
    extra.update(rec.get("extra", {}))
    if rec["calib"] == "percentile":
        extra["CalibPercentile"] = rec["pct"]
    cm = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy, "percentile": CalibrationMethod.Percentile}[rec["calib"]]
    kw = dict(quant_format=QuantFormat.QDQ, activation_type=getattr(QuantType, rec["act"]), weight_type=QuantType.QInt8, per_channel=True, calibrate_method=cm, extra_options=extra)
    if rec.get("ops"):
        kw["op_types_to_quantize"] = rec["ops"]
    excl = []
    for g in rec.get("exclude", []):
        excl += groups[0] if g == "depthwise" else groups[1]
    if excl:
        kw["nodes_to_exclude"] = sorted(set(excl))
    quantize_static(src, out_path, calib_reader_factory(), **kw)


def run_model(label, c, weights_path, out_dir, n_calib=256, do_test=True, recipes=None):
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader
    c = get_cfg(c)
    D = prepare(c)
    X, last, span, img_idx, T = D.X, D.last, D.span, D.img_idx, D.T
    net = RPMNet(dict(c, pretrained="none"), T.nb, T.ns).eval()
    sd = torch.load(weights_path, map_location="cpu")
    miss = net.load_state_dict(sd, strict=False)
    print(label, "load", "missing", miss.missing_keys, "unexpected", miss.unexpected_keys, flush=True)
    work = os.path.join(out_dir, label)
    os.makedirs(work, exist_ok=True)
    p32 = os.path.join(work, f"{label}_fp32.onnx")
    export_fp32(net, c, p32)
    rs = np.random.RandomState(0)
    calib_rows = rs.choice(D.tr_idx, min(n_calib, len(D.tr_idx)), replace=False)   # images of the model's own TRAIN part only (no validation or test image, no labels)

    class Rd(CalibrationDataReader):
        def __init__(s):
            s.it = iter([{"x": resize_u8(torch.from_numpy(np.stack([X[img_idx[i]]])), c["res"]).permute(0, 3, 1, 2).float().numpy()} for i in calib_rows])

        def get_next(s):
            return next(s.it, None)

    def logits(path, idxs, threads=4):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        outs = []
        for i in range(0, len(idxs), 64):
            xb = resize_u8(torch.from_numpy(np.stack([X[j] for j in idxs[i:i + 64]])), c["res"]).permute(0, 3, 1, 2).float().numpy()
            outs.append(s.run(None, {"x": xb})[0])
        z = np.concatenate(outs)
        return z[:, :T.nb], z[:, T.nb:]

    def hits_of(path, idxs, last_, rng_):
        zb, zs = logits(path, idxs)
        y = decode_window(softmax(zb), T.centers, c["half_width_bins"])
        return official_hits(y * T.class_to_span(zs.argmax(-1)), last_, rng_)

    def lat(path, n=100):
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        x = np.random.rand(1, 3, c["res"], c["res"]).astype(np.float32) * 255.0
        for _ in range(15):
            s.run(None, {"x": x})
        ts = []
        for _ in range(n):
            t = time.perf_counter(); s.run(None, {"x": x}); ts.append((time.perf_counter() - t) * 1e3)
        ts.sort()
        return {"median_ms": ts[len(ts) // 2], "p95_ms": ts[int(0.95 * len(ts)) - 1], "min_ms": ts[0]}

    va = D.va_idx
    known = ~np.isnan(span[va])
    vlast, vrng = last[va], np.where(known, span[va], 1.0)
    vimg = img_idx[va]
    fp32_val = float(hits_of(p32, vimg, vlast, vrng)[known].mean())
    res = {"label": label, "data_mode": D.mode, "n_val": int(known.sum()), "fp32_onnx_MB": os.path.getsize(p32) / 1e6, "fp32_val": fp32_val, "fp32_latency": lat(p32), "recipes": {}}
    print(label, "fp32 onnx val", fp32_val, "MB", res["fp32_onnx_MB"], flush=True)
    groups = node_groups(p32)
    print(label, "depthwise convs", len(groups[0]), "se nodes", len(groups[1]), flush=True)
    for rec in (recipes or RECIPES):
        out = os.path.join(work, f"{label}_{rec['name']}.onnx")
        r = {"kind": rec["kind"], "recipe": {k: v for k, v in rec.items() if k != "name"}}
        try:
            t0 = time.time()
            build_recipe(p32, out, rec, Rd, groups, work)
            r["build_s"] = round(time.time() - t0, 1)
            r["MB"] = os.path.getsize(out) / 1e6
            r["val"] = float(hits_of(out, vimg, vlast, vrng)[known].mean())
            r["latency"] = lat(out)
        except Exception:
            traceback.print_exc(); sys.stdout.flush()
            r["error"] = traceback.format_exc()[-300:]
        res["recipes"][rec["name"]] = r
        print(label, rec["name"], json.dumps({k: r.get(k) for k in ("MB", "val", "build_s", "error")}), flush=True)
    ok = {n: r for n, r in res["recipes"].items() if "val" in r and r["val"] >= fp32_val - TOL_POINTS / 100.0}
    order = {rec["name"]: i for i, rec in enumerate(recipes or RECIPES)}
    sel = min(ok, key=lambda n: (round(ok[n]["MB"], 3), -ok[n]["val"], order[n])) if ok else None
    res["selection_rule"] = f"smallest file among recipes within {TOL_POINTS} point of fp32 ONNX validation Acc_eps; ties higher validation accuracy, then list order"
    res["selected"] = sel
    if sel is None:
        good = [n for n, r in res["recipes"].items() if "val" in r]
        res["best_validation_only"] = max(good, key=lambda n: res["recipes"][n]["val"]) if good else None
    if sel is not None and do_test:
        tt = D.test
        k = tt.known
        h = hits_of(os.path.join(work, f"{label}_{sel}.onnx"), tt.img_idx, tt.last, np.where(k, tt.rng, 1.0))[k]
        lo, hi = wilson_interval(int(h.sum()), len(h))
        res["test"] = {"recipe": sel, "acc_eps": float(h.mean()), "wilson95": [lo, hi], "n": int(len(h))}
        print(label, "TEST selected", sel, json.dumps(res["test"]), flush=True)
    # keep only the fp32 file and the selected int8 file (the others are not needed after the sweep)
    for rec in (recipes or RECIPES):
        f = os.path.join(work, f"{label}_{rec['name']}.onnx")
        if os.path.exists(f) and rec["name"] != sel:
            os.remove(f)
    for f in glob.glob(os.path.join(work, "pre_*.onnx")):
        os.remove(f)
    json.dump(res, open(os.path.join(out_dir, f"{label}_int8_sweep.json"), "w"), indent=1, default=str)
    return res
