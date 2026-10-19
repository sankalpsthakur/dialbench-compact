#!/usr/bin/env python3
"""Edge numbers for an architecture: params, fp32/int8 ONNX size, latency at 1 thread, batch 1 on this Mac (CPU).
Latency depends on architecture/input size only (not on weight values), so random weights are fine here.
usage: latency.py arch res [arch res ...]   (timm names). Output: JSON lines + a table."""
import os, sys, time, json, tempfile, subprocess, statistics
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, timm
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

torch.set_num_threads(1)
N_WARM, N_RUN = 30, 300


class RandReader(CalibrationDataReader):
    def __init__(self, name, shape, n=8):
        self.it = iter([{name: np.random.RandomState(i).rand(*shape).astype(np.float32)} for i in range(n)])

    def get_next(self):
        return next(self.it, None)


def bench_ort(path, shape, name):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    x = np.random.rand(*shape).astype(np.float32)
    for _ in range(N_WARM):
        s.run(None, {name: x})
    ts = []
    for _ in range(N_RUN):
        t = time.perf_counter(); s.run(None, {name: x}); ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return {"median_ms": round(statistics.median(ts), 3), "p95_ms": round(ts[int(0.95 * len(ts)) - 1], 3), "min_ms": round(ts[0], 3)}


def bench_torch(m, shape):
    x = torch.rand(*shape)
    with torch.no_grad():
        for _ in range(N_WARM):
            m(x)
        ts = []
        for _ in range(N_RUN):
            t = time.perf_counter(); m(x); ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return {"median_ms": round(statistics.median(ts), 3), "p95_ms": round(ts[int(0.95 * len(ts)) - 1], 3)}


def run(arch, res, head=True):
    m = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="").eval()
    # mimic the RPM head so the measured graph is the deployed one: 1x1 reduce -> flatten -> MLP(512) -> bins+span outputs
    with torch.no_grad():
        o = m(torch.zeros(1, 3, res, res))
    ch, h, w = o.shape[1:]
    net = torch.nn.Sequential(m, torch.nn.Conv2d(ch, 64, 1, bias=False), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.Flatten(),
                              torch.nn.Linear(64 * h * w, 512), torch.nn.ReLU(), torch.nn.Linear(512, 505 + 9)).eval()
    params = sum(p.numel() for p in net.parameters())
    d = tempfile.mkdtemp(prefix="lat_")
    shape = (1, 3, res, res)
    p32 = os.path.join(d, "m_fp32.onnx"); p8 = os.path.join(d, "m_int8.onnx")
    torch.onnx.export(net, torch.zeros(*shape), p32, input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    out = {"arch": arch, "res": res, "params_M": round(params / 1e6, 2), "fp32_onnx_MB": round(os.path.getsize(p32) / 1e6, 2),
           "torch_eager_1t": bench_torch(net, shape), "ort_fp32_1t": bench_ort(p32, shape, "x")}
    for tag, kw in (("int8_all", {}), ("int8_convonly", {"op_types_to_quantize": ["Conv", "Gemm", "MatMul"]})):
        try:
            pq = os.path.join(d, f"m_{tag}.onnx")
            quantize_static(p32, pq, RandReader("x", shape), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            per_channel=True, extra_options={"QuantizeBias": False}, **kw)
            out[tag + "_onnx_MB"] = round(os.path.getsize(pq) / 1e6, 2)
            out["ort_" + tag + "_1t"] = bench_ort(pq, shape, "x")
        except Exception as e:
            out[tag + "_error"] = repr(e)[:200]
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    os.rmdir(d)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    la0 = os.getloadavg()
    res = []
    for a, r in zip(args[::2], args[1::2]):
        o = run(a, int(r)); o["loadavg_before"] = [round(x, 1) for x in la0]; o["loadavg_after"] = [round(x, 1) for x in os.getloadavg()]
        print(json.dumps(o), flush=True); res.append(o)
