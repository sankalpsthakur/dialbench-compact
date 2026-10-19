#!/usr/bin/env python3
"""Latency of exported ONNX files on this Mac: 1 intra-op thread, batch 1, warm-up then N timed runs.
usage: latency_onnx.py model.onnx [model2.onnx ...] [--runs 400] [--res 224]
Reports min / median / p95 / p99 in ms, load average before and after, ORT version, CPU model. Input is float32 0..255 NCHW (normalisation is inside the graph)."""
import os, sys, time, json, subprocess, statistics
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import onnxruntime as ort

args = [a for a in sys.argv[1:] if not a.startswith("--")]
runs = 400
res = 224
for i, a in enumerate(sys.argv):
    if a == "--runs":
        runs = int(sys.argv[i + 1]); args.remove(sys.argv[i + 1])
    if a == "--res":
        res = int(sys.argv[i + 1]); args.remove(sys.argv[i + 1])
cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
out = []
for p in args:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    x = (np.random.rand(1, 3, res, res) * 255).astype(np.float32)
    la0 = os.getloadavg()[0]
    for _ in range(40):
        s.run(None, {name: x})
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); s.run(None, {name: x}); ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    q = lambda f: ts[min(len(ts) - 1, int(f * len(ts)))]
    r = {"file": os.path.basename(p), "size_MB": round(os.path.getsize(p) / 1e6, 2), "runs": runs, "min_ms": round(ts[0], 3), "median_ms": round(statistics.median(ts), 3),
         "p95_ms": round(q(0.95), 3), "p99_ms": round(q(0.99), 3), "load_before": round(la0, 1), "load_after": round(os.getloadavg()[0], 1), "ort": ort.__version__, "cpu": cpu, "threads": 1, "batch": 1}
    print(json.dumps(r), flush=True)
    out.append(r)
