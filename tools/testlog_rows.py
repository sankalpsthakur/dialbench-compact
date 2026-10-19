#!/usr/bin/env python3
"""Print TEST-LOG.md rows for one finished run straight from its result JSONs (numbers copied as the kernel wrote them, nothing re-scored).
usage: testlog_rows.py RESULT_JSON EXPORT_RESULT_JSON "When (UTC)" "Benchmark / split" "run label"
Reads <tag>_result.json (trainer test block) and <tag>_export_result.json (torch_fp32, onnx_fp32, onnx_int8, onnx_int8_convonly).
EXPORT_RESULT_JSON may be "-" (GPU Stage B runs of FROZEN-4 have no export step: one row, the trainer evaluation, "evaluation 1 of 1")."""
import json, sys
res, exp, when, bench, label = sys.argv[1:6]
r = json.load(open(res)); e = {} if exp == "-" else json.load(open(exp))
N_EVALS = 1 if exp == "-" else 5
t = r["test"]; m = t["metrics"]
h = r["config_hash"]
n = m["n"]
rows = []
rows.append(f"| {when} | {bench} | cfg `{h}` | **{label}, evaluation 1 of {N_EVALS}**: PyTorch fp32 EMA weights at the last epoch, window decoder, inside the trainer; n = {n} scored rows, official scorer | **Acc_eps {m['acc_eps']}**, Wilson 95 % {t['wilson95']}, grouped bootstrap 95 % {t['grouped_bootstrap95']} ({t['n_test_groups']} test groups); Acc_theta {m['acc_theta']}, Ref {m['ref']}, Rel {m['rel']} | decoders argmax {t['decoders']['argmax']}, expect {t['decoders']['expect']}, window {t['decoders']['window']} (window was frozen) |")
names = [("torch_fp32", "PyTorch fp32 reload in the export step"), ("onnx_fp32", "ONNX fp32 (opset 17, ONNX Runtime on the Kaggle Xeon)"), ("onnx_int8", "ONNX int8, static QDQ all ops (256 random train images for calibration)"), ("onnx_int8_convonly", "ONNX int8, QDQ on Conv/Gemm/MatMul only")]
for i, (k, desc) in (enumerate(names, start=2) if exp != "-" else []):
    x = e.get(f"test_{k}")
    if x is None:
        rows.append(f"| {when} | {bench} | same | {label}, evaluation {i} of 5: {desc} | NOT PRODUCED (export step missing this variant) | |")
    else:
        rows.append(f"| {when} | {bench} | same | {label}, evaluation {i} of 5: {desc} | Acc_eps {x['acc_eps']}, Wilson 95 % {x['wilson95']} (n = {x.get('n', n)}) | |")
print("\n".join(rows))
print()
print("validation (not test):", {k: v for k, v in e.items() if k.startswith("val_")})
print("sizes MB:", {k: v for k, v in e.items() if k.endswith("_MB")}, "params", e.get("params"))
print("latency (Kaggle Xeon, 1 thread, batch 1):", {k: v for k, v in e.items() if k.startswith("kaggle_xeon")})
sf = t.get("split_file")
if sf:
    print("split_file:", json.dumps({k: sf[k] for k in ("n_test_rows", "n_scored", "n_excluded_unknown_range", "by_source_file", "by_range_source", "by_range_class", "knn_copy_baseline") if k in sf}))
print("by_train_similarity:", json.dumps(t["by_train_similarity"]))
print("per_meter_type:", json.dumps(t["per_meter_type"]))
