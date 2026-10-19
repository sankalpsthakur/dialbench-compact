#!/usr/bin/env python3
"""Print validation (and test, if present) summaries from *_result.json files written by the trainers."""
import json, sys, glob
for f in sorted(sum([glob.glob(a) for a in sys.argv[1:]], [])):
    r = json.load(open(f))
    if "config" not in r:  # e.g. *_export_result.json
        continue
    c = r["config"]
    h = r["history"]
    last_eval = [x for x in h if "val_ema" in x][-1] if any("val_ema" in x for x in h) else None
    print(f"== {c['tag']} arch={c['arch']} res={c['res']} params={r['params']/1e6:.2f}M epochs={c['epochs']} bs={c['bs']} lr={c['lr']} hash={r['config_hash']} secs/epoch~{sum(x['secs'] for x in h)/len(h):.1f}")
    for x in h:
        if "val_ema" in x:
            e, w = x["val_ema"], x["val_raw"]
            print(f"  ep{x['epoch']:3d} loss {x['loss']:.3f} | EMA window {e['window']:.4f} argmax {e['argmax']:.4f} expect {e['expect']:.4f} spanacc {e['span_cls_acc']:.3f} oracle-span {e['window_oracle_span']:.4f} | raw window {w['window']:.4f}")
    if "test" in r:
        t = r["test"]
        print("  TEST", json.dumps({k: t[k] for k in ("metrics", "wilson95", "grouped_bootstrap95", "per_meter_type", "decoders", "by_train_similarity") if k in t}))
        sf = t.get("split_file")
        if sf:   # split_file mode (split v3): scored rows, per-source/range tables and the no-learning copy baseline on the same rows
            print("  SPLIT-FILE", json.dumps({k: sf[k] for k in ("n_test_rows", "n_scored", "n_excluded_unknown_range", "by_source_file", "by_range_source", "by_range_class", "knn_copy_baseline") if k in sf}))
            print("  SPLIT-INFO", json.dumps({k: v for k, v in sf.get("info", {}).items() if k != "manifest_baselines"}))
        if "split_file_error" in t:
            print("  SPLIT-FILE ERROR", t["split_file_error"])
