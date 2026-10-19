"""Local smoke test of trainer + export on the synthetic fixture (no real data, random pixels). Run it through ../bin/guard.sh, one mode at a time.
usage: run_local_smoke.py FIXTURE_DIR [--split] [--export] [--res 224]
  --split   use the manifest FIXTURE_DIR/splits_v3.json (split_file mode) instead of the official-split logic
  --export  also run ONNX/int8 export + evaluation (heavier: needs onnx/onnxruntime and more RAM)"""
import sys, types, os, argparse
ap = argparse.ArgumentParser()
ap.add_argument("fixture", nargs="?", default="/tmp/gauges_synth")
ap.add_argument("--split", action="store_true"); ap.add_argument("--export", action="store_true"); ap.add_argument("--res", type=int, default=224)
a = ap.parse_args()
fx = a.fixture
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
m = types.ModuleType("jax_resnet"); m.flax_to_tv_state_dict = lambda *a_, **k: (_ for _ in ()).throw(RuntimeError("jax path not used")); sys.modules["jax_resnet"] = m
import hashlib, json
import resource
import torch
torch.backends.mkldnn.enabled = False   # oneDNN primitive JIT alone costs ~250 MiB RSS; irrelevant for checking our logic (keeps the smoke under the 512 MiB guard)
import rpm_train_torch
from rpm_train_torch import run
rpm_train_torch.predict_logits.__wrapped__.__defaults__ = (8,)   # local only: batch of 128 224-px images costs ~250 MiB in resize/interpolate temporaries
_log0 = rpm_train_torch.log
rpm_train_torch.log = lambda *a: _log0(f"[rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.0f} MiB]", *a)   # peak RSS at every log line (macOS: bytes)
cfg = dict(arch="mobilenetv3_small_100", pretrained="none", epochs=1, bs=16 if a.res >= 224 else 8, n_val=30, max_train=32 if a.res >= 224 else 16, threads=1, input_dir=fx,
           out_dir=fx + "/out", tag="smoke-split" if a.split else "smoke", run_test=True, eval_every=1, save_weights=True, hidden=32, res=a.res)
if a.split:
    cfg["split_file"] = fx + "/splits_v3.json"
    cfg["split_sha256"] = hashlib.sha256(open(cfg["split_file"], "rb").read()).hexdigest()
os.makedirs(fx + "/out", exist_ok=True)
res = run(cfg)
t = res["test"]
print("TEST n", t["metrics"]["n"], "acc", t["metrics"]["acc_eps"], "keys", sorted(t.keys()))
if a.split:
    sf = t["split_file"]
    assert sf["n_test_rows"] == 40 and sf["n_excluded_unknown_range"] == 1 and sf["n_scored"] == 39, sf
    assert t["metrics"]["n"] == 39
    assert sf["info"]["names_without_label_row"] == {"train": 1, "val": 0, "test": 0}, sf["info"]
    assert sf["info"]["n_rows"] == {"train": 84, "val": 15, "test": 40}, sf["info"]
    assert "knn_copy_baseline" in sf and sf["knn_copy_baseline"]["n"] == 39
    print("SPLIT-MODE CHECKS OK", json.dumps({k: sf[k] for k in ("by_source_file", "by_range_source", "knn_copy_baseline")}, default=str)[:600])
if a.export:
    from rpm_export import export_and_eval
    r = export_and_eval(cfg, fx + "/out/" + cfg["tag"] + "_ema.pt", "torch", run_test=True, out_dir=fx + "/out", n_calib=32)
    print("EXPORT OK", sorted(r.keys()))
print("SMOKE OK")
