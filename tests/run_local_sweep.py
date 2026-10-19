"""Local smoke test of the int8 sweep on the synthetic fixture (tiny random model; checks plumbing only).
Needs the weights written by `run_local_smoke.py FIXTURE --split --res 64` (FIXTURE/out/smoke-split_ema.pt).
usage: run_local_sweep.py FIXTURE_DIR [recipe_name ...]   (default: every recipe)"""
import sys, types, os, hashlib, resource
import torch
torch.backends.mkldnn.enabled = False
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
m = types.ModuleType("jax_resnet"); sys.modules["jax_resnet"] = m
fx = sys.argv[1]
names = sys.argv[2:]
import rpm_train_torch
rpm_train_torch.predict_logits.__wrapped__.__defaults__ = (8,)
import rpm_int8_sweep as S
cfg = dict(arch="mobilenetv3_small_100", pretrained="none", res=64, hidden=32, input_dir=fx, threads=1, split_file=fx + "/splits_v3.json",
           split_sha256=hashlib.sha256(open(fx + "/splits_v3.json", "rb").read()).hexdigest())
recs = [r for r in S.RECIPES if not names or r["name"] in names]
res = S.run_model("smoke", cfg, fx + "/out/smoke-split_ema.pt", fx + "/out", n_calib=24, do_test=True, recipes=recs)
print("SELECTED", res["selected"], "TEST", res.get("test"), "rss MiB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 2**20)
print("SWEEP SMOKE OK")
