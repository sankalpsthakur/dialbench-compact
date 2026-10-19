import sys as _sys, traceback as _tb
_PRELUDE = '# Prelude (non-fatal): tiny JAX training on CPU + numeric equivalence check of the JAX->torch weight mapping used for TPU/GPU exports.\nimport os, sys, json, traceback\nos.environ.setdefault("JAX_PLATFORMS", "cpu")\nprint("=== JAX SELFTEST (tiny, CPU) ===", flush=True)\ntry:\n    import numpy as np\n    from rpm_train_jax import run as jax_run, get_cfg as jget\n    from rpm_train_torch import load_data, build_train_arrays\n    from rpm_common import Targets\n    from rpm_jax_check import check_equivalence\n    cfg = dict(arch="resnet18", pretrained="npz:resnet18", res=224, epochs=2, bs=64, lr=5e-4, eval_every=1, tag="jaxself", max_train=256, n_val=128,\n               val_mode="split", run_test=False, bf16=False, save_weights=True, out_dir="/kaggle/working")\n    res = jax_run(cfg)\n    c = jget(cfg)\n    base, train, test, X = load_data(c)\n    last, span, how, fam, img_idx = build_train_arrays(train)\n    T = Targets(span)\n    eq = check_equivalence(c, "/kaggle/working/jaxself_ema.npz", X, img_idx[:8], T.nb, T.ns)\n    print("JAX_EQUIV", json.dumps(eq), flush=True)\n    print("JAX_SELFTEST val", json.dumps(res["history"][-1].get("val_ema")), "secs/epoch", [h["secs"] for h in res["history"]], flush=True)\nexcept BaseException:\n    traceback.print_exc()\nprint("=== JAX SELFTEST END ===", flush=True)\n'
try:
    exec(compile(_PRELUDE, 'prelude', 'exec'), {'__name__': 'prelude'})
except BaseException:
    _tb.print_exc()
_sys.stdout = _sys.__stdout__
import sys, traceback
from rpm_train_torch import run
from rpm_export import export_and_eval
CONFIGS = [{'arch': 'mobilenetv3_small_100',
  'bs': 32,
  'ema': 0.995,
  'epochs': 30,
  'eval_every': 5,
  'lr': 0.0007,
  'n_val': 1000,
  'pretrained': 'timm',
  'res': 224,
  'run_test': True,
  'tag': 'cpu-mnv3s-b',
  'threads': 4,
  'val_mode': 'split'}]
for _cfg in CONFIGS:
    try:
        run(_cfg)
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
        continue
    try:
        export_and_eval(_cfg, "/kaggle/working/" + _cfg["tag"] + "_ema.pt", "torch", run_test=_cfg.get("run_test", False))
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
print("ALL CONFIGS DONE", flush=True)
