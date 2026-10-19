# Prelude (non-fatal): tiny JAX training on CPU + numeric equivalence check of the JAX->torch weight mapping used for TPU/GPU exports.
import os, sys, json, traceback
os.environ.setdefault("JAX_PLATFORMS", "cpu")
print("=== JAX SELFTEST (tiny, CPU) ===", flush=True)
try:
    import numpy as np
    from rpm_train_jax import run as jax_run, get_cfg as jget
    from rpm_train_torch import load_data, build_train_arrays
    from rpm_common import Targets
    from rpm_jax_check import check_equivalence
    cfg = dict(arch="resnet18", pretrained="npz:resnet18", res=224, epochs=2, warmup_epochs=1, bs=64, lr=5e-4, eval_every=1, tag="jaxself", max_train=256, n_val=128,
               val_mode="split", run_test=False, bf16=False, save_weights=True, out_dir="/kaggle/working")
    res = jax_run(cfg)
    c = jget(cfg)
    base, train, test, X = load_data(c)
    last, span, how, fam, img_idx = build_train_arrays(train)
    T = Targets(span)
    eq = check_equivalence(c, "/kaggle/working/jaxself_ema.npz", X, img_idx[:8], T.nb, T.ns)
    print("JAX_EQUIV", json.dumps(eq), flush=True)
    print("JAX_SELFTEST val", json.dumps(res["history"][-1].get("val_ema")), "secs/epoch", [h["secs"] for h in res["history"]], flush=True)
except BaseException:
    traceback.print_exc()
print("=== JAX SELFTEST END ===", flush=True)
