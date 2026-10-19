
# ---- config part (edit here; the config hash is logged in the result)
CONFIGS = [dict(arch="mobilenetv3_small_100", pretrained="timm", res=224, epochs=6, lr=0.001, bs=64, eval_every=2, tag="cpu-mnv3s", threads=4, run_test=False, val_mode="split", n_val=1000)]
