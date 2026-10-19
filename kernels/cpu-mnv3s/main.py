import sys, traceback
from rpm_train_torch import run
CONFIGS = [
 {
  "arch": "mobilenetv3_small_100",
  "pretrained": "timm",
  "res": 224,
  "epochs": 6,
  "lr": 0.001,
  "bs": 64,
  "eval_every": 2,
  "tag": "cpu-mnv3s",
  "threads": 4,
  "run_test": false,
  "val_mode": "split",
  "n_val": 1000
 }
]
for _cfg in CONFIGS:
    try:
        run(_cfg)
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
print("ALL CONFIGS DONE", flush=True)
