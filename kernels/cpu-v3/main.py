import sys, traceback
from rpm_train_torch import run
from rpm_export import export_and_eval
CONFIGS = [{'arch': 'mobilenetv3_small_100',
  'bs': 32,
  'ema': 0.995,
  'epochs': 30,
  'eval_every': 5,
  'lr': 0.0007,
  'pretrained': 'timm',
  'res': 224,
  'run_test': True,
  'split_file': 'splits_v3.json',
  'split_sha256': 'c16e51f6b9bd25cfa9925f19412da2510f9eaad9944dcddfc4d3a261307fc2e3',
  'tag': 'cpu-v3',
  'threads': 4}]
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
