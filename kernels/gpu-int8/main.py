import sys, traceback, glob, json
from rpm_int8_sweep import run_model
MODELS = [{'cfg': {'arch': 'resnet18', 'n_val': 1000, 'pretrained': 'none', 'res': 224, 'tag': 'f-r18-224-off-s0', 'val_mode': 'split'},
  'label': 'p-off',
  'weights': 'f-r18-224-off-s0_ema.pt'},
 {'cfg': {'arch': 'resnet18',
          'pretrained': 'none',
          'res': 224,
          'split_file': 'splits_v3.json',
          'split_sha256': 'c16e51f6b9bd25cfa9925f19412da2510f9eaad9944dcddfc4d3a261307fc2e3',
          'tag': 'f-r18-224-v3-s0'},
  'label': 'p-v3',
  'weights': 'f-r18-224-v3-s0_ema.pt'}]
for _m in MODELS:
    try:
        _w = glob.glob("/kaggle/input/**/" + _m["weights"], recursive=True)[0]
        run_model(_m["label"], _m["cfg"], _w, "/kaggle/working")
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
print("SWEEP DONE", flush=True)
