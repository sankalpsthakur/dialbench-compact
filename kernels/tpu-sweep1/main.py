import sys, traceback
from rpm_train_jax import run
CONFIGS = [
 {
  "arch": "resnet18",
  "pretrained": "npz:resnet18",
  "res": 224,
  "epochs": 60,
  "bs": 256,
  "lr": 0.001,
  "eval_every": 10,
  "tag": "s1-r18-lr1e-3"
 },
 {
  "arch": "resnet18",
  "pretrained": "npz:resnet18",
  "res": 224,
  "epochs": 60,
  "bs": 256,
  "lr": 0.0005,
  "eval_every": 10,
  "tag": "s1-r18-lr5e-4"
 },
 {
  "arch": "resnet18",
  "pretrained": "none",
  "res": 224,
  "epochs": 100,
  "bs": 256,
  "lr": 0.002,
  "eval_every": 10,
  "tag": "s1-r18-scratch"
 },
 {
  "arch": "resnet18",
  "pretrained": "npz:resnet18",
  "res": 320,
  "epochs": 60,
  "bs": 256,
  "lr": 0.001,
  "eval_every": 10,
  "tag": "s1-r18-res320"
 },
 {
  "arch": "resnet34",
  "pretrained": "npz:resnet34",
  "res": 224,
  "epochs": 60,
  "bs": 256,
  "lr": 0.0007,
  "eval_every": 10,
  "tag": "s1-r34"
 },
 {
  "arch": "resnet10",
  "pretrained": "none",
  "res": 224,
  "epochs": 100,
  "bs": 256,
  "lr": 0.002,
  "eval_every": 10,
  "tag": "s1-r10-scratch"
 },
 {
  "arch": "resnet18x0.5",
  "pretrained": "none",
  "res": 224,
  "epochs": 100,
  "bs": 256,
  "lr": 0.002,
  "eval_every": 10,
  "tag": "s1-r18x0.5-scratch"
 }
]
for _cfg in CONFIGS:
    try:
        run(_cfg)
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
print("ALL CONFIGS DONE", flush=True)
