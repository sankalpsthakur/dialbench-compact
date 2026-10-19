import sys, traceback, json
import numpy as np
from rpm_train_torch import prepare, get_cfg, load_data
from rpm_common import knn_copy_pred, official_hits
JOBS = [{'cfg': {'arch': 'mobilenetv3_small_100', 'pretrained': 'none', 'res': 224}, 'label': 'official'},
 {'cfg': {'arch': 'mobilenetv3_small_100',
          'pretrained': 'none',
          'res': 224,
          'split_file': 'splits_v3.json',
          'split_sha256': 'c16e51f6b9bd25cfa9925f19412da2510f9eaad9944dcddfc4d3a261307fc2e3'},
  'label': 'v3'}]
for _job in JOBS:
    try:
        c = get_cfg(_job["cfg"]); D = prepare(c); tt = D.test
        src = np.arange(len(D.last)) if D.mode == "official" else D.tr_idx   # official: every train.json row (as the audit floor); v3: the split's train part
        pred = knn_copy_pred(D.X, D.img_idx[src], D.last[src], tt.img_idx)
        hits = official_hits(pred, tt.last, np.where(tt.known, tt.rng, 1.0))
        out = dict(hits=hits, known=tt.known, last=tt.last, rng=tt.rng, pred=pred, img_idx=tt.img_idx)
        if D.mode == "official":
            _b, _tr, _te, _X = load_data(c)
            out["names"] = np.array([it["image"] for it in _te])
        np.savez("/kaggle/working/copy_hits_" + _job["label"] + ".npz", **out)
        print(_job["label"], "copy floor", float(hits[tt.known].mean()), "n", int(tt.known.sum()), "source rows", int(len(src)), flush=True)
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
print("COPY DONE", flush=True)
