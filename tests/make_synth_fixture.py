"""Synthetic fixture shaped like the rpm-prep kernel output (random pixels, fake labels, ~25 MB). For local smoke tests only.
usage: make_synth_fixture.py OUT_DIR"""
import json, os, sys
import numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gauges_synth"
os.makedirs(d + "/cache", exist_ok=True)
rs = np.random.RandomState(0)
N = 140
np.save(d + "/cache/x224.npy", rs.randint(0, 255, (N, 224, 224, 3), dtype=np.uint8))
json.dump({"image_rel": [f"{i}_img.jpg" for i in range(N)], "orig_wh": [[100, 100]] * N}, open(d + "/images_index.json", "w"))
fams = [("0-3", 0.1, [0, 1, 2, 3]), ("0-6", 0.2, [0, 1, 2, 3, 4, 5, 6]), ("0-10", 0.2, [0, 2, 4, 6, 8, 10])]
train, test = [], []
for i in range(100):
    if i % 10 == 0:
        train.append({"image": f"{i}_img.jpg", "query": "q", "reading": "%.1f" % (rs.rand() * 100), "detail": "<think>The dial ranges from 0 to 100 ...</think>", "img_idx": i})
    elif i % 37 == 0:
        train.append({"image": f"{i}_img.jpg", "query": "q", "reading": "none", "detail": "no pointer", "img_idx": i})
    else:
        r, g, k = fams[i % 3]; hi = float(r.split("-")[1])
        train.append({"image": f"{i}_img.jpg", "query": "q", "reading": "%.2f" % (rs.rand() * hi), "range": r, "graduation value": g, "key scale": k, "detail": "d", "img_idx": i})
for i in range(100, 140):
    hi = [3, 6, 10][i % 3]
    test.append({"image": f"{i}_img.jpg", "query": "q", "reading": "%.2f" % (rs.rand() * hi), "range": hi, "meter_type": ["biao1", "biao2", "others"][i % 3],
                 "environment_conditions": "tilted", "img_idx": i})
# the image that is in both label files (like 7443_img.jpg): the train.json row has another reading than the test.json row
train.append({"image": "105_img.jpg", "query": "q", "reading": "9.99", "range": "0-10", "graduation value": 0.2, "key scale": [0, 2, 4, 6, 8, 10], "detail": "d", "img_idx": 105})
json.dump(train, open(d + "/meta__RPM-10K__labels__train.json", "w")); json.dump(test, open(d + "/meta__RPM-10K__labels__test.json", "w"))
# a split-v3 style manifest (names only): official-test rows land in train and official-train rows in test; image 37 has no determinable range; one name has no label row
nm = lambda r: [f"{i}_img.jpg" for i in r]
man = {"description": "synthetic", "B": 100, "seed": 1, "tau_twin": 0.98,
       "train": nm(range(1, 37)) + nm(range(38, 70)) + nm(range(100, 116)) + ["999_img.jpg"], "val": nm(range(70, 85)),
       "test": nm([37]) + nm(range(85, 100)) + nm(range(116, 140)),
       "baselines": {"knn_copy_v3_test": [0.1, 39]}}
json.dump(man, open(d + "/splits_v3.json", "w"), sort_keys=True)
print("fixture written to", d)
