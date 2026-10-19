#!/usr/bin/env python3
"""Check that the DialBench RPM-10K release you downloaded is the one the reported numbers were computed on.
usage: check_release.py LABELS_DIR [--pred predictions/official_seed0.json] [--zip RPM-10K.zip] [--images IMAGES_DIR]
  LABELS_DIR   the directory with the authors' train.json and test.json (RPM-10K/labels)
  --pred       a public prediction file; its image names must be exactly the names of test.json, in the same order
  --zip        the downloaded RPM-10K.zip: byte size and sha256 are compared with the file used on 2026-10-07
  --images     the extracted image directory: number of files and files not named by any label row
Recorded on 2026-10-07 from the release the experiments used (kernel log of kernels/rpm-prep): zip 3,185,292,936 bytes, sha256 97e09a89...; labels/test.json 491,106 bytes, 2,000 rows;
labels/train.json 6,253,664 bytes, 8,730 rows; 10,730 image files; the two label files name 10,729 distinct images and 7443_img.jpg is in both. The sha256 of the two label files
was not recorded; this script prints it so that a difference can be reported. Prints one line per check ("ok" or "DIFFERS") and exits 1 if anything differs.
Nothing is read except the paths you give; no label value is printed."""
import argparse, hashlib, json, os, sys

EXPECT = {
    "zip_bytes": 3185292936,
    "zip_sha256": "97e09a89381336316f3893709cee626f51de09892ff7a80d328767b3cac9fe6f",
    "test_bytes": 491106, "test_rows": 2000, "train_bytes": 6253664, "train_rows": 8730,
    "distinct_names": 10729, "in_both": ["7443_img.jpg"], "image_files": 10730, "unreferenced_files": 1,
}


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def check(labels_dir, pred=None, zip_path=None, images=None, expect=EXPECT):
    """Returns a list of (label, got, want, ok)."""
    res = []

    def add(label, got, want):
        res.append((label, got, want, got == want))

    tp, rp = os.path.join(labels_dir, "test.json"), os.path.join(labels_dir, "train.json")
    with open(tp, encoding="utf-8") as fh:
        test = json.load(fh)
    with open(rp, encoding="utf-8") as fh:
        train = json.load(fh)
    add("labels/test.json bytes", os.path.getsize(tp), expect["test_bytes"])
    add("labels/test.json rows", len(test), expect["test_rows"])
    add("labels/train.json bytes", os.path.getsize(rp), expect["train_bytes"])
    add("labels/train.json rows", len(train), expect["train_rows"])
    tn, rn = [it["image"] for it in test], [it["image"] for it in train]
    add("distinct image names in the two label files", len(set(tn) | set(rn)), expect["distinct_names"])
    add("names in both label files", sorted(set(tn) & set(rn)), expect["in_both"])
    if pred:
        with open(pred, encoding="utf-8") as fh:
            keys = [k for k, _ in json.load(fh, object_pairs_hook=lambda kv: kv)]
        add("test.json names equal %s, in the same order" % os.path.basename(pred), tn == keys, True)
    if zip_path:
        add("zip bytes", os.path.getsize(zip_path), expect["zip_bytes"])
        add("zip sha256", sha256_file(zip_path), expect["zip_sha256"])
    if images:
        files = sorted(f for f in os.listdir(images) if f.lower().endswith(".jpg"))
        add("image files", len(files), expect["image_files"])
        add("image files named by no label row", len(set(files) - (set(tn) | set(rn))), expect["unreferenced_files"])
        add("label rows whose image file is missing", len((set(tn) | set(rn)) - set(files)), 0)
    return res, {"test_sha256": sha256_file(tp), "train_sha256": sha256_file(rp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir"); ap.add_argument("--pred"); ap.add_argument("--zip"); ap.add_argument("--images")
    a = ap.parse_args()
    res, hashes = check(a.labels_dir, a.pred, a.zip, a.images)
    for label, got, want, ok in res:
        print("%-6s %s: got %s, expected %s" % ("ok" if ok else "DIFFERS", label, got, want))
    print("info   sha256 labels/test.json  %s" % hashes["test_sha256"])
    print("info   sha256 labels/train.json %s" % hashes["train_sha256"])
    bad = [r for r in res if not r[3]]
    print("RESULT " + ("this is the release the reported numbers were computed on" if not bad else "%d check(s) differ: the release may have changed since 2026-10-07; please report which" % len(bad)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
