"""The public per-seed prediction files (predictions/*.json) hold ONLY image file name -> predicted reading, and agree with the split manifests; the number formatter
used to write them never produces scientific notation (the official scorer reads the last number in the text, so '4.5e-05' would be read as -5)."""
import glob, json, os, re, sys, unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from dialbench_metrics import extract_last_number
from reduce_predictions import fmt

NAME = re.compile(r"^[0-9]+_img\.jpg$")
VALUE = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")
FORBIDDEN_WORDS = ("reading", "range", "hits", "known", "last", "rng", "meter", "detail", "label")


def load(p):
    with open(p, encoding="utf-8") as fh:
        raw = fh.read()
    return raw, json.loads(raw, object_pairs_hook=lambda kv: kv)    # keep order and duplicates


def load_plain(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


class T(unittest.TestCase):
    def test_formatter(self):
        rs = np.random.RandomState(0)
        vals = [0.0, 3.0, 100.0, 4.5e-05, -0.0035, 1e-12, 12345678.0, 0.1 + 0.2, 7.628288354539508] + list(rs.randn(300) * 10.0 ** rs.randint(-9, 6, 300))
        for v in vals:
            s = fmt(v)
            self.assertRegex(s, VALUE)
            self.assertNotIn("e", s.lower())
            self.assertEqual(extract_last_number(s), v)       # what the official scorer reads back
        self.assertEqual(extract_last_number(str(4.5e-05)), -5.0)   # why the formatter exists

    def test_files(self):
        off = sorted(glob.glob(os.path.join(ROOT, "predictions", "official_seed*.json")))
        v3 = sorted(glob.glob(os.path.join(ROOT, "predictions", "v3_seed*.json")))
        self.assertEqual(len(off), 5)
        self.assertEqual(len(v3), 3)
        man = load_plain(os.path.join(ROOT, "protocol", "splits_v3.json"))
        v2 = load_plain(os.path.join(ROOT, "protocol", "splits_v2.json"))
        pool = set(v2["group_id"])                                  # all 10,729 distinct image names of the release
        first = None
        for p in off + v3:
            raw, items = load(p)
            keys = [k for k, _ in items]
            self.assertEqual(len(set(keys)), len(keys), p + ": duplicate image names")
            for k, v in items:
                self.assertRegex(k, NAME)
                self.assertIsInstance(v, str)
                self.assertRegex(v, VALUE)
                self.assertTrue(np.isfinite(float(v)))
            low = raw.lower()
            for w in FORBIDDEN_WORDS:
                self.assertNotIn(w, low, "%s mentions %r" % (p, w))
            self.assertTrue(set(keys) <= pool, p + ": names outside the release")
            if os.path.basename(p).startswith("official"):
                self.assertEqual(len(keys), 2000)
                if first is None:
                    first = keys
                self.assertEqual(keys, first, p + ": row order differs between seeds")
            else:
                self.assertEqual(keys, man["test"], p + ": not the v3 test list in manifest order")
        # the 365 v3 test images that are also official test images
        self.assertEqual(len(set(first) & set(man["test"])), 365)


if __name__ == "__main__":
    unittest.main()
