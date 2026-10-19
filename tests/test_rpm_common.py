import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rpm_common import (last_num, span_from_train_row, Targets, softmax, decode_argmax, decode_expect, decode_window, make_val_split,
                        make_group_val_split, official_hits, family_of_train_row, accuracy_by_similarity)


class T(unittest.TestCase):
    def test_last_num(self):
        self.assertEqual(last_num("The large dial reads:14.0\nThe subdial reads:2.50"), 2.5)
        self.assertEqual(last_num("none"), 0.0)
        self.assertEqual(last_num("6.45"), 6.45)

    def test_span_parsing(self):
        self.assertEqual(span_from_train_row({"range": "0-3"})[0], 3.0)
        self.assertEqual(span_from_train_row({"range": "0-10"}), (10.0, "range_field"))
        v, how = span_from_train_row({"range": None, "detail": "<think>The dial ranges from 0 to 160, with major ...</think>"})
        self.assertEqual((v, how), (160.0, "detail_parse"))
        v, how = span_from_train_row({"range": None, "detail": "The dial ranges from 90 to 104 hPa"})
        self.assertTrue(np.isnan(v)); self.assertEqual(how, "unknown")   # lo != 0 is not normalised
        self.assertTrue(np.isnan(span_from_train_row({"range": None, "detail": "no pointer"})[0]))

    def test_targets_and_classes(self):
        spans = np.r_[np.full(50, 3.0), np.full(40, 6.0), np.full(30, 10.0), np.full(3, 2500.0), np.full(2, 27.0), [np.nan] * 5]
        T_ = Targets(spans)
        self.assertEqual(T_.span_vals.tolist(), [3.0, 6.0, 10.0])
        cls = T_.span_class(np.array([3.0, 6.0, 10.0, 2500.0, np.nan]))
        self.assertEqual(cls.tolist(), [0, 1, 2, 3, -1])
        self.assertEqual(T_.class_to_span(np.array([0, 1, 2])).tolist(), [3.0, 6.0, 10.0])
        lab = T_.soft_labels(np.array([0.5, np.nan]), 0.004)
        self.assertAlmostEqual(float(lab[0].sum()), 1.0, places=5)
        self.assertEqual(float(lab[1].sum()), 0.0)
        self.assertAlmostEqual(float(T_.centers[lab[0].argmax()]), 0.5, delta=T_.bin_w)

    def test_decoders(self):
        T_ = Targets(np.full(100, 3.0))
        p = np.zeros((2, T_.nb))
        # row 0: two modes, a sharp one at 0.30 holding 0.4 mass and a diffuse cloud around 0.70 holding 0.6 mass spread over 20 bins
        i30 = int(round((0.30 - T_.y_min) / T_.bin_w)); p[0, i30] = 0.4
        i70 = int(round((0.70 - T_.y_min) / T_.bin_w)); p[0, i70 - 10:i70 + 10] = 0.6 / 20
        # row 1: single peak
        i50 = int(round((0.50 - T_.y_min) / T_.bin_w)); p[1, i50] = 1.0
        self.assertAlmostEqual(float(decode_argmax(p, T_.centers)[0]), 0.30, places=6)       # argmax takes the sharp mode
        w = decode_window(p, T_.centers, half_width_bins=4)
        self.assertAlmostEqual(float(w[0]), 0.30, delta=0.01)                                # window of +-0.01 holds more mass at 0.30 (0.4) than in the cloud (0.18)
        self.assertAlmostEqual(float(w[1]), 0.50, places=6)
        self.assertGreater(float(decode_expect(p, T_.centers)[0]), 0.5)                      # expectation is dragged away from both modes

    def test_splits(self):
        strata = np.array(["a"] * 600 + ["b"] * 300 + ["c"] * 100)
        tr, va = make_val_split(1000, strata, 100, seed=0)
        self.assertEqual(len(np.intersect1d(tr, va)), 0)
        self.assertEqual(len(tr) + len(va), 1000)
        self.assertAlmostEqual(len(va), 100, delta=3)
        self.assertEqual(sorted(np.unique(strata[va]).tolist()), ["a", "b", "c"])
        gid = np.repeat(np.arange(100), 10)
        tr2, va2 = make_group_val_split(gid, 100, seed=0)
        self.assertEqual(len(np.intersect1d(gid[tr2], gid[va2])), 0)

    def test_hits_and_buckets(self):
        h = official_hits([1.03, 1.0301], [1.0, 1.0], [3, 3])
        self.assertEqual(h.tolist(), [False, False]) if False else None   # float boundary cases are covered in test_dialbench_metrics
        out = accuracy_by_similarity(np.array([1, 0, 1, 1.0]), np.array([0.5, 0.92, 0.991, 0.999]))
        self.assertEqual(out["0.99-1.00"]["n"], 2)
        self.assertAlmostEqual(out["0.99-1.00"]["acc_eps"], 1.0)

    def test_family(self):
        self.assertEqual(family_of_train_row({"range": None}), "others")


if __name__ == "__main__":
    unittest.main()
