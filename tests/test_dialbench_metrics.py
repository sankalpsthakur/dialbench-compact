import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from dialbench_metrics import dialbench_metrics, extract_last_number, wilson_interval, bootstrap_acc_ci
import official_reference as ref


def port_from_text(pred_texts, readings, ranges_raw):
    gt = np.array([extract_last_number(str(r)) for r in readings])
    rg = np.array([float(r) for r in ranges_raw])
    pr = np.array([extract_last_number(t) for t in pred_texts])
    return dialbench_metrics(pr, gt, rg)


def assert_same(tc, o, p):
    tc.assertAlmostEqual(o["accuracy_epsilon"], p["acc_eps"], places=15)
    tc.assertAlmostEqual(o["accuracy_theta"], p["acc_theta"], places=15)
    tc.assertAlmostEqual(float(o["gamma"]), p["ref"], places=12)
    tc.assertAlmostEqual(float(o["theta"]), p["rel"], places=12)


class T(unittest.TestCase):
    def test_random(self):
        rs = np.random.RandomState(0)
        for trial in range(50):
            n = 200
            rng = rs.choice([3, 6, 10, 100, 1.6], n)
            gt = np.round(rs.rand(n) * rng, 2)
            gt[rs.rand(n) < 0.05] = 0
            pred = gt + rs.randn(n) * 0.02 * rng
            if trial % 5 == 0:
                pred[rs.rand(n) < 0.02] += 1e4  # huge errors -> clip paths
            texts = [format(p, ".10f") for p in pred]
            readings = [format(g, ".2f") for g in gt]
            assert_same(self, ref.official_eval(texts, readings, rng), port_from_text(texts, readings, rng))

    def test_boundary_epsilon(self):
        # error exactly epsilon in decimal, float roundoff decides: both implementations must agree
        cases = [("1.50", "1.53", 3), ("1.50", "1.47", 3), ("0.10", "0.13", 3), ("5.00", "5.10", 10), ("2.30", "2.36", 6), ("0.00", "0.06", 6)]
        for g, p, r in cases:
            o = ref.official_eval([p], [g], [r]); q = port_from_text([p], [g], [r])
            self.assertEqual(o["accuracy_epsilon"], q["acc_eps"], (g, p, r))

    def test_text_edge_cases(self):
        readings = ["The upper dial reads:12.35\nThe lower dial reads:20.8", "0", "0.0", "-2.5", "N/A", "7"]
        preds = ["The reading is 20.7", "0.5", "none", "-2.49", "N/A", "7.0 units 12"]
        rng = [80, 3, 3, 10, 100, 10]
        assert_same(self, ref.official_eval(preds, readings, rng), port_from_text(preds, readings, rng))
        self.assertEqual(extract_last_number("N/A"), 0.0)
        self.assertEqual(extract_last_number("The upper dial reads:12.35\nThe lower dial reads:20.8"), 20.8)

    def test_all_zero_gt_and_empty(self):
        o = ref.official_eval(["1", "2"], ["0", "0.0"], [3, 3])
        q = port_from_text(["1", "2"], ["0", "0.0"], [3, 3])
        self.assertIsNone(o["theta"]); self.assertIsNone(o["accuracy_theta"])
        self.assertIsNone(q["rel"]); self.assertIsNone(q["acc_theta"])
        self.assertAlmostEqual(o["gamma"], q["ref"])

    def test_wilson_and_bootstrap(self):
        lo, hi = wilson_interval(1240, 2000)
        self.assertAlmostEqual((lo + hi) / 2, 0.6203, places=2)
        self.assertTrue(0.017 < (hi - lo) / 2 < 0.023)
        hits = np.r_[np.ones(60), np.zeros(40)]
        lo, hi = bootstrap_acc_ci(hits, n_boot=500)
        self.assertTrue(lo < 0.6 < hi)
        g = np.repeat(np.arange(20), 5)
        lo2, hi2 = bootstrap_acc_ci(hits, n_boot=500, groups=g)
        self.assertTrue(lo2 <= 0.6 <= hi2)


if __name__ == "__main__":
    unittest.main()
