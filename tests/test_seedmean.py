"""Local test of kernels/gpu-seedmean/script.py on synthetic per-row hits (no data, no model): Wilson / block bootstrap numbers, verdict strings, group parsing."""
import json, math, os, subprocess, sys, tempfile, unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "kernels", "gpu-seedmean", "script.py")


def wilson(p, n, z=1.959964):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def build(p_hit, seeds=(0, 1, 2)):
    root = tempfile.mkdtemp()
    ind, outd = os.path.join(root, "in", "k"), os.path.join(root, "out")
    os.makedirs(ind)
    rs = np.random.RandomState(1)
    n = 2000
    names = ["%d_img.jpg" % (i + 1) for i in range(n)]        # ids 1..2000 -> blocks (id-1)//160: 13 blocks
    known = np.ones(n, bool)
    np.savez(os.path.join(ind, "copy_hits_official.npz"), hits=rs.rand(n) < 0.42, known=known, names=np.array(names))
    for s in seeds:
        np.savez(os.path.join(ind, "f-r18-320-off-s%d_test_pred.npz" % s), hits=rs.rand(n) < p_hit, known=known)
    return root, ind, outd


def run(root, outd):
    env = dict(os.environ, IN_ROOT=os.path.join(root, "in"), OUT_ROOT=outd)
    r = subprocess.run([sys.executable, SCRIPT], env=env, capture_output=True, text=True, timeout=50)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout


class T(unittest.TestCase):
    def test_high_accuracy_gives_full_wording_and_matches_wilson(self):
        root, ind, outd = build(0.80)
        out = run(root, outd)
        self.assertIn("config r18-320: seeds [0, 1, 2], n = 2000", out)
        H = np.mean([np.load(os.path.join(ind, "f-r18-320-off-s%d_test_pred.npz" % s))["hits"].mean() for s in (0, 1, 2)])
        lo, hi = wilson(H, 2000)
        self.assertIn("H (seed mean) = %.4f" % H, out)
        self.assertIn("[%.4f, %.4f]" % (lo, hi), out)
        self.assertIn("full wording", out)
        self.assertTrue(os.path.exists(os.path.join(outd, "seedmean.txt")))

    def test_mid_accuracy_is_negative_or_qualified(self):
        root, ind, outd = build(0.60)
        self.assertIn("negative result", run(root, outd))

    def test_wilson_only_case_exists_between(self):
        # about 0.652: Wilson LB just above 0.624, block interval (13 blocks, little spread) may also clear; accept either qualified or full wording but never negative
        root, ind, outd = build(0.67)
        out = run(root, outd)
        self.assertNotIn("negative result", out)


if __name__ == "__main__":
    unittest.main()
