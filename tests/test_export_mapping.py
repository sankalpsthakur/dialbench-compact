import os, sys, types, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# rpm_export imports torch and jax_resnet (needs jax); stub jax_resnet for this pure-numpy test. Without torch the test is skipped (the public kit's CI installs numpy only).
if "jax_resnet" not in sys.modules:
    try:
        import jax  # noqa
    except Exception:
        m = types.ModuleType("jax_resnet"); m.flax_to_tv_state_dict = None; sys.modules["jax_resnet"] = m
try:
    from rpm_export import flax_fc1_to_torch
except ModuleNotFoundError as e:
    if e.name != "torch":
        raise
    flax_fc1_to_torch = None


@unittest.skipIf(flax_fc1_to_torch is None, "torch is not installed")
class T(unittest.TestCase):
    def test_fc1_layout(self):
        rs = np.random.RandomState(0)
        B, h, w, hc, hid = 3, 5, 4, 7, 11
        f_nhwc = rs.randn(B, h, w, hc).astype(np.float32)
        k1 = rs.randn(h * w * hc, hid).astype(np.float32)
        y_flax = f_nhwc.reshape(B, -1) @ k1                         # Flax: flatten NHWC
        f_nchw = f_nhwc.transpose(0, 3, 1, 2)
        W = flax_fc1_to_torch(k1, h, w, hc)                          # torch Linear weight [hid, hc*h*w]
        y_torch = f_nchw.reshape(B, -1) @ W.T                        # torch: flatten NCHW
        self.assertEqual(W.shape, (hid, hc * h * w))
        np.testing.assert_allclose(y_flax, y_torch, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
