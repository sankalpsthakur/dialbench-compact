import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rpm_common import row_span, row_family, pool_rows, rows_from_manifest, knn_copy_pred, thumb_feats


class T(unittest.TestCase):
    def test_row_span(self):
        self.assertEqual(row_span({"range": 300}), (300.0, "range_numeric"))          # test.json row
        self.assertEqual(row_span({"range": 0.6})[0], 0.6)
        self.assertEqual(row_span({"range": "0-3"}), (3.0, "range_field"))            # train.json row
        v, how = row_span({"range": None, "detail": "<think>The dial ranges from 0 to 160 ...</think>"})
        self.assertEqual((v, how), (160.0, "detail_parse"))
        self.assertTrue(np.isnan(row_span({"range": None, "detail": "no pointer"})[0]))
        self.assertTrue(np.isnan(row_span({"range": 0})[0]))                          # a zero range is not a range
        self.assertTrue(np.isnan(row_span({})[0]))

    def test_row_family(self):
        self.assertEqual(row_family({"range": 300}), "t:300")
        self.assertEqual(row_family({"range": None}), "others")
        self.assertTrue(row_family({"range": "0-3", "graduation value": 0.1, "key scale": [0, 1, 2, 3]}).startswith("0-3|0.1|"))

    def test_pool_and_manifest(self):
        tr = [{"image": "a.jpg", "reading": "1"}, {"image": "dup.jpg", "reading": "train-row"}]
        te = [{"image": "b.jpg", "reading": "2"}, {"image": "dup.jpg", "reading": "test-row"}]
        by = pool_rows(tr, te)
        self.assertEqual(len(by), 3)
        self.assertEqual(by["dup.jpg"]["reading"], "test-row")                        # test.json wins, as in rpm_split3.py
        parts, missing = rows_from_manifest({"train": ["a.jpg", "zzz.jpg"], "val": ["b.jpg"], "test": ["dup.jpg"]}, by)
        self.assertEqual([len(parts[k]) for k in ("train", "val", "test")], [1, 1, 1])
        self.assertEqual(missing, {"train": 1, "val": 0, "test": 0})

    def test_knn_copy(self):
        rs = np.random.RandomState(0)
        X = rs.randint(0, 255, (6, 224, 224, 3), dtype=np.uint8)
        X[4] = X[1]                                                                    # image 4 is a twin of image 1
        X[5] = X[2]
        src, lab = np.array([0, 1, 2]), np.array([10.0, 20.0, 30.0])
        pred = knn_copy_pred(X, src, lab, np.array([4, 5, 0]))
        self.assertEqual(pred.tolist(), [20.0, 30.0, 10.0])
        f = thumb_feats(X, np.array([0, 1]), "gray56")
        self.assertEqual(f.shape, (2, 56 * 56))
        self.assertAlmostEqual(float(np.linalg.norm(f[0])), 1.0, places=4)
        self.assertEqual(thumb_feats(X, np.array([0]), "rgb32").shape, (1, 32 * 32 * 3))


if __name__ == "__main__":
    unittest.main()
