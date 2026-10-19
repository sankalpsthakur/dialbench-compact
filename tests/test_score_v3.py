import json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCORER = os.path.join(HERE, "..", "protocol", "score_v3.py")


class T(unittest.TestCase):
    def test_scores_pooled_rows(self):
        train = [{"image": "1_img.jpg", "reading": "1.0", "range": "0-3"},
                 {"image": "2_img.jpg", "reading": "4.0", "range": "0-6"},
                 {"image": "3_img.jpg", "reading": "none", "range": None, "detail": "no pointer"},      # no determinable range: excluded
                 {"image": "4_img.jpg", "reading": "50", "range": None, "detail": "The dial ranges from 0 to 100"},
                 {"image": "5_img.jpg", "reading": "9.99", "range": "0-10"}]                              # also in test.json: the test.json row wins
        test = [{"image": "6_img.jpg", "reading": "2.0", "range": 300},
                {"image": "5_img.jpg", "reading": "7.0", "range": 10}]
        man = {"B": 160, "test": ["1_img.jpg", "2_img.jpg", "3_img.jpg", "4_img.jpg", "5_img.jpg", "6_img.jpg", "999_img.jpg"], "val": [],
               "block_of": {n: 0 for n in ["1_img.jpg", "2_img.jpg", "3_img.jpg", "4_img.jpg", "5_img.jpg", "6_img.jpg", "999_img.jpg"]}}
        # exact readings except image 2 (off by 0.5 on range 6, tolerance 0.06) and image 6 (off by 2 on range 300, tolerance 3: still a hit)
        pred = {"1_img.jpg": 1.0, "2_img.jpg": "about 4.5", "3_img.jpg": 0, "4_img.jpg": "50", "5_img.jpg": "7", "6_img.jpg": 4.0}
        with tempfile.TemporaryDirectory() as d:
            for name, obj in (("man.json", man), ("train.json", train), ("test.json", test), ("pred.json", pred)):
                json.dump(obj, open(os.path.join(d, name), "w"))
            r = subprocess.run([sys.executable, SCORER] + [os.path.join(d, x) for x in ("man.json", "train.json", "test.json", "pred.json")], capture_output=True, text=True, check=True)
        out = json.loads(r.stdout)
        self.assertEqual(out["n_names"], 7)
        self.assertEqual(out["n_rows"], 6)                       # 999_img.jpg has no label row
        self.assertEqual(out["n_excluded_unknown_range"], 1)     # 3_img.jpg
        self.assertEqual(out["n_scored"], 5)
        self.assertAlmostEqual(out["acc_eps"], 4 / 5)            # images 1, 4, 5 (test row: reading 7, range 10), 6 hit; image 2 misses
        self.assertEqual(out["missing_predictions"], 0)


if __name__ == "__main__":
    unittest.main()
