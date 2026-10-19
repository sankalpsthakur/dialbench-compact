"""protocol/check_release.py on a synthetic release (no data): it reports "ok" when the files match the recorded expectations and "DIFFERS" for every deviation."""
import hashlib, json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "protocol"))
import check_release as C


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def make_release(tmp, n_test=5, n_train=8, dup="3_img.jpg"):
    labels = os.path.join(tmp, "labels")
    images = os.path.join(tmp, "images")
    os.makedirs(labels)
    os.makedirs(images)
    test = [{"image": "%d_img.jpg" % i, "reading": "1.0", "range": 10} for i in range(1, n_test + 1)]
    train = [{"image": "%d_img.jpg" % i, "reading": "2.0"} for i in range(n_test + 1, n_test + n_train)] + [{"image": dup, "reading": "3.0"}]
    write_json(os.path.join(labels, "test.json"), test)
    write_json(os.path.join(labels, "train.json"), train)
    names = {it["image"] for it in test + train}
    for n in names | {"999_img.jpg"}:
        with open(os.path.join(images, n), "wb") as fh:
            fh.write(b"x")
    pred = os.path.join(tmp, "pred.json")
    with open(pred, "w") as fh:
        fh.write("{\n" + ",\n".join('"%s": "1.5"' % it["image"] for it in test) + "\n}\n")
    return labels, images, pred, test, train, len(names)


def expect_for(labels, test, train, n_names, dup="3_img.jpg"):
    return {"zip_bytes": 10, "zip_sha256": hashlib.sha256(b"0123456789").hexdigest(), "test_bytes": os.path.getsize(os.path.join(labels, "test.json")), "test_rows": len(test),
            "train_bytes": os.path.getsize(os.path.join(labels, "train.json")), "train_rows": len(train), "distinct_names": n_names, "in_both": [dup],
            "image_files": n_names + 1, "unreferenced_files": 1}


class T(unittest.TestCase):
    def test_ok_and_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels, images, pred, test, train, n_names = make_release(tmp)
            zp = os.path.join(tmp, "r.zip")
            with open(zp, "wb") as fh:
                fh.write(b"0123456789")
            exp = expect_for(labels, test, train, n_names)
            res, hashes = C.check(labels, pred, zp, images, exp)
            self.assertTrue(all(r[3] for r in res), [r for r in res if not r[3]])
            self.assertEqual(len(hashes["test_sha256"]), 64)
            # a wrong expectation per check is reported as DIFFERS
            for key, wrong in (("test_rows", 4), ("train_bytes", 1), ("distinct_names", 1), ("in_both", ["x"]), ("zip_sha256", "0" * 64), ("image_files", 3), ("unreferenced_files", 0)):
                res, _ = C.check(labels, pred, zp, images, dict(exp, **{key: wrong}))
                self.assertEqual(sum(1 for r in res if not r[3]), 1, key)
            # reordered predictions
            with open(pred, "w") as fh:
                fh.write('{\n"2_img.jpg": "1",\n"1_img.jpg": "1"\n}\n')
            res, _ = C.check(labels, pred, None, None, exp)
            self.assertEqual([r[0] for r in res if not r[3]], ["test.json names equal pred.json, in the same order"])

    def test_cli_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels, images, pred, test, train, n_names = make_release(tmp)
            r = subprocess.run([sys.executable, os.path.join(HERE, "..", "protocol", "check_release.py"), labels, "--pred", pred], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)                                 # the synthetic release is not the real one
            self.assertIn("DIFFERS", r.stdout)
            self.assertIn("RESULT", r.stdout)


if __name__ == "__main__":
    unittest.main()
