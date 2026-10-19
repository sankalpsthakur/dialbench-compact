"""protocol/rescore_seeds.py on synthetic labels and predictions (no data, no model): per-seed metrics against the verbatim oracle of the authors' scorer, the
missing-prediction rule, and the seed-mean statistic (H, Wilson, block bootstrap) against kernels/gpu-seedmean/script.py, the pre-registered implementation, on the official
split (160-id blocks) and on split v3 (manifest blocks, pooled label rows, rows without a determinable range excluded; per-seed numbers also against protocol/score_v3.py)."""
import json, os, subprocess, sys, tempfile, unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "protocol"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import rescore_seeds as R
import official_reference as ref

KERNEL = os.path.join(ROOT, "kernels", "gpu-seedmean", "script.py")
SCORE_V3 = os.path.join(ROOT, "protocol", "score_v3.py")


def pos(x):
    return np.format_float_positional(float(x), unique=True, trim="-")


def fixture(n=1700, seeds=3, p_hit=0.7, rs_seed=3):
    rs = np.random.RandomState(rs_seed)
    rng = rs.choice([3.0, 6.0, 10.0, 80.0, 300.0], n)
    gt = np.round(rs.rand(n) * rng, 2)
    gt[::97] = 0.0                                                    # zero readings: excluded from Acc_theta / Rel
    rows = [{"image": "%d_img.jpg" % (i + 1), "reading": repr(float(gt[i])), "range": float(rng[i])} for i in range(n)]
    preds = []
    for s in range(seeds):
        ok = rs.rand(n) < p_hit
        err = np.where(ok, (rs.rand(n) * 2 - 1) * 0.008 * rng, (rs.rand(n) * 2 - 1) * 0.2 * rng + np.sign(rs.rand(n) - 0.5) * 0.02 * rng)
        pr = np.maximum(gt + err, 0.0)
        preds.append({r["image"]: pos(v) for r, v in zip(rows, pr)})
    return rows, gt, rng, preds


class T(unittest.TestCase):
    def test_per_seed_metrics_match_the_authors_scorer(self):
        rows, gt, rng, preds = fixture()
        out = R.rescore(rows, preds)
        self.assertEqual(out["n"], len(rows))
        self.assertEqual(out["missing_predictions"], [0, 0, 0])
        for s, pr in enumerate(preds):
            o = ref.official_eval([pr[r["image"]] for r in rows], [r["reading"] for r in rows], [r["range"] for r in rows])
            self.assertAlmostEqual(out["per_seed"][s]["acc_eps"], o["accuracy_epsilon"], places=12)
            self.assertAlmostEqual(out["per_seed"][s]["acc_theta"], o["accuracy_theta"], places=12)
            self.assertAlmostEqual(out["per_seed"][s]["ref"], o["gamma"], places=12)
            self.assertAlmostEqual(out["per_seed"][s]["rel"], o["theta"], places=12)
        for k in ("acc_eps", "acc_theta", "ref", "rel"):
            self.assertAlmostEqual(out["mean"][k], float(np.mean([p[k] for p in out["per_seed"]])), places=12)
        self.assertAlmostEqual(out["H"], out["mean"]["acc_eps"], places=12)

    def test_missing_prediction_counts_as_zero(self):
        rows, gt, rng, preds = fixture(n=300, seeds=1)
        full = R.rescore(rows, preds)
        cut = dict(preds[0])
        victim = rows[5]["image"]
        del cut[victim]
        out = R.rescore(rows, [cut])
        self.assertEqual(out["missing_predictions"], [1])
        hit0 = abs(0.0 - gt[5]) <= 0.01 * rng[5]
        hit_full = abs(float(preds[0][victim]) - gt[5]) <= 0.01 * rng[5]
        self.assertAlmostEqual(out["per_seed"][0]["acc_eps"] - full["per_seed"][0]["acc_eps"], (int(hit0) - int(hit_full)) / 300.0, places=12)

    def test_sci_notation_would_misread_but_positional_does_not(self):
        rows = [{"image": "1_img.jpg", "reading": "0.0", "range": 10.0}, {"image": "2_img.jpg", "reading": "2.0", "range": 10.0}]
        tiny = 4.5e-05                                                  # a prediction that is a hit for reading 0.0
        good = R.rescore(rows, [{"1_img.jpg": pos(tiny), "2_img.jpg": "2.01"}])
        bad = R.rescore(rows, [{"1_img.jpg": str(tiny), "2_img.jpg": "2.01"}])   # '4.5e-05' is read as -5 by the official extraction
        self.assertEqual(good["per_seed"][0]["acc_eps"], 1.0)
        self.assertEqual(bad["per_seed"][0]["acc_eps"], 0.5)

    def test_seed_mean_statistic_equals_the_preregistered_kernel(self):
        rows, gt, rng, preds = fixture(n=1700, seeds=3)
        out = R.rescore(rows, preds)
        hits = [np.array([abs(float(p[r["image"]]) - g) <= 0.01 * rg for r, g, rg in zip(rows, gt, rng)]) for p in preds]
        root = tempfile.mkdtemp()
        ind, outd = os.path.join(root, "in", "k"), os.path.join(root, "out")
        os.makedirs(ind)
        known = np.ones(len(rows), bool)
        np.savez(os.path.join(ind, "copy_hits_official.npz"), hits=np.zeros(len(rows), bool), known=known, names=np.array([r["image"] for r in rows]))
        for s, h in enumerate(hits):
            np.savez(os.path.join(ind, "f-r18-224-off-s%d_test_pred.npz" % s), hits=h, known=known)
        env = dict(os.environ, IN_ROOT=os.path.join(root, "in"), OUT_ROOT=outd)
        r = subprocess.run([sys.executable, KERNEL], env=env, capture_output=True, text=True, timeout=100)
        self.assertEqual(r.returncode, 0, r.stderr[-600:])
        line = [l for l in r.stdout.splitlines() if l.strip().startswith("H (seed mean)")][0]
        want = "H (seed mean) = %.4f   Wilson 95%% at n=%d: [%.4f, %.4f]   block bootstrap 95%% over %d blocks: [%.4f, %.4f]" % (
            out["H"], out["n"], out["wilson95"][0], out["wilson95"][1], out["n_blocks"], out["block_bootstrap95"][0], out["block_bootstrap95"][1])
        self.assertEqual(line.strip(), want)
        self.assertEqual(out["n_blocks"], 11)                           # ids 1..1700 -> (id - 1) // 160 -> 11 blocks


def v3_fixture(seeds=3, rs_seed=5):
    """Synthetic split v3: train.json rows with '0-3'-style ranges, test.json rows with numeric ranges, rows with no determinable range, one name without a label row."""
    rs = np.random.RandomState(rs_seed)
    train, test, names, rng_true = [], [], [], {}
    for i in range(1, 641):
        name = "%d_img.jpg" % i
        names.append(name)
        if i <= 400:
            span = [3.0, 6.0, 10.0][i % 3]
            gt = round(float(rs.rand()) * span, 2)
            train.append({"image": name, "reading": repr(gt), "range": "0-%d" % span})
            rng_true[name] = (gt, span)
        elif i <= 560:
            span = float([3, 6, 10, 80, 300][i % 5])
            gt = round(float(rs.rand()) * span, 2)
            test.append({"image": name, "reading": repr(gt), "range": span, "meter_type": "biao1"})
            rng_true[name] = (gt, span)
        elif i <= 600:
            train.append({"image": name, "reading": "none", "range": None, "detail": "no pointer"})   # no determinable range: excluded
            rng_true[name] = (0.0, float("nan"))
        else:
            train.append({"image": name, "reading": repr(1.0), "range": None, "detail": "The dial ranges from 0 to 50"})   # range read from the text
            rng_true[name] = (1.0, 50.0)
    man = {"B": 100, "test": names, "val": [], "block_of": {n: (int(n.split("_")[0]) - 1) // 100 for n in names}}
    preds = []
    for s in range(seeds):
        pm = {}
        for n in names:
            gt, span = rng_true[n]
            sp = 1.0 if np.isnan(span) else span
            ok = rs.rand() < 0.7
            pm[n] = pos(max(gt + ((rs.rand() * 2 - 1) * 0.008 * sp if ok else (0.05 + 0.1 * rs.rand()) * sp), 0.0))
        preds.append(pm)
    return train, test, man, preds, rng_true


class V3(unittest.TestCase):
    def test_per_seed_equals_score_v3_and_counts(self):
        train, test, man, preds, rng_true = v3_fixture()
        man2 = dict(man, test=man["test"] + ["999_img.jpg"], block_of=dict(man["block_of"], **{"999_img.jpg": 9}))      # a name without a label row
        out = R.rescore_v3(man2, train, test, preds)
        self.assertEqual((out["n_names"], out["n_rows"], out["n_excluded_unknown_range"], out["n"]), (641, 640, 40, 600))
        with tempfile.TemporaryDirectory() as d:
            for name, obj in (("man.json", man2), ("train.json", train), ("test.json", test)):
                with open(os.path.join(d, name), "w") as fh:
                    json.dump(obj, fh)
            for s, pm in enumerate(preds):
                with open(os.path.join(d, "pred.json"), "w") as fh:
                    json.dump(pm, fh)
                r = subprocess.run([sys.executable, SCORE_V3] + [os.path.join(d, x) for x in ("man.json", "train.json", "test.json", "pred.json")], capture_output=True, text=True, check=True)
                o = json.loads(r.stdout)
                self.assertEqual((o["n_scored"], o["n_excluded_unknown_range"]), (600, 40))
                for k in ("acc_eps", "acc_theta", "ref", "rel"):
                    self.assertAlmostEqual(out["per_seed"][s][k], o[k], places=12)

    def test_seed_mean_statistic_equals_the_preregistered_kernel_on_v3(self):
        train, test, man, preds, rng_true = v3_fixture()
        out = R.rescore_v3(man, train, test, preds)
        names = man["test"]
        known = np.array([not np.isnan(rng_true[n][1]) for n in names])
        root = tempfile.mkdtemp()
        ind, outd = os.path.join(root, "in", "k"), os.path.join(root, "out")
        os.makedirs(ind)
        with open(os.path.join(ind, "splits_v3.json"), "w") as fh:
            json.dump(man, fh)
        np.savez(os.path.join(ind, "copy_hits_v3.npz"), hits=np.zeros(len(names), bool), known=known)
        for s, pm in enumerate(preds):
            h = np.array([(not np.isnan(rng_true[n][1])) and abs(float(pm[n]) - rng_true[n][0]) <= 0.01 * rng_true[n][1] for n in names])
            np.savez(os.path.join(ind, "f-r18-224-v3-s%d_test_pred.npz" % s), hits=h, known=known)
        env = dict(os.environ, IN_ROOT=os.path.join(root, "in"), OUT_ROOT=outd)
        r = subprocess.run([sys.executable, KERNEL], env=env, capture_output=True, text=True, timeout=100)
        self.assertEqual(r.returncode, 0, r.stderr[-600:])
        line = [l for l in r.stdout.splitlines() if l.strip().startswith("H (seed mean)")][0]
        want = "H (seed mean) = %.4f   Wilson 95%% at n=%d: [%.4f, %.4f]   block bootstrap 95%% over %d blocks: [%.4f, %.4f]" % (
            out["H"], out["n"], out["wilson95"][0], out["wilson95"][1], out["n_blocks"], out["block_bootstrap95"][0], out["block_bootstrap95"][1])
        self.assertEqual(line.strip(), want)
        self.assertEqual(out["n_blocks"], 7)                           # ids 1..640 -> (id - 1) // 100 -> 7 blocks


if __name__ == "__main__":
    unittest.main()
