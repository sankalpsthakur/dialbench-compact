"""Rule test for tools/gpu_stage_b.py (FROZEN-4 sections 2 and 3) on synthetic Stage A result JSONs: no data, no model."""
import json, os, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import gpu_stage_b as G

CFG = os.path.join(HERE, "..", "kernels", "gpu-sweep1", "configs.json")
PARAMS = {"a1-mnv3s224-f100": 3.45e6, "a2-r18-224": 13.1e6, "a3-r18-320": 14.8e6, "a4-mnv3s320": 4.8e6, "a5-r34-224": 23.2e6, "a6-mnv3l224": 6.1e6, "a7-mnv3l320": 7.4e6}


def make_dir(scores, epoch_secs, nan_tag=None, short_tag=None, skip=()):
    d = tempfile.mkdtemp()
    with open(CFG) as fh:
        cands = json.load(fh)
    for c in cands:
        t = c["tag"]
        if t in skip:
            continue
        n = c["epochs"] - (5 if t == short_tag else 0)
        hist = [{"epoch": i + 1, "loss": 1.0, "secs": epoch_secs.get(t, 10.0)} for i in range(n)]
        if t == nan_tag:
            hist[3]["loss"] = float("nan")
        hist[-1]["val_ema"] = {"window": scores[t]}
        with open(os.path.join(d, t + "_result.json"), "w") as fh:
            json.dump({"config": c, "params": PARAMS[t], "history": hist}, fh)
    return d


BASE = {"a1-mnv3s224-f100": 0.62, "a2-r18-224": 0.66, "a3-r18-320": 0.70, "a4-mnv3s320": 0.695, "a5-r34-224": 0.67, "a6-mnv3l224": 0.65, "a7-mnv3l320": 0.69}


class T(unittest.TestCase):
    def test_primary_edge_and_full_list(self):
        o = G.plan(make_dir(BASE, {}), CFG)            # 10 s/epoch: every T_run about 14 min or less; default hardware = 2 GPUs billed, 2 workers
        self.assertEqual(o["P"]["tag"], "a3-r18-320")
        self.assertEqual(o["E"]["tag"], "a4-mnv3s320")  # 0.695 is within 1.0 point of 0.70 and has fewer parameters (4.8 M < 14.8 M)
        self.assertFalse(o["futility"])
        self.assertEqual(len(o["runs"]), 10)           # R11 and R12 would take the caps past 14,400 billed s (7,800 + 5,700 now, 9,900 + 5,700 with R11)
        self.assertEqual([k["tag"] for k in o["runs"]][:4], ["f-r18-320-off-s0", "f-r18-320-off-s1", "f-r18-320-off-s2", "f-r18-320-v3-s0"])
        c = o["runs"][3]["cfg"]
        self.assertTrue(c["run_test"] and c["split_file"] == "splits_v3.json" and c["split_sha256"] == G.PIN and c["seed"] == 0 and c["amp"] is True)
        self.assertEqual(o["runs"][0]["cfg"]["val_mode"], "split")
        self.assertNotIn("split_file", o["runs"][0]["cfg"])
        self.assertLessEqual(o["sum_caps_billed_s"], 25200)
        self.assertEqual((o["kernels"]["gpu-final-off"]["cap_billed_s"], o["kernels"]["gpu-final-v3"]["cap_billed_s"]), (7800, 5700))
        m = {x["label"]: x for x in o["int8_models"]}
        self.assertEqual(sorted(m), ["e-off", "e-v3", "p-off", "p-v3"])        # seed-0 models only
        self.assertEqual(m["p-v3"]["weights"], "f-r18-320-v3-s0_ema.pt")
        self.assertEqual(m["p-v3"]["cfg"]["split_sha256"], G.PIN)
        self.assertEqual(m["p-off"]["cfg"]["val_mode"], "split")
        self.assertEqual(m["p-off"]["cfg"]["pretrained"], "none")

    def test_budget_cutoff_at_25_minutes(self):
        # a3: 80 epochs x 17.5 s = 23.33 min + 1.5 = 24.83 -> 25 min charge; E (a4, 80 x 17.5 too) the same
        secs = {"a3-r18-320": 17.5, "a4-mnv3s320": 17.5}
        o = G.plan(make_dir(BASE, secs), CFG)           # 2 GPUs billed, 2 workers: R1..R5 (caps 8,400 + 4,800 = 13,200); R6 would give 12,000 + 4,800 > 14,400
        self.assertEqual(len(o["runs"]), 5)
        self.assertEqual(sum(1 for k in o["runs"] if k["role"] == "P" and k["split"] == "off"), 4)
        o1 = G.plan(make_dir(BASE, secs), CFG, billed_gpus=1, workers=1)   # one GPU (an L4): R1..R7 (caps 11,400 + 2,400); R8 would add 1,800 more
        self.assertEqual(len(o1["runs"]), 7)
        self.assertEqual(sum(1 for k in o1["runs"] if k["role"] == "P" and k["split"] == "off"), 5)

    def test_hardware_is_read_from_the_stage_a_wrapper_summary(self):
        d = make_dir(BASE, {})
        with open(os.path.join(d, "wrapper_summary.json"), "w") as fh:
            json.dump({"gpus": 1, "workers": 1}, fh)
        o = G.plan(d, CFG)
        self.assertEqual((o["billed_gpus"], o["workers"]), (1, 1))
        self.assertEqual(len(o["runs"]), 12)           # 12 x 15 min on one GPU: caps 9,300 + 5,100 = 14,400, exactly the allowance

    def test_makespan_and_cap_formula(self):
        self.assertEqual(G.makespan_s([900, 900, 900], 2), 1800)
        self.assertEqual(G.makespan_s([900, 900, 900], 1), 2700)
        self.assertEqual(G.kernel_cap_billed_s([1500], 2, 2), 4800)      # 2 x (600 + 1.2 x 1500) = 4,800
        self.assertEqual(G.kernel_cap_billed_s([], 2, 2), 0)

    def test_edge_equals_primary_drops_edge_runs(self):
        sc = dict(BASE, **{"a4-mnv3s320": 0.60, "a7-mnv3l320": 0.60})   # no other candidate within 1.0 point of a3
        o = G.plan(make_dir(sc, {}), CFG)
        self.assertEqual(o["E"]["tag"], o["P"]["tag"])
        self.assertEqual([k["role"] for k in o["runs"]], ["P"] * 8)     # 2 GPUs billed, 2 workers: caps 7,800 + 5,700 fit

    def test_futility_stop(self):
        sc = {k: v - 0.08 for k, v in BASE.items()}   # best 0.62 < 0.64
        o = G.plan(make_dir(sc, {}), CFG)
        self.assertTrue(o["futility"])
        self.assertEqual([(k["role"], k["split"], k["seed"]) for k in o["runs"]], [("P", "off", 0), ("P", "off", 1), ("P", "off", 2), ("P", "v3", 0)])

    def test_invalid_incomplete_slow_and_missing_candidates_are_not_eligible(self):
        secs = {"a3-r18-320": 40.0}                    # 80 x 40 s = 53.3 min + 1.5 > 45: valid but not eligible
        d = make_dir(BASE, secs, nan_tag="a7-mnv3l320", short_tag="a4-mnv3s320", skip=("a2-r18-224",))
        o = G.plan(d, CFG, invalid=["a5-r34-224"])
        by = {r["tag"]: r for r in o["rows"]}
        self.assertFalse(by["a3-r18-320"]["eligible"] and True)
        self.assertTrue(by["a3-r18-320"]["valid"])
        self.assertFalse(by["a7-mnv3l320"]["valid"])
        self.assertFalse(by["a4-mnv3s320"]["valid"])
        self.assertFalse(by["a2-r18-224"]["valid"])
        self.assertFalse(by["a5-r34-224"]["valid"])
        self.assertEqual(o["P"]["tag"], "a6-mnv3l224")  # best remaining: 0.65 (a1 0.62)

    def test_slow_primary_gets_the_minimum_plan_and_caps_stay_inside_seven_hours(self):
        secs = {"a3-r18-320": 28.0}                    # 80 x 28 s = 37.3 min + 1.5 = 38.8 -> charge 39: eligible under the 45 min cap
        o = G.plan(make_dir(BASE, secs), CFG)
        self.assertEqual(o["P"]["tag"], "a3-r18-320")
        self.assertEqual([(k["split"], k["seed"]) for k in o["runs"]], [("off", 0), ("off", 1), ("off", 2)])   # 2 GPUs billed: 3 official seeds fill the 14,400 billed s
        self.assertLessEqual(o["sum_caps_billed_s"], 25200)
        o1 = G.plan(make_dir(BASE, secs), CFG, billed_gpus=1, workers=1)
        self.assertEqual([(k["split"], k["seed"]) for k in o1["runs"]], [("off", 0), ("off", 1), ("off", 2), ("v3", 0)])
        self.assertLessEqual(o1["sum_caps_billed_s"], 25200)

    def test_combined_kernel_and_actual_stage_a_usage(self):
        secs = {"a2-r18-224": 19.5, "a3-r18-320": 1.0, "a4-mnv3s320": 1.0, "a1-mnv3s224-f100": 1.0}   # a2: 80 x 19.5 s = 26 min + 1.5 -> 28 min charge
        sc = dict(BASE, **{"a2-r18-224": 0.70, "a4-mnv3s320": 0.66})                                    # a2 is the primary and no candidate is within a point of it (E = P), as in the real Stage A
        d = make_dir(sc, secs, skip=("a3-r18-320", "a5-r34-224", "a6-mnv3l224", "a7-mnv3l320"))
        lit = G.plan(d, CFG)                                                                             # amendment 5 as written: Stage A booked at its cap, two kernels
        self.assertEqual(lit["P"]["tag"], "a2-r18-224")
        self.assertEqual([(k["split"], k["seed"]) for k in lit["runs"]], [("off", 0), ("off", 1), ("off", 2)])
        o = G.plan(d, CFG, stage_a_billed_s=7800, combined=True)                                         # amendment 6: actual Stage A usage, one kernel
        self.assertEqual(list(o["kernels"]), ["gpu-final"])
        k = o["kernels"]["gpu-final"]
        self.assertEqual([(r["split"], r["seed"]) for r in o["runs"]], [("off", 0), ("off", 1), ("off", 2), ("v3", 0), ("off", 3), ("off", 4), ("v3", 1), ("v3", 2)])
        self.assertTrue(k["has_v3"] and k["has_off"])
        self.assertEqual(k["makespan_s"], 4 * 1680)                                                      # 8 runs on 2 workers = 4 rounds
        self.assertEqual(k["cap_billed_s"], 17400)
        self.assertLessEqual(o["sum_caps_billed_s"], 25200)
        o2 = G.plan(d, CFG, stage_a_billed_s=8000, combined=True)                                        # 200 billed s more used by Stage A: the 8-run cap no longer fits
        self.assertEqual(len(o2["runs"]), 6)
        self.assertEqual(o2["kernels"]["gpu-final"]["cap_billed_s"], 13500)

    def test_tie_goes_to_fewer_parameters(self):
        sc = dict(BASE, **{"a3-r18-320": 0.70, "a4-mnv3s320": 0.70})
        o = G.plan(make_dir(sc, {}), CFG)
        self.assertEqual(o["P"]["tag"], "a4-mnv3s320")


if __name__ == "__main__":
    unittest.main()
