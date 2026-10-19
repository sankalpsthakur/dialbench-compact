"""Logic test of the GPU wrapper (kernels/gpu-sweep1/main.py, kind torch-train-gpu) with a fake nvidia-smi and stub torch / trainer modules: no torch, no GPU, no data.
The workers really are forked processes (as on Kaggle), so the claim queue, the log forwarding, the hard limit and the summary merge are exercised.
Checks: GPU counting (billing is per GPU), 2 workers on 2 GPUs / 1 worker on 1 GPU, CUDA / npz pre-flight aborts, budget skips, a failed config, the hard limit."""
import json, os, re, stat, sys, tempfile, types, unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "..", "kernels", "gpu-sweep1", "main.py")


def run_wrapper(n_gpu=2, cuda=True, missing=(), fail_tag=None, run_sleep=0.05, edits=()):
    d = tempfile.mkdtemp()
    for sub in ("input/x", "working", "bin", "ran"):
        os.makedirs(os.path.join(d, sub))
    for n in ("resnet18", "resnet34"):
        np.savez(os.path.join(d, "input", "x", n + "_imagenet_tv.npz"), w=np.zeros(2))
    smi = os.path.join(d, "bin", "nvidia-smi")
    with open(smi, "w") as fh:
        fh.write('#!/bin/bash\nif [ "$1" = "-L" ]; then for ((i=0; i<%d; i++)); do echo "GPU $i: Stub T4 (UUID: GPU-$i)"; done; else echo "stub nvidia-smi table"; fi\n' % n_gpu)
    os.chmod(smi, os.stat(smi).st_mode | stat.S_IEXEC)
    with open(MAIN) as fh:
        src = fh.read()
    src = src.replace('"/kaggle/input/', '"%s/input/' % d).replace('OUT = "/kaggle/working"', 'OUT = "%s/working"' % d).replace("POLL_S = 3", "POLL_S = 0.05")
    for old, new in edits:
        assert old in src, old
        src = src.replace(old, new)
    cudamod = types.SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda i: "Stub T4 vis=" + os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
                                    device_count=lambda: 1, empty_cache=lambda: None)
    torch = types.ModuleType("torch")
    torch.__version__ = "stub"; torch.cuda = cudamod; torch.from_numpy = lambda x: x; torch.version = types.SimpleNamespace(cuda="stub-12.8")
    torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False))
    trainer = types.ModuleType("rpm_train_torch")

    def run(cfg):
        import time
        if cfg["tag"] == fail_tag:
            raise RuntimeError("boom")
        open(os.path.join(d, "ran", "%s__%d" % (cfg["tag"], os.getpid())), "w").close()
        time.sleep(run_sleep)

    class BB:
        def load_state_dict(self, sd, strict=False):
            return types.SimpleNamespace(missing_keys=list(missing), unexpected_keys=[])
    trainer.run = run; trainer.make_backbone = lambda arch: BB()
    saved = {k: sys.modules.get(k) for k in ("torch", "rpm_train_torch")}
    old_path = os.environ["PATH"]
    sys.modules["torch"], sys.modules["rpm_train_torch"] = torch, trainer
    os.environ["PATH"] = os.path.join(d, "bin") + os.pathsep + old_path
    err = None
    try:
        try:
            exec(compile(src, "main.py", "exec"), {"__name__": "wrapper"})
        except BaseException as e:   # the wrapper ends with an assertion when nothing completed
            err = e
    finally:
        os.environ["PATH"] = old_path
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    ran = sorted(os.listdir(os.path.join(d, "ran")))
    summ = None
    if os.path.exists(os.path.join(d, "working", "wrapper_summary.json")):
        with open(os.path.join(d, "working", "wrapper_summary.json")) as fh:
            summ = json.load(fh)
    logs = {}
    for k in (0, 1):
        f = os.path.join(d, "working", "worker%d.log" % k)
        if os.path.exists(f):
            with open(f) as fh:
                logs[k] = fh.read()
    return dict(err=err, ran=ran, summary=summ, logs=logs)


class T(unittest.TestCase):
    def test_two_gpus_two_workers_each_config_once(self):
        r = run_wrapper(n_gpu=2, run_sleep=0.15)
        self.assertIsNone(r["err"])
        tags = [x.split("__")[0] for x in r["ran"]]
        self.assertEqual(sorted(tags), sorted(set(tags)))            # every config ran exactly once (atomic claims)
        self.assertEqual(len(tags), 7)
        self.assertEqual(len({x.split("__")[1] for x in r["ran"]}), 2)   # both worker processes did work
        s = r["summary"]
        self.assertEqual((s["gpus"], s["workers"], s["exit_codes"]), (2, 2, [0, 0]))
        self.assertTrue(all(v["status"] == "done" for v in s["status"].values()))
        self.assertIn("DEVICE w0 Stub T4 vis=0", r["logs"][0])
        self.assertIn("DEVICE w1 Stub T4 vis=1", r["logs"][1])       # each worker sees only its own GPU
        self.assertAlmostEqual(s["hard_s"], (10800 - 600) / 2.0)     # billed cap divided by the GPUs held

    def test_one_gpu_one_worker(self):
        r = run_wrapper(n_gpu=1)
        self.assertIsNone(r["err"])
        s = r["summary"]
        self.assertEqual((s["gpus"], s["workers"]), (1, 1))
        self.assertEqual(len(r["ran"]), 7)
        self.assertNotIn(1, r["logs"])
        self.assertAlmostEqual(s["hard_s"], 10800 - 600)             # one GPU held: the wall limit is the whole billed cap

    def test_no_gpu_listed_aborts_before_any_training(self):
        r = run_wrapper(n_gpu=0)
        self.assertIsInstance(r["err"], AssertionError)
        self.assertEqual(r["ran"], [])
        self.assertIsNone(r["summary"])

    def test_no_cuda_in_the_workers_ends_in_error_without_training(self):
        r = run_wrapper(n_gpu=2, cuda=False)
        self.assertIsInstance(r["err"], AssertionError)
        self.assertEqual(r["ran"], [])
        self.assertTrue(all(v["status"] == "not_started" for v in r["summary"]["status"].values()))
        self.assertIn("no CUDA device", r["logs"][0])

    def test_missing_backbone_keys_abort_the_workers(self):
        r = run_wrapper(n_gpu=2, missing=["layer1.0.conv1.weight"])
        self.assertIsInstance(r["err"], AssertionError)
        self.assertEqual(r["ran"], [])
        self.assertIn("backbone keys missing", r["logs"][0])

    def test_budget_skips_everything_when_the_cap_is_tiny(self):
        r = run_wrapper(n_gpu=2, edits=[("CAP_BILLED_S = 10800", "CAP_BILLED_S = 1000")])   # wall limit 350 s, start budget negative
        self.assertIsInstance(r["err"], AssertionError)
        self.assertEqual(r["ran"], [])
        self.assertTrue(all(v["status"] == "skipped_budget" for v in r["summary"]["status"].values()))

    def test_cpu_twin_runs_the_same_orchestration_without_cuda(self):
        r = run_wrapper(n_gpu=0, cuda=False, edits=[("ALLOW_CPU = False", "ALLOW_CPU = True")])
        self.assertIsNone(r["err"])
        s = r["summary"]
        self.assertEqual((s["gpus"], s["workers"], s["exit_codes"]), (2, 2, [0, 0]))     # two emulated GPUs, two forked workers, no CUDA assertion
        self.assertEqual(len(r["ran"]), 7)
        self.assertIn("cpu-twin", r["logs"][0])

    def test_failed_config_does_not_stop_the_queue(self):
        r = run_wrapper(n_gpu=2, fail_tag="a3-r18-320")
        self.assertIsNone(r["err"])
        st = r["summary"]["status"]
        self.assertEqual(st["a3-r18-320"]["status"], "failed")
        self.assertEqual(sum(1 for v in st.values() if v["status"] == "done"), 6)

    def test_hard_limit_terminates_workers_and_marks_claimed_configs_interrupted(self):
        r = run_wrapper(n_gpu=2, run_sleep=30,
                        edits=[("HARD_S = (CAP_BILLED_S - 600.0) / _B", "HARD_S = 1.5"), ("BUDGET_S = HARD_S - 600.0", "BUDGET_S = 5000.0")])
        self.assertIsInstance(r["err"], AssertionError)              # nothing completed
        s = r["summary"]
        self.assertTrue(s["killed_at_hard_limit"])
        self.assertEqual(sum(1 for v in s["status"].values() if v["status"] == "interrupted"), 2)   # one claimed config per worker


if __name__ == "__main__":
    unittest.main()
