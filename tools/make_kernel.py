#!/usr/bin/env python3
"""Generate kernels/<name>/{main.py,script.py,kernel-metadata.json} for the gauges track.
kinds: torch-train | torch-train-gpu | torch-train-export | jax-train | export | jax-check | int8-sweep
usage: make_kernel.py NAME KIND --config JSON_LIST [--sources a,b] [--internet] [--tpu] [--cpu-jax]
  GPU (FROZEN-4 amendment 5): make_kernel.py NAME torch-train-gpu --config-file configs.json --est-json est_s.json --cap-billed-s N --workers 2 --internet --gpu NvidiaTeslaT4 [--sources rpm-prep,model-port[,rpm-split3]]"""
import argparse, json, os, subprocess, sys
from pprint import pformat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("kind")
ap.add_argument("--config", default="[]", help="JSON list of config dicts (train) or export jobs")
ap.add_argument("--config-file", default=None, help="path of a JSON file holding the list (overrides --config)")
ap.add_argument("--est-json", default=None, help="torch-train-gpu: JSON {tag: expected seconds}, used only by the start/stop budget")
ap.add_argument("--cap-billed-s", type=int, default=None, help="torch-train-gpu: billed GPU-seconds the kernel may use (= the push -t value); wall limit = cap / GPUs held")
ap.add_argument("--workers", type=int, default=1, help="torch-train-gpu: training processes, one per GPU (1 or 2)")
ap.add_argument("--slug", default=None, help="kernel slug after bench-gauges- (default: NAME); lets several kernel directories push the same kernel ref")
ap.add_argument("--allow-cpu", action="store_true", help="torch-train-gpu: CPU twin for orchestration smoke tests on a Kaggle CPU kernel (no GPU needed, workers run on CPU); never for the GPU kernels")
ap.add_argument("--sources", default="rpm-prep,model-port")
ap.add_argument("--internet", action="store_true")
ap.add_argument("--tpu", action="store_true")
ap.add_argument("--gpu", default=None, help="enable GPU with this machine_shape, e.g. NvidiaTeslaT4 or NvidiaL4 (needs locks/kaggle-gpu and quota)")
ap.add_argument("--cpu-jax", action="store_true", help="force JAX_PLATFORMS=cpu (smoke tests)")
ap.add_argument("--prelude", default=None, help="standalone script exec'd first inside try/except (its stdout Tee is reset afterwards)")
ap.add_argument("--notebook", action="store_true", help="emit an .ipynb (single code cell) instead of a script kernel")
a = ap.parse_args()
kdir = os.path.join(ROOT, "kernels", a.name)
os.makedirs(kdir, exist_ok=True)
cfgs = json.load(open(a.config_file)) if a.config_file else json.loads(a.config)
S = lambda f: os.path.join(ROOT, "src", f)
pre = None
if a.kind == "torch-train":
    mods = ["dialbench_metrics.py", "rpm_common.py", "rpm_train_torch.py"]
    main = f"""import sys, traceback
from rpm_train_torch import run
CONFIGS = {pformat(cfgs, width=140)}
for _cfg in CONFIGS:
    try:
        run(_cfg)
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
print("ALL CONFIGS DONE", flush=True)
"""
elif a.kind == "torch-train-gpu":
    # GPU trainer wrapper (FROZEN-4 amendment 5): the parent process never touches CUDA; it counts the GPUs held (billing is per GPU), forks one training
    # process per GPU that pulls configs from a shared claim queue, enforces a hard wall limit derived from the billed cap, merges the summaries.
    # The template uses placeholders, not an f-string.
    if not a.gpu and not a.allow_cpu:
        sys.exit("torch-train-gpu needs --gpu NvidiaTeslaT4 (or another machine shape)")
    if a.gpu and a.allow_cpu:
        sys.exit("--allow-cpu is only for the CPU twin: do not combine it with --gpu")
    if not a.est_json:
        sys.exit("torch-train-gpu needs --est-json")
    if not a.cap_billed_s:
        sys.exit("torch-train-gpu needs --cap-billed-s (billed GPU-seconds; also the push -t value)")
    if a.workers not in (1, 2):
        sys.exit("--workers must be 1 or 2")
    est = json.load(open(a.est_json))
    tags = [c["tag"] for c in cfgs]
    if len(set(tags)) != len(tags):
        sys.exit("duplicate tags")
    for c in cfgs:
        if c.get("amp") is not True:
            sys.exit("config %s: amp must be true (the trainer uses CUDA only then)" % c["tag"])
        if c["tag"] not in est:
            sys.exit("config %s: no expected seconds in --est-json" % c["tag"])
        if c.get("pretrained") == "timm" and not a.internet:
            sys.exit("config %s uses timm weights: pass --internet" % c["tag"])
    mods = ["dialbench_metrics.py", "rpm_common.py", "rpm_train_torch.py"]
    GPU_MAIN = """import os, sys, time, json, traceback, subprocess, glob, gc
import multiprocessing as mp
import numpy as np
import torch
from rpm_train_torch import run, make_backbone

CONFIGS = __CONFIGS__
EST_S = __EST__   # expected seconds per config on ONE GPU (FROZEN-4 / its addendum S); used only by the start/stop budget below
CAP_BILLED_S = __CAP__   # billed GPU-seconds this kernel may use (the push -t value); the quota is billed per GPU held, so wall limit = cap / GPUs held
WORKERS = __WORKERS__   # most training processes, one per GPU
ALLOW_CPU = __ALLOW_CPU__   # True only in the CPU twin used to smoke-test this orchestration on a Kaggle CPU kernel; False in every GPU kernel
OUT = "/kaggle/working"
POLL_S = 3
_T0 = time.time()
print("WRAPPER torch", torch.__version__, flush=True)   # the parent never calls torch.cuda: CUDA must first be touched in the forked workers
try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout, flush=True)
except Exception as _e:
    print("nvidia-smi failed", repr(_e), flush=True)


def _gpu_names():
    try:
        _out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        _n = [l.strip() for l in _out.splitlines() if l.startswith("GPU ")]
        if _n:
            return _n
    except Exception as _e:
        print("nvidia-smi -L failed", repr(_e), flush=True)
    try:   # fallback that does not touch CUDA in this process: count the devices in a child interpreter
        _c = int(subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True, timeout=180).stdout.split()[-1])
        return ["GPU %d (counted by torch)" % _i for _i in range(_c)]
    except Exception as _e:
        print("torch device count failed", repr(_e), flush=True)
    return []


_names = _gpu_names()
if ALLOW_CPU and not _names:
    _names = ["cpu-twin"] * WORKERS   # emulate WORKERS GPUs so that the billing arithmetic and the two-worker path are exercised
assert _names, "no GPU listed by nvidia-smi -L or counted by torch: exit before any training so that no GPU quota is spent on a CPU run"
assert all(_c.get("amp") is True for _c in CONFIGS), "every config needs amp=True (the trainer uses CUDA only then)"
_B = len(_names)   # billed GPU-seconds per wall second
_W = max(1, min(WORKERS, _B))
HARD_S = (CAP_BILLED_S - 600.0) / _B   # the parent stops the workers at this wall time (600 s of margin for the session start before the script and for the final merge)
BUDGET_S = HARD_S - 600.0   # a config is started only if it is expected to end before this many seconds after the wrapper started
print("WRAPPER gpus %d %s | workers %d | billed rate %dx | hard wall limit %.0f s, start budget %.0f s (cap %d billed s)" % (_B, _names, _W, _B, HARD_S, BUDGET_S, CAP_BILLED_S), flush=True)
os.makedirs(OUT + "/claims", exist_ok=True)


def _worker(k):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(k)   # before the first CUDA call of this process
    sys.stdout = sys.stderr = open("%s/worker%d.log" % (OUT, k), "a", buffering=1)   # the parent forwards this file to the kernel log
    print("WORKER", k, "of", _W, "pid", os.getpid(), "CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
    if ALLOW_CPU:
        print("DEVICE w%d cpu-twin (no CUDA, orchestration smoke test)" % k, "torch", torch.__version__, flush=True)
    else:
        assert torch.cuda.is_available(), "no CUDA device: exit before any training so that no GPU quota is spent on a CPU run"
        print("DEVICE w%d" % k, torch.cuda.get_device_name(0), "count", torch.cuda.device_count(), "torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
        torch.backends.cudnn.benchmark = True
    _seen = set()
    for _c in CONFIGS:   # pre-flight: an npz backbone must load without missing keys (the trainer loads it with strict=False)
        if str(_c["pretrained"]).startswith("npz:") and (_c["arch"], _c["pretrained"]) not in _seen:
            _seen.add((_c["arch"], _c["pretrained"]))
            _f = glob.glob("/kaggle/input/**/" + _c["pretrained"][4:] + "_imagenet_tv.npz", recursive=True)
            assert _f, "npz weights not found for " + _c["pretrained"]
            _sd = {n: torch.from_numpy(v) for n, v in np.load(_f[0]).items()}
            _m = make_backbone(_c["arch"]).load_state_dict(_sd, strict=False)
            print("PREFLIGHT", _c["arch"], _f[0], "missing", _m.missing_keys, "unexpected", _m.unexpected_keys, flush=True)
            assert not _m.missing_keys, "backbone keys missing: the run would train from random init"
    status = {}
    scale = 1.0
    for cfg in CONFIGS:
        tag = cfg["tag"]
        el = time.time() - _T0
        est = EST_S[tag] * scale
        if el + est > BUDGET_S:
            print("SKIPPED (budget): %s elapsed %.0f s + expected %.0f s > %.0f s" % (tag, el, est, BUDGET_S), flush=True)
            status[tag] = {"status": "skipped_budget", "elapsed_s": round(el), "expected_s": round(est)}
            continue
        try:
            os.mkdir(OUT + "/claims/" + tag)   # atomic claim: the first worker to get here runs the config
        except FileExistsError:
            continue
        status.pop(tag, None)
        t = time.time()
        try:
            run(cfg)
            dt = time.time() - t
            scale = min(3.0, max(0.5, dt / EST_S[tag]))
            status[tag] = {"status": "done", "secs": round(dt), "expected_s": EST_S[tag]}
            print("CONFIG DONE %s in %.0f s (expected %.0f s, scale now %.2f)" % (tag, dt, EST_S[tag], scale), flush=True)
        except Exception:
            traceback.print_exc(); sys.stdout.flush()
            status[tag] = {"status": "failed", "secs": round(time.time() - t)}
        gc.collect(); torch.cuda.empty_cache()
        with open("%s/wrapper_summary_w%d.json" % (OUT, k), "w") as fh:
            json.dump({"worker": k, "status": status}, fh, indent=1)
    with open("%s/wrapper_summary_w%d.json" % (OUT, k), "w") as fh:
        json.dump({"worker": k, "status": status}, fh, indent=1)
    print("WORKER", k, "finished", flush=True)


_ctx = mp.get_context("fork")
_procs = [_ctx.Process(target=_worker, args=(_k,)) for _k in range(_W)]
for _p in _procs:
    _p.start()
_pos, _buf = [0] * _W, [""] * _W


def _drain(final=False):
    for _k in range(_W):
        _f = "%s/worker%d.log" % (OUT, _k)
        if not os.path.exists(_f):
            continue
        with open(_f) as _fh:
            _fh.seek(_pos[_k]); _chunk = _fh.read(); _pos[_k] = _fh.tell()
        _lines = (_buf[_k] + _chunk).split("\\n")
        _buf[_k] = _lines.pop()
        if final and _buf[_k]:
            _lines.append(_buf[_k]); _buf[_k] = ""
        for _ln in _lines:
            print("[w%d] %s" % (_k, _ln), flush=True)


_killed = False
while any(_p.is_alive() for _p in _procs):
    _drain()
    if time.time() - _T0 > HARD_S:
        print("HARD LIMIT: %.0f s reached, terminating the workers" % HARD_S, flush=True)
        for _p in _procs:
            if _p.is_alive():
                _p.terminate()
        _killed = True
        break
    time.sleep(POLL_S)
for _p in _procs:
    _p.join(timeout=60)
_drain(final=True)
_rank = {"done": 3, "failed": 2, "skipped_budget": 1}
_merged = {}
for _k in range(_W):
    _f = "%s/wrapper_summary_w%d.json" % (OUT, _k)
    if os.path.exists(_f):
        with open(_f) as _fh:
            for _tag, _st in json.load(_fh)["status"].items():
                if _tag not in _merged or _rank[_st["status"]] > _rank[_merged[_tag]["status"]]:
                    _merged[_tag] = dict(_st, worker=_k)
for _c in CONFIGS:
    if _c["tag"] not in _merged or _merged[_c["tag"]]["status"] == "skipped_budget":
        if os.path.isdir(OUT + "/claims/" + _c["tag"]):
            _merged[_c["tag"]] = {"status": "interrupted"}
        elif _c["tag"] not in _merged:
            _merged[_c["tag"]] = {"status": "not_started"}
_summary = {"elapsed_s": round(time.time() - _T0), "cap_billed_s": CAP_BILLED_S, "gpus": _B, "gpu_names": _names, "workers": _W, "hard_s": HARD_S, "budget_s": BUDGET_S,
            "killed_at_hard_limit": _killed, "exit_codes": [_p.exitcode for _p in _procs], "status": _merged}
with open(OUT + "/wrapper_summary.json", "w") as _fh:
    json.dump(_summary, _fh, indent=1)
print("WRAPPER SUMMARY", json.dumps(_summary), flush=True)
print("ALL CONFIGS DONE", flush=True)
assert any(_s["status"] == "done" for _s in _merged.values()), "no config completed: see the [w*] lines above"
"""
    main = GPU_MAIN.replace("__CONFIGS__", pformat(cfgs, width=140)).replace("__EST__", pformat(est, width=140)).replace("__CAP__", str(a.cap_billed_s)).replace("__WORKERS__", str(a.workers)).replace("__ALLOW_CPU__", str(bool(a.allow_cpu)))
elif a.kind == "torch-train-export":
    mods = ["dialbench_metrics.py", "rpm_common.py", "jax_resnet.py", "rpm_train_torch.py", "rpm_export.py", "rpm_train_jax.py", "rpm_jax_check.py"]
    pre = os.path.join(ROOT, "kernels", "_common", "pip_ort_header.py")
    main = f"""import sys, traceback
from rpm_train_torch import run
from rpm_export import export_and_eval
CONFIGS = {pformat(cfgs, width=140)}
for _cfg in CONFIGS:
    try:
        run(_cfg)
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
        continue
    try:
        export_and_eval(_cfg, "/kaggle/working/" + _cfg["tag"] + "_ema.pt", "torch", run_test=_cfg.get("run_test", False))
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
print("ALL CONFIGS DONE", flush=True)
"""
elif a.kind == "jax-train":
    mods = ["dialbench_metrics.py", "rpm_common.py", "jax_resnet.py", "rpm_train_jax.py"]
    main = f"""import sys, traceback
from rpm_train_jax import run
CONFIGS = {pformat(cfgs, width=140)}
for _cfg in CONFIGS:
    try:
        run(_cfg)
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
print("ALL CONFIGS DONE", flush=True)
"""
elif a.kind == "int8-sweep":
    mods = ["dialbench_metrics.py", "rpm_common.py", "rpm_train_torch.py", "rpm_int8_sweep.py"]
    pre = os.path.join(ROOT, "kernels", "_common", "pip_ort_header.py")
    main = f"""import sys, traceback, glob, json
from rpm_int8_sweep import run_model
MODELS = {pformat(cfgs, width=140)}
for _m in MODELS:
    try:
        _w = glob.glob("/kaggle/input/**/" + _m["weights"], recursive=True)[0]
        run_model(_m["label"], _m["cfg"], _w, "/kaggle/working")
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
print("SWEEP DONE", flush=True)
"""
elif a.kind == "copy-hits":
    mods = ["dialbench_metrics.py", "rpm_common.py", "rpm_train_torch.py"]
    main = f"""import sys, traceback, json
import numpy as np
from rpm_train_torch import prepare, get_cfg, load_data
from rpm_common import knn_copy_pred, official_hits
JOBS = {pformat(cfgs, width=140)}
for _job in JOBS:
    try:
        c = get_cfg(_job["cfg"]); D = prepare(c); tt = D.test
        src = np.arange(len(D.last)) if D.mode == "official" else D.tr_idx   # official: every train.json row (as the audit floor); v3: the split's train part
        pred = knn_copy_pred(D.X, D.img_idx[src], D.last[src], tt.img_idx)
        hits = official_hits(pred, tt.last, np.where(tt.known, tt.rng, 1.0))
        out = dict(hits=hits, known=tt.known, last=tt.last, rng=tt.rng, pred=pred, img_idx=tt.img_idx)
        if D.mode == "official":
            _b, _tr, _te, _X = load_data(c)
            out["names"] = np.array([it["image"] for it in _te])
        np.savez("/kaggle/working/copy_hits_" + _job["label"] + ".npz", **out)
        print(_job["label"], "copy floor", float(hits[tt.known].mean()), "n", int(tt.known.sum()), "source rows", int(len(src)), flush=True)
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
print("COPY DONE", flush=True)
"""
elif a.kind in ("export", "jax-check"):
    mods = ["dialbench_metrics.py", "rpm_common.py", "jax_resnet.py", "rpm_train_torch.py", "rpm_train_jax.py", "rpm_export.py", "rpm_jax_check.py"]
    pre = os.path.join(ROOT, "kernels", "_common", "pip_ort_header.py")
    main = f"""import sys, traceback, json
from rpm_export import export_and_eval
JOBS = {pformat(cfgs, width=140)}
for _job in JOBS:
    try:
        if _job.get("check_equivalence"):
            from rpm_jax_check import check_equivalence
            from rpm_train_torch import load_data, build_train_arrays, get_cfg
            from rpm_common import Targets
            c = get_cfg(_job["cfg"]); base, train, test, X = load_data(c)
            last, span, how, fam, img_idx = build_train_arrays(train); T = Targets(span)
            print("EQUIVALENCE", json.dumps(check_equivalence(c, _job["weights"], X, img_idx[:8], T.nb, T.ns)), flush=True)
        export_and_eval(_job["cfg"], _job["weights"], _job["kind"], run_test=_job.get("run_test", False))
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
print("EXPORT DONE", flush=True)
"""
else:
    sys.exit("bad kind")
if a.prelude:
    pre_src = open(os.path.join(ROOT, a.prelude)).read()
    main = ("import sys as _sys, traceback as _tb\n_PRELUDE = " + repr(pre_src) + "\ntry:\n    exec(compile(_PRELUDE, 'prelude', 'exec'), {'__name__': 'prelude'})\nexcept BaseException:\n    _tb.print_exc()\n"
            "_sys.stdout = _sys.__stdout__\n" + main)
open(os.path.join(kdir, "main.py"), "w").write(main)
pre_path = None
if a.cpu_jax:
    pre_path = os.path.join(kdir, "pre.py")
    open(pre_path, "w").write('import os\nos.environ["JAX_PLATFORMS"] = "cpu"\n' + (open(pre).read() if pre else ""))
elif pre:
    pre_path = pre
cmd = [sys.executable, os.path.join(ROOT, "tools", "build_bundle.py"), kdir, "script.py", "--mods"] + [S(m) for m in mods] + ["--main", os.path.join(kdir, "main.py")]
if pre_path:
    cmd += ["--pre", pre_path]
subprocess.check_call(cmd)
code_file = "script.py"
if a.notebook:
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": open(os.path.join(kdir, "script.py")).read().splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    json.dump(nb, open(os.path.join(kdir, "script.ipynb"), "w"))
    code_file = "script.ipynb"
slug = a.slug or a.name
meta = {"id": f"sankalpsthakur/bench-gauges-{slug}", "title": f"bench-gauges-{slug}", "code_file": code_file, "language": "python",
        "kernel_type": "notebook" if a.notebook else "script", "is_private": "true", "enable_gpu": "true" if a.gpu else "false", "enable_tpu": "true" if a.tpu else "false",
        "enable_internet": "true" if a.internet else "false", "machine_shape": "TpuV5E8" if a.tpu else (a.gpu or ""),
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [f"sankalpsthakur/bench-gauges-{s}" for s in a.sources.split(",") if s],
        "model_sources": []}
json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
print("kernel", a.name, "ready:", kdir)
