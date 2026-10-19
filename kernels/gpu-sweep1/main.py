import os, sys, time, json, traceback, subprocess, glob, gc
import multiprocessing as mp
import numpy as np
import torch
from rpm_train_torch import run, make_backbone

CONFIGS = [{'amp': True,
  'arch': 'mobilenetv3_small_100',
  'bs': 32,
  'ema': 0.995,
  'epochs': 100,
  'eval_every': 5,
  'lr': 0.0007,
  'n_val': 1000,
  'pretrained': 'timm',
  'res': 224,
  'run_test': False,
  'seed': 0,
  'tag': 'a1-mnv3s224-f100',
  'threads': 4,
  'val_mode': 'split',
  'warmup_epochs': 1,
  'wd': 0.02},
 {'amp': True,
  'arch': 'resnet18',
  'bs': 64,
  'ema': 0.995,
  'epochs': 80,
  'eval_every': 5,
  'lr': 0.0006,
  'n_val': 1000,
  'pretrained': 'npz:resnet18',
  'res': 224,
  'run_test': False,
  'seed': 0,
  'tag': 'a2-r18-224',
  'threads': 4,
  'val_mode': 'split',
  'warmup_epochs': 2,
  'wd': 0.02},
 {'amp': True,
  'arch': 'resnet18',
  'bs': 64,
  'ema': 0.995,
  'epochs': 80,
  'eval_every': 5,
  'lr': 0.0006,
  'n_val': 1000,
  'pretrained': 'npz:resnet18',
  'res': 320,
  'run_test': False,
  'seed': 0,
  'tag': 'a3-r18-320',
  'threads': 4,
  'val_mode': 'split',
  'warmup_epochs': 2,
  'wd': 0.02},
 {'amp': True,
  'arch': 'mobilenetv3_small_100',
  'bs': 64,
  'ema': 0.995,
  'epochs': 80,
  'eval_every': 5,
  'lr': 0.001,
  'n_val': 1000,
  'pretrained': 'timm',
  'res': 320,
  'run_test': False,
  'seed': 0,
  'tag': 'a4-mnv3s320',
  'threads': 4,
  'val_mode': 'split',
  'warmup_epochs': 2,
  'wd': 0.02},
 {'amp': True,
  'arch': 'resnet34',
  'bs': 64,
  'ema': 0.995,
  'epochs': 80,
  'eval_every': 5,
  'lr': 0.0005,
  'n_val': 1000,
  'pretrained': 'npz:resnet34',
  'res': 224,
  'run_test': False,
  'seed': 0,
  'tag': 'a5-r34-224',
  'threads': 4,
  'val_mode': 'split',
  'warmup_epochs': 2,
  'wd': 0.02},
 {'amp': True,
  'arch': 'mobilenetv3_large_100',
  'bs': 64,
  'ema': 0.995,
  'epochs': 80,
  'eval_every': 5,
  'lr': 0.0008,
  'n_val': 1000,
  'pretrained': 'timm',
  'res': 224,
  'run_test': False,
  'seed': 0,
  'tag': 'a6-mnv3l224',
  'threads': 4,
  'val_mode': 'split',
  'warmup_epochs': 2,
  'wd': 0.02},
 {'amp': True,
  'arch': 'mobilenetv3_large_100',
  'bs': 64,
  'ema': 0.995,
  'epochs': 80,
  'eval_every': 5,
  'lr': 0.0008,
  'n_val': 1000,
  'pretrained': 'timm',
  'res': 320,
  'run_test': False,
  'seed': 0,
  'tag': 'a7-mnv3l320',
  'threads': 4,
  'val_mode': 'split',
  'warmup_epochs': 2,
  'wd': 0.02}]
EST_S = {'a1-mnv3s224-f100': 690,
 'a2-r18-224': 810,
 'a3-r18-320': 1530,
 'a4-mnv3s320': 750,
 'a5-r34-224': 1470,
 'a6-mnv3l224': 630,
 'a7-mnv3l320': 1050}   # expected seconds per config on ONE GPU (FROZEN-4 / its addendum S); used only by the start/stop budget below
CAP_BILLED_S = 10800   # billed GPU-seconds this kernel may use (the push -t value); the quota is billed per GPU held, so wall limit = cap / GPUs held
WORKERS = 2   # most training processes, one per GPU
ALLOW_CPU = False   # True only in the CPU twin used to smoke-test this orchestration on a Kaggle CPU kernel; False in every GPU kernel
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
        _lines = (_buf[_k] + _chunk).split("\n")
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
