#!/usr/bin/env python3
"""Pre-push checks for a generated gauges GPU/CPU kernel directory (stdlib only; run it under ../bin/guard.sh).
usage: check_kernel.py KERNEL_DIR [--gpu] [--v3 | --mixed]   (--mixed: official and v3 runs in one kernel; every v3 config must pin the manifest, rpm-split3 must be a source)
Checks: kernel-metadata.json (private, GPU flags, internet when timm weights are used, sources incl. rpm-prep/model-port[/rpm-split3]), script.py < 1 MB and
compiles (also under the Python 3.10 grammar), main.py literals parse (CONFIGS / EST_S: Python True/False, not JSON true/false), every config key is a known trainer
key (a typo would be ignored silently), amp true in GPU kernels, tags unique, v3 configs pin the manifest hash, and the modules embedded in script.py are the current
src/ files (sha256 comments). Prints one line per check and exits 1 on any failure."""
import ast, hashlib, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIN = "c16e51f6b9bd25cfa9925f19412da2510f9eaad9944dcddfc4d3a261307fc2e3"
kdir = os.path.abspath(sys.argv[1])
gpu, v3, mixed = "--gpu" in sys.argv, "--v3" in sys.argv, "--mixed" in sys.argv
bad = []


def ok(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        bad.append(msg)


meta = json.load(open(os.path.join(kdir, "kernel-metadata.json")))
name = os.path.basename(kdir)
slug = sys.argv[sys.argv.index("--slug") + 1] if "--slug" in sys.argv else name
ok(meta.get("id") == "sankalpsthakur/bench-gauges-" + slug, "id = bench-gauges-" + slug)
ok(meta.get("is_private") == "true", "is_private true")
ok(meta.get("kernel_type") == "script" and meta.get("language") == "python", "script / python")
ok(meta.get("enable_gpu") == ("true" if gpu else "false"), "enable_gpu " + meta.get("enable_gpu", "?"))
ok(meta.get("enable_tpu") == "false", "enable_tpu false")
if gpu:
    ok(meta.get("machine_shape") == "NvidiaTeslaT4", "machine_shape " + str(meta.get("machine_shape")))
src_ids = meta.get("kernel_sources", [])
need = ["rpm-prep", "model-port"] + (["rpm-split3"] if (v3 or mixed) else [])
for s in need:
    ok("sankalpsthakur/bench-gauges-" + s in src_ids, "kernel_sources has " + s)
ok(all(s.startswith("sankalpsthakur/bench-gauges-") for s in src_ids), "kernel_sources are all own bench-gauges-* kernels: " + ", ".join(src_ids))
ok(not meta.get("dataset_sources") and not meta.get("competition_sources") and not meta.get("model_sources"), "no dataset/competition/model sources")
code = os.path.join(kdir, meta["code_file"])
size = os.path.getsize(code)
ok(size < 1_000_000, "script size %d bytes < 1 MB" % size)
text = open(code).read()
try:
    compile(text, code, "exec"); ast.parse(text, feature_version=(3, 10))
    ok(True, "script.py compiles (and parses under the 3.10 grammar)")
except SyntaxError as e:
    ok(False, "script.py syntax: %r" % (e,))
# modules embedded in the bundle are the current src files
stale = []
for m in re.finditer(r"^# module (\w+) from (\S+) sha256=([0-9a-f]{16})$", text, re.M):
    cur = hashlib.sha256(open(os.path.join(ROOT, "src", os.path.basename(m.group(2)))).read().encode()).hexdigest()[:16]
    if cur != m.group(3):
        stale.append(m.group(1))
ok(not stale, "embedded modules equal current src/" + (" (stale: %s)" % ", ".join(stale) if stale else ""))
# main.py literals
main = ast.parse(open(os.path.join(kdir, "main.py")).read(), feature_version=(3, 10))
lit = {}
for node in main.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("CONFIGS", "EST_S", "CAP_BILLED_S", "WORKERS", "ALLOW_CPU"):
        lit[node.targets[0].id] = ast.literal_eval(node.value)
cfgs = lit.get("CONFIGS", [])
ok(bool(cfgs), "CONFIGS literal parses (%d configs)" % len(cfgs))
known = set()
tr = ast.parse(open(os.path.join(ROOT, "src", "rpm_train_torch.py")).read())
for node in tr.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "DEFAULTS":
        known = {kw.arg for kw in node.value.keywords}
extra = {"tag", "drop_test_dups"}
unknown = sorted({k for c in cfgs for k in c} - known - extra)
ok(not unknown, "all config keys are trainer keys" + (" (unknown: %s)" % unknown if unknown else ""))
tags = [c.get("tag") for c in cfgs]
ok(len(set(tags)) == len(tags) and all(tags), "tags unique: " + ", ".join(map(str, tags)))
if gpu:
    ok(all(c.get("amp") is True for c in cfgs), "amp true in every config")
    est = lit.get("EST_S", {})
    ok(set(est) == set(tags) and all(isinstance(v, (int, float)) and v > 0 for v in est.values()), "EST_S covers every tag")
    ok(isinstance(lit.get("CAP_BILLED_S"), int) and 600 < lit["CAP_BILLED_S"] <= 25200, "CAP_BILLED_S = %s billed s (the push -t value; at most 7 h)" % lit.get("CAP_BILLED_S"))
    ok(lit.get("WORKERS") in (1, 2), "WORKERS = %s" % lit.get("WORKERS"))
    ok(lit.get("ALLOW_CPU") is False, "ALLOW_CPU is False in a GPU kernel")
    ok(any(str(c.get("pretrained")) == "timm" for c in cfgs) <= (meta.get("enable_internet") == "true"), "internet on when timm weights are used")
nt = sum(1 for c in cfgs if c.get("run_test"))
print("info run_test true in %d of %d configs; epochs %s; split_file in %d" % (nt, len(cfgs), sorted({c.get("epochs") for c in cfgs}), sum(1 for c in cfgs if c.get("split_file"))))
if mixed:
    v3c = [c for c in cfgs if c.get("split_file")]
    ok(all(c["split_file"] == "splits_v3.json" and c.get("split_sha256") == PIN for c in v3c), "every v3 config pins splits_v3.json and its sha256")
    ok(0 < len(v3c) < len(cfgs), "mixed kernel holds both official and v3 runs (%d v3 of %d)" % (len(v3c), len(cfgs)))
elif v3:
    ok(all(c.get("split_file") == "splits_v3.json" and c.get("split_sha256") == PIN for c in cfgs), "every config pins splits_v3.json and its sha256")
else:
    ok(not any(c.get("split_file") for c in cfgs), "no split_file (official split)")
print("RESULT", "FAIL (%d)" % len(bad) if bad else "all checks passed")
sys.exit(1 if bad else 0)
