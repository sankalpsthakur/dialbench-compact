#!/usr/bin/env python3
"""FROZEN-4 Stage A -> Stage B planner (stdlib only; run it under ../bin/guard.sh). Applies the pre-registered rules of FROZEN-4 sections 2 and 3 as amended by
amendments 4 and 5 (budgets in BILLED GPU-seconds: the quota is billed per GPU held) to the Stage A result JSONs, prints the table, the primary P, the edge pick E,
the futility verdict, the kept run list, the caps and the commands, and with --write creates the Stage B config files. It reads validation numbers only.
usage: gpu_stage_b.py STAGE_A_DIR [--invalid TAG,TAG] [--billed-gpus B] [--workers W] [--stage-a-billed-s N] [--combined] [--write]
  --stage-a-billed-s N: bill Stage A at what it really used (billed seconds) instead of its cap when computing the Stage B allowance (amendment 6).
  --kernels-only: with --write, write only the kernel directory (no plan json, no int8 models).
  --max-runs N / --kernel-name NAME: keep at most the first N runs of the list and name the combined kernel directory NAME (quota tiers of amendment 7).
  --combined: one Stage B kernel `gpu-final` holding the official and the v3 runs, so that both GPUs stay busy (amendment 6); default: two kernels as in amendment 5.
  STAGE_A_DIR: the fetched outputs of kernel gpu-sweep1 (a*_result.json, wrapper_summary.json). B (GPUs held = billed rate) and W (parallel workers) default to what
  wrapper_summary.json recorded, else 2 and 2. --invalid: tags the lead found invalid from the log (no CUDA device line, PREFLIGHT / trainer 'missing' not empty).
  --write: kernels/gpu-final-off/{configs,est_s}.json, kernels/gpu-final-v3/{configs,est_s}.json, kernels/gpu-int8/models.json, docs/stage_b_plan.json."""
import json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIN = "c16e51f6b9bd25cfa9925f19412da2510f9eaad9944dcddfc4d3a261307fc2e3"
FUTILITY = 0.64                  # FROZEN-4 section 2: best validation below this -> reduced Stage B
EDGE_TOL = 0.01                  # one percentage point
MAX_TRUN_MIN = 45.0              # eligibility (amendment 4 raised it from 30)
CAPS_TOTAL_BILLED_S = 25200      # amendment 5: caps book at most 7 of the 8 billed GPU-hours; 1 h stays as the retry reserve
STAGE_A_CAP_BILLED_S = 10800     # amendment 5: Stage A cap, 3 billed GPU-hours
OVERHEAD_S = 600                 # wall seconds of container start, input mounts and downloads per session
RUNS = [("P", "off", 0), ("P", "off", 1), ("P", "off", 2), ("P", "v3", 0), ("P", "off", 3), ("P", "off", 4), ("E", "off", 0), ("E", "v3", 0),
        ("P", "v3", 1), ("P", "v3", 2), ("E", "off", 1), ("E", "off", 2)]


def load_stage_a(stage_dir, cfg_path, invalid=()):
    with open(cfg_path) as fh:
        cfgs = json.load(fh)
    rows = []
    for i, c in enumerate(cfgs):
        row = dict(idx=i, tag=c["tag"], cfg=c, valid=False, eligible=False, note="")
        f = os.path.join(stage_dir, c["tag"] + "_result.json")
        if not os.path.exists(f):
            row["note"] = "no result (skipped, failed or not run)"
            rows.append(row); continue
        with open(f) as fh:
            r = json.load(fh)
        h = r["history"]
        row["params"] = r["params"]
        row["t_run_min"] = sum(x["secs"] for x in h) / 60.0 + 1.5
        finite = all(math.isfinite(x["loss"]) for x in h)
        done = len(h) == c["epochs"] and "val_ema" in h[-1]
        row["score"] = h[-1]["val_ema"]["window"] if done else None
        row["valid"] = bool(done and finite and c["tag"] not in invalid)
        row["note"] = ("" if done else "incomplete (%d of %d epochs)" % (len(h), c["epochs"])) + ("" if finite else " NaN/inf loss") + (" marked invalid by the lead" if c["tag"] in invalid else "")
        row["eligible"] = row["valid"] and row["t_run_min"] <= MAX_TRUN_MIN
        if row["valid"] and not row["eligible"]:
            row["note"] += " T_run %.1f min > %.0f" % (row["t_run_min"], MAX_TRUN_MIN)
        rows.append(row)
    return rows


def select(rows):
    el = [r for r in rows if r["eligible"]]
    if not el:
        return None, None
    P = max(el, key=lambda r: (r["score"], -r["params"], -r["idx"]))
    near = [r for r in el if r["score"] >= P["score"] - EDGE_TOL - 1e-9]
    E = min(near, key=lambda r: (r["params"], -r["score"], r["idx"]))
    return P, E


def makespan_s(charges_s, workers):
    """Wall seconds to finish the runs when `workers` processes take them in list order from a shared queue (what the wrapper does)."""
    free = [0.0] * max(1, workers)
    for c in charges_s:
        i = free.index(min(free))
        free[i] += c
    return max(free) if charges_s else 0.0


def kernel_cap_billed_s(charges_s, billed_gpus, workers):
    """Billed-seconds cap of a kernel: the old cap formula (600 s overhead + 1.2 x the schedule length, rounded up to 300 s) in wall seconds, times the GPUs held."""
    if not charges_s:
        return 0
    return int(300 * math.ceil(billed_gpus * (OVERHEAD_S + 1.2 * makespan_s(charges_s, workers)) / 300.0))


def hardware_from_summary(stage_dir):
    f = os.path.join(stage_dir, "wrapper_summary.json")
    if os.path.exists(f):
        with open(f) as fh:
            s = json.load(fh)
        return s.get("gpus"), s.get("workers")
    return None, None


def plan(stage_dir, cfg_path, invalid=(), billed_gpus=None, workers=None, stage_a_billed_s=None, combined=False, max_runs=None, kernel_name="gpu-final"):
    g, w = hardware_from_summary(stage_dir)
    B = billed_gpus or g or 2
    W = workers or w or min(2, B)
    rows = load_stage_a(stage_dir, cfg_path, invalid)
    P, E = select(rows)
    stage_a = int(stage_a_billed_s) if stage_a_billed_s else STAGE_A_CAP_BILLED_S
    out = dict(rows=rows, P=P, E=E, futility=False, runs=[], kernels={}, billed_gpus=B, workers=W, stage_a_billed_s=stage_a, combined=bool(combined))
    if P is None:
        out["verdict"] = "no eligible candidate: Stage B is not run; report Stage A only"
        return out
    out["futility"] = P["score"] < FUTILITY
    wanted = list(RUNS)
    if E["tag"] == P["tag"] or out["futility"]:
        wanted = [r for r in wanted if r[0] == "P"]
    if out["futility"]:
        wanted = wanted[:4]
    if max_runs:
        wanted = wanted[:max_runs]
    allowance = CAPS_TOTAL_BILLED_S - stage_a
    kept = []

    def caps_of(runs):
        if combined:
            return {"all": kernel_cap_billed_s([k["charge_min"] * 60 for k in runs], B, W)}
        return {sp: kernel_cap_billed_s([k["charge_min"] * 60 for k in runs if k["split"] == sp], B, W) for sp in ("off", "v3")}

    for role, split, seed in wanted:
        cand = P if role == "P" else E
        charge = int(math.ceil(cand["t_run_min"]))
        name = cand["tag"].split("-", 1)[1]
        c = dict(cand["cfg"], seed=seed, run_test=True, tag="f-%s-%s-s%d" % (name, split, seed))
        if split == "v3":
            c.update(split_file="splits_v3.json", split_sha256=PIN)
        trial = kept + [dict(role=role, split=split, seed=seed, tag=c["tag"], charge_min=charge, cfg=c)]
        if sum(caps_of(trial).values()) > allowance:
            break
        kept = trial
    out["runs"] = kept
    out["total_charge_min"] = sum(k["charge_min"] for k in kept)
    caps = caps_of(kept)
    for split, kname in ((("all", kernel_name),) if combined else (("off", "gpu-final-off"), ("v3", "gpu-final-v3"))):
        rs = [k for k in kept if combined or k["split"] == split]
        if rs:
            ch = [k["charge_min"] * 60 for k in rs]
            m = makespan_s(ch, W)
            out["kernels"][kname] = dict(runs=[k["tag"] for k in rs], configs=[k["cfg"] for k in rs], est={k["tag"]: k["charge_min"] * 60 for k in rs}, charge_s=sum(ch),
                                         has_v3=any(k["split"] == "v3" for k in rs), has_off=any(k["split"] == "off" for k in rs),
                                         makespan_s=m, cap_billed_s=caps[split], expected_wall_min=(m + OVERHEAD_S) / 60.0, expected_billed_h=B * (m + OVERHEAD_S) / 3600.0)
    out["sum_caps_billed_s"] = stage_a + sum(caps.values())
    models = []
    for k in kept:
        if k["seed"] != 0:
            continue
        keep = {kk: k["cfg"][kk] for kk in ("arch", "res", "tag") if kk in k["cfg"]}
        keep["pretrained"] = "none"
        if k["split"] == "v3":
            keep.update(split_file="splits_v3.json", split_sha256=PIN)
        else:
            keep.update(val_mode="split", n_val=1000)
        models.append(dict(label="%s-%s" % (k["role"].lower(), k["split"]), cfg=keep, weights=k["tag"] + "_ema.pt"))
    out["int8_models"] = models
    out["verdict"] = "ok" if out["sum_caps_billed_s"] <= CAPS_TOTAL_BILLED_S else "WARNING: caps above 25,200 billed s (7 h)"
    return out


def report(o):
    print("| tag | params (M) | validation Acc_eps | T_run (min) | valid | eligible | note |\n|---|---|---|---|---|---|---|")
    for r in o["rows"]:
        print("| %s | %s | %s | %s | %s | %s | %s |" % (r["tag"], "%.2f" % (r["params"] / 1e6) if "params" in r else "-", "%.4f" % r["score"] if r.get("score") is not None else "-",
                                                      "%.1f" % r["t_run_min"] if "t_run_min" in r else "-", r["valid"], r["eligible"], r["note"].strip()))
    if o["P"] is None:
        print(o["verdict"]); return
    print("\nhardware assumed for Stage B: %d GPU(s) held (billed rate %dx), %d worker(s)" % (o["billed_gpus"], o["billed_gpus"], o["workers"]))
    print("P (primary) =", o["P"]["tag"], "score %.4f" % o["P"]["score"], "| E (edge pick) =", o["E"]["tag"], "score %.4f" % o["E"]["score"], "(same as P)" if o["E"]["tag"] == o["P"]["tag"] else "")
    print("futility stop (best validation < %.2f):" % FUTILITY, o["futility"])
    print("kept runs (%d, %d run-minutes of work):" % (len(o["runs"]), o["total_charge_min"]), ", ".join("%s(%dm)" % (k["tag"], k["charge_min"]) for k in o["runs"]))
    for kn, k in o["kernels"].items():
        v3 = " --sources rpm-prep,model-port,rpm-split3" if k["has_v3"] else " --sources rpm-prep,model-port"
        chk = " --mixed" if (k["has_v3"] and k["has_off"]) else (" --v3" if k["has_v3"] else "")
        label = "official + v3" if (k["has_v3"] and k["has_off"]) else ("v3" if k["has_v3"] else "official")
        print("\n%s: %d runs, %d s of work, schedule %d s on %d worker(s), cap %d billed s (= -t), expected %.0f min wall, about %.2f billed GPU-h" % (
            kn, len(k["runs"]), k["charge_s"], k["makespan_s"], o["workers"], k["cap_billed_s"], k["expected_wall_min"], k["expected_billed_h"]))
        slug = " --slug gpu-final" if (o["combined"] and kn != "gpu-final") else ""
        ref = "gpu-final" if o["combined"] else kn
        print("  ../bin/guard.sh -- .venv/bin/python tools/make_kernel.py %s torch-train-gpu --config-file kernels/%s/configs.json --est-json kernels/%s/est_s.json --cap-billed-s %d --workers %d --internet --gpu NvidiaTeslaT4%s%s" % (
            kn, kn, kn, k["cap_billed_s"], o["workers"], v3, slug))
        print("  ../bin/guard.sh -- .venv/bin/python tools/check_kernel.py kernels/%s --gpu%s%s" % (kn, chk, slug))
        print("  tools/gpu_push_wait.sh kernels/%s sankalpsthakur/bench-gauges-%s logs/kaggle-output/%s %d %d \"FROZEN-4 Stage B %s: %d runs (%s), reads the private rpm-prep and model-port%s outputs\"   # run in the background" % (
            kn, ref, ref, k["cap_billed_s"], int(k["expected_wall_min"]), label, len(k["runs"]), ", ".join(k["runs"]), " and rpm-split3" if k["has_v3"] else ""))
        print("  # the push script's defaults (T4 shape, -t = cap / 2 + 600 = %d) apply; a single-GPU shape would need GPU_SHAPES=... and its own -t" % int(300 * math.ceil((k["cap_billed_s"] / 2.0 + 600) / 300.0)))
    if o["int8_models"]:
        src = "rpm-prep" + (",rpm-split3" if any(m["cfg"].get("split_file") for m in o["int8_models"]) else "") + "," + ",".join(sorted(["gpu-final"] if o["combined"] else o["kernels"]))
        print("\nCPU follow-up (no GPU hours; after the Stage B kernels finished; FROZEN-3 sweep on the seed-0 weights):")
        print("  ../bin/guard.sh -- .venv/bin/python tools/make_kernel.py gpu-int8 int8-sweep --config-file kernels/gpu-int8/models.json --internet --sources " + src)
        print("  tools/slot_push_wait.sh kernels/gpu-int8 sankalpsthakur/bench-gauges-gpu-int8 logs/kaggle-output/gpu-int8 '^(?!.*fp32).*\\.(json|txt|onnx)$' 7200 90 \"FROZEN-4 section 5 int8 sweep\"")
    print("\nsum of caps: Stage A %d (%s) + Stage B -> %d billed s (%.2f GPU-h of the 7 h allowed; the 8 h cap keeps at least 1 h as the retry reserve); %s" % (
        o["stage_a_billed_s"], "booked at its cap" if o["stage_a_billed_s"] == STAGE_A_CAP_BILLED_S else "as actually billed", o["sum_caps_billed_s"], o["sum_caps_billed_s"] / 3600.0, o["verdict"]))


def write(o, kernels_only=False):
    for kn, k in o["kernels"].items():
        d = os.path.join(ROOT, "kernels", kn)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "configs.json"), "w") as fh:
            json.dump(k["configs"], fh, indent=1)
        with open(os.path.join(d, "est_s.json"), "w") as fh:
            json.dump(k["est"], fh, indent=1)
    if kernels_only:
        print("written: kernels/%s/{configs,est_s}.json only" % ", ".join(o["kernels"]))
        return
    slim = dict(P=o["P"]["tag"], E=o["E"]["tag"], futility=o["futility"], billed_gpus=o["billed_gpus"], workers=o["workers"], stage_a_billed_s=o["stage_a_billed_s"], combined=o["combined"],
                total_charge_min=o["total_charge_min"],
                sum_caps_billed_s=o["sum_caps_billed_s"], table=[{k: v for k, v in r.items() if k != "cfg"} for r in o["rows"]],
                kernels={kn: {k: v for k, v in kk.items() if k != "configs"} for kn, kk in o["kernels"].items()})
    with open(os.path.join(ROOT, "docs", "stage_b_plan.json"), "w") as fh:
        json.dump(slim, fh, indent=1)
    if o["int8_models"]:
        os.makedirs(os.path.join(ROOT, "kernels", "gpu-int8"), exist_ok=True)
        with open(os.path.join(ROOT, "kernels", "gpu-int8", "models.json"), "w") as fh:
            json.dump(o["int8_models"], fh, indent=1)
    print("written: kernels/gpu-final-*/{configs,est_s}.json, kernels/gpu-int8/models.json and docs/stage_b_plan.json")


if __name__ == "__main__":
    args = sys.argv[1:]
    inv = args[args.index("--invalid") + 1].split(",") if "--invalid" in args else []
    b = int(args[args.index("--billed-gpus") + 1]) if "--billed-gpus" in args else None
    w = int(args[args.index("--workers") + 1]) if "--workers" in args else None
    sa = int(args[args.index("--stage-a-billed-s") + 1]) if "--stage-a-billed-s" in args else None
    mr = int(args[args.index("--max-runs") + 1]) if "--max-runs" in args else None
    kn = args[args.index("--kernel-name") + 1] if "--kernel-name" in args else "gpu-final"
    o = plan(args[0], os.path.join(ROOT, "kernels", "gpu-sweep1", "configs.json"), inv, b, w, sa, "--combined" in args, mr, kn)
    report(o)
    if "--write" in args and o["P"] is not None:
        write(o, "--kernels-only" in args)
