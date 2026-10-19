"""DialBench / RPM-10K official metric, ported verbatim from the authors' evaluation script.

Source: https://github.com/Event-AHU/DialBench  benchmark/benchmark.py  (functions
`extract_last_number`, `load_data`, and the accumulation inside `main` + `calculate_metrics`).
Fetched 2026-10-07; repo default branch `main`, pushed_at 2026-08-27.

Definitions as implemented there (not as the prose of the paper, where "Ref" and "Rel"
are described loosely):
  error      = |pred - gt|
  Acc_eps    = mean( error <= 0.01 * range )                    over ALL samples
  Acc_theta  = mean( error / gt < 0.05 )                        over samples with gt != 0
  Ref (Gamma)= mean( min(error / range, 10) )                   (clip at 10)
  Rel (Theta)= mean( min(error / gt, 100) )                     over samples with gt != 0 (clip at 100)

Notes kept from the authors' code on purpose:
  * the ground truth is `extract_last_number(str(item['reading']))` (last number in the string);
  * `range` is read from the annotation file and is used ONLY to scale epsilon, never fed to the model;
  * a missing number in a prediction becomes 0.0.
This module has no torch dependency so the same file runs on the Mac and in Kaggle kernels.
"""
import re
import numpy as np

_NUM = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def extract_last_number(text):
    """Last int/float in a string (authors' regex). Returns 0.0 if none."""
    matches = _NUM.findall(str(text))
    if matches:
        return float(matches[-1])
    return 0.0


def dialbench_metrics(pred, gt, rng):
    """pred, gt, rng: 1-D float arrays. Returns dict with the four authors' metrics (fractions, not %)."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    rng = np.asarray(rng, dtype=np.float64)
    n = len(gt)
    err = np.abs(pred - gt)
    eps = 0.01 * rng
    acc_eps_hits = err <= eps  # same `<=` as the authors' code
    nz = gt != 0
    theta = err[nz] / gt[nz]
    out = {
        "n": int(n),
        "acc_eps": float(acc_eps_hits.sum() / n) if n else None,
        "acc_theta": float((theta < 0.05).sum() / nz.sum()) if nz.sum() else None,
        "ref": float(np.mean(np.clip(err / rng, None, 10))) if n else None,
        "rel": float(np.mean(np.clip(theta, None, 100))) if nz.sum() else None,
        "n_nonzero_gt": int(nz.sum()),
    }
    return out


def wilson_interval(k, n, z=1.959964):
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def bootstrap_acc_ci(hits, n_boot=2000, seed=0, groups=None):
    """Percentile bootstrap CI for a mean of 0/1 hits. If `groups` is given, resample whole groups
    (use for frames from the same video / meter instance, which are not independent)."""
    rs = np.random.RandomState(seed)
    hits = np.asarray(hits, dtype=np.float64)
    if groups is None:
        idx = rs.randint(0, len(hits), size=(n_boot, len(hits)))
        means = hits[idx].mean(1)
    else:
        groups = np.asarray(groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        sums = np.bincount(inv, weights=hits, minlength=len(uniq))
        cnts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
        gi = rs.randint(0, len(uniq), size=(n_boot, len(uniq)))
        means = sums[gi].sum(1) / cnts[gi].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
