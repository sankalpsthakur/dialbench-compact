# Numeric equivalence check: JAX RPMModel vs torch RPMNet built from the same exported npz weights (needs jax + torch in one process).
import numpy as np
import torch
import jax.numpy as jnp
from rpm_train_jax import RPMModel, MEAN as JMEAN, STD as JSTD
from rpm_train_torch import RPMNet as RPMNet_t
from rpm_export import jax_npz_to_torch_state


def unflatten(flat, prefix):
    out = {}
    for k, v in flat.items():
        if not k.startswith(prefix + "/"):
            continue
        parts = k[len(prefix) + 1:].split("/")
        d = out
        for q in parts[:-1]:
            d = d.setdefault(q, {})
        d[parts[-1]] = jnp.asarray(v)
    return out


def check_equivalence(c, npz_path, X, idxs, nb, ns):
    z = dict(np.load(npz_path))
    P_, S_ = unflatten(z, "p"), unflatten(z, "s")
    m = RPMModel(c["arch"], nb, ns, c["head_ch"], c["hidden"], c["drop"], c.get("width_mult", 1.0), jnp.float32)
    imgs = np.stack([X[j] for j in idxs])
    xb = (jnp.asarray(imgs).astype(jnp.float32) / 255.0 - JMEAN) / JSTD
    zb_j, zs_j = m.apply({"params": P_, "batch_stats": S_}, xb, False)
    # torch side
    net = RPMNet_t(dict(c, pretrained="none"), nb, ns).eval()
    with torch.no_grad():
        o = net.backbone(torch.zeros(1, 3, c["res"], c["res"]))
    sd = jax_npz_to_torch_state(npz_path, c, o.shape[2], o.shape[3])
    net.load_state_dict(sd, strict=False)
    with torch.no_grad():
        t = torch.from_numpy(np.asarray(xb)).permute(0, 3, 1, 2).contiguous()
        zb_t, zs_t = net(t)
    return {"max_abs_diff_bins": float(np.abs(np.asarray(zb_j) - zb_t.numpy()).max()), "max_abs_diff_span": float(np.abs(np.asarray(zs_j) - zs_t.numpy()).max()),
            "logit_absmax": float(np.abs(zb_t.numpy()).max())}
