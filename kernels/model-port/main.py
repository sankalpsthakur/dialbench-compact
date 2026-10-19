# ---- bench-gauges-model-port: CPU kernel, NO dataset. Verifies Flax<->torchvision weight conversion and
# exports ImageNet-pretrained ResNet-18/34 weights as npz for later TPU kernels.
import os, sys, json, time
import numpy as np
import torch, torchvision

OUT = "/kaggle/working"
jax.config.update("jax_default_matmul_precision", "highest")
report = {}
for arch in ["resnet18", "resnet34"]:
    t = time.time()
    tvm = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1").eval()
    sd = {k: v.numpy() for k, v in tvm.state_dict().items()}
    params, stats = convert_tv_to_flax(sd, arch)
    net = ResNetBackbone(arch=arch, dtype=jnp.float32)
    x = np.random.RandomState(0).randn(2, 224, 224, 3).astype(np.float32)
    y_flax = np.asarray(net.apply({"params": params, "batch_stats": stats}, jnp.asarray(x), False))
    with torch.no_grad():
        xt = torch.from_numpy(x.transpose(0, 3, 1, 2))
        f = tvm.maxpool(tvm.relu(tvm.bn1(tvm.conv1(xt))))
        f = tvm.layer4(tvm.layer3(tvm.layer2(tvm.layer1(f))))
        y_tv = f.numpy().transpose(0, 2, 3, 1)
    d1 = float(np.abs(y_flax - y_tv).max())
    # inverse conversion
    sd2 = flax_to_tv_state_dict(params, stats, arch)
    tvm2 = getattr(torchvision.models, arch)(weights=None).eval()
    missing = tvm2.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in sd2.items()}, strict=False)
    with torch.no_grad():
        f2 = tvm2.maxpool(tvm2.relu(tvm2.bn1(tvm2.conv1(xt))))
        f2 = tvm2.layer4(tvm2.layer3(tvm2.layer2(tvm2.layer1(f2))))
    d2 = float(np.abs(f2.numpy().transpose(0, 2, 3, 1) - y_tv).max())
    nparams = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))
    report[arch] = {"max_abs_diff_flax_vs_tv": d1, "max_abs_diff_roundtrip": d2, "unexpected_keys": list(missing.unexpected_keys),
                    "missing_keys": list(missing.missing_keys), "backbone_params": nparams, "feat_absmax": float(np.abs(y_tv).max()),
                    "secs": round(time.time() - t, 1)}
    print(arch, report[arch], flush=True)
    flat = {}
    for k, v in sd.items():
        if not k.startswith("fc."):
            flat[k] = v
    np.savez(os.path.join(OUT, f"{arch}_imagenet_tv.npz"), **flat)
json.dump(report, open(os.path.join(OUT, "model_port_report.json"), "w"), indent=1)
print("done", flush=True)
