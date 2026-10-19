# Flax (linen) ResNet-18/34 whose submodule names mirror torchvision, so ImageNet weights and
# trained weights convert 1:1 in both directions (see convert_tv_to_flax / flax_to_tv_state_dict).
from functools import partial
from typing import Any, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

ARCH = {"resnet10": [1, 1, 1, 1], "resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3]}
WIDTHS = [64, 128, 256, 512]


class BasicBlock(nn.Module):
    features: int
    stride: int = 1
    has_down: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype, param_dtype=jnp.float32)
        norm = partial(nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5,
                       dtype=self.dtype, param_dtype=jnp.float32)
        y = conv(self.features, (3, 3), (self.stride, self.stride), padding=((1, 1), (1, 1)), name="conv1")(x)
        y = norm(name="bn1")(y)
        y = nn.relu(y)
        y = conv(self.features, (3, 3), (1, 1), padding=((1, 1), (1, 1)), name="conv2")(y)
        y = norm(name="bn2")(y)
        if self.has_down:
            x = conv(self.features, (1, 1), (self.stride, self.stride), padding=((0, 0), (0, 0)), name="down_conv")(x)
            x = norm(name="down_bn")(x)
        return nn.relu(x + y)


class ResNetBackbone(nn.Module):
    arch: str = "resnet18"
    width_mult: float = 1.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool):
        """x: [B,H,W,3] already normalised. Returns the final feature map [B,H/32,W/32,C]."""
        widths = [int(round(w * self.width_mult)) for w in WIDTHS]
        stem = widths[0]
        x = x.astype(self.dtype)
        x = nn.Conv(stem, (7, 7), (2, 2), padding=((3, 3), (3, 3)), use_bias=False, dtype=self.dtype,
                    param_dtype=jnp.float32, name="conv1")(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=self.dtype,
                         param_dtype=jnp.float32, name="bn1")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), (2, 2), padding=((1, 1), (1, 1)))
        in_f = stem
        for li, (nb, f) in enumerate(zip(ARCH[self.arch], widths)):
            for bi in range(nb):
                stride = 2 if (bi == 0 and li > 0) else 1
                has_down = (bi == 0) and (stride != 1 or in_f != f)
                x = BasicBlock(f, stride, has_down, self.dtype, name=f"layer{li + 1}_{bi}")(x, train)
                in_f = f
        return x


# ---------------------------------------------------------------- torchvision <-> flax weight conversion
def _conv_to_flax(w):  # [O,I,kh,kw] -> [kh,kw,I,O]
    return np.asarray(w).transpose(2, 3, 1, 0)


def _conv_to_tv(k):
    return np.asarray(k).transpose(3, 2, 0, 1)


def convert_tv_to_flax(sd, arch="resnet18"):
    """torchvision ResNet state_dict (tensors or numpy) -> (params, batch_stats) for ResNetBackbone (fc dropped)."""
    g = lambda k: np.asarray(sd[k].detach().cpu().numpy() if hasattr(sd[k], "detach") else sd[k])
    params, stats = {}, {}

    def bn(dst_p, dst_s, name, prefix):
        dst_p[name] = {"scale": g(prefix + ".weight"), "bias": g(prefix + ".bias")}
        dst_s[name] = {"mean": g(prefix + ".running_mean"), "var": g(prefix + ".running_var")}

    params["conv1"] = {"kernel": _conv_to_flax(g("conv1.weight"))}
    bn(params, stats, "bn1", "bn1")
    for li, nb in enumerate(ARCH[arch]):
        for bi in range(nb):
            tv = f"layer{li + 1}.{bi}"
            nm = f"layer{li + 1}_{bi}"
            p, s = {}, {}
            p["conv1"] = {"kernel": _conv_to_flax(g(tv + ".conv1.weight"))}
            bn(p, s, "bn1", tv + ".bn1")
            p["conv2"] = {"kernel": _conv_to_flax(g(tv + ".conv2.weight"))}
            bn(p, s, "bn2", tv + ".bn2")
            if (tv + ".downsample.0.weight") in sd:
                p["down_conv"] = {"kernel": _conv_to_flax(g(tv + ".downsample.0.weight"))}
                bn(p, s, "down_bn", tv + ".downsample.1")
            params[nm], stats[nm] = p, s
    return params, stats


def flax_to_tv_state_dict(params, stats, arch="resnet18"):
    """Inverse of convert_tv_to_flax: returns a dict of numpy arrays with torchvision names (no fc)."""
    sd = {}
    a = lambda x: np.asarray(x)

    def bn(prefix, p, s):
        sd[prefix + ".weight"] = a(p["scale"])
        sd[prefix + ".bias"] = a(p["bias"])
        sd[prefix + ".running_mean"] = a(s["mean"])
        sd[prefix + ".running_var"] = a(s["var"])
        sd[prefix + ".num_batches_tracked"] = np.array(0, dtype=np.int64)

    sd["conv1.weight"] = _conv_to_tv(a(params["conv1"]["kernel"]))
    bn("bn1", params["bn1"], stats["bn1"])
    for li, nb in enumerate(ARCH[arch]):
        for bi in range(nb):
            tv = f"layer{li + 1}.{bi}"
            nm = f"layer{li + 1}_{bi}"
            p, s = params[nm], stats[nm]
            sd[tv + ".conv1.weight"] = _conv_to_tv(a(p["conv1"]["kernel"]))
            bn(tv + ".bn1", p["bn1"], s["bn1"])
            sd[tv + ".conv2.weight"] = _conv_to_tv(a(p["conv2"]["kernel"]))
            bn(tv + ".bn2", p["bn2"], s["bn2"])
            if "down_conv" in p:
                sd[tv + ".downsample.0.weight"] = _conv_to_tv(a(p["down_conv"]["kernel"]))
                bn(tv + ".downsample.1", p["down_bn"], s["down_bn"])
    return sd
