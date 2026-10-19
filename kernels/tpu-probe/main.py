# ---- bench-gauges-tpu-probe: TPU kernel, NO dataset, short. Checks JAX on the Kaggle TPU and ResNet-18 train-step throughput.
import os, sys, time, json, subprocess
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def sh(c, t=60):
    try:
        r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERR {e}"

T0 = time.time()
print("== host", flush=True)
print(sh("nproc; free -g | head -2; python -V; df -h /kaggle/working | tail -1"))
print(sh("pip list 2>/dev/null | grep -i -E '^(jax|jaxlib|libtpu|torch|torch-xla|torch_xla|flax|optax|numpy|opencv|timm|onnx)'"))
print("== jax", jax.__version__, flush=True)
devs = jax.devices()
print("devices", devs, "local", jax.local_device_count(), "process_count", jax.process_count(), "t=%.1f" % (time.time() - T0), flush=True)
print("platform", devs[0].platform, getattr(devs[0], "device_kind", "?"))
try:
    print("memory_stats", devs[0].memory_stats())
except Exception as e:
    print("memory_stats err", e)

n = len(devs)
mesh = Mesh(np.array(devs), ("data",))
shard_b = NamedSharding(mesh, P("data"))
repl = NamedSharding(mesh, P())

# matmul sanity
a = jnp.ones((8192, 8192), jnp.bfloat16)
f = jax.jit(lambda x: x @ x)
f(a).block_until_ready()
t = time.time()
for _ in range(10):
    r = f(a)
r.block_until_ready()
dt = (time.time() - t) / 10
print("single-chip bf16 matmul 8192^3: %.1f TFLOPs" % (2 * 8192 ** 3 / dt / 1e12), flush=True)

class Net(nn.Module):
    @nn.compact
    def __call__(self, x, train):
        f = ResNetBackbone("resnet18", dtype=jnp.bfloat16)(x, train)
        f = f.mean((1, 2))
        return nn.Dense(10, dtype=jnp.float32)(f.astype(jnp.float32))

net = Net()
B = 64 * n
x0 = jnp.zeros((B, 224, 224, 3), jnp.float32)
vars_ = net.init(jax.random.PRNGKey(0), x0[:2], True)
params, stats = vars_["params"], vars_["batch_stats"]
tx = optax.adamw(1e-3)
opt = tx.init(params)
params, stats, opt = jax.device_put((params, stats, opt), repl)

@jax.jit
def step(params, stats, opt, xb, yb):
    def loss_fn(p):
        out, upd = net.apply({"params": p, "batch_stats": stats}, (xb.astype(jnp.float32) / 255.0 - 0.45) / 0.225, True, mutable=["batch_stats"])
        loss = optax.softmax_cross_entropy_with_integer_labels(out, yb).mean()
        return loss, upd["batch_stats"]
    (loss, new_stats), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt2 = tx.update(g, opt, params)
    return optax.apply_updates(params, updates), new_stats, opt2, loss

xb_np = np.random.randint(0, 255, (B, 224, 224, 3), dtype=np.uint8)
yb_np = np.random.randint(0, 10, (B,), dtype=np.int32)
t = time.time()
xb = jax.device_put(xb_np, shard_b); yb = jax.device_put(yb_np, shard_b)
xb.block_until_ready()
print("h2d %.1f MB in %.3fs" % (xb_np.nbytes / 1e6, time.time() - t), flush=True)
t = time.time()
params, stats, opt, loss = step(params, stats, opt, xb, yb)
loss.block_until_ready()
print("compile+first step %.1fs loss %.3f" % (time.time() - t, float(loss)), flush=True)
t = time.time()
for _ in range(30):
    params, stats, opt, loss = step(params, stats, opt, xb, yb)
loss.block_until_ready()
dt = (time.time() - t) / 30
print("resnet18 bf16 train step global batch %d: %.1f ms -> %.0f img/s (all %d chips)" % (B, dt * 1e3, B / dt, n), flush=True)
# with per-step host->device transfer
t = time.time()
for _ in range(10):
    xb = jax.device_put(xb_np, shard_b)
    params, stats, opt, loss = step(params, stats, opt, xb, yb)
loss.block_until_ready()
dt = (time.time() - t) / 10
print("with h2d each step: %.1f ms -> %.0f img/s" % (dt * 1e3, B / dt), flush=True)
print("total secs %.1f" % (time.time() - T0))
