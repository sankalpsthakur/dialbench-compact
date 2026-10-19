import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], capture_output=True, text=True)
print("pip onnxruntime:", r.returncode, (r.stdout + r.stderr)[-300:], flush=True)
