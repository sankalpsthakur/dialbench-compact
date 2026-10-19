"""The generated kernel bundles (kernels/*/script.py, built by tools/build_bundle.py) embed the src modules as string literals and end with the kernel's main.py.
This test decodes them with `ast` (nothing is executed) and checks that
  1. every `# module NAME from PATH sha256=H` comment carries the first 16 hex digits of the sha256 of the literal that follows it,
  2. the `# main: PATH sha256=H` comment matches kernels/<kernel>/main.py and the script ends with that file,
  3. for the kernels built from the current src/ (the Stage A/B GPU kernels, the int8 sweeps, the split-v3 run, the copy-floor and JAX self-test kernels) every embedded module is
     byte-identical to src/<module>.py.
The run-1 kernel (cpu-mnv3s-b), the first CPU kernel (cpu-mnv3s) and the stale TPU kernel (tpu-sweep1) were built from older src/ files and are only checked for 1 and 2.
Kernels that are plain concatenations (build_concat.py: no `_load` calls) are skipped. The path in the comment lines may be redacted in the public kit; only the hash is used."""
import ast, glob, hashlib, os, re, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CURRENT = {"copy-hits", "cpu-v3", "gpu-final", "gpu-final4", "gpu-final6", "gpu-int8", "gpu-smoke", "gpu-sweep1", "int8-sweep", "jax-selftest"}


def h16(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def loads(script_text):
    """[(module name, source, line number of the _load call)] found in a bundle."""
    out = []
    for node in ast.walk(ast.parse(script_text)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_load" and len(node.args) == 2:
            a, b = node.args
            if isinstance(a, ast.Constant) and isinstance(b, ast.Constant) and isinstance(a.value, str) and isinstance(b.value, str):
                out.append((a.value, b.value, node.lineno))
    return sorted(out, key=lambda c: c[2])


class T(unittest.TestCase):
    def test_bundles(self):
        checked = 0
        for sp in sorted(glob.glob(os.path.join(ROOT, "kernels", "*", "script.py"))):
            kernel = os.path.basename(os.path.dirname(sp))
            text = read(sp)
            calls = loads(text)
            if not calls:
                continue
            lines = text.split("\n")
            for name, src, ln in calls:
                m = re.match(r"^# module (\w+) from (.+) sha256=([0-9a-f]{16})$", lines[ln - 2])
                self.assertIsNotNone(m, "%s: no module comment above the _load of %s" % (kernel, name))
                self.assertEqual((m.group(1), m.group(3)), (name, h16(src)), "%s: module comment of %s does not match its literal" % (kernel, name))
                if kernel in CURRENT:
                    self.assertEqual(src, read(os.path.join(ROOT, "src", name + ".py")), "%s: embedded %s differs from src/%s.py" % (kernel, name, name))
            mm = re.search(r"^# main: (.+?) sha256=([0-9a-f]{16})$", text, re.M)
            self.assertIsNotNone(mm, kernel + ": no main comment")
            main = read(os.path.join(ROOT, "kernels", kernel, "main.py"))
            self.assertEqual(mm.group(2), h16(main), kernel + ": main comment does not match main.py")
            self.assertTrue(text.endswith(main + "\n"), kernel + ": the script does not end with main.py")
            checked += 1
        self.assertGreaterEqual(checked, 10)


if __name__ == "__main__":
    unittest.main()
