import json, sys, torch
a, b = (torch.load(p, map_location="cpu", weights_only=True, mmap=True) for p in sys.argv[1:3])
def diff(x, y, path, out):
    if isinstance(x, torch.Tensor):
        assert x.shape == y.shape and x.dtype == y.dtype, path
        if not torch.equal(x, y):
            out[path] = float((x.double() - y.double()).abs().max())
    elif isinstance(x, dict):
        assert x.keys() == y.keys(), path
        for k in x:
            diff(x[k], y[k], f"{path}/{k}", out)
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), path
        for i, (p, q) in enumerate(zip(x, y)):
            diff(p, q, f"{path}/{i}", out)
    elif x != y:
        out[path] = [str(x), str(y)]
report = {}
for key in ("iteration", "policy", "reference", "optimizer", "generator", "cpu_rng", "cuda_rng"):
    out = {}
    diff(a[key], b[key], key, out)
    report[key] = {"equal": not out, "differences": len(out),
                   "max_abs": max((v for v in out.values() if isinstance(v, float)), default=0.0),
                   "examples": dict(list(out.items())[:5])}
print(json.dumps(report, indent=2))
