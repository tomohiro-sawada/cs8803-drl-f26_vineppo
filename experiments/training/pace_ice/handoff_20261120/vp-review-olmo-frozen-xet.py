"""Read-only: bucket content hash of the local OLMo frozen base, to compare with the bucket's copy."""
import hashlib, inspect, json, sys
import hf_xet

path = sys.argv[1]
print("hash_files doc:", (inspect.getdoc(hf_xet.hash_files) or "")[:400].replace("\n", " "), flush=True)
result = hf_xet.hash_files([path])
print("result type:", type(result).__name__, [type(r).__name__ for r in result][:2], flush=True)
for item in result:
    fields = {name: getattr(item, name) for name in dir(item) if not name.startswith("_") and not callable(getattr(item, name))}
    print(json.dumps({k: (v if isinstance(v, (int, str)) else str(v)) for k, v in fields.items()}), flush=True)
