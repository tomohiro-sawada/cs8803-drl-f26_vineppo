from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
import time

worker = Path.cwd() / "src/core/_verifier_worker.py"
pairs = [(("0",), (r"\frac{\sqrt{7}}{3}",))] * 24
def run(pair):
    start = time.monotonic()
    result = subprocess.run([sys.executable, "-I", str(worker)], input=json.dumps(pair),
                            text=True, env={}, capture_output=True, timeout=60, check=True)
    return json.loads(result.stdout), time.monotonic() - start
for parallelism in (1, 4):
    with ThreadPoolExecutor(parallelism) as pool:
        results = list(pool.map(run, pairs))
    assert all(correct is False for correct, _ in results)
    durations = sorted(seconds for _, seconds in results)
    print(parallelism, durations, flush=True)
