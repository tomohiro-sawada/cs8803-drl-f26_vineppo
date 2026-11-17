"""Single-use bounded grading worker, invoked by verifier with Python -I.

Resource limits are defense against expensive mathematics, NOT an OS security
sandbox. Code execution is prevented by safe_symbolic's restricted construction.
"""

import json
import resource
import sys
from pathlib import Path


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 11))
    # Darwin rejects address-space/data limits; CPU and wall limits still apply.
    # Production Linux workers additionally have a hard 1 GiB address-space cap.
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # -I ignores PYTHONPATH; admit only the directory containing our vendor package.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vendor.math_grader_minerva import eval_math

    predictions, references = json.load(sys.stdin)
    correct = bool(eval_math({"prediction": predictions, "answer": references}))
    print(json.dumps(correct))


if __name__ == "__main__":
    main()
