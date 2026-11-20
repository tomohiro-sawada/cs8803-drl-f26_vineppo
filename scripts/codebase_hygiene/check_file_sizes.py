"""Check committed Git blobs against the repository's 500 KiB limit."""

import argparse
import os
import subprocess


def git(*args):
    return subprocess.check_output(["git", *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", help="Check files added or modified since the merge base"
    )
    args = parser.parse_args()
    paths = (
        set(
            git(
                "diff",
                "--name-only",
                "--diff-filter=ACMRT",
                "-z",
                f"{args.base}...HEAD",
            ).split(b"\0")
        )
        if args.base is not None
        else None
    )
    over_limit = 0
    for row in git("ls-tree", "-r", "-l", "-z", "HEAD").split(b"\0")[:-1]:
        info, name = row.split(b"\t", 1)
        _, kind, _, size = info.split()
        if (
            kind != b"blob"
            or name == b"uv.lock"
            or (paths is not None and name not in paths)
        ):
            continue
        if int(size) > 500 * 1024:
            print(f"{os.fsdecode(name)!r} is {size.decode()} bytes (limit 500 KiB)")
            over_limit += 1
    return int(over_limit > 0)


if __name__ == "__main__":
    raise SystemExit(main())
