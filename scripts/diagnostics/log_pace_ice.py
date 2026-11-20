"""Hourly, read-only Slurm snapshots locally or over SSH; Python 3.9+ standard library only."""

# ruff: noqa: UP017 -- timezone.utc keeps this standalone script Python 3.9 compatible.

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/diagnostics/pace_ice/snapshots.jsonl"
COMMANDS = {
    "nodes": "scontrol --oneliner --details show nodes",
    "partitions": "scontrol --oneliner show partitions",
    "reservations": "scontrol --oneliner show reservations",
    "pending_jobs": "squeue --noheader --states=PENDING --format='%i|%P|%T|%V|%l|%D|%b|%r'",
}
EASTERN = ZoneInfo("America/New_York")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ssh_prefix(host):
    return [
        "ssh", "-T", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
        "-o", "StrictHostKeyChecking=yes", host,
    ]


def collect(prefix, timeout):
    """Keep raw output: missing/failed observations must never become zero GPUs."""
    results = {}
    for name, command in COMMANDS.items():
        started = now()
        # Login shell loads the site's Slurm environment; UTC makes queue dates explicit.
        remote = "bash -lc " + shlex.quote("export TZ=UTC LC_ALL=C; " + command)
        try:
            result = subprocess.run(
                [*prefix, remote], capture_output=True, text=True, timeout=timeout, check=False,
            )
            results[name] = {
                "started_at": started, "finished_at": now(), "command": command,
                "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as error:
            # TimeoutExpired carries bytes even when subprocess.run uses text=True.
            results[name] = {
                "started_at": started, "finished_at": now(), "command": command,
                "returncode": None, "stdout": (error.stdout or b"").decode(),
                "stderr": (error.stderr or b"").decode(), "timed_out": True,
            }
    return results


def append(stream, record):
    stream.write(json.dumps(record) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--host", help="Your existing SSH alias or user@host")
    transport.add_argument("--local", action="store_true", help="Run on the cluster without SSH")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Per scheduler command (including SSH when used)")
    parser.add_argument("--once", action="store_true", help="One snapshot; nonzero exit on failure")
    args = parser.parse_args()
    if not args.local and (not args.host or args.host.startswith("-") or any(c.isspace() for c in args.host)):
        parser.error("--host must be an SSH destination, not options or a command")
    if args.interval_seconds < 300 or args.timeout_seconds <= 0:
        parser.error("interval must be >=300 seconds; timeout must be positive")
    if args.timeout_seconds * len(COMMANDS) >= args.interval_seconds:
        parser.error("interval must exceed the combined command timeouts")

    provenance = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": os.uname().nodename if args.local else args.host,
        "transport": "local" if args.local else "ssh",
    }
    prefix = ["bash", "-c"] if args.local else ssh_prefix(args.host)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        # flock is available on both macOS and Linux; the OS releases it on exit/crash.
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"Logging to {args.output.resolve()} (Ctrl-C to stop)", flush=True)
        while True:
            cycle_start = time.monotonic()
            started = datetime.now(timezone.utc)
            results = collect(prefix, args.timeout_seconds)
            failed = [name for name, result in results.items() if result["returncode"] != 0]
            append(stream, {
                "schema_version": 1, **provenance,
                "started_at": started.isoformat(timespec="seconds"),
                "started_at_eastern": started.astimezone(EASTERN).isoformat(timespec="seconds"),
                "finished_at": now(), "interval_seconds": args.interval_seconds,
                "ok": not failed, "commands": results,
            })
            print(f"{now()} {'FAILED: ' + ', '.join(failed) if failed else 'OK'}", flush=True)
            for name in failed:
                result = results[name]
                print(f"  {name}: timeout={result['timed_out']} "
                      f"exit={result['returncode']} {result['stderr'].strip()}", flush=True)
            if args.once:
                return int(bool(failed))
            # No catch-up burst after a slow poll or suspended laptop.
            time.sleep(max(0, args.interval_seconds - (time.monotonic() - cycle_start)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped. Completed snapshots are saved.", flush=True)
