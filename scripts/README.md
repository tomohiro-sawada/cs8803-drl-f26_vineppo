# README. 



## PACE-ICE scheduler logger

On a PACE-ICE login node, start a remote `tmux` session and run `python3 scripts/diagnostics/log_pace_ice.py --local` from the repository root. This executes the read-only commands locally without SSH authentication; add `--once` for a single snapshot. The remote tmux session survives your SSH disconnect, but not a login-node reboot; reconnect to the same login node to reattach. Follow the site's policy for persistent login-node processes.

Alternatively, from the repository root on your laptop, run `python3 scripts/diagnostics/log_pace_ice.py --host YOUR_SSH_DESTINATION --once` to check connectivity, then omit `--once` to collect immediately and hourly until Ctrl-C. Use your existing SSH alias or `user@host`; Python 3.9+ is sufficient, with no project dependencies required.

SSH must work noninteractively (key/agent or an authenticated multiplexed connection), and the host key must already be trusted. Authentication failures are recorded, not prompted for. Each of four read-only Slurm commands has a 60-second timeout. No jobs are submitted. Errors remain visible and the next scheduled poll still runs.

Snapshots append to `artifacts/diagnostics/pace_ice/snapshots.jsonl` (gitignored). Each includes UTC/Eastern timestamps, commit and script hash, command outputs, return codes, and timeout flags. A file lock prevents duplicate writers to the same output. Restarting appends; use `--output` to separate studies. Only completed polls are written; an interrupted poll can leave a coverage gap.

Node records preserve GPU types/counts, `CfgTRES`, `AllocTRES`, `GresUsed`, CPU/RAM allocation, states and partitions. Partition/reservation records preserve access restrictions. Pending-job rows have `job_id|partition|state|submit_time_UTC|time_limit|nodes|tres_per_node|reason`; `tres_per_node` is not a complete GPU-request description (per-job/per-task requests may not appear). Snapshots include visible CPU and GPU resources, not just `ice-gpu`, so shared GPU nodes and other GPU partitions remain identifiable. Count each node once, not once per partition.

These are sequential scheduler observations, not an atomic snapshot, GPU compute utilization, measured queue time, or a guarantee that an unallocated GPU is usable by your account. Retain failed observations as missing data. After collection, use these records for a timestamped report under `prose/knowledge-bases/gpu-cluster-info/pace-ice/`; distinguish allocation pressure from actual access difficulty. Do not publish raw reservation metadata without reviewing it.

## Quickstart. 


- `setup` installs the training and inference environments and stages the paper assets.
- `codebase_hygiene` checks the repository's Git file-size policy.
- `remote_sync` mirrors training run directories to the private Hugging Face
  bucket `tsawada/cs8803-drl-f26_vineppo` and verifies them by SHA256
  (`remote_sync/hf_bucket_sync.py`, which never deletes anything).
  `remote_sync/prune_verified.py` removes local checkpoints of a verified run.
  `remote_sync/delete_planned.py` deletes from the bucket only the paths a
  reviewed plan names, after checking each against the live listing and the
  off-site backup receipt; without `--apply` it only runs the checks.
- `paper_artifact_replications` reproduces the paper's evaluation artifacts.

Run `python3 scripts/codebase_hygiene/check_file_sizes.py` to audit every
tracked file in `HEAD`. Pull-request CI checks newly added or modified Git
blobs against the same 500 KiB limit. The exact `uv.lock` path is exempt so the
repository can retain its generated dependency lockfile.
