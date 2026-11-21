#!/bin/bash
#SBATCH --partition=ice-cpu
#SBATCH --account=math
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pinned worktree}"
export TMPDIR=/tmp
python=${PYTHON:?Set the absolute learner Python}
run=$1
receipt=$2
verified=$receipt.verified.log
if [[ ! -e $verified ]]; then
    "$python" -u scripts/remote_sync/hf_bucket_sync.py push "$run"
    "$python" -u scripts/remote_sync/hf_bucket_sync.py verify "$run" | tee "$verified.pending"
    mv "$verified.pending" "$verified"
fi
"$python" -m scripts.remote_sync.prune_verified "$run" "$verified" "$receipt"
