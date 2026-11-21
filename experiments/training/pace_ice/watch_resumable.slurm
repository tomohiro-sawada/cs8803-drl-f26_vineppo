#!/bin/bash
#SBATCH --job-name=vp26-watch
#SBATCH --partition=ice-cpu
#SBATCH --account=math
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pinned worktree}"
exec "${PYTHON:?Set the absolute learner Python}" -u -m experiments.training.pace_ice.watch "$1" --slurm
