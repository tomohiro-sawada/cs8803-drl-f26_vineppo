#!/bin/bash
#SBATCH --job-name=vineppo-pace-chain
#SBATCH --partition=ice-gpu
#SBATCH --account=math
#SBATCH --qos=coc-ice
#SBATCH --constraint=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=artifacts/training/pace_ice/slurm/%x-%j.out
#SBATCH --error=artifacts/training/pace_ice/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pinned worktree}"
exec "${PYTHON:-.venv/bin/python}" -m experiments.training.pace_ice.chain "$@"
