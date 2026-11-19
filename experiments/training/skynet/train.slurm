#!/bin/bash
#SBATCH --job-name=vineppo-train-skynet
#SBATCH --partition=kargo-lab
#SBATCH --account=kargo-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/training/skynet/slurm/%x-%j.out
#SBATCH --error=artifacts/training/skynet/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
exec bash experiments/training/launch.sh skynet train "$@"
