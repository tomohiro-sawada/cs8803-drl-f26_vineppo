#!/bin/bash
#SBATCH --job-name=vineppo-train-pace
#SBATCH --partition=ice-gpu
#SBATCH --account=math
#SBATCH --qos=coc-ice
#SBATCH --constraint=gpu-h100|gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/training/pace_ice/slurm/%x-%j.out
#SBATCH --error=artifacts/training/pace_ice/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
exec bash experiments/training/launch.sh pace_ice train "$@"
