#!/bin/bash
#SBATCH --job-name=rho-sft-eval
#SBATCH --partition=kargo-lab
#SBATCH --account=kargo-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/evaluation/slurm/%x-%j.out
#SBATCH --error=artifacts/evaluation/slurm/%x-%j.err

# Submit from the repository root; create the log directory before sbatch.
# Arguments are forwarded to evaluate_sft.py (task and output are required).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
test -x .venv/bin/python
test -f scripts/paper_artifact_replications/evaluate_sft.py
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
source bash.sh
timestamp
git --no-pager rev-parse --short HEAD
nvidia-smi
srun .venv/bin/python scripts/paper_artifact_replications/evaluate_sft.py --expected-gpu A40 "$@"
