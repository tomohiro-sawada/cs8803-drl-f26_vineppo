#!/usr/bin/env bash
#SBATCH --job-name=rho-vllm-eval
#SBATCH --partition=kargo-lab
#SBATCH --account=kargo-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/evaluation/slurm/vllm-%j.out
#SBATCH --error=artifacts/evaluation/slurm/vllm-%j.err

set -euo pipefail
# Submit from the repository root; use the already-installed isolated environment.
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
exec .venv-inference-glibc2.39/bin/python -u \
    scripts/paper_artifact_replications/evaluate_sft_vllm.py --expected-gpu A40 "$@"
