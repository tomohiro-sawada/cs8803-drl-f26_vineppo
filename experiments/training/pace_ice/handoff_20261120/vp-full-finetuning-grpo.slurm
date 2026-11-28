#!/bin/bash
#SBATCH --job-name=vp-full-finetuning-grpo
#SBATCH --partition=ice-gpu
#SBATCH --account=math
#SBATCH --qos=coc-ice
#SBATCH --constraint=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#
# Full fine-tuning control for RhoMath GSM8K GRPO, on node-local disk.
#
#   sbatch vp-full-finetuning-grpo.slurm NAME SEED ITERATIONS CHECKPOINT_EVERY WANDB_MODE ARCHIVE
#
# Environment: PYTHON, VLLM_PYTHON, TRAINING_CACHE, TMPDIR (node-local), and
# RUNS (.../training/pace_ice on scratch), MODEL, DATASET.
#
# A full-weights checkpoint holds the actor, the reference and the AdamW state,
# and several do not fit under the scratch quota. So the run lives under TMPDIR:
#   1. train with the HF backend, all parameters in FP32, the paper's learning
#      rate (the runner's default for full tuning), selecting by validation;
#   2. export the selected actor as a model directory;
#   3. test it for 16 rounds through the vLLM evaluator, as every LoRA run is,
#      into NAME/exported-test, beside the export record that ties the tested
#      weights to the selected checkpoint;
#   4. copy everything but weights to RUNS/NAME;
#   5. with ARCHIVE=yes, mirror the run, weights included, to the bucket and
#      verify. ARCHIVE=no is for smokes, whose weights are dropped.
# The node-local copy is removed last, so a failed upload leaves it on the node.
# The paths recorded in the scratch copy point at the removed directory.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
name=$1 seed=$2 iterations=$3 every=$4 wandb=$5 archive=$6
[[ $archive == yes || $archive == no ]]
: "${PYTHON:?}" "${VLLM_PYTHON:?}" "${TRAINING_CACHE:?}" "${RUNS:?}" "${MODEL:?}" "${DATASET:?}"
[[ ! -e $RUNS/$name ]]
local=$(mktemp -d "${TMPDIR:?}/vp-full-XXXXXX")
runs=$local/training/pace_ice
mkdir -p "$runs"
df -h "$local"
common=(--algorithm grpo --task gsm8k --model-protocol rhomath --dataset-path "$DATASET"
    --eval-seed 42 --wandb-mode "$wandb" --wandb-entity tsawada --wandb-group handoff-20261120)

bash experiments/training/launch.sh pace_ice train "${common[@]}" --model-path "$MODEL" \
    --backend hf --finetuning full --dtype float32 --seed "$seed" \
    --num-iterations "$iterations" --prompts-per-iteration 16 --num-samples 8 \
    --minibatch-size 32 --microbatch-size 4 --update-epochs 2 \
    --checkpoint-every "$every" --keep-checkpoints 1 --checkpoint-format full \
    --evaluation validation --validation-rounds 4 --eval-batch-size 64 \
    --lr-schedule constant --kl-coef 0.003 \
    --wandb-name "$name" --output "$runs/$name"
du -ab "$runs/$name" | sort -n | tail -8

"$PYTHON" scripts/diagnostics/export_checkpoint_model.py \
    "$runs/$name/selection.json" "$MODEL" "$local/model"
TRAINING_ENTRY=scripts/diagnostics/evaluate_model_baseline.py \
    bash experiments/training/launch.sh pace_ice train "${common[@]}" --model-path "$local/model" \
    --dtype bfloat16 --finetuning lora --lora-rank 16 --lora-alpha 32 \
    --backend vllm --num-workers 1 --worker-placement colocated --eval-batch-size 256 \
    --wandb-name "$name-test" --output "$runs/$name/exported-test"
cp "$local/model/export.json" "$runs/$name/exported-test/"
du -sb "$runs/$name" "$local/model"

rsync -a --exclude='*.pt' "$runs/$name" "$RUNS/"
if [[ $archive == yes ]]; then
    export PYTHONPATH=src:.
    for action in push verify; do
        "$PYTHON" scripts/remote_sync/hf_bucket_sync.py "$action" --root "$local" "$runs/$name"
    done
fi
rm -rf "$local"
