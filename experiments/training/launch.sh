#!/bin/bash
# Shared entry point for the cluster wrappers; all runner options take a value.
set -euo pipefail
fail() { printf 'launch: %s\n' "$*" >&2; exit 2; }
[[ $# -ge 2 ]] || fail 'Expected cluster and mode from a SLURM wrapper'
cluster=$1
mode=$2
shift 2
case "$cluster" in skynet|pace_ice) ;; *) fail "Unknown cluster: $cluster" ;; esac
case "$mode" in train|benchmark) ;; *) fail "Unknown mode: $mode" ;; esac
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"

args=("$@")
task= model= dataset= output= resume= expected_gpu=
while (( $# )); do
    [[ $# -ge 2 && -n $2 && $2 != --* ]] || fail "Missing value for $1"
    case "$1" in
        --algorithm)
            case "$2" in
                vineppo|grpo|rloo) ;;
                ppo) [[ $cluster == pace_ice ]] || fail 'PPO is not supported on Skynet; use PACE ICE' ;;
                *) fail "Unsupported algorithm: $2 (choose vineppo, ppo, grpo, or rloo)" ;;
            esac ;;
        --wandb-mode|--wandb-project|--wandb-entity|--wandb-group|--wandb-name) ;;
        --task) task=$2 ;;
        --model-path) model=$2 ;;
        --dataset-path) dataset=$2 ;;
        --output) output=$2 ;;
        --resume) resume=$2 ;;
        --expected-gpu) expected_gpu=$2 ;;
        --num-iterations|--warmup-iterations|--prompts-per-iteration|--num-samples|--mc-num-samples|--update-epochs|--minibatch-size|--microbatch-size|--max-sequence-length|--max-new-tokens|--learning-rate|--critic-learning-rate|--seed|--checkpoint-every|--keep-checkpoints|--dtype|--finetuning|--lora-rank|--lora-alpha) ;;
        *) fail "Unsupported option: $1 (mode is fixed by the SLURM wrapper; use --option value)" ;;
    esac
    shift 2
done
case "$task" in gsm8k|math) ;; *) fail '--task gsm8k|math is required' ;; esac
[[ -n $model && -d $model ]] || fail '--model-path must name an existing local model directory'
[[ -n $dataset && -e $dataset ]] || fail '--dataset-path must name an existing local dataset'
[[ -n $output ]] || fail '--output is required; choose a unique artifacts/training/<cluster>/<runname>'
[[ ! -e $output && ! -L $output ]] || fail "Output already exists; choose a new run directory: $output"
[[ -z $resume || -e $resume ]] || fail "Resume checkpoint does not exist: $resume"
case "$cluster" in
    skynet)
        expected_gpu=${expected_gpu:-A40}
        [[ $expected_gpu == A40 ]] || fail 'Skynet requests A40; --expected-gpu must be A40' ;;
    pace_ice)
        case "$expected_gpu" in H100|H200) ;; *) fail 'PACE requires --expected-gpu H100 or H200 and a matching sbatch --constraint' ;; esac ;;
esac

[[ -n ${TRAINING_CACHE:-} && $TRAINING_CACHE == /* ]] || fail 'Set TRAINING_CACHE to an absolute shared cache path outside run outputs'
python=${PYTHON:-.venv/bin/python}
[[ -x $python ]] || fail "Python executable not found: $python (set PYTHON to an explicit executable path)"
[[ -f experiments/training/run.py ]] || fail 'experiments/training/run.py must be provided by the runner implementation'
# Resolve paths before checking containment, including existing symlink parents.
"$python" -c 'import pathlib, sys; cache, output = map(lambda p: pathlib.Path(p).resolve(), sys.argv[1:]); assert cache != output and output not in cache.parents, "TRAINING_CACHE must be outside the run output"' "$TRAINING_CACHE" "$output"
export HF_HOME="$TRAINING_CACHE/huggingface"
export XDG_CACHE_HOME="$TRAINING_CACHE/xdg"
export TORCH_HOME="$TRAINING_CACHE/torch"
export TRITON_CACHE_DIR="$TRAINING_CACHE/triton"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?Expected a SLURM allocation}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

defaults=(--mode "$mode" --algorithm vineppo --expected-gpu "$expected_gpu"
    --prompts-per-iteration 2 --num-samples 2 --mc-num-samples 2
    --update-epochs 1 --minibatch-size 2 --microbatch-size 1
    --max-sequence-length 2048 --max-new-tokens 1024
    --keep-checkpoints 2 --dtype float32 --wandb-mode online)
case "$mode" in
    train) defaults+=(--num-iterations 100 --warmup-iterations 0 --checkpoint-every 10) ;;
    benchmark) defaults+=(--num-iterations 3 --warmup-iterations 1) ;;
esac
# argparse uses the last occurrence, so explicit CLI values override defaults.
exec srun --ntasks=1 "$python" experiments/training/run.py "${defaults[@]}" "${args[@]}"
