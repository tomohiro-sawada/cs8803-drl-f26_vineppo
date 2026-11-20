#!/bin/bash
#SBATCH --job-name=vineppo-train-resumable
#SBATCH --partition=overcap
#SBATCH --account=kargo-lab
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=artifacts/training/skynet/slurm/%x-%j.out
#SBATCH --error=artifacts/training/skynet/slurm/%x-%j.err

# Usage: sbatch train_resumable.slurm <run-root> <runner options>
# Every start (first submission, preemption requeue, or resubmitting the same
# command after the time limit) trains into a new <run-root>/attempt-NN and
# resumes from the latest checkpoint until selection freezes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
[[ $# -ge 1 && $1 != --* ]] || { echo 'Expected a run root directory first' >&2; exit 2; }
root=$1
shift
for option in "$@"; do
    case "$option" in
        --output|--resume|--resume-selection)
            echo "$option is chosen by this wrapper" >&2; exit 2 ;;
    esac
done
mkdir -p "$root"
exec 9>"$root/.resume.lock"
flock -n 9
shopt -s nullglob
frozen=("$root"/attempt-*/selection-frozen.json)
if (( ${#frozen[@]} )); then
    (( ${#frozen[@]} == 1 )) || { echo 'Multiple frozen selections for this run' >&2; exit 2; }
    exec bash experiments/training/launch.sh skynet evaluate "$@" \
        --output "$root" --resume-selection "${frozen[0]}"
fi
attempt=$(find "$root" -maxdepth 1 -type d -name 'attempt-*' | wc -l)
output=$(printf '%s/attempt-%02d' "$root" "$attempt")
# Highest completed iteration; the newest attempt wins a tie. *.pt.tmp never match.
latest=$(find "$root" -path '*/checkpoints/iteration-*.pt' |
    awk -F'iteration-' '{print $NF+0 "\t" $0}' | sort -k1,1n -k2,2 | tail -n 1 | cut -f2-)
resume=()
if [[ -n $latest ]]; then
    resume=(--resume "$latest")
    # Present exactly when the source attempt ran automatic selection; the runner
    # rejects an inconsistent combination.
    selection=$(dirname "$(dirname "$latest")")/selection.json
    [[ ! -e $selection ]] || resume+=(--resume-selection "$selection")
fi
exec bash experiments/training/launch.sh skynet train "$@" --output "$output" "${resume[@]}"
