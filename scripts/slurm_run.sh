#!/bin/bash
# Run a command from the repo root as a SLURM GPU job, with logs in slurm_logs/.
#
#   scripts/slurm_run.sh <job name> <command ...>
#   scripts/slurm_run.sh block python train_dynamics.py --run-dir runs/block
#
# Edit the partition / resources below for your cluster. The command runs in the conda env
# named by $CONDA_ENV (default lpwm) with the repo root as the working directory.
set -euo pipefail
NAME=$1; shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/slurm_logs"
DEP=${DEPENDENCY:+--dependency=afterok:$DEPENDENCY}   # DEPENDENCY=<job id>: start after that job succeeds
sbatch --parsable -p "${PARTITION:-all}" --gres=gpu:1 --cpus-per-task="${CPUS:-8}" --mem="${MEM:-48G}" \
    --time="${TIME:-23:00:00}" --job-name="$NAME" --output="$ROOT/slurm_logs/${NAME}_%j.out" $DEP \
    --wrap "cd $ROOT && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV:-lpwm} && $*"
