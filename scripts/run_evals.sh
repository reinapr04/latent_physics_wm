#!/bin/bash
# Evaluate a checkpoint on unseen physics (as one SLURM job; without SLURM run the three commands):
#   1. eval_latent_inference.py  prediction error vs. number of probes
#   2. eval_mpc.py               goal reaching, up to 10 pushes (standard)
#   3. eval_mpc.py --hard        goal 15-30 cm away, 2 pushes only (the "harder version" of the README)
# Results land next to the checkpoint (latent_inference.json, mpc_eval.json, mpc_eval_hard.json, videos).
#
#   scripts/run_evals.sh runs/block/checkpoint.pt
#   scripts/run_evals.sh runs/block/ablation_arch/pointnet_res_wide_s0/model.pt
#   scripts/run_evals.sh runs/block_ensemble            # an ensemble directory (train_ensemble.py)
set -euo pipefail
CK=${1:?checkpoint file or ensemble dir}
ENVS=${ENVS:-64}
cd "$(dirname "$0")/.."
case "$CK" in *.pt) OUT=$(dirname "$CK");; *) OUT=$CK;; esac   # a checkpoint file, or an ensemble dir (may not exist yet)
NAME=eval_$(echo "$OUT" | tr '/' '_' | sed 's/^runs_//')
scripts/slurm_run.sh "$NAME" "python eval_latent_inference.py --checkpoint $CK --num-envs $ENVS \
    && python eval_mpc.py --checkpoint $CK --num-envs $ENVS \
    && python eval_mpc.py --checkpoint $CK --num-envs $ENVS --max-pushes 2 --goal-dist 0.15 0.30 --video-envs 0 --out $OUT/mpc_eval_hard.json"
