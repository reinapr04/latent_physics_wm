#!/bin/bash
# Off-centre COM experiments after training (results: docs/com_offset.md). Run from the repo root, after
#   python train_dynamics.py --run-dir runs/uniform_nomirror --no-mirror-aug
#   python train_dynamics.py --run-dir runs/com015 --com-offset 0.015 --no-mirror-aug
# Each stage is one Modal job, e.g.
#   modal run --detach modal_app.py --name offline --cmd "bash scripts/com_experiments.sh offline"
set -e
EVAL_ENVS=64   # more test settings -> steadier success rates and R^2

case "$1" in
  offline)  # which encoder, and whether an ensemble helps, on the COM dataset (no simulation)
    python scripts/compare_encoders.py --dataset runs/com015/dataset.pt --out runs/com015_encoders \
      --encoders pointnet pointnet_early cross_attn transformer --no-mirror-aug
    python scripts/train_ensemble.py --dataset runs/com015/dataset.pt --out runs/com015_ensemble \
      --members 5 --no-mirror-aug
    ;;
  eval)  # prediction error vs number of fitting pushes + latent R^2, then MPC with mean / fitted / fitted_N
    # runs to evaluate (default both); results that already exist are skipped
    for r in ${2:-uniform_nomirror com015}; do
      [ -f runs/$r/latent_inference.json ] || python eval_latent_inference.py --checkpoint runs/$r/checkpoint.pt --num-envs $EVAL_ENVS
      [ -f runs/$r/mpc_eval.json ] || python eval_mpc.py --checkpoint runs/$r/checkpoint.pt --num-envs $EVAL_ENVS
    done
    # does training on uneven blocks matter? the uniform model tested on COM blocks
    [ -f runs/uniform_nomirror_on_com015/mpc_eval.json ] || python eval_mpc.py --checkpoint runs/uniform_nomirror/checkpoint.pt --com-offset 0.015 --num-envs $EVAL_ENVS \
      --out runs/uniform_nomirror_on_com015/mpc_eval.json
    ;;
  eval_ensemble)  # after "offline"
    python eval_mpc.py --checkpoint runs/com015_ensemble --com-offset 0.015 --num-envs $EVAL_ENVS
    ;;
  *) echo "usage: $0 offline|eval|eval_ensemble"; exit 1 ;;
esac
