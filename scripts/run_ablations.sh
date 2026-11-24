#!/bin/bash
# The offline ablations of the README on a dataset written by train_dynamics.py, as SLURM jobs:
# one job computes the DINOv2 features, then one job per variant group (scripts/ablate.py GROUPS),
# each writing runs/<run>/ablation_<group>/. Without SLURM, run the same commands directly.
#
#   scripts/run_ablations.sh block                 # groups: image_vs_pc encoders arch latent_training data
#   scripts/run_ablations.sh ball
#   scripts/run_ablations.sh block_big big         # data scaling on ~9000 transitions + the best variants
set -euo pipefail
RUN=${1:?run name under runs/}
shift
GRPS=${*:-image_vs_pc encoders arch latent_training data}
SEEDS=${SEEDS:-0 1}
cd "$(dirname "$0")/.."
D=runs/$RUN/dataset.pt
# one job prepares every precomputed input: world-frame DINOv2 tokens, action-frame crops, their tokens
FEAT=$(scripts/slurm_run.sh dino_$RUN "python scripts/dino_features.py --dataset $D \
    && python scripts/image_crops.py --dataset $D \
    && python scripts/dino_features.py --dataset $D --images runs/$RUN/image_af.pt --key af224 --out runs/$RUN/dino_af_feats.pt")
echo "features: job $FEAT"
for G in $GRPS; do
    J=$(DEPENDENCY=$FEAT scripts/slurm_run.sh abl_${RUN}_$G python scripts/ablate.py --dataset $D \
        --out runs/$RUN/ablation_$G --group $G --seeds $SEEDS)
    echo "$G: job $J"
done
