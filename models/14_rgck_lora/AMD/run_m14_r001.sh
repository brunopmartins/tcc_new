#!/usr/bin/env bash
# =====================================================================
# Model 14: RGCK-Net + LoRA — R001 launch wrapper
#
# M14 R001 = the proven M12 data+head recipe (symmetric forward +
# comparison-only fusion + relation-type aux + role-matched hard negatives
# at 30%) with the AdaFace backbone adapted by LoRA (rank 16, alpha 16 on
# stage 4 + output_layer) INSTEAD of the stage-4 unfreeze.
#
# Why: the M12 diagnosis showed Val AUC peaks at ~epoch 3 and then memorises
# (train loss keeps falling, Val AUC drops). Full stage-4 unfreeze (~31 M
# trainable) gives lots of room to memorise training families. LoRA adds a
# small, regularised, low-rank adaptation surface (~0.9 M backbone params)
# to the same layers, aiming to adapt the kinship representation with much
# less memorisation and a higher generalisable ceiling.
#
# Single LR (no differential LR — there are no unfrozen base backbone params),
# no consistency loss, no extra stage-3 unfreeze. Only the tokenizer's
# backbone-adaptation mechanism changes vs the proven recipe.
#
# Usage:
#   bash models/14_rgck_lora/AMD/run_m14_r001.sh
# =====================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_INSTALL=1 \
ALIGNED_ROOT="${ALIGNED_ROOT:-/home/bruno/Desktop/tcc_new/datasets/FIW_aligned}" \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
FREEZE_BACKBONE=1 \
UNFREEZE_LAST_STAGE=0 \
LEARNING_RATE=1e-4 \
RELATION_AUX_WEIGHT=0.05 \
SYMMETRIC_FORWARD=1 \
COMPARISON_ONLY_FUSION=1 \
TRAIN_NEGATIVE_STRATEGY=relation_matched \
HARD_NEGATIVE_RATIO=0.30 \
LORA_RANK=16 \
LORA_ALPHA=16 \
NUM_WORKERS=4 SEED=42 \
bash "${SCRIPT_DIR}/run_pipeline.sh"
