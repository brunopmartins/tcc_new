#!/bin/bash
# =============================================================================
# Model 01: Age Synthesis Comparison — NVIDIA CUDA pipeline runner
# Called by docker-compose.nvidia.yml to avoid YAML shell escaping issues.
# Environment variables (EPOCHS, BATCH_SIZE, etc.) are set by docker-compose.
# =============================================================================
set -e

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_DATASET="${TRAIN_DATASET:-fiw}"
USE_AGE_SYNTHESIS="${USE_AGE_SYNTHESIS:-1}"

echo '============================================'
echo 'Model 01: Age Synthesis Comparison'
echo 'Platform: NVIDIA CUDA'
echo '============================================'
echo "Epochs:        ${EPOCHS}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "GPU ID:        ${GPU_ID:-0}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Dataset:       ${TRAIN_DATASET}"
echo "Age Synthesis: $([ "${USE_AGE_SYNTHESIS}" = "1" ] && echo 'enabled' || echo 'disabled')"
echo '============================================'

mkdir -p /app/output/checkpoints /app/output/results /app/output/logs

AGE_FLAG=""
if [ "${USE_AGE_SYNTHESIS}" = "1" ]; then
    AGE_FLAG="--use_age_synthesis"
fi

echo '[1/3] Training...'
python train.py \
  --train_dataset "${TRAIN_DATASET}" \
  --test_dataset kinface \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LEARNING_RATE}" \
  --checkpoint_dir /app/output/checkpoints \
  ${AGE_FLAG} \
  2>&1 | tee /app/output/logs/train.log

echo '[2/3] Testing...'
python test.py \
  --checkpoint /app/output/checkpoints/best.pt \
  --dataset kinface \
  --output_dir /app/output/results \
  --save_predictions \
  2>&1 | tee /app/output/logs/test.log

echo '[3/3] Evaluating...'
python evaluate.py \
  --checkpoint /app/output/checkpoints/best.pt \
  --dataset kinface \
  --output_dir /app/output/results \
  --full_analysis \
  --ablation \
  2>&1 | tee /app/output/logs/evaluate.log

echo ''
echo '============================================'
echo 'Model 01 completed!'
echo 'Results saved to /app/output/'
echo '============================================'
