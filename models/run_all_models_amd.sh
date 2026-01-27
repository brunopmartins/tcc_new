#!/bin/bash
#
# Run all kinship classification models on AMD GPUs (ROCm)
# Training: FIW dataset
# Testing: KinFaceW dataset (cross-dataset evaluation)
#

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

EPOCHS=${1:-50}
BATCH_SIZE=${2:-8}
GPU_ID=${3:-0}

echo "=============================================="
echo "KINSHIP CLASSIFICATION - AMD ROCm"
echo "Training on: FIW"
echo "Testing on: KinFaceW"
echo "Epochs: $EPOCHS, Batch size: $BATCH_SIZE, GPU: $GPU_ID"
echo "=============================================="

# Set ROCm device
export HIP_VISIBLE_DEVICES=$GPU_ID
export ROCR_VISIBLE_DEVICES=$GPU_ID

# ROCm optimizations
export MIOPEN_FIND_MODE=FAST
export HSA_FORCE_FINE_GRAIN_PCIE=1

# Check ROCm installation
echo ""
echo "Checking ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname
else
    echo "Warning: rocm-smi not found. Make sure ROCm is properly installed."
fi

# Model 01: Age Synthesis Comparison
echo ""
echo "=== MODEL 01: Age Synthesis Comparison (AMD ROCm) ==="
cd 01_age_synthesis_comparison/AMD
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --rocm_device $GPU_ID \
    --checkpoint_dir ../checkpoints_amd
cd ../..

# Model 02: ViT + FaCoR Cross-Attention
echo ""
echo "=== MODEL 02: ViT + FaCoR Cross-Attention (AMD ROCm) ==="
cd 02_vit_facor_crossattn/AMD
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --rocm_device $GPU_ID \
    --checkpoint_dir ../checkpoints_amd
cd ../..

# Model 03: ConvNeXt + ViT Hybrid
echo ""
echo "=== MODEL 03: ConvNeXt + ViT Hybrid (AMD ROCm) ==="
cd 03_convnext_vit_hybrid/AMD
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --rocm_device $GPU_ID \
    --checkpoint_dir ../checkpoints_amd
cd ../..

# Model 04: Unified Kinship Model
echo ""
echo "=== MODEL 04: Unified Kinship Model (AMD ROCm) ==="
cd 04_unified_kinship_model/AMD
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --rocm_device $GPU_ID \
    --checkpoint_dir ../checkpoints_amd
cd ../..

echo ""
echo "=============================================="
echo "ALL AMD ROCm MODELS COMPLETE"
echo "=============================================="
echo "Results saved in each model's checkpoints_amd/ directory"
