#!/bin/bash
#
# Run all kinship classification models on NVIDIA GPUs
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
echo "KINSHIP CLASSIFICATION - NVIDIA CUDA"
echo "Training on: FIW"
echo "Testing on: KinFaceW"
echo "Epochs: $EPOCHS, Batch size: $BATCH_SIZE, GPU: $GPU_ID"
echo "=============================================="

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Model 01: Age Synthesis Comparison
echo ""
echo "=== MODEL 01: Age Synthesis Comparison (NVIDIA) ==="
cd 01_age_synthesis_comparison/Nvidia
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --checkpoint_dir ../checkpoints_nvidia
cd ../..

# Model 02: ViT + FaCoR Cross-Attention
echo ""
echo "=== MODEL 02: ViT + FaCoR Cross-Attention (NVIDIA) ==="
cd 02_vit_facor_crossattn/Nvidia
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --checkpoint_dir ../checkpoints_nvidia
cd ../..

# Model 03: ConvNeXt + ViT Hybrid
echo ""
echo "=== MODEL 03: ConvNeXt + ViT Hybrid (NVIDIA) ==="
cd 03_convnext_vit_hybrid/Nvidia
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --checkpoint_dir ../checkpoints_nvidia
cd ../..

# Model 04: Unified Kinship Model
echo ""
echo "=== MODEL 04: Unified Kinship Model (NVIDIA) ==="
cd 04_unified_kinship_model/Nvidia
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --checkpoint_dir ../checkpoints_nvidia
cd ../..

echo ""
echo "=============================================="
echo "ALL NVIDIA MODELS COMPLETE"
echo "=============================================="
echo "Results saved in each model's checkpoints_nvidia/ directory"
