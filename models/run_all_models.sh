#!/bin/bash
#
# Run all kinship classification models
# Training: FIW dataset
# Testing: KinFaceW dataset (cross-dataset evaluation)
#

# Activate virtual environment
source venv/bin/activate

EPOCHS=${1:-50}
BATCH_SIZE=${2:-8}

echo "=============================================="
echo "KINSHIP CLASSIFICATION - CROSS-DATASET EVAL"
echo "Training on: FIW"
echo "Testing on: KinFaceW"
echo "Epochs: $EPOCHS, Batch size: $BATCH_SIZE"
echo "=============================================="

# Model 01: Age Synthesis Comparison
echo ""
echo "=== MODEL 01: Age Synthesis Comparison ==="
cd 01_age_synthesis_comparison
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --checkpoint_dir checkpoints
cd ..

# Model 02: ViT + FaCoR Cross-Attention
echo ""
echo "=== MODEL 02: ViT + FaCoR Cross-Attention ==="
cd 02_vit_facor_crossattn
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --checkpoint_dir checkpoints
cd ..

# Model 03: ConvNeXt + ViT Hybrid
echo ""
echo "=== MODEL 03: ConvNeXt + ViT Hybrid ==="
cd 03_convnext_vit_hybrid
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --checkpoint_dir checkpoints
cd ..

# Model 04: Unified Kinship Model
echo ""
echo "=== MODEL 04: Unified Kinship Model ==="
cd 04_unified_kinship_model
python train.py \
    --train_dataset fiw \
    --test_dataset kinface \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --checkpoint_dir checkpoints
cd ..

echo ""
echo "=============================================="
echo "ALL MODELS COMPLETE"
echo "=============================================="
echo "Results saved in each model's checkpoints/ directory"
