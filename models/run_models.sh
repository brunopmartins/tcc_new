#!/bin/bash
#
# Universal runner for kinship classification models
# Automatically detects GPU platform (NVIDIA/AMD) or allows manual selection
#
# Usage:
#   ./run_models.sh                    # Auto-detect platform
#   ./run_models.sh nvidia             # Force NVIDIA
#   ./run_models.sh amd                # Force AMD ROCm
#   ./run_models.sh nvidia 100 16 0    # NVIDIA with epochs=100, batch=16, gpu=0
#   ./run_models.sh amd 50 8 1         # AMD with epochs=50, batch=8, gpu=1
#

PLATFORM=${1:-auto}
EPOCHS=${2:-50}
BATCH_SIZE=${3:-8}
GPU_ID=${4:-0}

# Auto-detect platform
detect_platform() {
    # Check for NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo "nvidia"
            return
        fi
    fi

    # Check for AMD ROCm
    if command -v rocm-smi &> /dev/null; then
        if rocm-smi &> /dev/null; then
            echo "amd"
            return
        fi
    fi

    # Check PyTorch build
    python3 -c "import torch; print('amd' if hasattr(torch.version, 'hip') and torch.version.hip else 'nvidia')" 2>/dev/null
}

echo "=============================================="
echo "KINSHIP CLASSIFICATION MODEL RUNNER"
echo "=============================================="

# Determine platform
if [ "$PLATFORM" = "auto" ]; then
    echo "Auto-detecting GPU platform..."
    PLATFORM=$(detect_platform)
    echo "Detected platform: $PLATFORM"
fi

# Validate platform
if [ "$PLATFORM" != "nvidia" ] && [ "$PLATFORM" != "amd" ]; then
    echo "Error: Invalid platform '$PLATFORM'. Use 'nvidia', 'amd', or 'auto'"
    exit 1
fi

echo ""
echo "Platform: $PLATFORM"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "GPU ID: $GPU_ID"
echo "=============================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run appropriate script
if [ "$PLATFORM" = "nvidia" ]; then
    echo ""
    echo "Running NVIDIA CUDA training..."
    bash run_all_models_nvidia.sh $EPOCHS $BATCH_SIZE $GPU_ID
else
    echo ""
    echo "Running AMD ROCm training..."
    bash run_all_models_amd.sh $EPOCHS $BATCH_SIZE $GPU_ID
fi

echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "Platform: $PLATFORM"
echo "=============================================="
