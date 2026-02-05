# Docker Setup Guide

This guide explains how to run the kinship classification models using Docker on Windows (with Linux containers) or Linux.

## Prerequisites

### Windows (NVIDIA GPU)

1. **Docker Desktop for Windows** with WSL2 backend
   - Download from https://www.docker.com/products/docker-desktop
   - Enable WSL2 integration in settings

2. **NVIDIA GPU Driver** (Windows driver, not Linux)
   - Download from https://www.nvidia.com/drivers

3. **NVIDIA Container Toolkit** (automatically included with Docker Desktop 4.1+)

4. **WSL2** with Ubuntu (optional but recommended)
   ```powershell
   wsl --install -d Ubuntu
   ```

### Linux (NVIDIA GPU)

1. **Docker Engine**
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   ```

2. **NVIDIA Container Toolkit**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

### Linux (AMD GPU with ROCm)

1. **ROCm Drivers** (version 5.4+)
   ```bash
   # Follow AMD's ROCm installation guide for your distribution
   # https://rocm.docs.amd.com/
   ```

2. **Docker Engine** (same as above)

## Quick Start

### Build the Docker Image

```bash
# NVIDIA
docker build -f Dockerfile.nvidia -t kinship-nvidia .

# AMD
docker build -f Dockerfile.amd -t kinship-amd .
```

### Run Full Pipeline (Train + Test + Evaluate)

The docker-compose configuration runs the complete pipeline automatically:

```bash
# NVIDIA GPU - Run with default settings (50 epochs, batch size 16)
docker-compose --profile nvidia up

# AMD GPU (Linux only)
docker-compose --profile amd up
```

### Custom Training Parameters

Use environment variables to customize training:

```bash
# Custom epochs and batch size
EPOCHS=100 BATCH_SIZE=32 docker-compose --profile nvidia up

# All available parameters
EPOCHS=100 \
BATCH_SIZE=32 \
GPU_ID=0 \
LEARNING_RATE=1e-4 \
TRAIN_DATASET=fiw \
docker-compose --profile nvidia up
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EPOCHS` | 50 | Number of training epochs |
| `BATCH_SIZE` | 16 | Training batch size |
| `GPU_ID` | 0 | GPU device ID to use |
| `LEARNING_RATE` | 1e-4 | Learning rate |
| `TRAIN_DATASET` | fiw | Dataset to train on (fiw, kinface1, kinface2) |

### Interactive Shell

```bash
# NVIDIA
docker-compose --profile nvidia-shell run nvidia-shell

# AMD
docker-compose --profile amd-shell run amd-shell
```

## Directory Structure

```
project/
├── data/                     # Datasets (mount point)
│   ├── FIW/                  # FIW dataset
│   │   ├── FIW_PIDs_v2.csv
│   │   └── FIDs/
│   ├── KinFaceW-I/           # KinFaceW-I dataset
│   │   └── images/
│   └── KinFaceW-II/          # KinFaceW-II dataset (optional)
│       └── images/
├── models/                   # Model code
│   ├── 01_age_synthesis_comparison/
│   │   ├── Nvidia/
│   │   ├── AMD/
│   │   └── output/           # Generated during training
│   │       ├── checkpoints/  # Model weights
│   │       ├── results/      # Evaluation results
│   │       └── logs/         # Training logs
│   ├── 02_vit_facor_crossattn/
│   │   └── ...
│   ├── 03_convnext_vit_hybrid/
│   │   └── ...
│   └── 04_unified_kinship_model/
│       └── ...
├── Dockerfile.nvidia
├── Dockerfile.amd
├── docker-compose.yml
├── docker-run.bat
└── docker-run.ps1
```

## Dataset Setup

Place your datasets in the `data/` folder at the project root:

**FIW Dataset:**
```
data/FIW/
├── FIW_PIDs_v2.csv
└── FIDs/
    ├── F0001/
    │   ├── MID1/
    │   │   └── *.jpg
    │   └── MID2/
    │       └── *.jpg
    └── ...
```

**KinFaceW-I/II Dataset:**
```
data/KinFaceW-I/
└── images/
    ├── father-dau/
    │   ├── fd_001_1.jpg
    │   ├── fd_001_2.jpg
    │   └── ...
    ├── father-son/
    ├── mother-dau/
    └── mother-son/
```

## Output Structure

After running the pipeline, each model will have an `output/` folder:

```
models/01_age_synthesis_comparison/output/
├── checkpoints/
│   ├── best.pt              # Best model weights
│   ├── last.pt              # Last epoch weights
│   └── checkpoint_*.pt      # Periodic checkpoints
├── results/
│   ├── metrics.json         # Evaluation metrics
│   ├── roc_curve.png        # ROC curve plot
│   ├── confusion_matrix.png # Confusion matrix
│   └── ...                  # Additional analysis files
└── logs/
    ├── train.log            # Training output
    ├── test.log             # Test output
    └── evaluate.log         # Evaluation output
```

## Using Windows Scripts

### PowerShell

```powershell
# Build the Docker image
.\docker-run.ps1 build

# Train all models (default: 50 epochs, batch size 16)
.\docker-run.ps1 train

# Train with custom settings
.\docker-run.ps1 train -Epochs 100 -BatchSize 32 -GPU 0

# Evaluate models
.\docker-run.ps1 eval

# Open interactive shell
.\docker-run.ps1 shell

# Check status
.\docker-run.ps1 status
```

### Command Prompt

```batch
REM Build the Docker image
docker-run.bat build

REM Train all models
docker-run.bat train 50 16 0

REM Evaluate models
docker-run.bat eval

REM Open interactive shell
docker-run.bat shell
```

## Running Individual Models

Inside the container, you can train individual models:

```bash
# Model 01: Age Synthesis
cd /app/models/01_age_synthesis_comparison/Nvidia
python train.py --train_dataset fiw --epochs 100 --batch_size 32 --checkpoint_dir ../output/checkpoints

# Model 02: ViT-FaCoR
cd /app/models/02_vit_facor_crossattn/Nvidia
python train.py --train_dataset fiw --epochs 100 --batch_size 32 --checkpoint_dir ../output/checkpoints

# Model 03: ConvNeXt-ViT Hybrid
cd /app/models/03_convnext_vit_hybrid/Nvidia
python train.py --train_dataset fiw --epochs 100 --batch_size 32 --checkpoint_dir ../output/checkpoints

# Model 04: Unified Model
cd /app/models/04_unified_kinship_model/Nvidia
python train.py --train_dataset fiw --epochs 100 --batch_size 16 --checkpoint_dir ../output/checkpoints
```

## GPU Memory Requirements

| Model | Minimum VRAM | Recommended VRAM |
|-------|--------------|------------------|
| 01_age_synthesis | 6 GB | 8 GB |
| 02_vit_facor | 8 GB | 12 GB |
| 03_convnext_vit | 10 GB | 16 GB |
| 04_unified | 14 GB | 24 GB |

Reduce batch size if you encounter OOM errors:

```bash
BATCH_SIZE=8 docker-compose --profile nvidia up
```

## Troubleshooting

### Docker: Cannot connect to Docker daemon

**Windows:**
- Ensure Docker Desktop is running
- Check WSL2 integration is enabled

### NVIDIA: GPU not detected

**Windows:**
```powershell
# Check GPU is visible
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

If this fails:
1. Update NVIDIA drivers
2. Restart Docker Desktop
3. Restart Windows

### AMD ROCm: Device not found

ROCm containers require:
1. Linux host (not WSL2)
2. ROCm drivers installed on host
3. User in `video` and `render` groups

### Out of Memory

```bash
# Reduce batch size
BATCH_SIZE=8 docker-compose --profile nvidia up

# Or for individual training
python train.py --batch_size 8
```

### Dataset not found

Make sure datasets are in the correct location:
```bash
# Check data is visible inside container
docker-compose --profile nvidia-shell run nvidia-shell ls -la /app/data
```

## Building Custom Images

### Add custom dependencies

Edit the Dockerfile and rebuild:

```dockerfile
# Add to Dockerfile.nvidia
RUN pip install your-custom-package
```

Then rebuild:
```bash
docker build -f Dockerfile.nvidia -t kinship-nvidia:custom .
```

### Use a different CUDA version

Change the base image in `Dockerfile.nvidia`:

```dockerfile
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04
```

And update the PyTorch installation accordingly.
