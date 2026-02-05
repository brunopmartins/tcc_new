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

### Windows (PowerShell)

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

### Windows (Command Prompt)

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

### Linux (NVIDIA)

```bash
# Build
docker build -f Dockerfile.nvidia -t kinship-nvidia .

# Train
docker run --gpus all -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    kinship-nvidia \
    bash -c "cd /app/models && ./run_all_models_nvidia.sh 50 16 0"

# Interactive shell
docker run --gpus all -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    kinship-nvidia bash
```

### Linux (AMD ROCm)

```bash
# Build
docker build -f Dockerfile.amd -t kinship-amd .

# Train
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
    -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    kinship-amd \
    bash -c "cd /app/models && ./run_all_models_amd.sh 50 16 0"
```

## Using Docker Compose

### NVIDIA GPU

```bash
# Build and train
docker-compose --profile nvidia up nvidia-train

# Evaluate
docker-compose --profile nvidia up nvidia-eval

# Interactive shell
docker-compose --profile nvidia run nvidia-shell
```

### AMD GPU (Linux only)

```bash
# Build and train
docker-compose --profile amd up amd-train

# Evaluate
docker-compose --profile amd up amd-eval
```

## Directory Structure

Before running, create the following directory structure:

```
project/
├── data/
│   ├── FIW/                  # FIW dataset
│   │   ├── FIW_PIDs_v2.csv
│   │   └── FIDs/
│   ├── KinFaceW-I/           # KinFaceW-I dataset
│   │   └── images/
│   │       ├── father-dau/
│   │       ├── father-son/
│   │       ├── mother-dau/
│   │       └── mother-son/
│   └── KinFaceW-II/          # KinFaceW-II dataset (optional)
├── checkpoints/              # Model checkpoints (created automatically)
├── results/                  # Evaluation results (created automatically)
├── models/                   # Model code
├── Dockerfile.nvidia
├── Dockerfile.amd
├── docker-compose.yml
├── docker-run.bat
└── docker-run.ps1
```

## Dataset Configuration

The container uses the following default paths:

| Dataset | Container Path |
|---------|----------------|
| FIW | `/app/data/FIW` |
| KinFaceW-I | `/app/data/KinFaceW-I` |
| KinFaceW-II | `/app/data/KinFaceW-II` |

Mount your dataset directories accordingly:

```bash
docker run --gpus all -it --rm \
    -v /path/to/your/FIW:/app/data/FIW:ro \
    -v /path/to/your/KinFaceW-I:/app/data/KinFaceW-I:ro \
    kinship-nvidia bash
```

## Training Individual Models

Inside the container, you can train individual models:

```bash
# Model 01: Age Synthesis
cd /app/models/01_age_synthesis_comparison/Nvidia
python train.py --train_dataset fiw --epochs 100 --batch_size 32

# Model 02: ViT-FaCoR
cd /app/models/02_vit_facor_crossattn/Nvidia
python train.py --train_dataset fiw --epochs 100 --batch_size 32

# Model 03: ConvNeXt-ViT Hybrid
cd /app/models/03_convnext_vit_hybrid/Nvidia
python train.py --train_dataset fiw --epochs 100 --batch_size 32

# Model 04: Unified Model
cd /app/models/04_unified_kinship_model/Nvidia
python train.py --train_dataset fiw --epochs 100 --batch_size 16
```

## Evaluation

```bash
# Evaluate a specific model
cd /app/models/01_age_synthesis_comparison/Nvidia
python evaluate.py --checkpoint /app/checkpoints/best.pt --full_analysis

# Evaluate all models (from container)
cd /app/models
for model in 01_* 02_* 03_* 04_*; do
    cd $model/Nvidia
    python evaluate.py --checkpoint ../../checkpoints/$model/best.pt --full_analysis
    cd ../..
done
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
python train.py --batch_size 8  # or even 4
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
python train.py --batch_size 8

# Disable AMP (uses more memory but may help with some GPUs)
python train.py --disable_amp
```

### Dataset not found

Make sure to mount the data directory correctly:
```bash
# Check data is visible inside container
docker run --gpus all -it --rm -v $(pwd)/data:/app/data kinship-nvidia ls -la /app/data
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
