# GPU Platform Support

This project supports both NVIDIA CUDA and AMD ROCm GPU platforms for training and inference.

## Directory Structure

Each model directory now contains two platform-specific subfolders:

```
models/
├── 01_age_synthesis_comparison/
│   ├── Nvidia/           # NVIDIA CUDA implementation
│   │   ├── train.py
│   │   ├── test.py
│   │   └── evaluate.py
│   ├── AMD/              # AMD ROCm implementation
│   │   ├── train.py
│   │   ├── test.py
│   │   └── evaluate.py
│   ├── model.py          # Shared model architecture
│   └── README.md
│
├── 02_vit_facor_crossattn/
│   ├── Nvidia/
│   └── AMD/
│
├── 03_convnext_vit_hybrid/
│   ├── Nvidia/
│   └── AMD/
│
├── 04_unified_kinship_model/
│   ├── Nvidia/
│   └── AMD/
│
├── shared/
│   ├── Nvidia/           # NVIDIA-specific utilities
│   │   └── trainer.py
│   ├── AMD/              # AMD ROCm-specific utilities
│   │   ├── __init__.py
│   │   ├── rocm_utils.py
│   │   └── trainer.py
│   ├── config.py         # Shared configuration
│   ├── dataset.py        # Shared data loading
│   ├── losses.py         # Shared loss functions
│   └── evaluation.py     # Shared metrics
│
├── run_all_models_nvidia.sh   # Run all models on NVIDIA
├── run_all_models_amd.sh      # Run all models on AMD ROCm
└── run_models.sh              # Universal runner (auto-detect)
```

## Running Models

### Automatic Platform Detection

The universal runner script will automatically detect your GPU platform:

```bash
./run_models.sh
```

### Manual Platform Selection

#### NVIDIA CUDA

```bash
# Run all models
./run_all_models_nvidia.sh [epochs] [batch_size] [gpu_id]

# Example
./run_all_models_nvidia.sh 100 32 0

# Or use the universal runner
./run_models.sh nvidia 100 32 0
```

#### AMD ROCm

```bash
# Run all models
./run_all_models_amd.sh [epochs] [batch_size] [gpu_id]

# Example
./run_all_models_amd.sh 100 32 0

# Or use the universal runner
./run_models.sh amd 100 32 0
```

### Running Individual Models

#### NVIDIA

```bash
cd models/01_age_synthesis_comparison/Nvidia
python train.py --train_dataset fiw --epochs 100 --batch_size 32
python test.py --checkpoint ../checkpoints_nvidia/best.pt --dataset kinface
```

#### AMD ROCm

```bash
cd models/01_age_synthesis_comparison/AMD
python train.py --train_dataset fiw --epochs 100 --batch_size 32 --rocm_device 0
python test.py --checkpoint ../checkpoints_amd/best.pt --dataset kinface --rocm_device 0
```

## AMD ROCm-Specific Features

The AMD implementation includes several ROCm-specific optimizations:

### ROCm Utilities (`shared/AMD/rocm_utils.py`)

- `setup_rocm_environment()`: Configure ROCm environment variables
- `check_rocm_availability()`: Verify ROCm/HIP installation
- `get_rocm_device()`: Get ROCm device with error handling
- `optimize_for_rocm()`: Apply ROCm-specific model optimizations
- `clear_rocm_cache()`: Clear GPU memory cache
- `print_rocm_info()`: Display detailed ROCm system information

### Command Line Arguments (AMD scripts)

| Argument | Description | Default |
|----------|-------------|---------|
| `--rocm_device` | ROCm GPU device ID | 0 |
| `--disable_amp` | Disable mixed precision | False |
| `--gfx_version` | Override GFX version for compatibility | None |

### Environment Variables

The AMD scripts automatically set these optimizations:

```bash
MIOPEN_FIND_MODE=FAST          # Faster convolution algorithm selection
HSA_FORCE_FINE_GRAIN_PCIE=1    # Better memory management
HIP_VISIBLE_DEVICES=<id>       # GPU device selection
```

### GFX Version Override

For newer AMD GPUs that may not be officially supported, you can override the GFX version:

```bash
python train.py --gfx_version "10.3.0" ...
```

## Requirements

### NVIDIA CUDA

- PyTorch with CUDA support
- CUDA Toolkit 11.0+
- cuDNN 8.0+

### AMD ROCm

- PyTorch with ROCm support (install from https://pytorch.org/)
- ROCm 5.0+ (recommended: 5.4+)
- AMD GPU with GFX9 or newer architecture

Install PyTorch with ROCm:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
```

## Checkpoints

Checkpoints are saved in platform-specific directories:

- NVIDIA: `<model>/checkpoints_nvidia/`
- AMD: `<model>/checkpoints_amd/`

Checkpoints include a `platform` field to track which platform was used for training.

## Cross-Platform Compatibility

Models trained on one platform can be loaded on another platform for inference, as the model architecture is shared. However, for best performance, it's recommended to use the same platform for training and inference.

```python
# Load NVIDIA-trained model on AMD
checkpoint = torch.load("checkpoints_nvidia/best.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
```

## Troubleshooting

### AMD ROCm Issues

1. **"ROCm not available"**: Ensure PyTorch is installed with ROCm support
2. **Memory errors**: Try reducing batch size or enabling gradient accumulation
3. **GFX version errors**: Use `--gfx_version` to override

### NVIDIA CUDA Issues

1. **CUDA out of memory**: Reduce batch size or enable gradient checkpointing
2. **cuDNN errors**: Update CUDA and cuDNN to compatible versions
