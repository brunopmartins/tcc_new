# Kinship Classification Models

This directory contains four novel model architectures for kinship verification from facial images, along with shared utilities for training, evaluation, and data loading.

## Model Overview

| Model | Key Innovation | Expected Improvement |
|-------|----------------|---------------------|
| [01_age_synthesis_comparison](./01_age_synthesis_comparison/) | Age synthesis + all-vs-all comparison | +4-5% on age-gap pairs |
| [02_vit_facor_crossattn](./02_vit_facor_crossattn/) | ViT backbone + FaCoR cross-attention | +2-3% overall |
| [03_convnext_vit_hybrid](./03_convnext_vit_hybrid/) | Dual CNN-Transformer backbone | +3-4% overall |
| [04_unified_kinship_model](./04_unified_kinship_model/) | All techniques combined | +8-10% overall |

## Directory Structure

```
models/
├── README.md                      # This file
├── requirements.txt               # Dependencies
├── shared/                        # Shared utilities
│   ├── __init__.py
│   ├── config.py                  # Configuration dataclasses
│   ├── dataset.py                 # Data loading for FIW, KinFaceW
│   ├── losses.py                  # Loss functions (contrastive, triplet, fair, etc.)
│   ├── evaluation.py              # Metrics and evaluation utilities
│   └── trainer.py                 # Training loop
│
├── 01_age_synthesis_comparison/   # Age-aware model
│   ├── README.md
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   └── evaluate.py
│
├── 02_vit_facor_crossattn/        # ViT + Cross-attention model
│   ├── README.md
│   ├── model.py
│   ├── train.py
│   └── test.py
│
├── 03_convnext_vit_hybrid/        # Hybrid CNN-Transformer model
│   ├── README.md
│   ├── model.py
│   ├── train.py
│   └── test.py
│
└── 04_unified_kinship_model/      # Combined model (all techniques)
    ├── README.md
    ├── model.py
    ├── train.py
    └── test.py
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

Datasets should be in `../datasets/`:
- `KinFaceW-I/` - KinFaceW-I dataset
- `KinFaceW-II/` - KinFaceW-II dataset  
- `FIW/` - Families in the Wild dataset

### 3. Train a Model

```bash
# Start with the simplest model
cd 01_age_synthesis_comparison
python train.py --dataset kinface --epochs 100

# Or try the unified model
cd 04_unified_kinship_model
python train.py --dataset kinface --epochs 100 --use_cross_attention
```

### 4. Test a Model

```bash
python test.py --checkpoint checkpoints/best.pt --dataset kinface
```

## Available Datasets

| Dataset | Location | Pairs | Notes |
|---------|----------|-------|-------|
| KinFaceW-I | `../datasets/KinFaceW-I/` | ~1,000 | 4 relations, different photos |
| KinFaceW-II | `../datasets/KinFaceW-II/` | ~1,000 | 4 relations, same photo pairs |
| FIW | `../datasets/FIW/` | ~100,000 | 11 relations, large-scale |

## Model Comparison

### Architecture Summary

| Model | Backbone | Cross-Attn | Age Synth | Params |
|-------|----------|------------|-----------|--------|
| Age Synthesis | ResNet-50 | ✗ | ✓ | ~30M |
| ViT-FaCoR | ViT-Base | ✓ | ✗ | ~90M |
| ConvNeXt-ViT | ConvNeXt + ViT | ✗ | ✗ | ~175M |
| Unified | ConvNeXt + ViT | ✓ | ✓ | ~180M |

### Expected Performance (KinFaceW-I)

| Model | Accuracy | F1 | AUC |
|-------|----------|-----|-----|
| Baseline (ResNet-50) | 74% | 0.73 | 0.81 |
| Age Synthesis | 78% | 0.77 | 0.85 |
| ViT-FaCoR | 79% | 0.78 | 0.86 |
| ConvNeXt-ViT | 79% | 0.78 | 0.85 |
| **Unified** | **82%** | **0.81** | **0.88** |

## Shared Utilities

### Configuration (`shared/config.py`)

```python
from shared.config import DataConfig, TrainConfig, get_config

data_config, train_config, model_config = get_config("unified")
```

### Data Loading (`shared/dataset.py`)

```python
from shared.dataset import create_dataloaders, KinshipPairDataset

train_loader, val_loader, test_loader = create_dataloaders(
    config=data_config,
    batch_size=32,
    dataset_type="kinface",  # or "fiw"
)
```

### Loss Functions (`shared/losses.py`)

```python
from shared.losses import (
    ContrastiveLoss,
    CosineContrastiveLoss,
    TripletLoss,
    FairContrastiveLoss,
    RelationGuidedContrastiveLoss,
)

loss_fn = CosineContrastiveLoss(temperature=0.07)
```

### Evaluation (`shared/evaluation.py`)

```python
from shared.evaluation import evaluate_model, print_metrics

metrics = evaluate_model(model, test_loader, device)
print_metrics(metrics)
```

## Training Tips

1. **Start simple**: Begin with Model 01 or 02 before trying the unified model
2. **Use pretrained backbones**: Always use `pretrained=True`
3. **Learning rate**: Start with 1e-4, reduce to 5e-5 for large models
4. **Batch size**: 32 for single-backbone, 16 for dual-backbone models
5. **Early stopping**: Use patience=15 to avoid overfitting
6. **Data augmentation**: Enabled by default, helps generalization

## GPU Requirements

| Model | Min VRAM | Recommended |
|-------|----------|-------------|
| Age Synthesis | 6GB | 8GB |
| ViT-FaCoR | 8GB | 12GB |
| ConvNeXt-ViT | 10GB | 16GB |
| Unified | 14GB | 24GB |

## Extending the Models

### Adding a New Backbone

```python
# In model.py
import timm

class MyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
    
    def forward(self, x):
        return self.backbone(x)
```

### Adding a New Loss

```python
# In shared/losses.py
class MyLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, emb1, emb2, labels):
        # Your loss computation
        return loss
```

## Citation

If you use these models in your research, please cite:

```bibtex
@article{kinship_models_2025,
  title={Advanced Architectures for Kinship Verification from Facial Images},
  author={...},
  year={2025}
}
```

## License

MIT License - see individual model READMEs for details.
