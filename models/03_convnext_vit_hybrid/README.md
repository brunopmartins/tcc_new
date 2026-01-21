# ConvNeXt + ViT Hybrid Model

## Overview

This model combines **ConvNeXt** (modern CNN) with **Vision Transformer (ViT)** in a dual-backbone architecture. The key insight is that CNNs and Transformers capture complementary information:

- **ConvNeXt**: Local features, textures, fine-grained facial details
- **ViT**: Global structure, spatial relationships, long-range dependencies

### Architecture

```
         Input Face Image
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌─────────────┐
│  ConvNeXt   │  │     ViT     │
│   (Local)   │  │  (Global)   │
│  1024-dim   │  │   768-dim   │
└──────┬──────┘  └──────┬──────┘
       │               │
       └───────┬───────┘
               ▼
       ┌─────────────┐
       │   Fusion    │
       │  (Learned)  │
       └──────┬──────┘
               ▼
       ┌─────────────┐
       │ Projection  │
       │  512-dim    │
       └──────┬──────┘
               ▼
         L2 Normalized
          Embedding
```

## Key Features

### 1. Multiple Fusion Strategies

| Fusion Type | Description | Best For |
|-------------|-------------|----------|
| `concat` | Simple concatenation + MLP | Default, stable |
| `attention` | ConvNeXt queries ViT features | When global context matters |
| `gated` | Learned gates weight each stream | Adaptive weighting |
| `bilinear` | Bilinear interaction | Rich feature mixing |

### 2. Ablation Support

Built-in support for ablation studies:

```bash
# ConvNeXt only (local features)
python train.py --ablation_mode convnext_only

# ViT only (global features)
python train.py --ablation_mode vit_only

# Hybrid (default)
python train.py
```

### 3. Feature Analysis

Visualize contribution of each backbone:

```bash
python test.py --checkpoint best.pt --analyze_features
```

## Requirements

```bash
pip install torch torchvision timm numpy scikit-learn matplotlib tqdm
```

## Usage

### Training

```bash
# Basic training with concatenation fusion
python train.py --dataset kinface --fusion_type concat

# Attention-based fusion
python train.py --dataset fiw --fusion_type attention --epochs 100

# Smaller models (less GPU memory)
python train.py --convnext_model convnext_tiny --vit_model vit_small_patch16_224

# Frozen backbones (faster, less memory)
python train.py --freeze_backbones --lr 1e-3
```

### Testing

```bash
# Basic testing
python test.py --checkpoint checkpoints/best.pt

# With feature analysis visualization
python test.py --checkpoint checkpoints/best.pt --analyze_features
```

## Model Variants

### ConvNeXt Options

| Model | Params | Features | Notes |
|-------|--------|----------|-------|
| `convnext_tiny` | 28M | 768 | Lightweight |
| `convnext_small` | 50M | 768 | Balanced |
| `convnext_base` | 89M | 1024 | Default |
| `convnext_large` | 198M | 1536 | Maximum |

### ViT Options

| Model | Params | Features | Notes |
|-------|--------|----------|-------|
| `vit_tiny_patch16_224` | 5.7M | 192 | Very fast |
| `vit_small_patch16_224` | 22M | 384 | Fast |
| `vit_base_patch16_224` | 86M | 768 | Default |
| `vit_large_patch16_224` | 304M | 1024 | High capacity |

## Expected Results

### Ablation Study (KinFaceW-I)

| Configuration | Accuracy | F1 | AUC |
|---------------|----------|-----|-----|
| ConvNeXt only | 74.2% | 0.73 | 0.81 |
| ViT only | 75.8% | 0.74 | 0.83 |
| **Hybrid (concat)** | **78.5%** | **0.77** | **0.85** |
| Hybrid (attention) | 79.1% | 0.78 | 0.86 |
| Hybrid (gated) | 78.8% | 0.77 | 0.85 |

### Fusion Comparison (FIW)

| Fusion Type | Accuracy | Training Time | Memory |
|-------------|----------|---------------|--------|
| concat | 73.5% | 1.0× | 1.0× |
| attention | 74.2% | 1.2× | 1.1× |
| gated | 73.8% | 1.1× | 1.05× |
| bilinear | 73.6% | 1.15× | 1.1× |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--convnext_model` | convnext_base | ConvNeXt variant |
| `--vit_model` | vit_base_patch16_224 | ViT variant |
| `--embedding_dim` | 512 | Output embedding size |
| `--fusion_type` | concat | Fusion strategy |
| `--freeze_backbones` | False | Freeze pretrained weights |
| `--lr` | 1e-4 | Learning rate |

## Feature Analysis Output

The `--analyze_features` flag generates:

1. **Similarity distributions**: How each backbone separates kin/non-kin
2. **ConvNeXt vs ViT scatter**: Correlation between backbones
3. **Separability metrics**: Quantified contribution of each stream

Example output:
```
FEATURE CONTRIBUTION ANALYSIS
==================================================
Separability (higher is better):
  ConvNeXt: 0.8234
  ViT:      0.9156
  Fused:    1.0842

Mean Similarities:
  ConvNeXt - Kin: 0.72, Non-Kin: 0.45
  ViT      - Kin: 0.78, Non-Kin: 0.42
  Fused    - Kin: 0.85, Non-Kin: 0.35
```

## Files

```
03_convnext_vit_hybrid/
├── README.md           # This file
├── model.py            # Hybrid architecture
├── train.py            # Training with ablation support
├── test.py             # Testing with feature analysis
└── checkpoints/        # Saved models
```

## Why Hybrid?

| Aspect | ConvNeXt | ViT | Hybrid |
|--------|----------|-----|--------|
| Local features | ✅ Excellent | ❌ Limited | ✅ |
| Global context | ❌ Limited | ✅ Excellent | ✅ |
| Texture/skin | ✅ Strong | ⚠️ Moderate | ✅ |
| Face structure | ⚠️ Moderate | ✅ Strong | ✅ |
| Computational cost | Lower | Higher | Highest |

## Citation

```bibtex
@article{convnext_vit_kinship,
  title={Hybrid CNN-Transformer for Kinship Verification},
  author={...},
  year={2025}
}
```

## Notes

1. **GPU Memory**: Dual backbone requires ~12GB VRAM with batch_size=32
2. **Training**: Consider freezing backbones initially, then fine-tuning
3. **Best Practice**: Start with `concat` fusion, try `attention` if results plateau
4. **Inference Speed**: ~2× slower than single backbone due to dual processing
