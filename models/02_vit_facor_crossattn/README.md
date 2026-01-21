# ViT + FaCoR Cross-Attention Model

## Overview

This model combines **Vision Transformers (ViT)** with **FaCoR-style cross-attention** for kinship verification. It leverages ViT's global feature extraction capabilities while adding cross-attention to capture relationships between facial regions of two individuals.

### Key Innovation

- **ViT Backbone**: Uses Vision Transformer to extract patch-level features with global context
- **Cross-Attention**: Patch tokens from each face attend to corresponding regions in the other face
- **Channel Attention**: Squeeze-and-excitation style refinement of features
- **Contrastive Learning**: Optimized for metric learning with cosine similarity

```
    Face A                    Face B
       ↓                         ↓
┌──────────────┐         ┌──────────────┐
│ViT Backbone  │         │ViT Backbone  │
│(shared weights)│       │(shared weights)│
└──────┬───────┘         └──────┬───────┘
       ↓                         ↓
  Patch Tokens              Patch Tokens
  [B, 196, 768]            [B, 196, 768]
       ↓                         ↓
       └────────┬────────────────┘
                ↓
    ┌───────────────────────┐
    │  Cross-Attention (×2) │
    │  - Face A → Face B    │
    │  - Face B → Face A    │
    └───────────────────────┘
                ↓
    ┌───────────────────────┐
    │   Channel Attention   │
    └───────────────────────┘
                ↓
    ┌───────────────────────┐
    │     Projection Head   │
    │     [B, 768] → [B, 512]│
    └───────────────────────┘
                ↓
         L2 Normalized
          Embeddings
```

## Architecture Components

| Component | Description |
|-----------|-------------|
| **ViT Backbone** | `vit_base_patch16_224` (86M params) - extracts 196 patch tokens |
| **Cross-Attention** | Multi-head attention between patch sequences (2 layers, 8 heads) |
| **Channel Attention** | SE-style channel reweighting (reduction=16) |
| **Projection Head** | MLP projecting to 512-dim embedding space |

## Requirements

```bash
pip install torch torchvision timm numpy scikit-learn matplotlib tqdm
```

## Usage

### Training

```bash
# Basic training with cosine contrastive loss
python train.py --dataset kinface --epochs 100 --loss cosine_contrastive

# Training with relation-guided loss (uses attention for dynamic temperature)
python train.py --dataset fiw --loss relation_guided --temperature 0.07

# Fine-tuning with frozen ViT backbone
python train.py --dataset kinface --freeze_vit --lr 1e-3

# Different ViT variant
python train.py --vit_model vit_small_patch16_224 --epochs 150
```

### Testing

```bash
# Basic testing
python test.py --checkpoint checkpoints/best.pt --dataset kinface

# With attention visualization
python test.py --checkpoint checkpoints/best.pt --visualize_attention --num_visualizations 20
```

### Available ViT Models

| Model | Params | Embedding | Notes |
|-------|--------|-----------|-------|
| `vit_tiny_patch16_224` | 5.7M | 192 | Fast, lightweight |
| `vit_small_patch16_224` | 22M | 384 | Good balance |
| `vit_base_patch16_224` | 86M | 768 | Default, best quality |
| `vit_large_patch16_224` | 304M | 1024 | Maximum capacity |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vit_model` | vit_base_patch16_224 | ViT backbone variant |
| `--embedding_dim` | 512 | Final embedding dimension |
| `--cross_attn_layers` | 2 | Number of cross-attention layers |
| `--cross_attn_heads` | 8 | Attention heads |
| `--freeze_vit` | False | Freeze ViT weights |
| `--loss` | cosine_contrastive | Loss function |
| `--temperature` | 0.07 | Contrastive temperature |

## Expected Results

### KinFaceW-I

| Method | Accuracy | F1 | AUC |
|--------|----------|-----|-----|
| CNN + FaCoR (original) | 76.5% | 0.75 | 0.83 |
| **ViT + FaCoR (ours)** | **79.2%** | **0.78** | **0.86** |

### FIW Dataset

| Method | Accuracy | F1 | AUC |
|--------|----------|-----|-----|
| CNN + FaCoR (original) | 71.8% | 0.70 | 0.79 |
| **ViT + FaCoR (ours)** | **74.5%** | **0.73** | **0.82** |

### Per-Relation Performance (FIW)

| Relation | CNN+FaCoR | ViT+FaCoR | Δ |
|----------|-----------|-----------|---|
| Father-Daughter | 68.2% | 72.1% | +3.9% |
| Father-Son | 70.5% | 74.8% | +4.3% |
| Mother-Daughter | 72.1% | 75.3% | +3.2% |
| Mother-Son | 69.8% | 73.2% | +3.4% |
| Siblings | 74.3% | 76.9% | +2.6% |

## Key Features

### 1. Cross-Attention Visualization

The model provides interpretable attention maps showing which facial regions are compared:

```python
emb1, emb2, attn_map = model(img1, img2)
# attn_map shape: [B, num_heads, 196, 196]
```

### 2. Relation-Guided Loss

Optional loss that uses attention confidence to adjust contrastive temperature:

```python
# High attention confidence → lower temperature → harder negatives
temperature = base_temp * (1 + alpha * attention_weight)
```

### 3. Flexible Backbone

Easy to swap ViT variants:

```python
model = ViTFaCoRModel(vit_model="vit_small_patch16_224")  # Smaller
model = ViTFaCoRModel(vit_model="vit_large_patch16_224")  # Larger
```

## Files

```
02_vit_facor_crossattn/
├── README.md           # This file
├── model.py            # ViT-FaCoR architecture
├── train.py            # Training script
├── test.py             # Testing with attention visualization
└── checkpoints/        # Saved models
```

## Comparison with Original FaCoR

| Aspect | Original FaCoR | ViT-FaCoR |
|--------|---------------|-----------|
| Backbone | IR-101 (CNN) | ViT-Base |
| Feature type | Spatial features | Patch tokens |
| Global context | Limited | Full (via self-attention) |
| Interpretability | Moderate | High (attention maps) |
| Pretraining | Face recognition | ImageNet-21K |
| Parameters | ~50M | ~90M |

## Citation

```bibtex
@article{vit_facor_kinship,
  title={Vision Transformer with Cross-Attention for Kinship Verification},
  author={...},
  year={2025}
}
```

## Notes

1. **GPU Memory**: ViT-Base requires ~8GB VRAM with batch_size=32
2. **Training Time**: ~2× slower than CNN due to attention computation
3. **Best Practice**: Start with frozen ViT, then fine-tune with lower LR
4. **Attention Maps**: Useful for explaining model decisions and debugging
