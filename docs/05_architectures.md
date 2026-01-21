# Neural Network Architectures for Kinship Classification

## 1. Architecture Overview

```
Architecture Taxonomy for Kinship Recognition

├── Siamese Architectures
│   ├── Weight-sharing encoders
│   ├── Feature comparison heads
│   └── Distance-based prediction
│
├── Backbone Networks (Encoders)
│   ├── VGGFace / VGGFace2
│   ├── FaceNet (Inception-ResNet)
│   ├── ArcFace (IR-ResNet)
│   ├── ResNet variants
│   ├── EfficientNet
│   └── Vision Transformer (ViT)
│
├── Attention Mechanisms
│   ├── Self-attention (Transformer)
│   ├── Cross-attention (FaCoRNet)
│   ├── Channel attention (SE, CBAM)
│   └── Spatial attention
│
├── Multi-Task Heads
│   ├── Kinship verification head
│   ├── Relationship classification head
│   ├── Auxiliary heads (age, gender, race)
│   └── Gradient reversal layers
│
└── Graph-Based
    ├── Face component graphs
    └── Message passing networks
```

---

## 2. Siamese Network Architecture

### 2.1 Basic Siamese Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     SIAMESE NETWORK                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Image 1 ───┐                                               │
│              │     ┌─────────────┐                           │
│              ├────►│   Encoder   │────► Embedding 1 ─┐       │
│              │     │     φ       │                   │       │
│   Image 2 ───┤     └─────────────┘                   │       │
│              │          ║                            │       │
│              │    (shared weights)                   ├──► Comparison ──► Output
│              │          ║                            │       │
│              │     ┌─────────────┐                   │       │
│              └────►│   Encoder   │────► Embedding 2 ─┘       │
│                    │     φ       │                           │
│                    └─────────────┘                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseKinshipNetwork(nn.Module):
    """
    Basic Siamese architecture for kinship verification.
    """
    def __init__(self, encoder, embedding_dim=512):
        super().__init__()
        self.encoder = encoder  # Pretrained backbone
        
        # Comparison head
        self.comparison_head = nn.Sequential(
            nn.Linear(embedding_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, img1, img2):
        # Extract embeddings
        emb1 = self.encoder(img1)  # (B, D)
        emb2 = self.encoder(img2)  # (B, D)
        
        # L2 normalize
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        
        # Feature interactions
        diff = emb1 - emb2                    # Difference
        diff_sq = (emb1 - emb2) ** 2          # Squared difference
        product = emb1 * emb2                  # Element-wise product
        concat = torch.cat([emb1, emb2], dim=1)  # Concatenation
        
        # Combined features
        combined = torch.cat([diff, diff_sq, product, concat[:, :512]], dim=1)
        
        # Prediction
        output = self.comparison_head(combined)
        return output
    
    def get_embeddings(self, img1, img2):
        """Return normalized embeddings for contrastive loss."""
        emb1 = F.normalize(self.encoder(img1), p=2, dim=1)
        emb2 = F.normalize(self.encoder(img2), p=2, dim=1)
        return emb1, emb2
```

---

## 3. Backbone Encoders

### 3.1 ArcFace (IR-ResNet) - Recommended

**Source:** InsightFace (https://github.com/deepinsight/insightface)

```python
from insightface.recognition.arcface_torch.backbones import get_model

class ArcFaceEncoder(nn.Module):
    def __init__(self, model_name='ir_101'):
        super().__init__()
        self.backbone = get_model(model_name, dropout=0.0, fp16=False)
        # Load pretrained weights
        self.backbone.load_state_dict(torch.load('arcface_ir101.pth'))
        
    def forward(self, x):
        return self.backbone(x)  # Returns 512-dim embedding
```

**Architecture Details:**
- **IR-50:** 50 layers, ~44M parameters
- **IR-101:** 101 layers, ~65M parameters
- **IR-152:** 152 layers, ~85M parameters
- **Output:** 512-dimensional embedding

### 3.2 FaceNet (Inception-ResNet-V1)

**Source:** facenet-pytorch (https://github.com/timesler/facenet-pytorch)

```python
from facenet_pytorch import InceptionResnetV1

class FaceNetEncoder(nn.Module):
    def __init__(self, pretrained='vggface2'):
        super().__init__()
        self.backbone = InceptionResnetV1(
            pretrained=pretrained,
            classify=False
        )
        
    def forward(self, x):
        return self.backbone(x)  # Returns 512-dim embedding
```

### 3.3 Vision Transformer (ViT)

**Source:** Hugging Face Transformers

```python
from transformers import ViTModel, ViTImageProcessor

class ViTEncoder(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224'):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
    def forward(self, x):
        outputs = self.vit(x)
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding  # Returns 768-dim embedding
```

### 3.4 Backbone Comparison

| Backbone | Params | Embedding Dim | FLOPs | Kinship Accuracy |
|----------|--------|---------------|-------|------------------|
| **ArcFace IR-101** | 65M | 512 | 12.1G | **Best baseline** |
| FaceNet (IRMV1) | 23M | 512 | 5.0G | Good |
| VGGFace | 138M | 4096→512 | 15.5G | Moderate |
| ViT-Base | 86M | 768 | 17.6G | **92% (2025)** |
| EfficientNet-B0 | 5.3M | 1280 | 0.4G | Efficient |
| ConvNeXt-Base | 89M | 1024 | 15.4G | Hybrid partner |

---

## 4. Cross-Attention Architecture (FaCoRNet)

### 4.1 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FaCoRNet ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Image 1 ───► ArcFace Encoder ───► Feature Map F₁ ─┐                 │
│                                                     │                 │
│                                                     ├──► Cross-Attention
│                                                     │       Module    │
│  Image 2 ───► ArcFace Encoder ───► Feature Map F₂ ─┘        │        │
│                    ║                                         │        │
│              (shared weights)                                │        │
│                                                              ▼        │
│                                                   ┌──────────────┐   │
│                                                   │ Attention    │   │
│                                                   │    Maps      │   │
│                                                   └──────┬───────┘   │
│                                                          │           │
│                                                          ▼           │
│                                        ┌─────────────────────────┐   │
│                                        │ Channel Attention (CCA) │   │
│                                        └───────────┬─────────────┘   │
│                                                    │                 │
│                                                    ▼                 │
│                                        ┌─────────────────────────┐   │
│                                        │  Relation-Guided        │   │
│                                        │  Contrastive Loss       │   │
│                                        └─────────────────────────┘   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Cross-Attention Module Implementation

```python
class CrossAttentionModule(nn.Module):
    """
    Cross-attention between two face feature maps.
    Learns to align and compare corresponding facial regions.
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, f1, f2):
        """
        f1, f2: Feature maps (B, C, H, W)
        Returns: Enhanced features and attention map
        """
        B, C, H, W = f1.shape
        
        # Project to Q, K, V
        Q = self.query(f1).view(B, self.num_heads, self.head_dim, H*W)
        K = self.key(f2).view(B, self.num_heads, self.head_dim, H*W)
        V = self.value(f2).view(B, self.num_heads, self.head_dim, H*W)
        
        # Compute attention
        attention = torch.softmax(
            Q.transpose(-2, -1) @ K / (self.head_dim ** 0.5),
            dim=-1
        )
        
        # Apply attention
        out = (attention @ V.transpose(-2, -1)).transpose(-2, -1)
        out = out.view(B, C, H, W)
        out = self.out_proj(out)
        
        # Residual connection
        out = f1 + out
        
        return out, attention


class FSFNet2(nn.Module):
    """
    Face Similarity Feature Network with cross-attention.
    From FaCoR implementation.
    """
    def __init__(self, channels, use_channel_attention=True):
        super().__init__()
        self.cross_attn = CrossAttentionModule(channels)
        self.use_ca = use_channel_attention
        if use_channel_attention:
            self.channel_attn = CALayer(channels)
        
    def forward(self, f1, f2):
        # Cross-attention in both directions
        f1_enhanced, attn_12 = self.cross_attn(f1, f2)
        f2_enhanced, attn_21 = self.cross_attn(f2, f1)
        
        if self.use_ca:
            f1_enhanced = self.channel_attn(f1_enhanced)
            f2_enhanced = self.channel_attn(f2_enhanced)
        
        return f1_enhanced, f2_enhanced, (attn_12, attn_21)
```

### 4.3 Channel Attention Layer

```python
class CALayer(nn.Module):
    """
    Channel Attention Layer.
    Adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class CALayer2(nn.Module):
    """
    Channel Attention for feature fusion.
    Learns weights to combine multiple feature sources.
    """
    def __init__(self, channels, num_sources=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, num_sources * (channels // num_sources)),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """x: (B, C, 1, 1) concatenated features"""
        B, C, _, _ = x.size()
        weights = self.fc(x.view(B, C))
        return weights
```

---

## 5. Multi-Task Architecture (KFC)

### 5.1 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         KFC ARCHITECTURE                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Image Pair ───► Shared ResNet-101 Encoder ───► Feature F             │
│                                                      │                │
│                              ┌───────────────────────┼────────────┐  │
│                              │                       │            │  │
│                              ▼                       ▼            ▼  │
│                    ┌──────────────┐      ┌──────────────┐  ┌─────────┐
│                    │  Attention   │      │   Kinship    │  │  Race   │
│                    │   Module     │      │     Head     │  │  Head   │
│                    └──────┬───────┘      └──────┬───────┘  └────┬────┘
│                           │                     │               │    │
│                           ▼                     ▼               │    │
│                    ┌──────────────┐      ┌──────────────┐       │    │
│                    │   Attended   │      │  Fair        │       │    │
│                    │   Features   │──────│ Contrastive  │◄──────┘    │
│                    └──────────────┘      │    Loss      │   Gradient │
│                                          └──────────────┘   Reversal │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Implementation

```python
class KFCModel(nn.Module):
    """
    KFC: Kinship Verification with Fair Contrastive Loss.
    Multi-task architecture with attention and fairness.
    """
    def __init__(self, num_races=4):
        super().__init__()
        
        # Shared backbone
        self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classifier
        
        # Attention module
        self.attention = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.Sigmoid()
        )
        
        # Kinship projection head
        self.kinship_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Race classification head (for gradient reversal)
        self.race_head = nn.Sequential(
            GradientReversalLayer(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_races)
        )
        
    def forward(self, img1, img2):
        # Extract features
        f1 = self.backbone(img1)
        f2 = self.backbone(img2)
        
        # Apply attention
        att1 = self.attention(f1)
        att2 = self.attention(f2)
        f1_att = f1 * att1
        f2_att = f2 * att2
        
        # Kinship embeddings
        z1 = self.kinship_head(f1_att)
        z2 = self.kinship_head(f2_att)
        
        # Race predictions (for fairness training)
        race1 = self.race_head(f1)
        race2 = self.race_head(f2)
        
        return z1, z2, race1, race2, att1, att2


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    Forward: identity
    Backward: negate gradients
    """
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
```

---

## 6. Vision Transformer for Kinship

### 6.1 ViT-Based Architecture

```python
from transformers import ViTModel

class ViTKinshipModel(nn.Module):
    """
    Vision Transformer for kinship verification.
    Achieves 92% accuracy on FIW (2025).
    """
    def __init__(self, vit_model='google/vit-base-patch16-224'):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model)
        
        # Kinship comparison head
        self.comparison_head = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, img1, img2):
        # Get CLS token embeddings
        out1 = self.vit(img1).last_hidden_state[:, 0, :]
        out2 = self.vit(img2).last_hidden_state[:, 0, :]
        
        # Concatenate and compare
        combined = torch.cat([out1, out2], dim=1)
        output = self.comparison_head(combined)
        
        return output
    
    def get_embeddings(self, img):
        return self.vit(img).last_hidden_state[:, 0, :]
```

### 6.2 Hybrid CNN-ViT Architecture

```python
class HybridKinshipModel(nn.Module):
    """
    Hybrid architecture combining CNN and ViT features.
    Leverages complementary strengths of both.
    """
    def __init__(self):
        super().__init__()
        
        # CNN branch (local features)
        self.convnext = timm.create_model(
            'convnext_base', 
            pretrained=True, 
            num_classes=0
        )
        
        # ViT branch (global features)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Feature fusion
        cnn_dim = 1024  # ConvNeXt-Base
        vit_dim = 768   # ViT-Base
        
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        
        # Comparison head
        self.comparison = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        
    def extract_features(self, img):
        # CNN features
        cnn_feat = self.convnext(img)
        
        # ViT features
        vit_feat = self.vit(img).last_hidden_state[:, 0, :]
        
        # Fused features
        fused = self.fusion(torch.cat([cnn_feat, vit_feat], dim=1))
        return F.normalize(fused, p=2, dim=1)
    
    def forward(self, img1, img2):
        f1 = self.extract_features(img1)
        f2 = self.extract_features(img2)
        
        # Multiple interactions
        combined = torch.cat([
            f1, f2,
            f1 - f2,
            f1 * f2
        ], dim=1)
        
        return self.comparison(combined)
```

---

## 7. Architecture Selection Guide

### 7.1 By Use Case

| Use Case | Recommended Architecture | Rationale |
|----------|--------------------------|-----------|
| **Baseline** | Siamese + ArcFace | Proven, well-documented |
| **SOTA Performance** | FaCoRNet or ViT | Best accuracy |
| **Fairness Required** | KFC architecture | Built-in debiasing |
| **Resource Constrained** | Siamese + EfficientNet | Low compute |
| **Research/Interpretability** | Cross-attention models | Attention visualization |
| **Ensemble** | CNN + ViT hybrid | Complementary features |

### 7.2 Compute Requirements

| Architecture | GPU Memory (batch=32) | Training Time (FIW) | Inference Speed |
|--------------|----------------------|---------------------|-----------------|
| Siamese + FaceNet | ~4 GB | ~2 hours | Fast |
| Siamese + ArcFace IR-101 | ~8 GB | ~4 hours | Moderate |
| FaCoRNet | ~12 GB | ~8 hours | Moderate |
| ViT-Base | ~16 GB | ~12 hours | Slow |
| Hybrid CNN-ViT | ~20 GB | ~16 hours | Slow |

---

## 8. Implementation Checklist

### 8.1 Essential Components

- [ ] Pretrained backbone (ArcFace or ViT)
- [ ] L2 normalization of embeddings
- [ ] Feature interaction layer (diff, product, concat)
- [ ] Comparison/classification head
- [ ] Appropriate loss function (contrastive/BCE)

### 8.2 Recommended Enhancements

- [ ] Attention mechanism (channel or spatial)
- [ ] Multi-task heads (if applicable)
- [ ] Data augmentation (horizontal flip, crop)
- [ ] Learning rate scheduling
- [ ] Mixed precision training (fp16)

### 8.3 Advanced Features

- [ ] Cross-attention for face components
- [ ] Gradient reversal for fairness
- [ ] Relation-guided contrastive loss
- [ ] Ensemble of CNN and ViT

---

*This architecture guide provides the technical foundation for implementing kinship classification systems.*
