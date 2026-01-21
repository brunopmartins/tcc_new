# Methodology Survey: Kinship Classification Approaches

## 1. Historical Evolution

```
Timeline: Kinship Recognition Methods

2010-2014: Handcrafted Features Era
    └── SIFT, LBP, HOG + SVM/metric learning

2014-2017: Early Deep Learning
    └── Siamese CNNs with contrastive/triplet loss

2017-2020: Deep Metric Learning
    └── SphereFace → CosFace → ArcFace transfer learning
    └── DDML, denoising metric learning

2020-2022: Contrastive Learning
    └── Supervised contrastive, multi-task learning
    └── k-tuple metric networks

2022-2024: Attention & Transformers
    └── Vision Transformers, cross-attention
    └── FaCoRNet, hybrid architectures

2024-2025: Emerging Paradigms
    └── Graph Neural Networks, fairness-aware methods
    └── Self-supervised learning (nascent)
```

---

## 2. Handcrafted Feature Methods (2010-2014)

### 2.1 Common Features

| Feature | Description | Kinship Relevance |
|---------|-------------|-------------------|
| **SIFT** | Scale-Invariant Feature Transform | Local keypoint descriptors |
| **LBP** | Local Binary Patterns | Texture patterns around facial landmarks |
| **HOG** | Histogram of Oriented Gradients | Shape and edge information |
| **Gabor** | Gabor wavelets | Multi-scale, multi-orientation texture |
| **Geometric** | Facial landmark distances | Structural similarity |

### 2.2 Typical Pipeline

```
Image → Face Detection → Alignment → Feature Extraction → 
    → Feature Concatenation → Classifier (SVM, LDA) → Kin/Non-kin
```

### 2.3 Limitations

- Fixed, non-learnable representations
- Sensitive to illumination, pose, expression
- Shallow features miss complex genetic patterns
- Accuracy ceiling ~70-75% on benchmarks

---

## 3. Siamese Network Approaches (2014-2020)

### 3.1 Core Architecture

```
           ┌─────────────────┐
    I₁ ───►│   CNN Encoder   │───► φ(I₁) ─┐
           │   (VGGFace,     │            │
           │    FaceNet,     │            ├──► Distance/Similarity ──► Prediction
           │    ResNet)      │            │
           └─────────────────┘            │
                   │                      │
           (shared weights)               │
                   │                      │
           ┌─────────────────┐            │
    I₂ ───►│   CNN Encoder   │───► φ(I₂) ─┘
           └─────────────────┘
```

### 3.2 Key Implementations

#### VGGFace-based Siamese

```python
# From kinship_classifier/model.py pattern
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = InceptionResnetV1(pretrained='vggface2')
        
        emb_len = 512
        self.last = nn.Sequential(
            nn.Linear(4*emb_len, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        
    def forward(self, input1, input2):
        emb1 = self.encoder(input1)
        emb2 = self.encoder(input2)
        
        # Multiple feature combinations
        x1 = torch.pow(emb1, 2) - torch.pow(emb2, 2)  # Squared difference
        x2 = torch.pow(emb1 - emb2, 2)                 # Element-wise squared diff
        x3 = emb1 * emb2                               # Element-wise product
        x4 = emb1 + emb2                               # Sum
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return self.last(x)
```

**Key insight:** Combining multiple feature interaction types (difference, product, sum) captures different aspects of similarity.

### 3.3 Pretrained Backbone Comparison

| Backbone | Source | Kinship Performance | Notes |
|----------|--------|---------------------|-------|
| **ArcFace (IR-101)** | InsightFace | **Best** | Recommended baseline |
| SphereFace | Original paper | Moderate | Outperformed by ArcFace |
| CosFace | Original paper | Moderate | Outperformed by ArcFace |
| VGGFace | Oxford VGG | Moderate | Classic, well-tested |
| FaceNet | Google | Moderate | Good generalization |

**Critical finding (Shadrikov, 2020):** ArcFace pretrained models substantially outperform alternatives for kinship transfer learning.

---

## 4. Deep Metric Learning (2017-2021)

### 4.1 Discriminative Deep Metric Learning (DDML)

**Core idea:** Learn hierarchical nonlinear transformations to project face pairs into a latent space optimized for kinship verification.

```
                    ┌─────────────────┐
    φ(I₁), φ(I₂) ──►│ Nonlinear       │──► z₁, z₂ ──► d(z₁, z₂) ──► Loss
                    │ Transformation  │
                    │ (MLP layers)    │
                    └─────────────────┘
```

**Loss objective:**
- Minimize: $d(z_i, z_j)$ for kin pairs $(i, j)$
- Maximize: $d(z_i, z_k)$ for non-kin pairs $(i, k)$

### 4.2 k-Tuple Metric Networks (CVPR 2021)

**Innovation:** Move beyond duplet (pairwise) and triplet losses to exploit multiple negative samples.

**Traditional Triplet:**
```
Anchor (a) ──┬──► Positive (p)  [kin]
             └──► Negative (n)  [non-kin]
             
Loss = max(0, d(a,p) - d(a,n) + margin)
```

**k-Tuple (AW k-TMN):**
```
Anchor (a) ──┬──► Positive (p)
             ├──► Negative₁ (n₁)
             ├──► Negative₂ (n₂)
             ├──► ...
             └──► Negativeₖ (nₖ)
             
Loss = Σᵢ wᵢ · max(0, d(a,p) - d(a,nᵢ) + margin)
```

**Advantage:** Richer gradient signal from multiple negative comparisons.

### 4.3 Marginalized Denoising Metric Learning

**Addresses:** Noise robustness in real-world kinship data.

**Approach:** Add noise during training and learn to be robust to variations:
- Dropout on features
- Data augmentation
- Margin adjustment based on confidence

---

## 5. Contrastive Learning (2020-Present)

### 5.1 Supervised Contrastive Loss

**Key paper:** arXiv:2302.09556 (Feb 2023) - 81.1% accuracy on FIW

**Formulation:**
$$\mathcal{L} = -\sum_{i} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_p)/\tau)}{\sum_{a \neq i} \exp(\text{sim}(z_i, z_a)/\tau)}$$

**Intuition:** Pull together embeddings of related individuals, push apart unrelated.

### 5.2 Implementation Pattern

```python
def supervised_contrastive_loss(embeddings, labels, temperature=0.08):
    """
    embeddings: (2*batch_size, dim) - concatenated [z1; z2]
    labels: (batch_size,) - kinship labels
    """
    batch_size = embeddings.size(0) // 2
    
    # Cosine similarity matrix
    sim_matrix = F.cosine_similarity(
        embeddings.unsqueeze(1), 
        embeddings.unsqueeze(0), 
        dim=2
    ) / temperature
    
    # Mask for positive pairs (kin pairs and their symmetric)
    positive_mask = create_positive_mask(labels)
    
    # Compute log-softmax for each sample
    log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
    
    # Mean over positive pairs
    loss = -(positive_mask * log_prob).sum() / positive_mask.sum()
    
    return loss
```

### 5.3 Fair Contrastive Loss (KFC, BMVC 2023)

**Innovation:** Add fairness term to reduce performance disparity across racial groups.

**Architecture:**
```
                    ┌─────────────────┐
    Image ─────────►│    Encoder      │────┬──► Kinship Head ──► Contrastive Loss
                    │   (ResNet101)   │    │
                    └─────────────────┘    │
                                           └──► Race Classifier ──► Cross-Entropy
                                                      ↑
                                              Gradient Reversal
```

**Loss function:**
```python
# From KFC/losses.py (simplified)
def fair_contrastive_loss(x1, x2, kinship, race, bias_map, beta=0.08):
    # Standard contrastive computation
    x1x2 = torch.cat([x1, x2], dim=0)
    x2x1 = torch.cat([x2, x1], dim=0)
    
    cosine_mat = torch.cosine_similarity(
        x1x2.unsqueeze(1), x1x2.unsqueeze(0), dim=2
    ) / beta
    
    # Compute per-race debias margin
    debias_margin = torch.sum(bias_map, axis=1) / len(bias_map)
    
    # Adjusted numerator with debias term
    diagonal_cosine = torch.cosine_similarity(x1x2, x2x1, dim=1)
    numerators = torch.exp((diagonal_cosine - debias_margin) / beta)
    
    # Standard denominator
    mask = 1.0 - torch.eye(2 * x1.size(0))
    denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)
    
    return -torch.mean(torch.log(numerators / denominators))
```

**Key components:**
1. **Debias margin:** Per-race adjustment to equalize difficulty
2. **Gradient reversal:** Force encoder to learn race-invariant features
3. **Multi-task:** Joint kinship + race classification

---

## 6. Multi-Task Learning

### 6.1 Correlation-Based Multi-Task Learning (CCMTL)

**Motivation:** Different kinship types (FS, FD, MS, MD) share common patterns but also have unique characteristics.

**Architecture:**
```
                    ┌─────────────────┐
    Image Pair ────►│  Shared Encoder │────┬──► FS Head
                    │                 │    ├──► FD Head
                    │                 │    ├──► MS Head
                    │                 │    └──► MD Head
                    └─────────────────┘
```

**Benefits:**
- Shared features capture common kinship patterns
- Task-specific heads learn relationship-specific cues
- Improved generalization through information sharing

### 6.2 Multi-Scale Feature Concatenation

```python
def multi_scale_features(encoder, image):
    """Extract features from multiple layers."""
    features = []
    x = image
    
    for i, layer in enumerate(encoder.layers):
        x = layer(x)
        if i in [3, 4, 5]:  # Selected layers
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            features.append(pooled)
    
    return torch.cat(features, dim=1)
```

---

## 7. Attention Mechanisms (2022-Present)

### 7.1 Cross-Attention for Face Components (FaCoRNet)

**Key insight:** Kinship manifests in specific facial regions (eyes, nose, mouth). Cross-attention learns to align and compare corresponding regions.

**Architecture:**
```
    I₁ ──► Encoder ──► F₁ ─┐
                           ├──► Cross-Attention ──► Attention Maps ──► Weighted Features
    I₂ ──► Encoder ──► F₂ ─┘                              │
                                                          ▼
                                               Relation-Guided Contrastive Loss
```

**Implementation pattern (from FaCoR/models.py):**
```python
class CrossAttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
    def forward(self, f1, f2):
        # Compute attention between face features
        Q = self.query(f1)  # Query from first face
        K = self.key(f2)    # Key from second face
        V = self.value(f2)  # Value from second face
        
        # Attention weights
        attention = torch.softmax(Q @ K.transpose(-2, -1) / sqrt(d_k), dim=-1)
        
        # Attended features
        output = attention @ V
        
        return output, attention
```

### 7.2 Channel Attention Layer

```python
# From FaCoR pattern
class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

### 7.3 Relation-Guided Contrastive Loss

**Innovation:** Use attention maps to weight the contrastive loss.

```python
def relation_guided_contrastive_loss(z1, z2, attention_map, beta):
    """
    attention_map: learned importance weights for facial regions
    """
    # Attention-weighted temperature
    beta_weighted = (attention_map**2).sum([1, 2]) / scale_factor
    
    # Standard contrastive with adaptive temperature
    similarity = F.cosine_similarity(z1, z2) / beta_weighted
    
    return contrastive_loss(similarity)
```

---

## 8. Vision Transformers (2023-Present)

### 8.1 Pure ViT Approach

**Result:** 92% accuracy on FIW (2025)

**Architecture:**
```
    Image ──► Patch Embedding ──► Transformer Blocks ──► [CLS] Token ──► Kinship Head
                                      │
                              Self-Attention
```

**Advantages:**
- Global context from first layer
- Learns long-range dependencies
- Pretrained models readily available (Hugging Face)

### 8.2 Hybrid CNN-ViT Architectures

**ConvNeXt + EfficientNet + ViT Fusion (2025):**

```
                    ┌──────────────────┐
    Image ─────────►│   ConvNeXt-Base  │──► Local texture features
                    └──────────────────┘
                    ┌──────────────────┐
    Image ─────────►│  EfficientNet-B0 │──► Efficient features
                    └──────────────────┘
                    ┌──────────────────┐
    Image ─────────►│       ViT        │──► Global context features
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Feature Fusion  │──► Kinship Prediction
                    │   (Attention)    │
                    └──────────────────┘
```

**Key finding (Kaggle 2022):** ViT models exhibit low correlation with CNN predictions, making them excellent ensemble partners.

---

## 9. Graph Neural Networks (2024-2025)

### 9.1 Reasoning Graph Networks

**Concept:** Model facial features as graph nodes, use graph reasoning to infer kinship.

```
Face 1:                    Face 2:
   [Eye]                      [Eye]
    / \                        / \
[Nose]─[Mouth]            [Nose]─[Mouth]
                    
        ↓ Graph Matching ↓
        
    Kinship Similarity Score
```

### 9.2 Forest Neural Network (2025)

**Innovation:** First dedicated GNN for kinship verification.

**Approach:**
1. Treat face components as disjoint nodes
2. Information exchange through GNN message passing
3. Aggregate node features for kinship prediction

**Advantages:**
- Explicitly models compositional nature of facial features
- Allows interpretable feature importance
- Novel paradigm beyond CNN/Transformer

---

## 10. Age-Invariant Methods

### 10.1 Age-Invariant Adversarial Feature (AIAF)

**Problem:** Large age gaps reduce visual similarity between kin.

**Solution:** Disentangle identity features from age features using adversarial training.

```
                    ┌─────────────────┐
    Image ─────────►│    Encoder      │────┬──► Identity Features ──► Kinship
                    │                 │    │
                    └─────────────────┘    └──► Age Features ──► Age Classifier
                                                      ↑
                                              Gradient Reversal
                                      (decorrelate identity and age)
```

### 10.2 Face Age Transformation

**Approach:** Generate synthetic images at different ages to augment training.

```
Parent (age 45) ──► Age Transform ──► Parent (age 25)
Child (age 20)                        Child (age 20)
                                           │
                                    Same-age comparison
                                    (higher similarity)
```

---

## 11. Method Comparison Summary

| Method Category | Representative | Best Accuracy | Key Advantage | Limitation |
|-----------------|----------------|---------------|---------------|------------|
| **Handcrafted** | LBP + SVM | ~70-75% | Interpretable | Accuracy ceiling |
| **Siamese CNN** | VGGFace Siamese | ~78-82% | Simple, proven | Limited expressiveness |
| **ArcFace Transfer** | ArcFace baseline | ~80-85% | Strong baseline | Not kinship-optimized |
| **Contrastive** | Supervised Contrastive | 81.1% | Better representations | Complex training |
| **Fair Contrastive** | KFC | Competitive + fair | Reduces bias | Requires race labels |
| **Cross-Attention** | FaCoRNet | +4.6% SOTA | Interpretable attention | Complex architecture |
| **Vision Transformer** | ViT | **92%** | Global context | Compute intensive |
| **GNN** | Forest NN | Emerging | Compositional | Very new |

---

## 12. Recommended Reading Order

1. **Start:** Siamese networks (understand baseline)
2. **Foundation:** ArcFace transfer learning (current best practice)
3. **Current SOTA:** Supervised contrastive learning
4. **Cutting edge:** FaCoRNet (attention), ViT approaches
5. **Emerging:** GNN methods, fairness-aware training

---

*This survey provides the methodological foundation for understanding and advancing kinship classification research.*
