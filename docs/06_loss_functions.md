# Loss Functions for Kinship Classification

## 1. Overview

Loss functions are critical for learning discriminative kinship representations. This document provides detailed analysis and implementation of losses used in kinship verification.

```
Loss Function Evolution

Binary Cross-Entropy (baseline)
        │
        ▼
Contrastive Loss (Siamese networks)
        │
        ▼
Triplet Loss (harder mining)
        │
        ▼
Supervised Contrastive Loss (batch-wise)
        │
        ▼
Fair Contrastive Loss (bias mitigation)
        │
        ▼
Relation-Guided Contrastive Loss (attention-weighted)
```

---

## 2. Binary Cross-Entropy Loss

### 2.1 Formulation

$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

where:
- $y_i \in \{0, 1\}$ is the ground truth kinship label
- $\hat{y}_i \in [0, 1]$ is the predicted probability

### 2.2 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEKinshipLoss(nn.Module):
    """
    Binary Cross-Entropy for kinship verification.
    Simple but effective baseline.
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, predictions, labels):
        """
        predictions: (B,) or (B, 1) logits
        labels: (B,) binary kinship labels
        """
        predictions = predictions.view(-1)
        labels = labels.float()
        return self.criterion(predictions, labels)
```

### 2.3 When to Use

- **Pros:** Simple, stable training, works with classification heads
- **Cons:** Doesn't explicitly learn embedding space structure
- **Best for:** Fine-tuning pretrained models, simple baselines

---

## 3. Contrastive Loss (Classic)

### 3.1 Formulation

$$\mathcal{L}_{contrastive} = y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2$$

where:
- $d = \|\phi(I_1) - \phi(I_2)\|_2$ is the L2 distance
- $m$ is the margin hyperparameter (typically 1.0-2.0)
- $y = 1$ for kin pairs, $y = 0$ for non-kin pairs

**Intuition:**
- **Kin pairs:** Minimize distance (pull together)
- **Non-kin pairs:** Push apart until distance > margin

### 3.2 Implementation

```python
class ContrastiveLoss(nn.Module):
    """
    Classic contrastive loss for Siamese networks.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, emb1, emb2, labels):
        """
        emb1, emb2: (B, D) normalized embeddings
        labels: (B,) binary kinship labels (1=kin, 0=non-kin)
        """
        # Euclidean distance
        distances = F.pairwise_distance(emb1, emb2)
        
        # Contrastive loss
        pos_loss = labels * distances.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        loss = (pos_loss + neg_loss).mean()
        return loss


class CosineContrastiveLoss(nn.Module):
    """
    Contrastive loss using cosine similarity.
    More stable for normalized embeddings.
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, emb1, emb2, labels):
        """
        emb1, emb2: (B, D) L2-normalized embeddings
        labels: (B,) binary kinship labels
        """
        # Cosine similarity
        similarity = F.cosine_similarity(emb1, emb2)
        
        # For kin pairs: maximize similarity (minimize 1 - sim)
        # For non-kin pairs: minimize similarity (push below margin)
        pos_loss = labels * (1 - similarity).pow(2)
        neg_loss = (1 - labels) * F.relu(similarity - self.margin).pow(2)
        
        loss = (pos_loss + neg_loss).mean()
        return loss
```

---

## 4. Triplet Loss

### 4.1 Formulation

$$\mathcal{L}_{triplet} = \max(0, d(a, p) - d(a, n) + m)$$

where:
- $a$ is the anchor sample
- $p$ is the positive sample (kin of anchor)
- $n$ is the negative sample (non-kin of anchor)
- $m$ is the margin

### 4.2 Implementation

```python
class TripletLoss(nn.Module):
    """
    Triplet loss with hard negative mining.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        anchor, positive, negative: (B, D) embeddings
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class TripletLossWithMining(nn.Module):
    """
    Triplet loss with online hard negative mining.
    """
    def __init__(self, margin=0.3, mining_type='hard'):
        super().__init__()
        self.margin = margin
        self.mining_type = mining_type
    
    def forward(self, embeddings, labels):
        """
        embeddings: (B, D) all embeddings in batch
        labels: (B,) family IDs or pair labels
        
        Mines hard triplets from batch.
        """
        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # Create positive/negative masks
        labels = labels.unsqueeze(0)
        positive_mask = (labels == labels.T) & ~torch.eye(
            len(labels[0]), device=labels.device, dtype=torch.bool
        )
        negative_mask = labels != labels.T
        
        triplet_loss = 0
        num_triplets = 0
        
        for i in range(len(embeddings)):
            # Find hardest positive
            pos_dists = dist_matrix[i][positive_mask[0, i]]
            if len(pos_dists) == 0:
                continue
            hardest_pos = pos_dists.max()
            
            # Find hardest negative
            neg_dists = dist_matrix[i][negative_mask[0, i]]
            if len(neg_dists) == 0:
                continue
            hardest_neg = neg_dists.min()
            
            loss = F.relu(hardest_pos - hardest_neg + self.margin)
            triplet_loss += loss
            num_triplets += 1
        
        if num_triplets == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        return triplet_loss / num_triplets
```

---

## 5. Supervised Contrastive Loss

### 5.1 Formulation

$$\mathcal{L}_{SupCon} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}$$

where:
- $P(i)$ = set of all positives for anchor $i$
- $A(i)$ = all samples except $i$
- $\tau$ = temperature (typically 0.07-0.1)
- $z$ = L2-normalized embedding

**Key insight:** Considers ALL positives and negatives in the batch, not just pairs/triplets.

### 5.2 Implementation

```python
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for kinship verification.
    Achieves 81.1% on FIW (2023).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        features: (B, D) normalized embeddings
        labels: (B,) kinship pair IDs (same ID = same family/kin)
        
        For kinship: typically structure as (img1_1, img2_1, img1_2, img2_2, ...)
        where pairs (img1_i, img2_i) are kin.
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-similarity
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask
        
        # Compute log-softmax
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        
        # Mean over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss


class KinshipContrastiveLoss(nn.Module):
    """
    Contrastive loss specifically designed for kinship pairs.
    Handles the paired structure of kinship data.
    """
    def __init__(self, temperature=0.08):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, emb1, emb2):
        """
        emb1: (B, D) embeddings of first person in each pair
        emb2: (B, D) embeddings of second person in each pair
        
        All pairs are assumed to be positive (kin pairs).
        Negatives are other samples in the batch.
        """
        batch_size = emb1.size(0)
        
        # Concatenate embeddings
        embeddings = torch.cat([emb1, emb2], dim=0)  # (2B, D)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Mask out self-similarity
        mask = 1.0 - torch.eye(2 * batch_size, device=embeddings.device)
        
        # Positive pairs: (i, i+B) and (i+B, i)
        # Numerator: similarity of positive pairs
        pos_sim = torch.cat([
            torch.diag(sim_matrix, batch_size),      # (i, i+B)
            torch.diag(sim_matrix, -batch_size)      # (i+B, i)
        ])
        
        numerator = torch.exp(pos_sim)
        denominator = (torch.exp(sim_matrix) * mask).sum(dim=1)
        
        loss = -torch.log(numerator / (denominator + 1e-6)).mean()
        
        return loss
```

---

## 6. Fair Contrastive Loss (KFC)

### 6.1 Formulation

The fair contrastive loss adds a debiasing term:

$$\mathcal{L}_{Fair} = \mathcal{L}_{contrastive} + \lambda \cdot \mathcal{L}_{debias}$$

where the debias margin is computed per-race:

$$\text{debias}_i = \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \text{bias}_{i,r}$$

### 6.2 Implementation (from KFC repository)

```python
def fair_contrastive_loss(x1, x2, kinship, race, bias_map, beta=0.08):
    """
    Fair contrastive loss with per-race debiasing.
    
    Args:
        x1, x2: (B, D) embeddings of kin pairs
        kinship: (B,) kinship labels (all 1 for positive pairs)
        race: (B,) race labels (0=African, 1=Asian, 2=Caucasian, 3=Indian)
        bias_map: learned bias adjustment per sample
        beta: temperature parameter
    
    Returns:
        loss: scalar loss value
        race_margins: per-race margin statistics for monitoring
    """
    batch_size = x1.size(0)
    
    # Track per-race statistics
    race_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # AA, A, C, I
    race_margins = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # Concatenate embeddings
    x1x2 = torch.cat([x1, x2], dim=0)
    x2x1 = torch.cat([x2, x1], dim=0)
    
    # Compute similarity matrix
    cosine_mat = torch.cosine_similarity(
        x1x2.unsqueeze(1),
        x1x2.unsqueeze(0),
        dim=2
    ) / beta
    
    # Self-similarity mask
    mask = 1.0 - torch.eye(2 * batch_size, device=x1.device)
    
    # Diagonal similarity (positive pairs)
    diagonal_cosine = torch.cosine_similarity(x1x2, x2x1, dim=1)
    
    # Compute debias margin per sample
    debias_margin = bias_map.sum(dim=1) / bias_map.size(1)
    
    # Accumulate per-race margins
    for i in range(batch_size):
        r = race[i].item()
        margin_val = debias_margin[i] + debias_margin[i + batch_size]
        race_margins[r] += margin_val
        race_counts[r] += 2
    
    # Normalize per-race margins
    for r in range(4):
        if race_counts[r] > 0:
            race_margins[r] = race_margins[r] / race_counts[r]
    
    # Adjusted numerator with debias term
    numerators = torch.exp((diagonal_cosine - debias_margin) / beta)
    
    # Denominator (all other samples)
    denominators = (
        torch.sum(torch.exp(cosine_mat) * mask, dim=1) 
        - torch.exp(diagonal_cosine / beta) 
        + numerators
    )
    
    # Final loss
    loss = -torch.mean(torch.log(numerators) - torch.log(denominators))
    
    return loss, [race_margins[i] for i in range(4)]


class KFCLoss(nn.Module):
    """
    Complete KFC loss with fairness components.
    """
    def __init__(self, temperature=0.08, lambda_race=0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_race = lambda_race
        self.race_criterion = nn.CrossEntropyLoss()
    
    def forward(self, z1, z2, race1, race2, race_labels, bias_map):
        """
        z1, z2: kinship embeddings
        race1, race2: race predictions
        race_labels: ground truth race
        bias_map: learned bias adjustment
        """
        # Fair contrastive loss
        kinship_labels = torch.ones(z1.size(0), device=z1.device)
        contrastive_loss, race_margins = fair_contrastive_loss(
            z1, z2, kinship_labels, race_labels, bias_map, self.temperature
        )
        
        # Race classification loss (with gradient reversal)
        race_loss = (
            self.race_criterion(race1, race_labels) +
            self.race_criterion(race2, race_labels)
        ) / 2
        
        total_loss = contrastive_loss + self.lambda_race * race_loss
        
        return total_loss, contrastive_loss, race_loss, race_margins
```

---

## 7. Relation-Guided Contrastive Loss (FaCoRNet)

### 7.1 Concept

Uses attention maps to weight the contrastive loss, making the temperature adaptive based on face component relationships.

### 7.2 Implementation

```python
def relation_guided_contrastive_loss(emb1, emb2, attention_maps, base_beta=0.08):
    """
    Contrastive loss with attention-based temperature.
    
    Args:
        emb1, emb2: (B, D) embeddings
        attention_maps: tuple of attention maps from cross-attention
        base_beta: base temperature
    
    Returns:
        loss: scalar
    """
    batch_size = emb1.size(0)
    
    # Compute adaptive temperature from attention maps
    beta1, beta2, beta_eye, beta_nose, beta_mouth = attention_maps
    
    # Aggregate attention strengths
    beta_eye = (beta_eye ** 2).sum([1, 2])
    beta_nose = (beta_nose ** 2).sum([1, 2])
    beta_mouth = (beta_mouth ** 2).sum([1, 2])
    
    # Average across components
    beta = (beta_eye + beta_nose + beta_mouth) / 3 / 12  # Normalization factor
    beta = torch.cat([beta, beta])  # For both directions
    
    # Concatenate embeddings
    x1x2 = torch.cat([emb1, emb2], dim=0)
    x2x1 = torch.cat([emb2, emb1], dim=0)
    
    # Similarity matrix with adaptive temperature
    sim_matrix = torch.cosine_similarity(
        x1x2.unsqueeze(1),
        x1x2.unsqueeze(0),
        dim=2
    ) / (beta.unsqueeze(1) + 1e-6)
    
    # Mask
    mask = 1.0 - torch.eye(2 * batch_size, device=emb1.device)
    
    # Positive pair similarity
    numerators = torch.exp(
        torch.cosine_similarity(x1x2, x2x1, dim=1) / (beta + 1e-6)
    )
    
    # All pairs
    denominators = torch.sum(torch.exp(sim_matrix) * mask, dim=1)
    
    loss = -torch.mean(torch.log(numerators / (denominators + 1e-6)))
    
    return loss


class FaCoRLoss(nn.Module):
    """
    Complete FaCoRNet loss.
    """
    def __init__(self, base_temperature=0.08, warmup_epochs=5):
        super().__init__()
        self.base_temp = base_temperature
        self.warmup_epochs = warmup_epochs
    
    def forward(self, emb1, emb2, attention_maps, epoch=None):
        if epoch is not None and epoch < self.warmup_epochs:
            # Use standard contrastive during warmup
            return self._standard_contrastive(emb1, emb2)
        
        return relation_guided_contrastive_loss(
            emb1, emb2, attention_maps, self.base_temp
        )
    
    def _standard_contrastive(self, emb1, emb2):
        """Fallback standard contrastive loss."""
        batch_size = emb1.size(0)
        embeddings = torch.cat([emb1, emb2], dim=0)
        
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.base_temp
        mask = 1.0 - torch.eye(2 * batch_size, device=emb1.device)
        
        pos_sim = torch.cat([
            torch.diag(sim_matrix, batch_size),
            torch.diag(sim_matrix, -batch_size)
        ])
        
        numerator = torch.exp(pos_sim)
        denominator = (torch.exp(sim_matrix) * mask).sum(dim=1)
        
        return -torch.log(numerator / (denominator + 1e-6)).mean()
```

---

## 8. Combined Loss Strategies

### 8.1 Multi-Task Loss

```python
class MultiTaskKinshipLoss(nn.Module):
    """
    Combined loss for multi-task kinship learning.
    """
    def __init__(self, 
                 temperature=0.08,
                 lambda_cls=1.0,
                 lambda_type=0.5,
                 lambda_fairness=0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_cls = lambda_cls
        self.lambda_type = lambda_type
        self.lambda_fairness = lambda_fairness
        
        self.contrastive = KinshipContrastiveLoss(temperature)
        self.classification = nn.BCEWithLogitsLoss()
        self.type_classification = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        """
        outputs: dict with 'embeddings', 'logits', 'type_logits'
        targets: dict with 'kinship', 'relation_type'
        """
        losses = {}
        
        # Contrastive loss on embeddings
        emb1, emb2 = outputs['embeddings']
        losses['contrastive'] = self.contrastive(emb1, emb2)
        
        # Binary classification loss
        if 'logits' in outputs:
            losses['classification'] = self.classification(
                outputs['logits'],
                targets['kinship'].float()
            )
        
        # Relationship type classification
        if 'type_logits' in outputs and 'relation_type' in targets:
            losses['type'] = self.type_classification(
                outputs['type_logits'],
                targets['relation_type']
            )
        
        # Total loss
        total = losses['contrastive']
        if 'classification' in losses:
            total += self.lambda_cls * losses['classification']
        if 'type' in losses:
            total += self.lambda_type * losses['type']
        
        losses['total'] = total
        
        return losses
```

---

## 9. Loss Selection Guide

| Scenario | Recommended Loss | Temperature | Notes |
|----------|------------------|-------------|-------|
| **Baseline** | Contrastive | - | Simple, stable |
| **Better embeddings** | Supervised Contrastive | 0.07-0.1 | Batch-wise learning |
| **Fairness required** | Fair Contrastive (KFC) | 0.08 | With race labels |
| **Attention-based** | Relation-Guided | Adaptive | Requires attention maps |
| **Classification head** | BCE + Contrastive | - | Joint training |
| **Multi-task** | Combined loss | - | Weight balancing needed |

### 9.1 Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| **Temperature (τ)** | 0.05-0.1 | Lower = sharper distribution |
| **Margin (m)** | 0.3-1.0 | For contrastive/triplet |
| **λ_cls** | 0.5-2.0 | Classification weight |
| **λ_fairness** | 0.1-0.5 | Fairness term weight |

---

## 10. Implementation Tips

### 10.1 Numerical Stability

```python
# Always add epsilon to prevent log(0)
loss = -torch.log(numerator / (denominator + 1e-6))

# Clip similarities to prevent overflow
similarity = torch.clamp(similarity / temperature, max=80)

# Use log-sum-exp for stability
log_denominator = torch.logsumexp(similarity, dim=1)
```

### 10.2 Embedding Normalization

```python
# Always L2 normalize for contrastive losses
embeddings = F.normalize(embeddings, p=2, dim=1)
```

### 10.3 Batch Construction

```python
# For contrastive learning, ensure diverse batches
# Include multiple families per batch
# Balance positive/negative ratios
```

---

*This comprehensive guide covers all major loss functions used in kinship classification research.*
