# Problem Formulation: Kinship Classification from Facial Images

## 1. Formal Problem Definition

### 1.1 Kinship Verification (Binary Classification)

**Input:** A pair of facial images $(I_1, I_2)$

**Output:** Binary label $y \in \{0, 1\}$ where:
- $y = 1$: The individuals are biologically related (kin)
- $y = 0$: The individuals are not biologically related (non-kin)

**Formal Definition:**
$$f: \mathcal{I} \times \mathcal{I} \rightarrow \{0, 1\}$$

where $\mathcal{I}$ is the space of facial images and $f$ is the learned kinship verification function.

### 1.2 Kinship Relationship Classification (Multi-class)

**Input:** A pair of facial images $(I_1, I_2)$

**Output:** Relationship type $r \in \mathcal{R}$ where:
$$\mathcal{R} = \{\text{FS}, \text{FD}, \text{MS}, \text{MD}, \text{BB}, \text{SS}, \text{BS}, \text{GFGD}, \text{GFGS}, \text{GMGD}, \text{GMGS}, \text{None}\}$$

| Code | Relationship | Description |
|------|--------------|-------------|
| FS | Father-Son | Male parent to male child |
| FD | Father-Daughter | Male parent to female child |
| MS | Mother-Son | Female parent to male child |
| MD | Mother-Daughter | Female parent to female child |
| BB | Brother-Brother | Male sibling pair |
| SS | Sister-Sister | Female sibling pair |
| BS | Brother-Sister | Mixed-gender sibling pair |
| GFGD | Grandfather-Granddaughter | Male grandparent to female grandchild |
| GFGS | Grandfather-Grandson | Male grandparent to male grandchild |
| GMGD | Grandmother-Granddaughter | Female grandparent to female grandchild |
| GMGS | Grandmother-Grandson | Female grandparent to male grandchild |
| None | No Relation | Unrelated individuals |

### 1.3 Tri-Subject Kinship Verification

**Input:** Three facial images $(I_F, I_M, I_C)$ representing Father, Mother, and Child

**Output:** Binary label $y \in \{0, 1\}$ indicating whether $I_C$ is the biological child of parents $(I_F, I_M)$

**Formal Definition:**
$$g: \mathcal{I} \times \mathcal{I} \times \mathcal{I} \rightarrow \{0, 1\}$$

### 1.4 Family Search and Retrieval

**Input:** 
- Query image $I_q$
- Gallery of images $\mathcal{G} = \{I_1, I_2, ..., I_n\}$

**Output:** Ranked list of gallery images by kinship likelihood:
$$\text{rank}(I_q, \mathcal{G}) = [I_{\pi(1)}, I_{\pi(2)}, ..., I_{\pi(n)}]$$

where $\pi$ is a permutation such that $P(\text{kin}|I_q, I_{\pi(i)}) \geq P(\text{kin}|I_q, I_{\pi(i+1)})$

---

## 2. Mathematical Framework

### 2.1 Embedding-Based Approach

The dominant paradigm learns an embedding function:
$$\phi: \mathcal{I} \rightarrow \mathbb{R}^d$$

where $d$ is the embedding dimension (typically 128, 256, or 512).

**Kinship decision** based on embedding similarity:
$$\hat{y} = \mathbb{1}[\text{sim}(\phi(I_1), \phi(I_2)) > \tau]$$

where $\text{sim}(\cdot, \cdot)$ is a similarity function and $\tau$ is a learned or tuned threshold.

### 2.2 Common Similarity Functions

| Function | Formula | Range |
|----------|---------|-------|
| **Cosine Similarity** | $\frac{\phi(I_1) \cdot \phi(I_2)}{\|\phi(I_1)\| \|\phi(I_2)\|}$ | $[-1, 1]$ |
| **L2 Distance** | $\|\phi(I_1) - \phi(I_2)\|_2$ | $[0, \infty)$ |
| **Squared L2** | $\|\phi(I_1) - \phi(I_2)\|_2^2$ | $[0, \infty)$ |

### 2.3 Siamese Network Formulation

The Siamese architecture uses weight-sharing:

```
           ┌─────────────┐
    I₁ ───►│   Encoder   │───► φ(I₁) ─┐
           │     φ       │            │
           └─────────────┘            ├───► sim(φ(I₁), φ(I₂)) ───► ŷ
           ┌─────────────┐            │
    I₂ ───►│   Encoder   │───► φ(I₂) ─┘
           │     φ       │
           └─────────────┘
         (shared weights)
```

### 2.4 Cross-Attention Formulation (FaCoRNet)

For attention-based approaches, given feature maps $F_1, F_2 \in \mathbb{R}^{C \times H \times W}$:

**Cross-Attention:**
$$Q = W_Q F_1, \quad K = W_K F_2, \quad V = W_V F_2$$
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

This allows the model to learn which facial regions in $I_1$ correspond to similar regions in $I_2$.

---

## 3. Loss Functions

### 3.1 Contrastive Loss (Classic)

$$\mathcal{L}_{\text{contrastive}} = y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2$$

where:
- $d = \|\phi(I_1) - \phi(I_2)\|_2$ is the embedding distance
- $m$ is the margin hyperparameter
- $y \in \{0, 1\}$ is the kinship label

### 3.2 Triplet Loss

$$\mathcal{L}_{\text{triplet}} = \max(0, d(a, p) - d(a, n) + m)$$

where:
- $a$ is the anchor sample
- $p$ is the positive sample (kin)
- $n$ is the negative sample (non-kin)
- $m$ is the margin

### 3.3 Supervised Contrastive Loss

$$\mathcal{L}_{\text{SupCon}} = -\sum_{i \in I} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_p)/\tau)}{\sum_{a \in A(i)} \exp(\text{sim}(z_i, z_a)/\tau)}$$

where:
- $P(i)$ is the set of positive samples for anchor $i$
- $A(i)$ is the set of all samples except $i$
- $\tau$ is the temperature parameter

### 3.4 Fair Contrastive Loss (KFC)

$$\mathcal{L}_{\text{FairCon}} = \mathcal{L}_{\text{SupCon}} + \lambda \cdot \mathcal{L}_{\text{debias}}$$

where $\mathcal{L}_{\text{debias}}$ is computed with gradient reversal for race classification:
$$\mathcal{L}_{\text{debias}} = -\sum_{r \in \text{races}} \log P(r | \phi(I))$$

The gradient reversal ensures the encoder learns race-invariant features.

### 3.5 Relation-Guided Contrastive Loss (FaCoRNet)

$$\mathcal{L}_{\text{Rel-Guide}} = \mathcal{L}_{\text{contrastive}} + \alpha \cdot \mathcal{L}_{\text{attention}}$$

where $\mathcal{L}_{\text{attention}}$ encourages the cross-attention to focus on semantically meaningful facial regions.

---

## 4. Evaluation Metrics

### 4.1 Primary Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall performance |
| **ROC-AUC** | Area under ROC curve | Threshold-independent |
| **F1 Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | Imbalanced datasets |
| **TAR@FAR** | True Accept Rate at Fixed False Accept Rate | Security applications |

### 4.2 Fairness Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Demographic Parity** | $P(\hat{y}=1|G=g_1) = P(\hat{y}=1|G=g_2)$ | Equal positive rate across groups |
| **Equalized Odds** | $P(\hat{y}=1|y=1,G=g_1) = P(\hat{y}=1|y=1,G=g_2)$ | Equal TPR/FPR across groups |
| **Accuracy Std Dev** | $\sigma(\{Acc_{g_1}, ..., Acc_{g_n}\})$ | Performance consistency across groups |

### 4.3 Cross-Dataset Metrics

| Metric | Description |
|--------|-------------|
| **Transfer Accuracy** | Accuracy when trained on Dataset A, tested on Dataset B |
| **Domain Gap** | $|Acc_{\text{same-domain}} - Acc_{\text{cross-domain}}|$ |

---

## 5. Challenge Formalization

### 5.1 Core Challenges

**Challenge 1: Subtle Genetic Similarity**
- Kinship features are far more subtle than identity features
- Shared genes manifest in small, distributed facial characteristics

**Challenge 2: Age Gap Confounding**
$$\text{sim}(\phi(I_{\text{parent}}), \phi(I_{\text{child}})) < \text{sim}(\phi(I_{\text{sibling}_1}), \phi(I_{\text{sibling}_2}))$$

Parent-child pairs with large age gaps show reduced visual similarity due to aging effects.

**Challenge 3: Same-Photo Bias**
$$P(y=1 | \text{same\_photo}(I_1, I_2) = \text{True}) \gg P(y=1 | \text{general})$$

When kin pairs come from the same photograph, models can exploit contextual (non-kinship) cues.

**Challenge 4: Class Imbalance**
In realistic scenarios:
$$|\text{non-kin pairs}| \gg |\text{kin pairs}|$$

This requires careful negative sampling and evaluation strategies.

### 5.2 Data Distribution

For a dataset with $F$ families, $N$ individuals:

**Positive pairs (kin):**
$$|\mathcal{P}^+| = \sum_{f=1}^{F} \binom{n_f}{2}$$

where $n_f$ is the number of individuals in family $f$.

**Negative pairs (non-kin):**
$$|\mathcal{P}^-| = \binom{N}{2} - |\mathcal{P}^+|$$

Typically $|\mathcal{P}^-| \gg |\mathcal{P}^+|$.

---

## 6. Notation Summary

| Symbol | Meaning |
|--------|---------|
| $I, I_1, I_2$ | Facial images |
| $\phi$ | Embedding function (encoder) |
| $y$ | Ground truth kinship label |
| $\hat{y}$ | Predicted kinship label |
| $d$ | Distance in embedding space |
| $\tau$ | Temperature / threshold |
| $m$ | Margin hyperparameter |
| $\mathcal{R}$ | Set of relationship types |
| $G$ | Demographic group (for fairness) |

---

## 7. Protocol Definitions

### 7.1 Standard Verification Protocol

1. **Train**: Learn $\phi$ on training pairs with kinship labels
2. **Validate**: Tune threshold $\tau$ on validation set
3. **Test**: Report metrics on held-out test set (family-disjoint)

### 7.2 Cross-Dataset Protocol

1. **Train**: Learn $\phi$ on source dataset (e.g., FIW)
2. **Test**: Evaluate directly on target dataset (e.g., KinFaceW-I)
3. **Report**: Both in-domain and cross-domain performance

### 7.3 Fairness Evaluation Protocol

1. Compute per-demographic-group accuracy: $\{Acc_{g_1}, ..., Acc_{g_n}\}$
2. Report mean: $\bar{Acc} = \frac{1}{n}\sum_{i=1}^{n} Acc_{g_i}$
3. Report std dev: $\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(Acc_{g_i} - \bar{Acc})^2}$
4. Goal: High $\bar{Acc}$, low $\sigma$

---

*This formulation provides the mathematical foundation for kinship classification research and implementation.*
