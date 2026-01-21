# State-of-the-Art Benchmarks: Kinship Classification Results

## 1. Overview

This document compiles state-of-the-art results across major kinship recognition benchmarks, organized by dataset, method category, and chronology.

---

## 2. FIW (Families In the Wild) Results

### 2.1 Kinship Verification (Overall)

| Method | Year | Accuracy | AUC | F1 | Source |
|--------|------|----------|-----|-----|--------|
| **Vision Transformer** | 2025 | **92.0%** | - | 0.85 | ScienceDirect |
| **FaCoRNet** | 2023 | +4.6% over prev | - | - | ICCV Workshop |
| Supervised Contrastive | 2023 | 81.1% | - | - | arXiv:2302.09556 |
| TeamCNU (RFIW Winner) | 2021 | Top across tracks | - | - | RFIW 2021 |
| ArcFace Baseline | 2020 | ~80% | - | - | arXiv:2006.11739 |
| Fine-tuned CNN | 2016 | ~70% | - | - | FIW Paper |

### 2.2 RFIW Challenge Results

#### Track 1: Kinship Verification

| Team/Method | Year | Performance | Key Technique |
|-------------|------|-------------|---------------|
| TeamCNU | 2021 | **1st Place** | Contrastive learning framework |
| Various | 2020 | Competitive | Deep metric learning |
| Various | 2019 | Baseline+ | Siamese networks |

#### Track 2: Tri-Subject Verification

| Team/Method | Year | Performance | Notes |
|-------------|------|-------------|-------|
| TeamCNU | 2021 | **1st Place** | Joint F-M-C reasoning |

#### Track 3: Search & Retrieval

| Team/Method | Year | Performance | Metric |
|-------------|------|-------------|--------|
| TeamCNU | 2021 | **1st Place** | mAP, Rank-1 |

### 2.3 Per-Relation Performance (Typical Ranges)

| Relation | Accuracy Range | Difficulty | Notes |
|----------|----------------|------------|-------|
| **Siblings (SS, BB)** | 78-85% | Easier | Similar age, more visual similarity |
| **Mother-Daughter** | 75-82% | Moderate | Same gender helps |
| **Mother-Son** | 72-80% | Moderate | Cross-gender |
| **Father-Daughter** | 70-78% | Harder | Cross-gender, age gap |
| **Father-Son** | 68-76% | Harder | Age gap effects |
| **Grandparent-Grandchild** | 60-70% | Hardest | Large age gap |

---

## 3. KinFaceW-I Results

### 3.1 Overall Performance

| Method | Year | FS | FD | MS | MD | Mean | Source |
|--------|------|----|----|----|----|------|--------|
| **ArcFace + Fine-tune** | 2020 | - | - | - | - | **~85%** | arXiv:2006.11739 |
| DDML | 2017 | 79.4 | 77.3 | 77.8 | 78.2 | 78.2% | IEEE TIP |
| MNRML (KinVer) | 2017 | 82.4 | 87.2 | 89.0 | 88.4 | 86.8% | GitHub |
| CNN Baseline | 2016 | ~72 | ~74 | ~73 | ~75 | ~74% | Various |
| LBP + SVM | 2014 | ~65 | ~67 | ~66 | ~68 | ~67% | Early work |

### 3.2 Protocol Comparison

| Setting | Typical Performance | Notes |
|---------|---------------------|-------|
| **Image-Unrestricted** | Highest | More negative pairs available |
| **Image-Restricted** | Moderate | Only given pairs |
| **Unsupervised** | Lowest | No labeled kin info |

---

## 4. KinFaceW-II Results

### 4.1 Overall Performance

| Method | Year | FS | FD | MS | MD | Mean | Source |
|--------|------|----|----|----|----|------|--------|
| **ArcFace + Fine-tune** | 2020 | - | - | - | - | **~90%** | arXiv:2006.11739 |
| Various Deep | 2018-20 | 85-92 | 86-93 | 85-91 | 86-92 | 88-92% | Various |
| DDML | 2017 | 82.1 | 85.3 | 83.7 | 84.6 | 83.9% | IEEE TIP |
| CNN Baseline | 2016 | ~78 | ~80 | ~79 | ~81 | ~80% | Various |

**⚠️ CAUTION:** KinFaceW-II results are inflated due to same-photo bias. Both kin in each pair come from the same photograph, allowing models to exploit non-kinship cues.

### 4.2 Bias Analysis

| Aspect | KinFaceW-I | KinFaceW-II |
|--------|------------|-------------|
| Same-photo pairs | No | **Yes** |
| Typical accuracy | 78-86% | 84-92% |
| Gap | - | +6-8% (inflated) |
| Real-world validity | Higher | Lower |

---

## 5. TSKinFace Results

### 5.1 Tri-Subject Verification

| Method | Year | FM-S | FM-D | Mean | Notes |
|--------|------|------|------|------|-------|
| Deep Fusion | 2018 | ~78% | ~76% | ~77% | Tri-subject specific |
| Pairwise → Tri | 2017 | ~75% | ~73% | ~74% | Decomposed approach |
| Baseline CNN | 2015 | ~70% | ~68% | ~69% | Original paper |

### 5.2 Derived Pairwise Results

When decomposed into pairwise comparisons:

| Derived Pair | Accuracy Range |
|--------------|----------------|
| Father-Child | 72-80% |
| Mother-Child | 74-82% |

---

## 6. Cross-Dataset Generalization

### 6.1 Transfer Performance

| Train → Test | Accuracy | Drop | Notes |
|--------------|----------|------|-------|
| FIW → FIW | 80-92% | - | In-domain |
| FIW → KinFaceW-I | 65-75% | -15% | Cross-dataset |
| FIW → KinFaceW-II | 70-82% | -10% | Less drop (bias helps) |
| KinFaceW-II → FIW | 55-65% | -25% | Poor generalization |

### 6.2 Key Findings

1. **FIW-trained models generalize better** than KinFaceW-trained models
2. **Same-photo training hurts generalization** significantly
3. **Domain adaptation helps** reduce the gap
4. **Cross-dataset evaluation is essential** for real-world validity

---

## 7. Fairness Benchmarks

### 7.1 Per-Race Performance (KFC Results)

| Race | Before KFC | After KFC | Improvement |
|------|------------|-----------|-------------|
| African | 72% | 78% | +6% |
| Asian | 78% | 79% | +1% |
| Caucasian | 82% | 80% | -2% |
| Indian | 74% | 78% | +4% |
| **Std Dev** | **4.2%** | **0.9%** | **-3.3%** |

**Key metric:** Standard deviation across races reduced from 4.2% to 0.9%.

### 7.2 Gender Performance Gap

| Gender Pair | Typical Gap | Notes |
|-------------|-------------|-------|
| Same-gender | +3-5% | Easier comparison |
| Cross-gender | -3-5% | Harder comparison |
| Female pairs | +1-2% | Slightly higher accuracy |
| Male pairs | Baseline | Reference |

---

## 8. Method Category Comparison

### 8.1 By Architecture

| Category | Best Accuracy (FIW) | Representative | Year |
|----------|---------------------|----------------|------|
| **Vision Transformer** | **92%** | ViT end-to-end | 2025 |
| **Cross-Attention** | +4.6% SOTA | FaCoRNet | 2023 |
| **Contrastive Learning** | 81.1% | Supervised Contrastive | 2023 |
| **Siamese + ArcFace** | ~80% | ArcFace transfer | 2020 |
| **Deep Metric Learning** | ~78% | DDML, k-TMN | 2017-21 |
| **Siamese + VGGFace** | ~75% | VGGFace Siamese | 2018 |
| **Handcrafted Features** | ~70% | LBP + SVM | 2014 |

### 8.2 By Loss Function

| Loss Function | Accuracy | Fairness | Complexity |
|---------------|----------|----------|------------|
| BCE (baseline) | ~75% | Low | Simple |
| Contrastive | ~78% | Low | Low |
| Triplet | ~79% | Low | Moderate |
| Supervised Contrastive | 81% | Low | Moderate |
| Fair Contrastive | 79-80% | **High** | High |
| Relation-Guided | SOTA | Moderate | High |

---

## 9. Computational Efficiency

### 9.1 Inference Speed

| Method | Model Size | FLOPs | Inference Time (per pair) |
|--------|------------|-------|---------------------------|
| EfficientNet-B0 Siamese | 11M | 0.8G | ~5ms |
| ArcFace IR-50 | 88M | 12.1G | ~15ms |
| ArcFace IR-101 | 130M | 24.2G | ~25ms |
| ViT-Base | 172M | 17.6G | ~30ms |
| FaCoRNet | ~200M | ~30G | ~40ms |
| Hybrid CNN-ViT | ~300M | ~50G | ~60ms |

### 9.2 Training Time (FIW, Single GPU)

| Method | V100 | A100 | Epochs to Converge |
|--------|------|------|-------------------|
| Siamese Baseline | 2h | 1h | 20 |
| ArcFace Fine-tune | 4h | 2h | 30 |
| Contrastive Learning | 6h | 3h | 50 |
| FaCoRNet | 10h | 5h | 80 |
| ViT Fine-tune | 16h | 8h | 50 |

---

## 10. Ablation Study Summary

### 10.1 Backbone Impact

| Backbone | Relative Performance | Notes |
|----------|---------------------|-------|
| ArcFace IR-101 | **+8%** vs SphereFace | Best face recognition transfer |
| VGGFace2 | +5% vs random init | Solid baseline |
| ImageNet pretrain | +3% vs random init | Modest improvement |
| Random initialization | Baseline | Poor performance |

### 10.2 Loss Function Impact

| Loss | Relative to BCE | Notes |
|------|-----------------|-------|
| Supervised Contrastive | **+5%** | Best embedding learning |
| Triplet | +3% | Good with hard mining |
| Contrastive | +2% | Classic approach |
| BCE | Baseline | Classification only |

### 10.3 Attention Mechanism Impact

| Component | Impact | Notes |
|-----------|--------|-------|
| Cross-attention | **+4.6%** | Face component alignment |
| Channel attention | +1-2% | Feature recalibration |
| Self-attention (ViT) | +3-5% | Global context |
| No attention | Baseline | - |

---

## 11. Benchmark Summary Table

| Dataset | Best Method | Accuracy | Year | Key Insight |
|---------|-------------|----------|------|-------------|
| **FIW** | ViT | **92%** | 2025 | Transformers excel |
| **FIW** | FaCoRNet | SOTA | 2023 | Cross-attention key |
| **KinFaceW-I** | ArcFace | ~86% | 2020 | Transfer learning essential |
| **KinFaceW-II** | ArcFace | ~92% | 2020 | ⚠️ Biased benchmark |
| **TSKinFace** | Deep Fusion | ~77% | 2018 | Tri-subject harder |

---

## 12. Recommendations for Comparison

### 12.1 Baseline Selection

1. **Minimum baseline:** ArcFace + Fine-tune (~80% FIW)
2. **Competitive baseline:** Supervised Contrastive (81.1% FIW)
3. **SOTA comparison:** FaCoRNet, ViT approaches

### 12.2 Reporting Guidelines

- **Always report:** FIW results (primary benchmark)
- **Include:** Per-relation breakdown
- **Recommended:** Cross-dataset evaluation
- **For fairness:** Per-demographic performance
- **Avoid:** KinFaceW-II alone (biased)

### 12.3 Statistical Significance

- Use 5-fold cross-validation where applicable
- Report mean ± std across runs
- Include confidence intervals for key results

---

## 13. Result Visualization Templates

### 13.1 Per-Relation Bar Chart

```
Accuracy by Relation Type (FIW Dataset)

SS (Sister-Sister)      ████████████████████░░ 85%
BB (Brother-Brother)    ███████████████████░░░ 82%
MD (Mother-Daughter)    ██████████████████░░░░ 80%
MS (Mother-Son)         █████████████████░░░░░ 78%
FD (Father-Daughter)    ████████████████░░░░░░ 75%
FS (Father-Son)         ███████████████░░░░░░░ 73%
GF-GC (Grandparent)     ████████████░░░░░░░░░░ 65%
```

### 13.2 Method Comparison

```
Method Performance on FIW (Accuracy %)

Vision Transformer (2025)  ████████████████████████████████████████████████ 92%
FaCoRNet (2023)           ██████████████████████████████████████████████░░ ~88%
Sup. Contrastive (2023)   ████████████████████████████████████████░░░░░░░░ 81%
ArcFace Baseline (2020)   ██████████████████████████████████████░░░░░░░░░░ 80%
Siamese VGGFace (2018)    ██████████████████████████████████░░░░░░░░░░░░░░ 75%
```

---

*This benchmark compilation provides the empirical foundation for evaluating and comparing kinship classification methods.*
