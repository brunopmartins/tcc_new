# Comprehensive Dataset Analysis: Kinship Recognition Benchmarks

## 1. Dataset Overview

### 1.1 Summary Comparison

| Dataset | Year | Families | Individuals | Images | Pairs | Relation Types | Primary Task |
|---------|------|----------|-------------|--------|-------|----------------|--------------|
| **FIW** | 2016 | 1,000 | 5,000+ | 26,000+ | Variable | 11 | Verification, Classification, Search |
| **KinFaceW-I** | 2014 | N/A | 1,066 | 1,066 | 533 | 4 | Verification |
| **KinFaceW-II** | 2014 | N/A | 2,000 | 2,000 | 1,000 | 4 | Verification |
| **TSKinFace** | 2015 | 1,015 | ~3,000 | 2,589 | 2,030 | 4+tri | Tri-subject Verification |
| **Cornell KinFace** | 2010 | 143 | 286 | 286 | 143 | 4 | Verification |
| **UB KinFace 1.0** | 2010 | 90 | 180 | 270 | N/A | 4 | Verification |
| **UB KinFace 2.0** | 2012 | 200 | N/A | 600 | N/A | 4 | Age-variant Verification |

---

## 2. Families In the Wild (FIW)

### 2.1 Overview

**FIW** is the largest and most comprehensive kinship recognition dataset, established as the gold standard benchmark.

**Source:** Northeastern University SMILE Lab  
**URL:** https://web.northeastern.edu/smilelab/fiw/  
**License:** Non-commercial research and educational use only

### 2.2 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Families** | 1,000 |
| **Individuals** | 5,000+ |
| **Face Images** | 26,000+ |
| **Family Photos** | 13,000+ |
| **Relationship Types** | 11 |
| **Annotation Format** | CSV + folder structure |

### 2.3 Relationship Types

| Code | Relationship | Direction |
|------|--------------|-----------|
| 1 | Parent | Parent → Child |
| 2 | Child | Child → Parent |
| 3 | Grandparent | Grandparent → Grandchild |
| 4 | Sibling | Bidirectional |
| 5 | Spouse | Bidirectional |
| 6 | Grandchild | Grandchild → Grandparent |
| 7+ | Extended relations | Various |

### 2.4 Data Structure

```
FIW/
├── FIW_PIDs.csv          # Photo lookup (PID, Name, URL, Metadata)
├── FIW_FIDs.csv          # Family lookup (FID, Surname)
├── FIW_RIDs.csv          # Relationship lookup (1-9 → type)
└── FIDs/
    └── FID0001/          # Family folder
        ├── MID1/         # Member 1 face images
        │   ├── P00001_face0.jpg
        │   ├── P00001_face1.jpg
        │   └── ...
        ├── MID2/         # Member 2 face images
        └── F0001.csv     # Relationship matrix
```

**Relationship Matrix Example (F0001.csv):**
```
     0     1     2     3     Name    Gender
     1     0     4     5     Alice   female
     2     1     0     1     Bob     female
     3     5     4     0     Carol   male
```

Interpretation: MID1→MID2 = 4 (Parent), MID2→MID1 = 1 (Child)

### 2.5 Evaluation Protocols

**Task 1: Kinship Verification**
- Binary classification (kin vs. non-kin)
- Family-disjoint train/val/test splits
- Reported metric: Accuracy, AUC

**Task 2: Tri-Subject Verification (RFIW Track 2)**
- Input: (Father, Mother, Child) triplet
- Output: Is Child the biological offspring?

**Task 3: Family Search & Retrieval (RFIW Track 3)**
- Query: Probe face
- Gallery: Set of candidate family members
- Output: Ranked retrieval list

### 2.6 RFIW Challenge Editions

| Edition | Year | Venue | Winner | Performance |
|---------|------|-------|--------|-------------|
| RFIW 2017 | 2017 | FG | Various | Baseline established |
| RFIW 2018 | 2018 | FG | Various | CNN methods dominate |
| RFIW 2019 | 2019 | FG | Various | Deep metric learning |
| RFIW 2020 | 2020 | FG | Multiple | 3 tracks introduced |
| RFIW 2021 | 2021 | FG | **TeamCNU** | Contrastive learning wins all tracks |

### 2.7 Strengths and Limitations

**Strengths:**
- ✅ Largest kinship dataset
- ✅ Diverse demographics (more than other datasets)
- ✅ Multiple relationship types
- ✅ Official evaluation protocols and challenges
- ✅ Active maintenance and toolbox (FIW_KRT)
- ✅ Extended with multimedia (video, audio, text)
- ✅ Images from different photos (reduces same-photo bias)

**Limitations:**
- ⚠️ Still predominantly Caucasian subjects
- ⚠️ Internet-scraped (consent considerations)
- ⚠️ Some label noise reported
- ⚠️ Download requires agreement to terms

---

## 3. KinFaceW Datasets

### 3.1 KinFaceW-I

**Overview:** Images cropped from **different photographs**, introducing natural variations.

| Relation | Pairs | Total Images |
|----------|-------|--------------|
| Father-Son (FS) | 156 | 312 |
| Father-Daughter (FD) | 134 | 268 |
| Mother-Son (MS) | 116 | 232 |
| Mother-Daughter (MD) | 127 | 254 |
| **Total** | **533** | **1,066** |

**Key Characteristic:** Natural illumination, pose, and age variations between kin images.

### 3.2 KinFaceW-II

**Overview:** Images cropped from **same photographs**.

| Relation | Pairs | Total Images |
|----------|-------|--------------|
| Father-Son (FS) | 250 | 500 |
| Father-Daughter (FD) | 250 | 500 |
| Mother-Son (MS) | 250 | 500 |
| Mother-Daughter (MD) | 250 | 500 |
| **Total** | **1,000** | **2,000** |

**⚠️ CRITICAL BIAS:** Same-photo pairs share background, lighting, and temporal context. Models can exploit these cues rather than learning kinship features.

### 3.3 Evaluation Protocol

**5-Fold Cross-Validation** with three settings:

| Setting | Description | Train Data |
|---------|-------------|------------|
| **Unsupervised** | No labeled kin information | Unlabeled pairs only |
| **Image-Restricted** | Only given pairs | Labeled kin pairs |
| **Image-Unrestricted** | Identity info available | Can generate additional negative pairs |

### 3.4 Comparison

| Aspect | KinFaceW-I | KinFaceW-II |
|--------|------------|-------------|
| **Same-Photo Bias** | Low | **High** |
| **Size** | 533 pairs | 1,000 pairs |
| **Realistic** | More | Less |
| **Recommended** | Yes | With caution |

---

## 4. TSKinFace (Tri-Subject)

### 4.1 Overview

**Unique contribution:** First dataset for **tri-subject kinship verification**.

| Metric | Value |
|--------|-------|
| **Images** | 2,589 |
| **Tri-Subject Groups** | 1,015 |
| **Father-Mother-Son (FM-S)** | 513 |
| **Father-Mother-Daughter (FM-D)** | 502 |
| **Derived Pairs** | 2,030 |

### 4.2 Tri-Subject Verification Task

**Input:** $(I_F, I_M, I_C)$ — Father, Mother, Child images

**Question:** Is $I_C$ the biological child of $(I_F, I_M)$?

This requires reasoning about **combined parental features** rather than individual pairwise similarity.

### 4.3 Derived Pairs

The tri-subject groups can be decomposed into traditional pairs:
- 1,015 Father-Child pairs
- 1,015 Mother-Child pairs
- Total: 2,030 parent-child pairs

---

## 5. Cornell KinFace

### 5.1 Overview

**Early benchmark** dataset, now considered small-scale.

| Metric | Value |
|--------|-------|
| **Images** | 286 |
| **Pairs** | 143 (or 150 in some sources) |
| **Relations** | 4 (FS, FD, MS, MD) |
| **Image Quality** | Frontal pose, neutral expression |

### 5.2 Limitations

- ⚠️ Very small scale
- ⚠️ Same-photo signal exists
- ⚠️ Limited demographic diversity
- ⚠️ Primarily historical use

---

## 6. UB KinFace

### 6.1 Version 1.0

| Metric | Value |
|--------|-------|
| **Images** | 270 |
| **Individuals** | 180 |
| **Groups** | 90 |
| **Source** | Public figures (celebrities) |

### 6.2 Version 2.0

| Metric | Value |
|--------|-------|
| **Images** | 600 |
| **Groups** | 200 |
| **Structure** | Children, young parents, old parents |
| **Focus** | Large age variations |

**Note:** Old parent photos are grayscale, introducing additional challenges.

---

## 7. Dataset Bias Analysis

### 7.1 Same-Photo Bias

**Definition:** When kin pairs are extracted from the same photograph, they share contextual features unrelated to kinship.

**Affected Datasets:**
- **High Risk:** KinFaceW-II, Cornell KinFace
- **Moderate Risk:** TSKinFace, KinFaceW-I (some pairs)
- **Low Risk:** FIW (designed to minimize this)

**Evidence:** Research by Oxford VGG (2018) demonstrated that models trained on same-photo pairs learn to recognize photographic similarity rather than genetic relationships.

**Mitigation:**
1. Use FIW as primary benchmark
2. Ensure family-disjoint train/test splits
3. Evaluate cross-dataset generalization
4. Report results on different-photo pairs separately

### 7.2 Demographic Bias

**Issue:** Datasets predominantly feature:
- Caucasian subjects
- Balanced age distributions (middle-age parents, young children)
- High-quality images

**Impact:** Models trained on biased data show:
- Higher error rates for underrepresented demographics
- Poor generalization to non-Western populations
- Reduced accuracy for elderly subjects

**Mitigation:**
- KFC dataset includes race annotations (African, Asian, Caucasian, Indian)
- Use fairness-aware training (e.g., KFC fair contrastive loss)
- Report per-demographic performance metrics

### 7.3 Age Gap Bias

**Issue:** Parent-child pairs with larger age gaps are underrepresented and harder to classify.

| Age Gap | Difficulty | Representation |
|---------|------------|----------------|
| 15-25 years | Moderate | Overrepresented |
| 25-40 years | High | Moderate |
| 40+ years | Very High | Underrepresented |
| Grandparent-Grandchild | Extreme | Rare |

**Mitigation:**
- Age-invariant feature learning (AIAF)
- Face age transformation augmentation
- Explicit age gap stratification in evaluation

---

## 8. Recommended Dataset Usage

### 8.1 For Standard Benchmarking

```
Primary:    FIW (official train/val/test splits)
Secondary:  KinFaceW-I (5-fold CV, image-unrestricted)
Avoid:      KinFaceW-II (same-photo bias)
```

### 8.2 For Cross-Dataset Evaluation

```
Train:      FIW
Test:       KinFaceW-I, TSKinFace (different distribution)
```

### 8.3 For Fairness Evaluation

```
Dataset:    FIW with KFC race annotations
Metrics:    Per-race accuracy, standard deviation
Protocol:   Report disaggregated results
```

### 8.4 For Tri-Subject Verification

```
Primary:    TSKinFace
Secondary:  FIW (derive tri-subject groups from families)
```

---

## 9. Data Access Summary

| Dataset | URL | Access |
|---------|-----|--------|
| **FIW** | https://web.northeastern.edu/smilelab/fiw/ | Registration required |
| **KinFaceW** | https://www.kinfacew.com/ | Direct download |
| **TSKinFace** | https://parnec.nuaa.edu.cn/~xtan/TSKinFace.html | Direct download |
| **Cornell** | http://chenlab.ece.cornell.edu/ | Research request |
| **UB KinFace** | http://www1.ece.neu.edu/~yunfu/research/Kinface/ | Direct download |

### 9.1 KFC Combined Dataset

The KFC paper provides a cleaned, combined dataset with race annotations:
- **Source:** https://drive.google.com/drive/folders/1r8yi1L6ues2gB6HGQD-UJhJwZl7Gtf_q
- **Contents:** CornellKin, UBKinFace, KinFaceW-I/II, Family101, FIW
- **Added Value:** Data cleaning + race annotations

---

## 10. Dataset Statistics Visualization

### 10.1 Size Comparison (Log Scale)

```
FIW:          ████████████████████████████████████████ 26,000 images
KinFaceW-II:  ██ 2,000 images
TSKinFace:    █ 2,589 images
KinFaceW-I:   █ 1,066 images
UB KinFace:   ░ 600 images
Cornell:      ░ 286 images
```

### 10.2 Relationship Type Coverage

```
Dataset       | FS | FD | MS | MD | Sib | GP-GC | Spouse |
--------------|----|----|----|----|-----|-------|--------|
FIW           | ✓  | ✓  | ✓  | ✓  | ✓   | ✓     | ✓      |
KinFaceW-I/II | ✓  | ✓  | ✓  | ✓  | ✗   | ✗     | ✗      |
TSKinFace     | ✓  | ✓  | ✓  | ✓  | ✗   | ✗     | ✗      |
Cornell       | ✓  | ✓  | ✓  | ✓  | ✗   | ✗     | ✗      |
UB KinFace    | ✓  | ✓  | ✓  | ✓  | ✗   | ✗     | ✗      |
```

---

## 11. Practical Recommendations

### For Your Paper

1. **Use FIW as primary benchmark** — largest, most diverse, standard protocol
2. **Report cross-dataset results** — train on FIW, test on KinFaceW-I
3. **Avoid KinFaceW-II for primary results** — document same-photo bias
4. **Include fairness analysis** — per-demographic performance if possible
5. **Use official splits** — ensure reproducibility via FIW_KRT toolbox

### Data Loading Example (FIW_KRT)

```python
# Using FIW_KRT toolbox
from src.data import FIWDataset

# Load kinship verification pairs
train_dataset = FIWDataset(
    root='path/to/FIW',
    split='train',
    task='verification'
)

# Each sample: (img1, img2, label, relation_type)
for img1, img2, label, rel in train_dataset:
    # label: 1 = kin, 0 = non-kin
    # rel: relationship type (FS, FD, MS, MD, etc.)
    pass
```

---

*This analysis provides the foundation for experimental design and dataset selection in kinship classification research.*
