# Run Log — Model 09 (DINOv2-Face + Retrieval)

Template for individual run notes. Copy to `run-NNN.md` and fill in.

---

## Run NNN — <short label>

**Date:** YYYY-MM-DD
**GPU:** <e.g. AMD Radeon RX 6750 XT (11.98 GB, gfx1031, ROCm 5.7)>
**Status:** <RUNNING | COMPLETED | ABORTED>

## Context

<Why this run; what is being tested vs prior runs; hypothesis>

## Launch Command

```bash
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
  DINOV2_WEIGHTS=models/09_dinov2face_retrieval/weights/dinov2_face.pth \
  SKIP_INSTALL=1 EPOCHS=20 \
  bash models/09_dinov2face_retrieval/AMD/run_pipeline.sh
```

(Or leave `DINOV2_WEIGHTS` unset for base DINOv2.)

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local pipeline (`AMD/run_pipeline.sh`) |
| Device | GPU - cuda:0 (gfx1031, ROCm 5.7) |
| Dataset | FIW (RFIW Track-I) |
| Aligned dataset | `datasets/FIW_aligned` (MTCNN preprocessed 224×224) |
| **Backbone** | DINOv2 ViT-B/14 (frozen) |
| Backbone source | `vit_base_patch14_dinov2.lvd142m` (timm) |
| DINOv2-Face overlay | (none / path) |
| Img size | 224×224 |
| Normalization | ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |
| Total params | TBD (~93 M = 86 M frozen + 7.2 M trainable) |
| **Trainable params** | TBD |
| Retrieval K | 32 |
| Cross-attn | 2 layers, 4 heads |
| Embedding dim | 512 |
| Pool | cls |
| Dropout | 0.1 |
| Batch size | 8 (grad_accum 4 → eff. 32) |
| Learning rate | 1e-4 (cosine, warmup 3, min_lr=1e-6) |
| Weight decay | 1e-4 |
| Patience | 10 |
| Loss | Combined (BCE + 0.3 × cosine-contrastive + 0.15 × relation-CE) |
| Temperature | 0.1 |
| Relation set | fiw (11 classes) |
| Gallery cap | 200,000 |
| Gallery on CPU | False |
| AMP | Enabled |
| Seed | 42 |
| Workers | 4 |
| Time per epoch | TBD |

## Training Progress

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR | Status |
|-------|-----------|---------|---------|-----|-----|--------|
| 1 | | | | | | |
| ... | | | | | | |

## Results

### Validation-selected threshold

| Metric | Value |
|--------|-------|
| Threshold | |
| Val AUC | |

### Test metrics (FIW, 13,425 pairs)

| Metric | Value |
|--------|------:|
| Test ROC-AUC | |
| Test Accuracy | |
| Balanced Accuracy | |
| Test F1 | |
| Test Precision | |
| Test Recall | |
| Average Precision | |
| TAR@FAR=0.001 | |
| TAR@FAR=0.01 | |
| TAR@FAR=0.1 | |
| Gallery size | |
| Val→Test gap | |

### Per-relation accuracy

| Relation | N | Accuracy |
|----------|---:|---------:|
| bb | | |
| ss | | |
| sibs | | |
| md | | |
| fs | | |
| ms | | |
| fd | | |
| gfgd | | |
| gmgd | | |
| gfgs | | |
| gmgs | | |

## Analysis

<Compare to M06 R001/R002, M08 R001, M02 R031. Hypothesis confirmed/refuted?>

## Next Step

<What to try next, if anything>

## Artifacts

- Checkpoints: `output/NNN/checkpoints/{best.pt, final.pt, epoch_*.pt}`
- Logs: `output/NNN/logs/{train,test,evaluate}.log`
- Results: `output/NNN/results/{test_metrics_rocm.json, metrics_rocm.json,
  per_relation.json, confusion_matrix_rocm.png, roc_curve_rocm.png,
  per_relation_rocm.png}`
- Weights used: `weights/dinov2_face.pth` (if overlay applied) or
  timm-cached base DINOv2 (~340 MB).
