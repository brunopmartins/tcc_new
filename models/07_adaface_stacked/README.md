# Model 07 — AdaFace + Stacking Architecture for Kinship Verification

**Status:** SCAFFOLD — implementation pending.
**Target:** Test ROC AUC ≥ 0.90 OR Test Accuracy (per-relation balanced) ≥ 80%.

## Motivation

After 7 runs of Model 05 (DINOv2 + LoRA + DiffAttn), the ceiling on FIW
Track-I stabilized at **Test AUC ≈ 0.81-0.82** for any frozen-backbone
configuration and **0.822 for full unfreeze**. M02 R031 with full
fine-tune of ImageNet-ViT + supervised contrastive + cross-attention
reaches **0.850**.

The literature shows a clear gap that we did not close:

- **FaCoRNet (Su et al. 2023, arXiv:2304.04546)** reaches **82.0% mean
  per-relation balanced accuracy** on FIW Track-I using:
  - AdaFace ResNet-100 backbone (pretrained on MS-Celeb-1M)
  - Relation-Guided Contrastive Loss with adaptive temperature
  - Face Componential Relation cross-attention
- **FRoundation (Boutros et al. 2024, arXiv:2410.23831)** establishes
  the canonical DINOv2 face-task recipe:
  - LoRA rank 16 on q/v
  - CosFace loss (margin=0.3, scale=64)
  - LR=1e-4, batch=512, 40 epochs
- Reaches 87-91% verification accuracy on standard face benchmarks
  (LFW/CALFW/CPLFW/etc.)

M07 is the **stacking model** that combines the proven literature
ingredients with our M05-style architectural contributions (differential
cross-attention) and post-hoc gains (ensemble, TTA, calibration).

## Architecture Plan

```
              Face Detection + Alignment (MTCNN / RetinaFace)
                                    ↓
        [face1 cropped + aligned to 112×112]   [face2 ditto]
                                    ↓
             AdaFace backbone (frozen OR LoRA r=16)
                                    ↓
                  Embeddings (B, 512) per face
                                    ↓
              ┌────────────────────────────────┐
              │ Differential cross-attention   │
              │ (from M05) — 2 layers, 8 heads │
              │ Operating on patch tokens      │
              └─────────────────┬──────────────┘
                                ↓
                     Pair embedding (B, 1024)
                                ↓
                  ArcFace-pair loss (margin=0.3, scale=64)
                          on cos(emb1, emb2)
                                ↓
              ┌────────────────────────────────┐
              │ Test time: TTA (5-crop + flip) │
              │ Score-level ensemble with      │
              │ M02 R031 + M05 R001            │
              └────────────────────────────────┘
```

## Tier 2 Stacking Ingredients (priority order)

1. **AdaFace backbone** (largest single lift, +0.02-0.04 AUC vs ViT-Base)
   - Use `insightface` or `face_recognition` library to load pretrained weights
   - Two options: AdaFace-IR50 (24M params) or AdaFace-IR100 (65M params)
   - Default: IR50 for VRAM, fall back to IR100 if VRAM permits
   - Frozen with LoRA r=16 on attention projections

2. **ArcFace-pair / CosFace-pair loss** (+0.01-0.02 AUC)
   - Loss = margin-based angular distance between paired embeddings
   - For kin pair: cos(emb1, emb2) - margin should exceed threshold
   - For non-kin: cos < threshold
   - Implementation: extend FaCoRNet's Relation-Guided Contrastive Loss
   - Adaptive temperature τ from cross-attention output (FaCoR style)

3. **Face-aligned crops** (+0.005-0.02 AUC)
   - Replace current center-crop preprocessing with:
     1. Detect face bbox + 5 landmarks (MTCNN)
     2. Apply similarity transform to canonical 112×112 face
   - Cache aligned crops on disk to avoid runtime cost
   - Affects all input pipeline — should be done once for FIW

4. **Differential cross-attention** (preserve M05's contribution)
   - Reuse `models/05_dinov2_lora_diffattn/model.py::DifferentialCrossAttention`
   - Architecture differentiator vs FaCoRNet (which uses regular cross-attn)

5. **Test-time augmentation** (+0.005-0.01 AUC, free at inference)
   - Horizontal flip → 2 predictions
   - 5-crop (center + 4 corners) → 5 predictions
   - Average scores across all 10 = 1 ensemble per image
   - Combined: 10 × 10 = 100 pair evaluations per test pair (slow but free)
   - Approximate: just horizontal flip = 2 × 2 = 4 evals (much faster)

6. **Ensemble with M02 R031 + M05 R001** (+0.01-0.02 AUC)
   - Score-level: simple average of sigmoid(logits) across the three models
   - Weighted average if one model is much better in val (e.g. 0.4 M07 + 0.3 M02 + 0.3 M05)
   - No training cost — just inference at test time

7. **5-fold cross-validation** (+0.005-0.015 AUC via averaging + statistical confidence)
   - Train 5 separate M07 checkpoints on disjoint family splits
   - Average their test predictions
   - Costly (~5×30h = 150h) but gives published-quality numbers
   - Defer until R001 of M07 confirms baseline AUC > 0.85

## Realistic Expected Outcome

Stacking the high-confidence wins (1+2+3) on a single run:

- AdaFace backbone alone: M05 R001's 0.806 + 0.04 = **~0.85**
- + ArcFace-pair loss: + 0.01 = **~0.86**
- + Face-aligned crops: + 0.01 = **~0.87**

Add lower-confidence stacking (5+6):
- + TTA + ensemble: + 0.015 = **~0.885**

**Realistic Test AUC target after full Tier 2 stack: 0.87-0.89.**
**Probability of hitting AUC 0.90: ~25-35%.**
**Probability of hitting Acc 80% (per-relation balanced): ~70-80%.**

## Hardware Plan (AMD RX 6750 XT, 12 GB VRAM)

- AdaFace-IR50 frozen: ~96 MB weights, ~2 GB activations
- LoRA r=16 on attention: ~3M trainable
- Differential cross-attn (2 layers, 8 heads, 512-d): ~5M trainable
- Total trainable: ~10-15M
- Total model: ~30-50M params

VRAM budget (batch=8, grad_accum=4, AMP, grad_ckpt):
- AdaFace forward: ~2 GB
- DiffAttn forward+backward: ~2 GB
- Optimizer state (Adam for ~15M trainable): ~120 MB
- DataLoader workers + buffers: ~2 GB
- **Total: ~6-8 GB, comfortable in 12 GB**

For larger image size (224 → 384 if AdaFace supports):
- 3× more attention compute, ~10 GB VRAM
- Drop to batch=4 grad_accum=8 if needed

## Implementation Roadmap (~3-4 weeks)

### Week 1 — Backbone + loss
- [ ] Install `insightface` and download AdaFace IR50/IR100 weights
- [ ] Implement `model.py::AdaFaceBackbone` wrapper
- [ ] Implement `losses.py::ArcFacePairLoss`
- [ ] Train R001: AdaFace + ArcFace-pair loss + minimal head
  - Target: validate single ingredient (AdaFace) lifts AUC vs ViT-base
  - Expected: 0.83-0.86 AUC

### Week 2 — Architecture stacking
- [ ] Add differential cross-attention head (reuse M05 code)
- [ ] Train R002: full architecture, no TTA/ensemble
  - Target: validate stack of (AdaFace + DiffAttn + ArcFace loss)
  - Expected: 0.86-0.88 AUC

### Week 3 — Preprocessing + post-hoc
- [ ] Face detection + alignment pipeline (MTCNN once, cache)
- [ ] Re-train R003 with aligned crops
- [ ] Implement TTA wrapper (`tools/tta_inference.py`)
- [ ] Implement ensemble (`tools/score_ensemble.py`)
- [ ] Evaluate R001-R003 with TTA + ensemble
  - Expected: 0.87-0.89 AUC

### Week 4 — 5-fold CV + decisions
- [ ] If R003 ≥ 0.87 with single fold: run 5-fold CV
- [ ] If R003 < 0.87: stop, document negative result, accept M05/M02 as final
- [ ] Final TCC text writing with stacked numbers

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| insightface AdaFace weights hard to load with ROCm | Medium | Convert to PyTorch state_dict offline if needed; fall back to `facenet-pytorch` ArcFace |
| Face alignment changes data distribution → val/test gap widens | Medium | Cache aligned crops; A/B test old-vs-new preprocessing on R001 |
| Compute budget overruns (3-4 weeks → 5-6) | High | Front-load high-leverage interventions (1+2 first) |
| AUC 0.90 unreachable | Medium-high | Accept Acc 80% as fallback goal; literature suggests 0.85-0.88 is realistic ceiling |

## Files

```
07_adaface_stacked/
├── README.md                            (this file)
├── IMPLEMENTATION_PLAN.md               (week-by-week tasks)
├── RUN_LOG.md                           (empty, ready for R001)
├── model.py                             (TODO)
├── losses.py                            (TODO)
├── docker-compose.amd.yml               (TODO)
├── docker-compose.nvidia.yml            (TODO)
├── AMD/
│   ├── train.py                         (TODO — adapt from M05)
│   ├── test.py                          (TODO)
│   ├── evaluate.py                      (TODO)
│   └── run_pipeline.sh                  (TODO)
├── Nvidia/
│   └── (mirror of AMD/)
├── output/.gitkeep
└── run-review/
    └── (empty — first run-001.md when implemented)
```

## Sources

- [FaCoR (arxiv:2304.04546)](https://arxiv.org/html/2304.04546)
- [FRoundation (arxiv:2410.23831)](https://arxiv.org/html/2410.23831v2)
- [AdaFace (CVPR 2022)](https://github.com/mk-minchul/AdaFace)
- [insightface library](https://github.com/deepinsight/insightface)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
