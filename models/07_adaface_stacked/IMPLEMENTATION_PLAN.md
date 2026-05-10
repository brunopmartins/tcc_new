# M07 — Implementation Plan (Concrete Steps)

This document tracks the exact implementation tasks for M07. Update as
tasks are completed. Each task has acceptance criteria.

---

## Phase 1 — Backbone integration (3-4 days)

### Task 1.1: Install AdaFace dependencies
- [ ] `pip install insightface onnxruntime` into `models/07_adaface_stacked/.venv`
- [ ] OR `pip install facenet-pytorch` if insightface is problematic on ROCm
- [ ] Download AdaFace pretrained weights (IR50, ~96 MB) or use insightface's auto-download
- **Acceptance:** can load AdaFace model and run forward on a dummy 112×112 input

### Task 1.2: Implement `AdaFaceBackbone` wrapper
- [ ] In `model.py`, add class that wraps the pretrained AdaFace model
- [ ] Output: 512-d embedding per face
- [ ] Option to freeze (default) or unfreeze last N blocks
- [ ] Option to add LoRA on attention (r=16, q+v projections)
- **Acceptance:** forward pass on (B, 3, 112, 112) returns (B, 512) embedding;
  smoke test confirms ~24M params; trainable params controllable by flag

### Task 1.3: Implement `ArcFacePairLoss`
- [ ] In `losses.py`, add class taking (emb1, emb2, labels) → loss
- [ ] Compute cos(emb1, emb2) per pair
- [ ] Apply margin penalty: kin pairs need cos > margin_pos, non-kin < margin_neg
- [ ] Scale logits by `scale` factor (typically 64)
- [ ] Optional: adaptive margin (FaCoRNet style, derived from cross-attn map)
- **Acceptance:** loss decreases monotonically on a 100-pair toy dataset
  with random init; final cosine separation kin vs non-kin > 0.3

### Task 1.4: Train R001 — backbone-only smoke test
- [ ] Config: AdaFace frozen, ArcFace-pair loss, minimal MLP head
- [ ] Hardware budget: batch=16, grad_accum=2 (eff=32), LR=1e-4, 20 epochs
- [ ] Compare val AUC against M02 R031 (ImageNet ViT, 0.881)
- **Acceptance:** Val AUC ≥ 0.87 (proves AdaFace > ImageNet ViT for our setup);
  Test AUC ≥ 0.83 (small backbone-swap lift)

---

## Phase 2 — Architecture stacking (3-5 days)

### Task 2.1: Port differential cross-attention from M05
- [ ] Copy `DifferentialCrossAttention` and `CrossAttnBlock` from
  `models/05_dinov2_lora_diffattn/model.py`
- [ ] Adapt for AdaFace embeddings (1×512 per face, not 196×768)
- [ ] Need patch-level tokens → use AdaFace's intermediate feature maps
  (typically before global pooling); shape ~(B, 49, 512) for 112×112 input
- **Acceptance:** integrates cleanly; forward pass works at batch=8

### Task 2.2: Auxiliary relation head (optional)
- [ ] Same design as M05 — 11-way classifier on positive pairs only
- [ ] Loss weight 0.0-0.4 (start at 0.0, ablate later)
- **Acceptance:** loss contributes < 20% of total during training

### Task 2.3: Train R002 — full stacked architecture
- [ ] AdaFace + LoRA r=16 + DiffAttn + ArcFacePair loss
- [ ] Config: LR=1e-4, warmup=5, cosine, 30 epochs, patience=15
- [ ] Track val AUC trajectory carefully
- **Acceptance:** Val AUC ≥ 0.90 (signals strong architecture); Test AUC ≥ 0.85
  matching M02 R031

---

## Phase 3 — Preprocessing + post-hoc (5-7 days)

### Task 3.1: Face detection + alignment pipeline
- [ ] Choose: `MTCNN` (insightface) or `RetinaFace` (insightface)
- [ ] Write `tools/align_fiw_dataset.py`:
  - Iterate every FIW image
  - Detect face bbox + 5 landmarks
  - Apply similarity transform to canonical 112×112 with eyes at (38, 51), (74, 51)
  - Save aligned crop to disk: `datasets/FIW_aligned/...`
- [ ] Handle failure cases: keep original image if detection fails
- **Acceptance:** ≥95% of FIW images successfully aligned; spot-check 10 random
  alignments look correct

### Task 3.2: Update dataset.py to support aligned data
- [ ] Add `--use_aligned_crops` flag
- [ ] Load from `FIW_aligned` instead of `FIW` when set
- **Acceptance:** R002 retraining with `--use_aligned_crops` runs without errors

### Task 3.3: Train R003 with aligned crops
- [ ] Same config as R002, just with `--use_aligned_crops`
- **Acceptance:** Test AUC ≥ R002 + 0.005 (proves alignment helps)

### Task 3.4: Implement TTA wrapper
- [ ] In `tools/tta_inference.py`:
  - Load model
  - For each test pair: generate 4 versions (orig, hflip, hflip+orig, orig+hflip)
  - Average sigmoid scores
- [ ] Or 5-crop variant for slower but stronger TTA
- **Acceptance:** Test AUC ≥ R003 + 0.005 with hflip TTA

### Task 3.5: Implement score-level ensemble
- [ ] In `tools/score_ensemble.py`:
  - Take checkpoints from M02 R031, M05 R001, M07 R003
  - Run inference on test set with each
  - Combine: weighted average of sigmoid(logits)
  - Weight tuning on val set
- **Acceptance:** Ensemble Test AUC ≥ best single + 0.01

---

## Phase 4 — Decision point (1-2 days)

### Task 4.1: Decide on 5-fold CV
- [ ] If R003 Test AUC ≥ 0.87:
  - Plan 5-fold CV run (~150h compute)
  - Set up KFold sampler in dataset.py
  - Train M07 R004-R008 on disjoint family splits
  - Report mean ± std for the TCC
- [ ] If R003 Test AUC < 0.87:
  - Stop M07 iteration
  - Accept current numbers, focus on TCC text writing
  - Document M07 as "stacked SOTA-style attempt, partial success"

---

## Status Log

| Date | Phase | Task | Status | Notes |
|------|-------|------|--------|-------|
| 2026-05-10 | Setup | Scaffold created | ✅ | README + this plan |
| TBD | 1.1 | Install AdaFace | ⏳ | Waiting on impl |
| TBD | 1.2 | AdaFaceBackbone | ⏳ | |
| TBD | 1.3 | ArcFacePairLoss | ⏳ | |
| TBD | 1.4 | R001 baseline | ⏳ | |
| TBD | 2.1 | DiffAttn port | ⏳ | |
| TBD | 2.3 | R002 full stack | ⏳ | |
| TBD | 3.x | Aligned crops + TTA + ensemble | ⏳ | |
| TBD | 4.1 | 5-fold CV decision | ⏳ | |
