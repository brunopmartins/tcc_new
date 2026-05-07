# M05 — Improvement Options Catalog

After R001-R005, the M05 ceiling at Test ROC AUC ≈ 0.81 is well-established.
Five hypotheses tested and rejected (defaults, regularization, class balancing,
partial unfreeze, full unfreeze with M02-style LR). The val→test gap holds at
~−0.10 across every plasticity regime.

This file catalogs improvement directions that were **not pursued** in this
phase. They remain valid future experiments, ranked by expected leverage and
cost.

---

## A — Change loss function (PURSUED as R006)

Switch from `BCE + 0.5×contrastive + 0.2×relation-CE` to **pure cosine
contrastive**. M02 R031 used pure supervised contrastive with margin=0.3 and
reached Test AUC=0.850. M05 has never tested this loss in isolation.

```bash
SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
  LOSS=contrastive RELATION_LOSS_WEIGHT=0 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

Status: launched as R006.

---

## E — Per-relation thresholds (PURSUED post-hoc on R001)

No retraining. Loads R001's best.pt, computes 11 separate thresholds (one per
FIW relation) by maximizing F1 on the validation set per relation, then
applies them on the test set.

Doesn't change AUC (threshold-independent), but can lift accuracy and F1 by
2-4 pp by exploiting the fact that score distributions differ across
relations (e.g., grandparent kin pairs have lower mean score than parent-child
pairs).

Status: pursued via `tools/per_relation_thresholds.py`.

---

## B — Two-stage training (NOT pursued)

Stage 1: train R001 setup (frozen backbone + LoRA + heads at LR=3e-4) for ~10
epochs to convergence. Stage 2: load stage-1 checkpoint, unfreeze last 4
DINOv2 blocks at LR=5e-6, fine-tune for ~10 more epochs.

**Why it could work:** R005 failed because head + backbone started random
together at LR=5e-6 (gradient underflow). Two-stage avoids this — head is
already converged when backbone unfreeze begins, so backbone has a stable
gradient signal to align with.

**Cost:** ~25h (10 + 10 epochs at ~75 min/ep). Probability of beating R004
(0.812 Test AUC): ~60%. Probability of breaking 0.85: ~15%.

**Implementation:** would require either:
- A `--resume_from` flag that loads checkpoint AND keeps optimizer state
- A dedicated `train_stage2.py` script
- Or split the run manually: stage 1 = R001 config, stop at convergence, then
  launch stage 2 with `RESUME=output/00X/checkpoints/best.pt UNFREEZE_BACKBONE_BLOCKS=4`

Existing `--resume` arg in train.py loads the model but resets optimizer; a
stage-2 launch would benefit from cosine warmup over the stage-2 LR
specifically.

```bash
# Stage 1: R001 baseline
SKIP_INSTALL=1 EPOCHS=15 PATIENCE=8 NUM_WORKERS=4 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh

# Stage 2: load stage 1 best.pt, unfreeze last 4 blocks
SKIP_INSTALL=1 EPOCHS=10 PATIENCE=5 NUM_WORKERS=4 \
  UNFREEZE_BACKBONE_BLOCKS=4 BACKBONE_LR_FACTOR=0.0167 \
  LEARNING_RATE=3e-4 WARMUP_EPOCHS=2 \
  RESUME=models/05_dinov2_lora_diffattn/output/00<stage1>/checkpoints/best.pt \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

---

## C — Higher input resolution (NOT pursued)

DINOv2 was pretrained at 518×518. Currently using 224 (2.3× downsample).
Higher resolution exposes fine-grained facial details (eye shape, jaw
structure) that may be averaged away at 224.

```bash
SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
  IMG_SIZE=448 BATCH_SIZE=2 GRAD_ACCUM=16 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

**Cost:** attention is O(n²) in number of tokens. 448×448 / 14×14 patches =
1024 tokens (vs 256 at 224). Memory + compute ~16× heavier. With batch=2 and
grad_ckpt, may fit in 12GB VRAM but tight. Time per epoch ~5-6h. Total run
~100-120h.

**Why deferred:** prohibitively expensive on consumer GPU for a single
experiment. Would only be worth doing if A and E both fail to move the
ceiling AND there's evidence that the bottleneck is feature granularity.

---

## D — 5-fold cross-validation on FIW (NOT pursued)

Run R001 (best M05 config so far) on 5 different family splits of FIW. If
val→test gap varies substantially across folds (e.g., −0.05 to −0.15),
confirms that the −0.10 gap is split-driven. If gap is constant, it's a
property of the model.

**Cost:** 5 × ~30h = ~150h.

**Why deferred:** scientifically the most rigorous test of the structural-gap
hypothesis, but expensive. Would be valuable if the M05 numbers go into a
publication and reviewers question the protocol. For the TCC, the three
consecutive runs (R001, R002, R003) showing gap ≈ −0.10 are already strong
evidence.

```bash
# Would require modifying split logic to accept a fold index 0-4
SEED=42 FOLD=0 bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
SEED=42 FOLD=1 ...  # etc.
```

Currently the dataset code uses `split_seed=42` with hard-coded splits. To
support k-fold, would need to:
- Add `--cv_fold N` and `--cv_total K` flags
- Modify `models/shared/dataset.py::_split_id_sets()` to compute splits via
  K-fold of family IDs instead of fixed proportions

---

## F — Stop iterating on M05, accept R001 as final (CONSIDER)

Pareto analysis says R001 is already a meaningful datapoint:

- Test AUC = 0.806 (within 0.044 of M02's 0.850)
- **Best TAR@FAR=0.01 of the entire project** (0.152 vs ~0.13 for M02/M03,
  0.06 for M06)
- 10× fewer trainable params than M02 (8.5M vs 86M)
- Per-relation accuracy 39-83% range (uniform enough to be useful)

Five M05 runs have already exhaustively demonstrated the ceiling at this
plasticity regime. The marginal information from a sixth run with the same
backbone is low.

**If F is the right call**, the time saved (~30h × N more runs) should be
spent on:

- **M07 (next architecture)** — the project benefits from another datapoint
  more than another M05 ablation. Candidates: ArcFace-pretrained backbone,
  CLIP image encoder, FaceNet, or a true multi-task model with auxiliary
  attribute heads (age, ethnicity).
- **5-fold CV on M02** — M02 R031 already has the best balanced results;
  cross-validation would solidify the comparison numbers for the TCC.
- **Cross-dataset eval** — train on FIW, test on KinFaceW. Both M02 and M05
  R001 trained checkpoints can be repurposed. Reveals which model
  generalizes better outside the training distribution.
- **TCC text writing** — at 5 model implementations + 7 runs documented +
  results consolidated, the data side is well past sufficient for the
  empirical chapter.

---

## Decision tree for R006+

If R006 (option A, contrastive loss) succeeds (Test AUC ≥ 0.83):

→ Investigate whether contrastive + partial unfreeze stacks (R007 = A + R004
  config combined).

If R006 fails or produces marginal gain (Test AUC 0.78-0.82):

→ The ceiling is genuinely the architecture + protocol combination. Move to
  F (stop iterating M05) and consider M07 with a different design philosophy.

If E (per-relation thresholds) lifts R001 accuracy by 3+ pp:

→ Apply same calibration to all M05 runs and report the calibrated numbers in
  the TCC. Doesn't change AUC narrative but improves the headline accuracy.

---

## Summary

| Option | Status | Cost | P(beat 0.812) | P(beat 0.85) | Notes |
|--------|--------|-----:|--------------:|-------------:|-------|
| A — Contrastive loss | **Pursued** (R006) | ~30h | 50% | 25% | Tests M02's loss recipe on M05 architecture |
| E — Per-relation thresholds | **Pursued** (post-hoc R001) | 30 min | n/a | n/a | Lifts accuracy/F1 only; AUC unchanged |
| B — Two-stage training | Deferred | ~25h | 60% | 15% | Avoids R005's underflow; needs --resume polish |
| C — 448×448 resolution | Deferred | ~120h | 30% | 10% | Heavy compute; only if A/E plateau |
| D — 5-fold CV | Deferred | ~150h | n/a | n/a | Validates gap structurally; doesn't move teto |
| F — Stop, ship R001 | Available | 0h | n/a | n/a | Pivot to M07 or text writing |

---

# Literature Review — DINOv2 for Face/Kinship Tasks

After R001-R006 plateaued at Test AUC ~0.81, conducted a literature review to
locate the recipe gap. Key finding: **M05 diverges from established DINOv2
face-task patterns in multiple dimensions simultaneously**, and **no
published work combines DINOv2 with kinship verification specifically**.

## What we found

### FRoundation paper (Boutros et al., 2024-2025)
[arxiv:2410.23831](https://arxiv.org/html/2410.23831v2) — "Are Foundation
Models Ready for Face Recognition?". Direct evaluation of DINOv2 for face
verification with the canonical fine-tuning recipe.

| Component | FRoundation (canonical) | M05 R001-R006 |
|---|---|---|
| Loss | **CosFace** (margin=0.3, scale=64) | BCE + contrastive + relation-CE |
| LoRA rank | **16** | 8 |
| LR initial | **1e-4** | 3e-4 |
| Batch size | **512** | 32 (effective) |
| Output token | **CLS token** → linear head | Patch tokens → DiffAttn → heads |
| LoRA targets | q, v projections | q, v ✓ |
| Image size | 224 ✓ | 224 ✓ |
| Epochs | 40 | 20 |

Results FRoundation reports for DINOv2 ViT-S:
- Frozen (linear probe): 64.70% verification accuracy on LFW/CALFW/CPLFW/CFP-FP/AgeDB30
- LoRA fine-tuned on 1k identities of CASIA-WebFace: **87.10%** (+22.4 pp)
- LoRA fine-tuned on 10k identities: **90.94%**

The recipe used is fundamentally different from M05's: **margin-based loss
instead of BCE+contrastive, double LoRA rank, 3× lower LR, simpler head
architecture**.

### FaCoR paper (M02's basis, 2023)
[arxiv:2304.04546](https://arxiv.org/html/2304.04546) — "Kinship
Representation Learning with Face Componential Relation". The paper that
inspired our M02 architecture **does not use DINOv2**:

> "We employ ArcFace (primary comparison baseline) and AdaFace
> (state-of-the-art face recognition model achieving best results)."

ArcFace and AdaFace are pre-trained with margin-based loss on face identity
datasets (MS-Celeb-1M). They produce embeddings already specialized for face
discrimination. **DINOv2 is generalist** — trained on 142M diverse images,
not face-specific. M02's 0.85 AUC ceiling came partly from ArcFace's
face-specific pretraining; ImageNet-ViT replacement (which is what M02 used)
still outperforms DINOv2 in our setup because of the loss + recipe match
with the architecture.

### Kinship state-of-the-art (Shadrikov 2020 and later)
[arxiv:2006.11739](https://arxiv.org/abs/2006.11739) explicitly notes:

> "Even the pre-trained ArcFace model performs much better than other
> competitors for kinship recognition."

Confirmation that **face-specific pretraining matters more than general
visual richness** for this task.

### The gap in literature

**No published paper combines DINOv2 + kinship verification** as of 2025.
M05 is exploring uncharted territory but **without the right recipe** — we
went with generic fine-tuning patterns (BCE + contrastive) instead of
adopting the face-specific recipe (CosFace/ArcFace) that the foundation
model literature has converged on.

---

## Why M02 outperforms M05 in our comparison

Combining the literature with our experimental data:

1. **Loss mismatch.** M05 uses BCE+contrastive — these don't apply margin
   penalties to angular distance, which is what face/kinship tasks reward.
   M02 used pure supervised contrastive with margin=0.3 — closer to
   CosFace philosophy.

2. **Backbone specialization.** DINOv2 is general; ArcFace/AdaFace are
   face-tuned. M02 used ImageNet-ViT but the cross-attention head was
   specifically designed around the face-pair geometry. M05's DiffAttn is
   more general-purpose attention.

3. **Hyperparameter calibration.** R001-R006 used LR=3e-4 (tuned for
   training-from-scratch heads) but the FRoundation recipe uses LR=1e-4 for
   adapter-based fine-tuning of foundation models. Higher LR + LoRA can
   overfit specific training patterns rather than adapt features cleanly.

---

# New options discovered through the literature review

The options below extend the original A-F catalog with directions
specifically informed by what the literature converged on.

## G — Apply FRoundation recipe verbatim (DERIVATIVE — not novel)

Replace M05's head entirely with FRoundation's: drop DiffAttn, drop relation
head, use DINOv2 CLS token → linear projection → CosFace classifier.
Hyperparameters per FRoundation: LoRA rank=16, LR=1e-4, margin=0.3,
scale=64, 40 epochs.

```bash
# Would require dedicated train.py path: --use_froundation_head
# - removes cross_attn_layers
# - replaces binary_head with CosFace head
# - replaces loss with CosFaceLoss
```

**Chance of beating M02:** 70-80% (proven recipe in face tasks).
**Novelty:** zero. Pure replication of FRoundation for kinship.
**Verdict:** not pursued — derivative work without contribution.

## H — Switch backbone to AdaFace/ArcFace (DERIVATIVE — not novel)

Replace DINOv2 with a pre-trained face recognition model (ArcFace or
AdaFace). This is what FaCoRNet did. We'd essentially be reproducing
FaCoRNet with our cross-attention variant.

**Chance of beating M02:** 75-85% (literature confirms).
**Novelty:** very low — direct replication of FaCoRNet.
**Verdict:** not pursued — would dilute M05's identity to "FaCoRNet
clone" without architectural contribution.

## I (REVISED) — M05 with face-recognition-aware loss + tuned hyperparams (MEDIUM novelty)

Keep M05's architecture (DINOv2 + LoRA + DiffAttn + relation head) but:

1. Replace BCE+contrastive with **CosFace-pair / ArcFace-pair loss**:
   apply margin penalty on `cosine(emb1, emb2)` for kin pairs.
2. LoRA rank 8 → **16** (matches FRoundation).
3. LR 3e-4 → **1e-4** (matches FRoundation).
4. Warmup 3 → 5 epochs.

```bash
# Requires ~40 lines to add ArcFacePairLoss class + flag wiring
LORA_RANK=16 LEARNING_RATE=1e-4 WARMUP_EPOCHS=5 \
LOSS=arcface_pair MARGIN=0.3 SCALE=64 \
EPOCHS=30 PATIENCE=12 \
bash run_pipeline.sh
```

**Why it's novel:** **Differential cross-attention has never been combined
with margin-based pair loss for kinship**. DiffAttn is from Ye et al. 2024
(LLM domain). FaCoRNet uses regular cross-attention. FRoundation uses
margin loss but with a simple CLS+linear head. M05 with this combination
would be the first such combination.

**Chance of beating M02:** 50-60%.
**Novelty:** medium-high (architectural component preserved + recipe
correction).
**Cost:** ~3h implementation + ~30h training.
**Risk:** low — both halves of the combination are individually validated;
this is testing whether they compose well.

## J — DINOv2 + ArcFace hybrid backbone (HIGH novelty)

**Architectural innovation, not present in literature.** Use both backbones
in parallel and fuse their outputs:

```
img1 → DINOv2 ViT-B/14 (frozen) → patch tokens (256, 768)
       ↓
img1 → ArcFace ResNet/ViT (frozen) → identity embedding (1, 512)
       ↓
       Project both to 512-d
       ↓
       Differential cross-attention BIDIRECTIONAL:
         (1) within face: dinov2 ↔ arcface  (intra-face fusion)
         (2) across pair: face1 ↔ face2     (kinship reasoning)
       ↓
       Pair embedding → CosFace-pair / ArcFace-pair loss
```

**Why it's novel:**
- M03 fused ConvNeXt + ViT (both ImageNet) — small benefit, both general.
- Combining a **self-supervised foundation model** with a **face-specific
  pretrained model** as complementary backbones for kinship is a
  combination not in any kinship paper found.
- Both backbones frozen, only fusion + heads trainable → ~10-12M trainable
  params, very efficient.
- Direct test of the hypothesis "DINOv2 lacks face-specificity" by
  injecting that specificity from ArcFace while keeping DINOv2's general
  visual richness.

**Chance of beating M02:** 60-70%. ArcFace alone reaches 0.85 AUC on FIW
in published work (Shadrikov 2020, FaCoRNet); adding DINOv2's complementary
features should provide marginal gain on hard pairs (extreme age
differences, twins, etc.).

**Novelty:** high — architectural contribution, no published precedent.

**Cost:** 1 day implementation (load ArcFace via `insightface` lib or
reuse the M02 R031 trained checkpoint as the face-specific backbone),
~30h training.

**Risk:** medium — if ArcFace dominates the fusion, it could degrade to
"M02 with DINOv2 as auxiliary" and the novelty weakens. **Mitigation:**
gating/attention weights on the fusion module to force balanced use of
both backbones, plus an ablation run with each backbone alone for the
TCC.

**Hardware:** both backbones frozen → ~5-7 GB VRAM, fits comfortably.

## L — Self-supervised pair pretraining on FIW (HIGHEST novelty, highest risk)

**Most novel direction.** Run an additional self-supervised pretraining
phase BEFORE supervised kinship training:

- **Stage 1 (~12h):** On FIW, sample image pairs where both are from the
  same family (positive views) vs different families (negative views).
  Apply DINO/iBOT objective adapted to pairs: student network learns to
  predict teacher's embedding for the positive pair, treating it as a
  multi-crop scenario.
- **Stage 2 (~30h):** Take the pair-pretrained DINOv2+LoRA checkpoint
  and fine-tune for binary kinship verification (current M05 supervised
  setup).

**Why it's novel:**
- DINOv2's self-supervised paradigm has been adapted to many domains, but
  **pair-aware self-supervised pretraining for kinship doesn't exist**.
- Treats family membership as a self-supervised pretext task — the model
  learns face-similarity and family-similarity simultaneously without
  labels for the specific kinship classes.
- Could reveal whether the supervised kinship signal is enough or whether
  additional pretraining helps.

**Chance of beating M02:** 40-50%. Real risk: the model might learn
shortcuts (background, photo metadata, ethnicity) that don't transfer to
kinship verification.

**Novelty:** very high — most publishable direction.

**Cost:** ~3 days implementation (DINO/iBOT objective adaptation + pair
sampler), ~50h total training (stage 1 + stage 2).

**Risk:** high — three failure modes: (a) shortcut learning, (b) compute
budget overrun, (c) implementation bugs in the multi-crop adapter.

---

## Decision matrix — novelty + chance to beat M02

For "test something at least somewhat novel that could beat M02":

| Option | Novelty | P(beat 0.85) | Cost | Recommended? |
|---|:---:|---:|:---:|:---:|
| **J — DINOv2 + ArcFace hybrid** | High | **60-70%** | 1 day + 30h | **Yes — first** |
| **I — DiffAttn + margin loss** | Medium | 50-60% | 3h + 30h | Yes — if J fails |
| **L — SSL pair pretraining** | Very high | 40-50% | 3 days + 50h | If time permits |
| G — FRoundation verbatim | Zero | 70-80% | 1 day + 30h | No — derivative |
| H — AdaFace backbone | Very low | 75-85% | 1 day + 30h | No — FaCoRNet clone |

### Top recommendation: Option J as R007

**Why:** attacks the diagnosed bottleneck (DINOv2 lacks face-specificity)
through architectural innovation rather than recipe replication. Generates
a publishable contribution. Risk is controlled — both backbones already
exist and are validated. Implementation is contained to a new model file
+ a fusion module. If it fails, fall back to Option I (lower-cost variant
that preserves M05's identity).

### If J wins, follow-up R008 = Option I

Apply the loss/recipe corrections to the same hybrid setup. Likely the
combined optimization reaches Test AUC > 0.86 — a clean SOTA-style
contribution for the TCC.

### If J fails, fall back to I

I is the safer-but-less-novel direction. Still a contribution because of
DiffAttn + margin-based loss combination, just less ambitious.

---

## Sources

- [FRoundation: Are Foundation Models Ready for Face Recognition?
  (arxiv:2410.23831)](https://arxiv.org/html/2410.23831v2)
- [Kinship Representation Learning with Face Componential Relation —
  FaCoR (arxiv:2304.04546)](https://arxiv.org/html/2304.04546)
- [Achieving Better Kinship Recognition Through Better Baseline
  (Shadrikov 2020, arxiv:2006.11739)](https://arxiv.org/abs/2006.11739)
- [Deep learning-based kinship verification: a comprehensive survey
  (Springer 2025)](https://link.springer.com/article/10.1007/s13748-025-00402-y)
- [Facial Kinship Verification: A Comprehensive Review and Outlook
  (IJCV 2022)](https://link.springer.com/article/10.1007/s11263-022-01605-9)
- [DINOv2 Meta AI page](https://dinov2.metademolab.com/)
- [Micro-Expression Recognition via LoRA-Enhanced DinoV2
  (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12846233/)
