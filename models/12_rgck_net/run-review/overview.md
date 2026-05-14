# Overview — Model 12: RGCK-Net (Region-Guided Cross Kinship Network)

**Model:** 12 — implementation of the proposal in
`proposta_rgck_net_kinship.md`. AdaFace IR-101 (WebFace4M) frozen +
5 fixed-region tokens + bidirectional cross-region attention + sigmoid
regional gating + BCE classifier head.
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset:** FIW (FIW_aligned 224×224 native — model crops 5 regions
internally, each resized to 112×112 for AdaFace)

---

## Hypothesis recap

After three M11 attempts and M09 R002 all showed that *sophisticating
the training distribution* over M09 R001's full-FT AdaFace stack
(balanced sampling, hard negatives, attention-driven contrastive)
consistently raised Val AUC but lowered Test AUC, M12 tested an
**architectural** direction:

> Does **reducing AdaFace's influence** (frozen backbone, lightweight
> region tokens) reduce family memorization enough to close the
> val→test gap, even at a lower Val AUC ceiling?

The structure:
- AdaFace IR-101 backbone is **frozen** (5.6 M / 70.7 M = 7.9 % trainable)
- 5 anatomical region tokens (global, eyes, nose, mouth, jaw) via fixed
  bounding boxes in 224×224 aligned-face coordinates, then resized to
  112×112 for AdaFace
- 1-layer × 4-head bidirectional cross-region attention (5×5 attention
  matrix — orders of magnitude cheaper than M09's 196×196 + 49×49)
- Per-region sigmoid gating (interpretable weights)
- 3-layer MLP classifier head on `[gA, gB, |diff|, prod, per-rel sims, weights, weighted score]`
- BCE on classifier logit

## Configuration baseline (R001 defaults)

| Knob | Default |
|---|---|
| Backbone | AdaFace IR-101 WebFace4M, **frozen** |
| Regions | 5: global, eyes, nose, mouth, jaw |
| Cross-attn layers | 1 |
| Cross-attn heads | 4 |
| Gate hidden dim | 128 |
| Classifier hidden dim | 512 |
| Loss | bce on classifier head |
| LR | 1e-4 peak (head-only), warmup 5, cosine, min_lr 1e-6 |
| Weight decay | 1e-4 |
| Dropout | 0.2 |
| Batch | 8 × grad-accum 4 (eff 32) |
| Train neg strategy | random |
| Eval neg strategy | random |
| Patience | 50 |
| Epochs (max) | 100 |
| Img size | 224 (native FIW_aligned) |
| Trainable params | **5,589,762 / 70,740,674 (7.90%)** |
| Time/epoch | ~20 min (half of full-FT models) |

## Run table

| | Run 001 | **Run 002** | Run 003 |
|---|---|---|---|
| **Date** | 2026-05-13 | 2026-05-13 | 2026-05-14 |
| **Phase** | 1 (frozen) | **2 (partial unfreeze)** | 4 (R002 + SupCon λ=0.05) |
| **Trainable** | 5.6 M (7.9 %) | 31.6 M (44.6 %) | 31.6 M (44.6 %) |
| **LR** | 1e-4 | 1e-5 | 1e-5 |
| **Status** | Stopped at ep 7 | Stopped at ep 7 | Stopped at ep 7 |
| **Best Val AUC** | 0.8351 (ep 3) | **0.9323 (ep 4)** — project max | 0.9306 (ep 4) |
| **Test ROC-AUC** | 0.7464 | **0.8564** ⭐ **PROJECT HEADLINE** | 0.8510 |
| **Test Accuracy** | 68.0 % | 76.8 % | 76.4 % |
| **Val→test gap** | -0.089 | -0.076 | -0.080 |
| **Notes** | Phase 1 frozen capped ceiling | **Phase 2 partial unfreeze beats M02 R031**. [run-002.md](run-002.md) | SupCon aux REJECTED — improves siblings but regresses grandparent, net Test AUC -0.005. [run-003.md](run-003.md) |

---

## Test metrics

| Metric | M02 R031 (prior best) | Run 001 (Phase 1) | **Run 002 (Phase 2)** | Run 003 (+SupCon) |
|---|---:|---:|---:|---:|
| **Test ROC-AUC** | 0.850 | 0.7464 | **0.8564** ⭐ | 0.8510 |
| Test Accuracy | 74.4 % | 68.00 % | **76.79 %** ⭐ | 76.37 % |
| Test Balanced Acc | 75.2 % | 67.41 % | **76.48 %** | 76.09 % |
| Test F1 | 0.779 | 0.6150 | 0.7402 | 0.7373 |
| Test Precision | 66.5 % | 72.60 % | **79.82 %** ⭐ | 78.88 % |
| Test Recall | 94.1 % | 53.34 % | 69.00 % | 69.22 % |
| **Avg Precision** | 0.817 | 0.7323 | **0.8389** ⭐ | 0.8305 |
| **TAR@FAR=0.001** | 2.5 % | 2.36 % | **4.18 %** ⭐ | 2.67 % |
| **TAR@FAR=0.01** | 14.0 % | 10.06 % | **17.58 %** ⭐ | 16.57 % |
| **TAR@FAR=0.1** | 49.9 % | 37.86 % | **57.11 %** ⭐ | 55.43 % |
| Best Val ROC-AUC | 0.881 | 0.8351 | **0.9323** | 0.9306 |
| Best Val Accuracy | 76.6 % | 75.4 % | 85.5 % | 85.7 % |
| **Val→Test AUC gap** | -0.031 | -0.089 | -0.076 | -0.080 |

⭐ = Run 002 wins on every threshold-invariant metric — remains the **project headline**.

⭐ = M12 R002 wins on **all** threshold-invariant metrics (AUC, Avg Precision, all three TAR@FAR levels).

---

## Issues Log

| # | Severity | Status | Title | Notes |
|---|----------|--------|-------|-------|
| I-01 | Medium | Partial | Frozen backbone caps Val AUC at ~0.835 | Peak Val AUC 0.8351 vs M09 R001's 0.8919. Architectural simplicity comes at the cost of representational capacity. Proposal Phase 2 (unfreeze last IR-101 stage) could lift this. |
| I-02 | Medium | Open | Test AUC 0.7464 below M09 R001's 0.7982 | The smallest val→test gap (-0.089) doesn't compensate for the lower ceiling. Net Test AUC -0.052 vs M09 R001. |
| I-03 | Info | Open | No `evaluate.py` for M12 | Only `test.py` exists. No ROC plot, confusion matrix, or attention visualisation generated. Can be added later if needed; the regional gating weights are the most interesting visual artefact for this architecture. |
| I-04 | Info | Workaround applied | `model_config` not saved when training killed mid-pipeline | Same pattern as M09/M10/M11. best.pt patched manually before test.py could rebuild the model. |
| I-05 | Info | Open | Region tokenizer re-runs AdaFace 5× per face | Phase 1 uses Strategy 2 from the proposal (`recortar regiões + backbone`). Phase 2 should consider Strategy 1 (ROI Align on a single feature-map pass), which would halve training time further. |

---

## Comparison with other models (FIW, Test ROC-AUC)

| Model | Test ROC-AUC | Test Acc | Backbone | Architecture / Recipe | Notes |
|---|---:|---:|---|---|---|
| **M12 R002** | **0.8564** ⭐ | **76.8 %** | **AdaFace IR-101 (stages 1-3 frozen, stage 4 unfrozen)** | **5 region tokens + cross-attn + gate, BCE, LR 1e-5** | **NEW PROJECT HEADLINE** |
| M02 R031 | 0.850 | 74.4 % | ViT-B/16 ImageNet (full FT) | FaCoR top-only + cosine_contrastive (works on ViT) | prior project best |
| M05 R007 | 0.810 | — | hybrid (DINOv2 + LoRA + diff-attn) | — | partial freeze |
| M09 R001 | 0.7982 | 71.9 % | AdaFace IR-101 (full FT) | Multistage + BCE + classifier head + random negs | best AdaFace full-FT |
| M09 R002 | 0.7824 | 71.6 % | M09 R001 + balanced sampler | balanced positives | val→test gap widened |
| M06 R001 | 0.776 | 69.8 % | ViT-B/16 (frozen) + retrieval | retrieval + cross-attn | best frozen-ViT |
| M11 R001 v4 | 0.7707 | 70.6 % | M09 R001 + `relation_matched` negs | hard negs on M09 stack | val→test gap widened most (-0.128) |
| M10 R003 | 0.7478 | 70.6 % | AdaFace IR-101 (full FT) | FaCoR top-only + BCE + classifier head | val→test gap -0.140 |
| M12 R001 | 0.7464 | 68.0 % | AdaFace IR-101 (FROZEN) | 5 region tokens + cross-attn + gate | smallest val→test gap (-0.089), lowest Val AUC ceiling |
| M08 R001 | 0.693 | 60.8 % | ArcFace IR-100 (frozen) + retrieval | retrieval + cross-attn | anti-kinship trap |

---

## Per-relation accuracy comparison (test)

| Relation | M09 R001 | M09 R002 | M11 v4 | **M12 R001** |
|----------|---------:|---------:|-------:|-------------:|
| bb | 64.2 % | 58.1 % | 56.6 % | 58.9 % |
| ss | 62.9 % | 59.9 % | 53.9 % | 57.1 % |
| sibs | 64.5 % | 59.8 % | 50.0 % | 61.5 % |
| fs | 59.1 % | 56.7 % | 54.1 % | 56.3 % |
| ms | 57.3 % | 55.5 % | 51.5 % | 51.6 % |
| md | 53.9 % | 52.6 % | 53.1 % | 54.1 % |
| fd | 63.6 % | 59.3 % | 61.7 % | **52.0 %** |
| gfgd | 31.2 % | 50.7 % | 37.7 % | 28.3 % |
| gfgs | 30.6 % | 33.7 % | 36.7 % | 26.5 % |
| **gmgd** | 31.7 % | 33.3 % | 22.0 % | **36.6 %** |
| gmgs | 37.2 % | 37.2 % | 28.9 % | 33.1 % |
| non-kin | 84.7 % | 86.2 % | 86.7 % | 81.5 % |

`md` and `gmgd` improved over M09 R001. `fd` regressed catastrophically (-11.6 pp). Other classes 3-6 pp below.

---

## Conclusion (as of R002)

**R002 is the new project headline. M02 R031's reign as best-in-project
(Test AUC 0.850, held since the early ViT experiments) is over.**

Key findings:

1. **Test AUC 0.8564 beats M02 R031 (0.850) by +0.006.** Direct ranking
   improvement, threshold-invariant. All three TAR@FAR levels and Avg
   Precision also exceed M02 R031.

2. **The single architectural change between R001 and R002** — unfreezing
   stage 4 + output_layer — lifted Test AUC by +0.110 (from 0.7464 to
   0.8564). The Phase 1 capacity bottleneck was real and severe.

3. **Val→test gap is -0.076** — wider than R001's -0.089 (the model is
   somewhat more capable of memorizing val families) but narrower than
   every other AdaFace full-FT model (M09 R001 -0.094, M11 v4 -0.128,
   M10 R003 -0.140). The trade-off is favourable: slightly more
   memorization, much more discrimination, net positive.

4. **The recipe stack that won:**
   - AdaFace IR-101 backbone
   - Stages 1-3 frozen, stage 4 (body[46:49]) + output_layer trainable
   - 5 region tokens (global, eyes, nose, mouth, jaw — fixed coords)
   - 1-layer × 4-head bidirectional cross-region attention
   - Sigmoid regional gating
   - 3-layer MLP classifier head over `[gA, gB, |diff|, prod, sims, weights, score]`
   - BCE loss on classifier logit
   - LR 1e-5 (10× lower than R001's 1e-4)
   - Random negatives, no auxiliary losses, no hard negative mining

5. **Two FaCoRNet-inspired interventions (balanced sampling in M09 R002,
   hard negatives in M11 v4) consistently hurt Test AUC** despite
   raising Val AUC. M12 R002 found a different path: less aggressive
   training distribution, but better architectural fit to the kinship
   problem.

### What R001 already validated, still standing

R001 (Phase 1, fully frozen) had Val→test gap -0.089 — smallest in the
AdaFace family. R002 keeps most of that gap benefit (-0.076) while
adding the capacity needed to raise the absolute Val AUC ceiling. The
Phase 1 → Phase 2 progression validated the proposal's experimental
sequence design.

### Open issues (still tracked from R001)

| # | Severity | Status | Title | Notes |
|---|----------|--------|-------|-------|
| I-01 | Medium | **Closed in R002** | Frozen backbone caps Val AUC at ~0.835 | R002 partial unfreeze lifts peak to 0.9323 |
| I-02 | **Closed in R002** | M12 Test AUC below M09 R001 | R002 Test AUC 0.8564 beats M09 R001 (0.7982) by +0.058 |
| I-03 | Info | Open | No `evaluate.py` for M12 | Still applies — only `test.py` exists. Visualisations (ROC, CM, attention maps) and especially per-region gate weights would be valuable. |
| I-04 | Info | Workaround applied | `model_config` not saved when training killed mid-pipeline | Same pattern as M09/M10/M11 — best.pt patched manually before test.py rebuild. |
| I-05 | Info | Open | Region tokenizer re-runs AdaFace 5× per face | Still using Strategy 2 (recortar regiões + backbone). Strategy 1 (ROI Align on shared feature map) would halve training time and may add the missing 1-2 points of Test AUC. |
| I-06 | New, Info | Open | Per-relation grandparent accuracies still 37-52 % at threshold 0.5 | gfgd 52.2 %, gmgs 44.6 %, gfgs 39.8 %, gmgd 36.6 %. Better than every other M*/AdaFace model in the project but still well below M02 R031 (88-96 %) at its threshold-0.9 operating point. Relation-conditional auxiliary head (proposal Phase 5) could help. |

### Next directions

- **R003 — Phase 4: supervised contrastive auxiliary loss** (λ=0.05).
  Plausible incremental gain on top of R002. Main risk: Val→test gap
  widening.
- **R004 — Phase 5: relation-type auxiliary head.** Could specifically
  improve the grandparent classes which still lag.
- **R005 — Phase 6: hard negatives via `relation_matched`.** M11 v4
  showed this hurts on full FT; M12 has partial FT — unknown effect.
  Lower priority given M11 v4's negative result.
- **R006 — Phase 3: full fine-tune.** Almost certainly regressive
  (the M11 v4 lesson). Skip unless other phases reveal a need.
- **R007 — Architecture switch to ROI Align** (proposal §15 Strategy 1):
  halves training time, may add small Test AUC boost.

The proposal's experimental sequence (§38) is largely vindicated. R002
is Phase 2 done well.

---

## Architectural notes

- **Fixed regions in 224×224 aligned-face coords:**
  - `global`: (0:224, 0:224) — whole face
  - `eyes`:   (40:100, 20:204) — top-third horizontal strip
  - `nose`:   (80:150, 70:154) — central
  - `mouth`:  (140:185, 50:174)
  - `jaw`:    (170:220, 20:204) — bottom strip
  Each crop is resized to 112×112 for AdaFace.

- **Cross-region adapter:** verbatim MultiHeadAttention with
  `batch_first=True`, both directions wired in a `ModuleDict` per layer.
  1 layer × 4 heads × 512-d. The bidirectional pass keeps tokens_A and
  tokens_B symmetric.

- **Regional gate:** MLP over `[rA, rB, |rA-rB|, rA*rB]` → 1 logit per
  region → sigmoid. Outputs (B, K) weights in [0, 1]. Sigmoid (not
  softmax) per proposal §22 — multiple regions can be salient at once.

- **Classifier head input:** `[gA(512), gB(512), |diff|(512), prod(512), per-rel sims(5), per-rel weights(5), regional_score(1)]` = 2049-d.
  Three Linear/BatchNorm/GELU layers down to 1 logit.

- **Output tuple:** `(logit, weights, attn_map)`. Logit is consumed by
  the BCE loss; weights are for interpretability; attn_map is the last
  cross-region attention map (B, 4, 5, 5).
