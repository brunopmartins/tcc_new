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

| | **Run 001** |
|---|---|
| **Date** | 2026-05-13 |
| **Purpose** | Phase 1 baseline — frozen backbone, BCE only, random negs |
| **Status** | **Stopped manually at ep 7** (4 epochs below peak) |
| **Best Val AUC** | **0.8351 (ep 3)** |
| **Test ROC-AUC** | **0.7464** |
| **Test Accuracy** | 68.0 % |
| **Val→test gap** | **-0.089** (smallest in AdaFace family) |
| **Notes** | Architectural hypothesis directionally validated (smallest gap of any AdaFace-based model) but absolute Test AUC below M09 R001 (-0.052) because frozen backbone caps Val AUC ceiling. See [run-001.md](run-001.md). |

---

## Test metrics

| Metric | **Run 001** |
|---|---:|
| **Test ROC-AUC** | **0.7464** |
| Test Accuracy | 68.00 % |
| Test F1 | 0.6150 |
| Test Precision | 72.60 % |
| Test Recall | 53.34 % |
| Avg Precision | 0.7323 |
| TAR@FAR=0.01 | 10.06 % |
| TAR@FAR=0.1 | 37.86 % |
| Best Val ROC-AUC | 0.8351 (ep 3) |
| Best Val Accuracy | 75.4 % (ep 3) |
| **Val→Test AUC gap** | **-0.089** |

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
| M02 R031 | **0.850** | 74.4 % | ViT-B/16 ImageNet (full FT) | FaCoR top-only + cosine_contrastive (works on ViT) | project best |
| M05 R007 | 0.810 | — | hybrid (DINOv2 + LoRA + diff-attn) | — | partial freeze |
| **M09 R001** | **0.7982** | 71.9 % | AdaFace IR-101 (full FT) | Multistage + BCE + classifier head + random negs | **best AdaFace-based** |
| M09 R002 | 0.7824 | 71.6 % | M09 R001 + balanced sampler | balanced positives | val→test gap widened |
| M11 R001 v4 | 0.7707 | 70.6 % | M09 R001 + `relation_matched` negs | hard negs on M09 stack | val→test gap widened most (-0.128) |
| M10 R003 | 0.7478 | 70.6 % | AdaFace IR-101 (full FT) | FaCoR top-only + BCE + classifier head | val→test gap -0.140 |
| **M12 R001** | **0.7464** | 68.0 % | **AdaFace IR-101 (FROZEN)** | **5 region tokens + cross-attn + gate** | **smallest val→test gap (-0.089), lowest Val AUC ceiling** |
| M06 R001 | 0.776 | 69.8 % | ViT-B/16 (frozen) + retrieval | retrieval + cross-attn | best frozen-encoder |
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

## Conclusion (as of R001)

R001 establishes the M12 baseline. The result is **directionally
interesting but absolutely underwhelming**:

1. **Val→test gap is the smallest in the AdaFace family** (-0.089).
   Three independent comparisons confirm: M09 R001 -0.094, M11 v4
   -0.128, M12 R001 -0.089. The architectural change (frozen backbone,
   region tokens) does reduce family memorization.

2. **Val AUC ceiling is too low for the smaller gap to pay off.**
   Peak 0.8351 vs M09 R001's 0.8919. The frozen backbone simply lacks
   capacity to push Val higher.

3. **Per-class distribution at peak is balanced** (75-92 % range
   across all 11 classes in val). But after test-time threshold and
   held-out families, the per-class accuracies regress to roughly the
   same pattern as other AdaFace-based models.

4. **The whole experiment cycle was ~3 h** (vs ~8-10 h for full-FT
   models). Cheap to iterate on.

**Next directions:**

- **R002 — Phase 2 of the proposal: unfreeze last IR-101 stage.**
  Lifts the Val AUC ceiling while keeping most of the backbone fixed.
  This is the highest-priority next experiment.
- **R003 — Phase 3: full fine-tune.** Probably reintroduces family
  memorization (M09 R001 / M11 v4 territory). Lower priority.
- **R004 — Phase 4: supervised contrastive loss as auxiliary** (per
  proposal §28, with λ=0.05).
- **R005 — Phase 5: relation-type auxiliary head.**
- **R006 — hard negatives via `relation_matched`.** M11 v4 already
  showed this hurts test AUC on full-FT; whether it helps on M12's
  frozen backbone is unknown.

The proposal §38 explicitly mandates this experimental sequence
(Fase 1 → Fase 6). R001 is Fase 1 and produced its expected result:
a slow but well-behaved baseline that doesn't beat the full-FT
incumbent but proves the architecture is viable.

**On the project headline metric (Test AUC), R001 is below M09 R001
(0.7464 vs 0.7982).** The architecture is promising for *generalization
quality* (gap, balance) but needs Phase 2+ to lift the ceiling.

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
