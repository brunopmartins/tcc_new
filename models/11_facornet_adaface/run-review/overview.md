# Overview — Model 11: AdaFace + FaCoRNet treatments

**Model:** 11 — AdaFace IR-101 (WebFace4M) + FaCoRNet hard negatives
(`relation_matched`) on M09 R001's stack (multi-stage cross-attn + BCE
classifier head). Originally framed as a full FaCoRNet recipe
implementation but reduced after a project-side loss bug was
discovered. See [discussao.md](../discussao.md).
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset:** FIW (FIW_aligned 224×224 → resized to 112 at load time)

---

## Hypothesis recap

The original M11 design was to apply the full FaCoRNet recipe on top of
M10's architecture: `relation_guided` (attention-driven contrastive)
loss + `relation_matched` hard negatives + temperature 0.07. Three
attempts (v1 with temp 0.07, v2 with temp 0.3, v3 with
`cosine_contrastive`) collapsed within 1-2 epochs.

The root cause turned out to be a project-side bug: both
`CosineContrastiveLoss` and `RelationGuidedContrastiveLoss` in
`models/shared/losses.py` *ignore the batch labels*. Every
`(emb1_i, emb2_i)` correspondence is treated as a positive in the
InfoNCE numerator regardless of whether the dataset labelled the pair
kin or non-kin. Combined with `relation_matched` hard negatives, the
model was actively trained to *pull non-kin pairs together*, producing
a `predict-all-kin` regime (Val Acc trapped at 50 %, per-relation 1.000
uniform).

After the bug discovery, M11 R001 was re-scoped to the **single
FaCoRNet treatment that doesn't require a loss rewrite**:
`relation_matched` hard negatives on top of the proven M09 R001 stack
(BCE classifier head + multistage cross-attention). This is v4.

The narrower hypothesis tested by M11 R001 v4: **"Does adding
`relation_matched` hard negatives to M09 R001's stack improve Test
AUC?"**

Result: **No.** Val AUC improved (0.8987 > 0.8919 M09 R001), but Test
AUC regressed (0.7707 < 0.7982 M09 R001). Val→test gap widened to
-0.128 (vs M09 R001's -0.094).

## Configuration baseline (post-bug, v4 defaults)

| Knob | Default |
|---|---|
| Backbone | AdaFace IR-101 WebFace4M, full fine-tune |
| Architecture | Multi-stage cross-attn (stages 3 + 4) |
| Head | BCE classifier (MLP on `[emb1, emb2, diff, product]`) |
| Loss | bce |
| Train negative strategy | `relation_matched` |
| Eval negative strategy | `relation_matched` |
| Dropout | 0.2 |
| LR (peak) | 5e-6, warmup 5, cosine, min_lr 1e-7 |
| Batch | 4 × grad-accum 8 (eff 32) |
| Patience | 50 |
| Epochs (max) | 100 |

`relation_guided` and `cosine_contrastive` are **not used** for M11 due
to the label-ignoring bug. A future faithful FaCoRNet implementation
would require rewriting those losses (Caminho A of `discussao.md`).

## Run table

| | Run 001 v1 | Run 001 v2 | Run 001 v3 | **Run 001 v4** |
|---|---|---|---|---|
| **Date** | 2026-05-12 | 2026-05-13 | 2026-05-13 | 2026-05-13 |
| **Loss** | relation_guided | relation_guided | cosine_contrastive | **bce + classifier head** |
| **Temperature** | 0.07 | 0.3 | 0.3 | (n/a) |
| **Neg strategy** | relation_matched | relation_matched | relation_matched | **relation_matched** |
| **Multistage** | yes | yes | yes | **yes** |
| **Classifier head** | no | no | no | **yes** |
| **Status** | KILLED ep 2 | KILLED ep 2 | KILLED ep 2 | **Stopped manually at ep 13** |
| **Outcome** | predict-all-kin collapse | predict-all-kin collapse | predict-all-kin collapse | Val AUC peak 0.8987 / Test AUC 0.7707 |

---

## Test metrics (v4)

| Metric | M09 R001 | M09 R002 | **M11 R001 v4** |
|---|---:|---:|---:|
| **Best Val ROC-AUC** | 0.8919 | 0.8894 | **0.8987** (highest AdaFace-based in project) |
| **Test ROC-AUC** | **0.7982** | 0.7824 | **0.7707** |
| Val→test AUC gap | -0.094 | -0.107 | **-0.128** (largest) |
| Test Accuracy | 70.55 % | 71.55 % | 70.57 % |
| Test F1 | 0.6262 | 0.6518 | 0.6333 |
| Test Precision | 79.92 % | 78.78 % | 78.56 % |
| Test Recall | 51.48 % | 55.58 % | 53.05 % |
| TAR@FAR=0.01 | 10.53 % | 7.76 % | 10.98 % |

---

## Issues Log

| # | Severity | Status | Title | Notes |
|---|----------|--------|-------|-------|
| I-01 | Critical | Closed by user (discussao.md) | `cosine_contrastive` and `relation_guided` losses ignore labels | Both treat every `(emb1_i, emb2_i)` correspondence as positive in InfoNCE numerator regardless of `label`. Caused v1/v2/v3 all-kin collapse. Pre-existing project bug, not specific to M11. M10 R002 misdiagnosed at the time. |
| I-02 | High   | Open | M11 v4 Test AUC -0.028 vs M09 R001 | Adding `relation_matched` hard negs on top of M09 R001's stack RAISED Val AUC (+0.007) but LOWERED Test AUC (-0.028). Val→test gap widened from -0.094 to -0.128. Hard negs over-fit the val-pool distribution. |
| I-03 | High   | Open | Sibling classes catastrophically regressed in v4 | Test bb -7.6 pp, ss -9.0 pp, sibs -14.5 pp vs M09 R001. `relation_matched` draws negatives from same-role pairs in different families, which over-trained the model to discriminate sibling-like pairs on the val pool. Doesn't transfer to held-out families. |
| I-04 | Info   | Workaround applied | Checkpoint missing `model_config` when training killed mid-pipeline | Same pattern as M09 R001/R002 and M10 R003 — best.pt patched manually. Long-term fix: write model_config in trainer's `save_checkpoint`. |
| I-05 | Info   | Open | Faithful FaCoRNet implementation pending | The bug discovery means we never tested the actual `relation_guided` Rel-Guide formulation faithfully. Caminho A from `discussao.md` requires positives-only batches and `M(beta)/s` with s=500. Deferred. |

---

## Comparison with other models (FIW, Test ROC-AUC)

| Model | Test ROC-AUC | Test Acc | Backbone | Recipe | Notes |
|---|---:|---:|---|---|---|
| M02 R031 | **0.850** | 74.4 % | ViT-B/16 ImageNet (full FT) | cosine_contrastive (works on ViT) | project best |
| M05 R007 | 0.810 | — | hybrid (DINOv2 + LoRA + diff-attn) | — | partial freeze |
| **M09 R001** | **0.7982** | 71.9 % | AdaFace IR-101 (full FT) + multistage | BCE + classifier head + random negs | best AdaFace-based |
| M09 R002 | 0.7824 | 71.6 % | M09 R001 + balanced sampler | balanced positives | val→test gap widened |
| **M11 R001 v4** | **0.7707** | 70.6 % | M09 R001 + `relation_matched` negs | hard negs | **gap widened most** |
| M10 R003 | 0.7478 | 70.6 % | AdaFace IR-101 (full FT) + top-only | BCE + classifier head + random negs | val→test gap -0.140 |
| M06 R001 | 0.776 | 69.8 % | ViT-B/16 (frozen) + retrieval | retrieval + cross-attn | best frozen-encoder |
| M08 R001 | 0.693 | 60.8 % | ArcFace IR-100 (frozen) + retrieval | retrieval + cross-attn | anti-kinship trap |

---

## Per-relation accuracy comparison (test)

| Relation | M09 R001 | M09 R002 | **M11 v4** | M02 R031 |
|----------|---------:|---------:|-----------:|---------:|
| bb | 64.2 % | 58.1 % | 56.6 % | 95.5 % |
| ss | 62.9 % | 59.9 % | 53.9 % | 94.7 % |
| sibs | 64.5 % | 59.8 % | **50.0 %** | 94.9 % |
| fs | 59.1 % | 56.7 % | 54.1 % | 95.3 % |
| ms | 57.3 % | 55.5 % | 51.5 % | 93.9 % |
| md | 53.9 % | 52.6 % | 53.1 % | 94.4 % |
| fd | 63.6 % | 59.3 % | 61.7 % | 91.7 % |
| **gfgd** | 31.2 % | 50.7 % | **37.7 %** | 89.9 % |
| **gfgs** | 30.6 % | 33.7 % | **36.7 %** | 95.9 % |
| gmgd | 31.7 % | 33.3 % | 22.0 % | 91.1 % |
| gmgs | 37.2 % | 37.2 % | 28.9 % | 88.4 % |
| non-kin | 84.7 % | 86.2 % | 86.7 % | (n/a) |

`relation_matched` negs improved 2 grandfather classes but hurt all
sibling classes and both grandmother classes. Pattern not uniform.

---

## Conclusion (as of R001 v4)

**Two consistent findings from M11 R001 v4:**

1. **The project's `cosine_contrastive` and `relation_guided` losses
   are broken** for mixed kin/non-kin batches (they ignore labels). This
   is documented in `discussao.md` and was the cause of v1/v2/v3
   degenerate collapse. It also retroactively explains M10 R002's
   instability. Pre-existing project bug, not M11-specific.

2. **`relation_matched` hard negatives on M09 R001's stack RAISE Val
   AUC but LOWER Test AUC.** Same pattern M09 R002's balanced sampling
   showed. Both interventions over-fit the val-pool family distribution
   and fail to transfer to held-out test families.

The M09 R001 stack (AdaFace full FT + multistage cross-attn + BCE +
classifier head + random negatives) is the local maximum we've found
for the AdaFace family of models in this pipeline. Further FaCoRNet-
inspired sophistication (hard negs, balanced sampling, label-aware
contrastive variants) might still help, but the immediate evidence
says these interventions are net negative.

**Next direction:** M12 (RGCK-Net Phase 1) shifts to a different
architecture entirely — frozen AdaFace backbone with region tokens and
regional gating. This tests whether *less* AdaFace influence (rather
than *more* sophisticated training of full-FT AdaFace) is the path
forward.

A faithful FaCoRNet Rel-Guide implementation (Caminho A) is also
deferred pending project priorities.
