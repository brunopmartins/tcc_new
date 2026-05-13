# Overview — Model 09: AdaFace IR-101 + Multi-Stage Cross-Attention

**Model:** 09 — AdaFace IR-101 (WebFace4M) + bidirectional FaCoR
cross-attention **after stage 3 AND stage 4** of the backbone + BCE
classifier head over `[emb1, emb2, diff, product]`
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset:** FIW (FIW_aligned 224×224 → resized to 112 at load time)

This file aggregates per-run results. Individual run details land in
`run-NNN.md` files alongside this overview.

---

## Hypothesis recap

M10 R003 established the AdaFace IR-101 + FaCoR top-only ceiling: Val AUC
0.8875 but **Test AUC 0.7478** with -0.140 val→test gap, caused by the
identity-cluster pretrain memorizing the training family pool.

M09 ports the **SAI/CI³Former "interaction-inside-the-backbone" idea**
to the same AdaFace backbone: bidirectional FaCoR cross-attention runs
after stage 3 (196 × 256-d tokens) AND after stage 4 (49 × 512-d tokens),
not only at the top. The hypothesis is that pair-aware processing
*inside* the backbone forces the IR-101 features to adapt away from pure
identity discrimination during fine-tune, reducing the family
memorization that crippled M10 R003 on held-out families.

## Configuration baseline (defaults from run_pipeline.sh)

| Knob                           | Default                  |
|--------------------------------|--------------------------|
| Backbone                       | AdaFace IR-101 WebFace4M |
| Input                          | 112×112, [-1, 1]         |
| Cross-attn stages              | 3, 4                     |
| Cross-attn layers per stage    | 1                        |
| Cross-attn heads               | 8                        |
| Per-stage pos embed            | learnable per stage      |
| Global embed                   | enabled (AdaFace pool)   |
| Head                           | classifier (BCE on `[emb1, emb2, diff, product]`) |
| Loss                           | bce                      |
| Dropout                        | 0.2                      |
| Backbone trainable             | yes (end-to-end)         |
| LR (peak)                      | 5e-6                     |
| Scheduler                      | cosine, warmup 5         |
| Min LR                         | 1e-7                     |
| Batch                          | 4 × grad-accum 8 (eff 32)|
| Patience                       | 50                       |
| Epochs (max)                   | 100                      |

## Run table

| | Run 001 | **Run 002** |
|---|---|---|
| **Date** | 2026-05-12 | 2026-05-12 |
| **Purpose** | M09 baseline — multi-stage stages 3+4, M10 R003 recipe | **Balanced sampler + per-rel train metrics** |
| **Status** | Stopped manually at ep 19 | **Stopped manually at ep 13** (user request) |
| **Best Val AUC** | 0.8919 (ep 9) | 0.8894 (ep 11) |
| **Test ROC-AUC** | **0.7982** | **0.7824** |
| **Test Accuracy** | 71.92 % | 71.55 % |
| **Val→test gap** | -0.094 (vs M10 R003 -0.140) | **-0.107** (gap widened vs R001) |
| **Notes** | Architectural hypothesis supported. Grandparent classes still collapse. See [run-001.md](run-001.md). | **Balanced sampler hypothesis rejected.** Only `gfgd` improved (+19.5 pp); all 7 majority classes regressed (-1 to -6 pp). See [run-002.md](run-002.md). |

---

## Test metrics side-by-side

| Metric | **Run 001** | **Run 002** |
|---|---:|---:|
| **Test ROC-AUC** | **0.7982** | **0.7824** (-0.016) |
| Test Accuracy | 71.92 % | 71.55 % |
| Test F1 | 0.6642 | 0.6518 |
| Test Precision | 77.75 % | 78.78 % |
| Test Recall | 57.98 % | 55.58 % |
| Avg Precision | 0.7742 | 0.7645 |
| TAR@FAR=0.01 | 10.53 % | 7.76 % |
| TAR@FAR=0.1 | 44.75 % | 45.68 % |
| Best Val ROC-AUC | 0.8919 (ep 9) | 0.8894 (ep 11) |
| Best Val Accuracy | 81.7 % (ep 9) | 80.3 % (ep 11) |
| **Val→Test AUC gap** | **-0.094** | **-0.107** (widened) |

---

## Issues Log

| # | Severity | Status | Title | Notes |
|---|----------|--------|-------|-------|
| I-01 | High   | Partially closed in R002 | M09 R001 grandparent classes collapse | gfgd 31 %, gmgd 32 %, gfgs 31 %, gmgs 37 % in R001. R002 balanced sampler fixed `gfgd` (+19.5 pp) but the other 3 grandparent classes only +1.6 to +3.1 pp. Net impact negative on Test AUC. |
| I-02 | Medium | Open | Test AUC 0.7982 below project best M02 R031 (0.850) | Multi-stage helps with the val→test gap but absolute test number doesn't beat the ViT baseline. R002 actually regressed Test AUC. Need follow-up regularisation (R003 augmentation) or partial freeze (R004) to close. |
| I-03 | Info   | Workaround applied | Checkpoint missing `model_config` when training is killed mid-pipeline | `train.py` writes `model_config` to best.pt only after the full pipeline completes. R001/R002 best.pt had to be patched manually before test.py could rebuild the multi-stage wrapper. Same fix as M10 I-05: write `model_config` in the trainer's `save_checkpoint`, not in train.py's finaliser. |
| I-04 | Info   | Open | best.pt min_delta check rejected ep 15 tied peak (R001) | R001 ep 9 Val AUC 0.8919 saved; ep 15 produced 0.8920 (tied) but `> best + 0.0001` evaluated as not greater. Functionally equivalent best, but worth noting that exact ties are possible at this precision. |
| I-05 | High   | New in R002 — Open | Balanced sampler widens val→test gap | Val→test gap moved from -0.094 (R001) to -0.107 (R002). Oversampling rare classes appears to *concentrate* family memorisation onto the over-sampled rare-class families rather than spreading it. Net Test AUC -0.0158. |

---

## Comparison with other models (FIW, Test ROC-AUC)

| Model | Test ROC-AUC | Test Acc | Backbone | Cross-pair signal | Notes |
|---|---:|---:|---|---|---|
| M02 R031 | **0.850** | 74.4 % | ViT-B/16 (ImageNet, full FT) | FaCoR top-only | project best |
| M05 R007 | 0.810 | — | hybrid (DINOv2 + LoRA + diff-attn) | — | partial freeze |
| M06 R001 | 0.776 | 69.8 % | ViT-B/16 (frozen) + retrieval | retrieval + cross-attn | best frozen-encoder |
| **M09 R001** | **0.7982** | 71.9 % | AdaFace IR-101 (full FT) + **multi-stage cross-attn (stages 3+4)** | inside-backbone pair interaction | +0.050 over M10 R003 |
| **M09 R002** | 0.7824 | 71.6 % | M09 R001 + **balanced sampler** | + per-rel train metrics | gap widened to -0.107; only `gfgd` benefitted |
| M10 R003 | 0.7478 | 70.6 % | AdaFace IR-101 (full FT) | FaCoR top-only | val→test gap -0.140 |
| M08 R001 | 0.693 | 60.8 % | ArcFace IR-100 (frozen) + retrieval | retrieval + cross-attn | anti-kinship trap |

---

## Per-relation accuracy comparison (test)

| Relation | M09 R001 | **M09 R002** | Δ R002 | M10 R003 | M02 R031 |
|----------|---------:|-------------:|-------:|---------:|---------:|
| bb       | 64.2 %   | **58.1 %**   | -6.1 ⚠ | 53.7 %   | 95.5 %   |
| ss       | 62.9 %   | 59.9 %       | -3.0   | 52.8 %   | 94.7 %   |
| sibs     | 64.5 %   | 59.8 %       | -4.7   | 56.0 %   | 94.9 %   |
| fs       | 59.1 %   | 56.7 %       | -2.4   | 53.1 %   | 95.3 %   |
| ms       | 57.3 %   | 55.5 %       | -1.8   | 52.6 %   | 93.9 %   |
| md       | 53.9 %   | 52.6 %       | -1.3   | 50.4 %   | 94.4 %   |
| fd       | 63.6 %   | 59.3 %       | -4.3   | 57.6 %   | 91.7 %   |
| **gfgd** | 31.2 %   | **50.7 %**   | **+19.5** 🎯 | 26.8 % | 89.9 %   |
| gmgd     | 31.7 %   | 33.3 %       | +1.6   | 21.1 %   | 91.1 %   |
| gfgs     | 30.6 %   | 33.7 %       | +3.1   | 28.6 %   | 95.9 %   |
| gmgs     | 37.2 %   | 37.2 %       | =      | 33.9 %   | 88.4 %   |
| non-kin  | 84.7 %   | 86.2 %       | +1.5   | 88.1 %   | (n/a)    |

Every kin relation improved over M10 R003 (+2.0 to +10.6 pp). All
classes still well below M02 R031 (-28 to -65 pp). Grandparent classes
still collapse to 30-37 %.

---

## Conclusion (as of R002)

**Two runs, two clear results — one positive, one negative for the project headline.**

### R001 conclusion (still valid)

R001 validated the M09 architectural direction. Multi-stage cross-attn
reduced AdaFace's family memorization gap from -0.140 (M10 R003) to
-0.094; Test AUC climbed +0.050 to 0.7982. Peak Val AUC (0.8919) reached
at ep 9 vs M10 R003's ep 17. All kin relations improved over M10 R003.

### R002 conclusion (new)

R002 tested the natural follow-up hypothesis (relation-balanced sampler
boosts grandparent classes by forcing exposure) and **rejected it** at
the Test AUC level:

1. **Test AUC regressed -0.0158** to 0.7824 vs R001's 0.7982.
2. **Val→test gap widened** from -0.094 to -0.107 — direct evidence
   that oversampling rare positives can *concentrate* family memorization
   onto over-sampled classes rather than spreading it.
3. **Only one rare class genuinely improved** (`gfgd` +19.5 pp). The
   other 3 grandparent classes (gfgs/gmgd/gmgs) moved within ±3 pp of
   R001 — noise margin. All 7 majority classes regressed by 1.3 to
   6.1 pp.

The R002 reading is unambiguous: **balanced sampling alone is not the
right lever** on this AdaFace setup. Net negative on the headline metric.

**Per-relation finding (R002):** R001's diagnosis ("rare classes suffer
because of low N") was *partially* correct. R002 proved that forcing
exposure alone doesn't fix the rare classes (only `gfgd` benefitted) and
*costs* majority class generalisation. The real issue is **AdaFace's
identity-cluster prior memorising whatever families it sees most**, not
the absolute frequency. Random sampling: it memorises majority families;
balanced sampling: it memorises rare families harder.

### Next directions (revised)

Given R002 invalidated the simplest rebalancing hypothesis, the
remaining queue:

- **M09 R003 — online augmentation pack** (hflip + brightness/contrast
  + JPEG + small RandomErasing). Operates orthogonally to class
  frequency. Forces invariance learning instead of pixel memorisation.
  Most promising next intervention.
- **M09 R004 — partial freeze stages 1-2.** Keeps AdaFace's
  identity-discriminative low-level features intact (preventing
  re-purposing of identity-cluster prior into family memorization);
  only stages 3-4 + cross-attn + head adapt for kinship.
- **M09 R005 — peak LR 1-2e-6.** Gentler fine-tune, reduces memorisation
  pressure regardless of class frequency.
- **Soft rebalancing as alternative R002** — interpolate `α` between
  random and balanced weights (α=0.3 might capture the `gfgd` gain
  without paying the full majority-class cost). Lower priority — not
  the highest-leverage knob.
- **M11 R001/R002 (FaCoRNet recipe)** — already scaffolded, tests
  whether FaCoR-style attention-driven loss closes the gap from a
  different angle.

---

## Architectural notes

- **Body partitioning**: IR-101 `body` is a flat `Sequential` of 49
  `BasicBlockIR` units. M09 splits it into four `nn.Sequential` views:
  stage 1 (units 0-2, 56×56×64), stage 2 (3-15, 28×28×128), stage 3
  (16-45, 14×14×256), stage 4 (46-48, 7×7×512). No parameter
  duplication — the views share underlying modules.

- **Per-stage cross-attention**: bidirectional FaCoR (verbatim from
  M02/M10) with `embedding_dim=512` after a `Linear(channels, 512)`
  projection per stage. Stage 3 has 196 tokens (14×14), stage 4 has 49
  (7×7). Cross-attn at stage 3 dominates memory (196² attention matrix
  per head) but stays within the 12 GB VRAM budget at batch=4.

- **Output assembly**: per-stage attended tokens → mean pool → 512-d
  vector per stage. AdaFace's own pooled output (from `output_layer`)
  is added as a "global" branch. The three vectors (stage-3 emb, stage-4
  emb, global emb) are summed → final 512-d face embedding.

- **Head**: `AdaFaceMultiStageKinshipClassifier` uses
  `[emb1, emb2, emb1-emb2, emb1*emb2]` → Linear(2048, 256) → LN → ReLU
  → Linear(256, 1) → logit. Same head as M10 R003.
