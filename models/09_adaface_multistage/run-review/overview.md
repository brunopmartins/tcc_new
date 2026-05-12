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

| | Run 001 |
|---|---|
| **Date** | 2026-05-12 |
| **Purpose** | M09 baseline — multi-stage stages 3+4, M10 R003 recipe |
| **Status** | **Stopped manually at ep 19** for early eval after 4 epochs below peak |
| **Best Val AUC** | **0.8919 (ep 9)** |
| **Test ROC-AUC** | **0.7982** |
| **Test Accuracy** | 71.92 % |
| **Val→test gap** | **-0.094** (vs M10 R003 -0.140, ~33 % smaller) |
| **Notes** | Architectural hypothesis supported but not enough to close gap to M02 R031. Grandparent classes still collapse. See [run-001.md](run-001.md). |

---

## Test metrics side-by-side

| Metric | **Run 001** |
|---|---:|
| **Test ROC-AUC** | **0.7982** |
| Test Accuracy | 71.92 % |
| Test F1 | 0.6642 |
| Test Precision | 77.75 % |
| Test Recall | 57.98 % |
| Avg Precision | 0.7742 |
| TAR@FAR=0.01 | 10.53 % |
| TAR@FAR=0.1 | 44.75 % |
| Best Val ROC-AUC | 0.8919 (ep 9) |
| Best Val Accuracy | 81.7 % (ep 9) |
| Val→Test AUC gap | **-0.094** |

---

## Issues Log

| # | Severity | Status | Title | Notes |
|---|----------|--------|-------|-------|
| I-01 | High   | Open | M09 R001 grandparent classes still collapse | gfgd 31 %, gmgd 32 %, gfgs 31 %, gmgs 37 % — improved over M10 R003 (~5-10 pp) but still well below random + M02 R031 (88-96 %). Likely needs class-balanced sampling (R002). |
| I-02 | Medium | Open | Test AUC 0.7982 below project best M02 R031 (0.850) | Multi-stage helps with the val→test gap (-0.094 vs -0.140) but absolute test number doesn't beat the ViT baseline. Need follow-up regularisation (R003) or partial freeze (R004) to close. |
| I-03 | Info   | Workaround applied | Checkpoint missing `model_config` when training is killed mid-pipeline | `train.py` writes `model_config` to best.pt only after the full pipeline completes. R001 best.pt had to be patched manually before test.py could rebuild the multi-stage wrapper. Same fix as M10 I-05: write `model_config` in the trainer's `save_checkpoint`, not in train.py's finaliser. |
| I-04 | Info   | Open | best.pt min_delta check rejected ep 15 tied peak | ep 9 Val AUC 0.8919 saved; ep 15 produced Val AUC 0.8920 (exactly tied) but `> best + 0.0001` evaluated as not greater. Functionally equivalent best, but worth noting for future runs that exact ties are possible at this precision. |

---

## Comparison with other models (FIW, Test ROC-AUC)

| Model | Test ROC-AUC | Test Acc | Backbone | Cross-pair signal | Notes |
|---|---:|---:|---|---|---|
| M02 R031 | **0.850** | 74.4 % | ViT-B/16 (ImageNet, full FT) | FaCoR top-only | project best |
| M05 R007 | 0.810 | — | hybrid (DINOv2 + LoRA + diff-attn) | — | partial freeze |
| M06 R001 | 0.776 | 69.8 % | ViT-B/16 (frozen) + retrieval | retrieval + cross-attn | best frozen-encoder |
| **M09 R001** | **0.7982** | 71.9 % | AdaFace IR-101 (full FT) + **multi-stage cross-attn (stages 3+4)** | inside-backbone pair interaction | **+0.050 over M10 R003** |
| M10 R003 | 0.7478 | 70.6 % | AdaFace IR-101 (full FT) | FaCoR top-only | val→test gap -0.140 |
| M08 R001 | 0.693 | 60.8 % | ArcFace IR-100 (frozen) + retrieval | retrieval + cross-attn | anti-kinship trap |

---

## Per-relation accuracy comparison (test)

| Relation | M09 R001 | M10 R003 | M02 R031 | Notes |
|----------|---------:|---------:|---------:|-------|
| bb       | 64.2 %   | 53.7 %   | 95.5 %   | +10.5 pp over M10 R003 |
| ss       | 62.9 %   | 52.8 %   | 94.7 %   | +10.1 pp |
| sibs     | 64.5 %   | 56.0 %   | 94.9 %   | +8.5 pp  |
| fs       | 59.1 %   | 53.1 %   | 95.3 %   | +6.0 pp  |
| ms       | 57.3 %   | 52.6 %   | 93.9 %   | +4.7 pp  |
| md       | 53.9 %   | 50.4 %   | 94.4 %   | +3.5 pp  |
| fd       | 63.6 %   | 57.6 %   | 91.7 %   | +6.0 pp  |
| gfgd     | 31.2 %   | 26.8 %   | 89.9 %   | +4.4 pp  |
| gmgd     | 31.7 %   | 21.1 %   | 91.1 %   | **+10.6 pp** (best grandparent improvement) |
| gfgs     | 30.6 %   | 28.6 %   | 95.9 %   | +2.0 pp  |
| gmgs     | 37.2 %   | 33.9 %   | 88.4 %   | +3.3 pp  |
| non-kin  | 84.7 %   | 88.1 %   | (n/a)    | -3.4 pp  |

Every kin relation improved over M10 R003 (+2.0 to +10.6 pp). All
classes still well below M02 R031 (-28 to -65 pp). Grandparent classes
still collapse to 30-37 %.

---

## Conclusion (as of R001)

R001 validates the M09 architectural direction:

1. **Multi-stage cross-attn reduces AdaFace's family memorization
   on held-out FIW families.** Val→test gap dropped from -0.140 (M10
   R003) to -0.094 — ~33 % reduction. Test AUC climbed +0.050.

2. **The unlock is faster.** Val AUC crosses 0.80 at ep 5 (peak LR) in
   M09 vs ep 7 in M10 R003. Peak Val AUC (0.8919) reached at ep 9 in
   M09 vs ep 17 in M10 R003 (with similar value 0.8875).

3. **All kin relations improve over M10 R003 uniformly**, with the
   largest gains on same-generation classes (bb, ss: +10 pp) and on
   `gmgd` (+10.6 pp). The improvement pattern is consistent, not
   driven by a single outlier class.

**But the gap to M02 R031 remains substantial (-0.052 absolute Test AUC).**
M09 is now the **best AdaFace-based model** in the project, but the ViT-B/16
ImageNet baseline still wins overall. AdaFace's identity-cluster prior is
mitigated by multi-stage cross-attn but not eliminated; grandparent
classes remain catastrophic.

**Per-relation finding (R001):** the relation imbalance pattern in FIW
Track-I (gfgs N=98, gmgs N=121, gmgd N=123, gfgd N=138, vs ~1000 for
parent-child classes) directly correlates with the gap-to-M02-R031.
Same-generation classes (better-data) close ~30 pp of the gap;
grandparent classes (worst-data) close <5 pp. **This points
unambiguously to class-balanced sampling as the next intervention.**

**Next directions:**
- **M09 R002 — relation-balanced sampling + per-relation train metrics.**
  Highest expected impact on grandparent classes. Per-relation metrics
  in the train phase give epoch-by-epoch visibility into class-level
  regressions.
- **M09 R003 — online augmentation pack** (hflip + brightness/contrast
  + JPEG + small RandomErasing). Combats the residual memorization that
  R001 still shows.
- **M09 R004 — partial freeze stages 1-2.** Keeps AdaFace's
  identity-discriminative low-level features intact while letting
  stages 3-4 + cross-attn + head adapt for kinship.
- **M09 R005 (if needed) — peak LR 1-2e-6.** Gentler fine-tune,
  reducing memorization pressure.

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
