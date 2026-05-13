# Run Log — Model 09: AdaFace IR-101 + Multi-Stage Cross-Attention

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

---

## Run 002 — 2026-05-12 — Stopped at epoch 13 (Val AUC 0.8894 peak ep 11; Test AUC 0.7824, val→test gap -0.107)

**Status:** Stopped manually at end of epoch 13 (user request — pause for review before next run)
**Outcome:** Balanced sampler + per-relation train metrics. Val AUC peak 0.8894 close to R001 (0.8919). **Test AUC = 0.7824 — worse than R001 (0.7982) by -0.0158**. Val→test gap **widened to -0.107** (R001 was -0.094). Per-relation test: only `gfgd` improved substantially (+19.5 pp); all majority classes (bb, fs, ms, md, sibs, ss, fd) regressed by 1.3 to 6.1 pp. Hypothesis "balanced sampler closes the val→test gap by exposing rare classes" was NOT confirmed — gap widened.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=4 \
GRAD_ACCUM=8 \
USE_CLASSIFIER_HEAD=1 \
LOSS=bce \
CROSS_ATTN_STAGES=3,4 \
CROSS_ATTN_LAYERS_PER_STAGE=1 \
USE_BALANCED_SAMPLER=1 \
NUM_WORKERS=4 \
SEED=42 \
bash models/09_adaface_multistage/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw (Track-I, split_seed=42) |
| Aligned root | datasets/FIW_aligned (224 → 112 at load) |
| Backbone | AdaFace IR-101 (WebFace4M, **full fine-tune**) |
| Cross-attn placement | After stage 3 (196×256-d) AND stage 4 (49×512-d) |
| Cross-attn layers per stage | 1 |
| Cross-attn heads | 8 |
| Positional embedding | learnable per-stage |
| Global embedding | enabled (AdaFace pool) |
| Head | `AdaFaceMultiStageKinshipClassifier` MLP on `[emb1, emb2, diff, product]` → logit |
| Loss | **bce** (same as R001) |
| **Training sampler** | **`WeightedRandomSampler`** — equalises relation classes among positives, maintains 2:1 effective neg ratio (R001 used `shuffle=True`) |
| **Per-relation val metrics** | **enabled per-epoch** in trainer log line (new in R002) |
| Batch size | 4 (eff. 32 with grad_accum=8) |
| Grad accum | 8 |
| LR | 5e-6 peak, warmup 5, cosine, min_lr=1e-7 |
| Weight decay | 1e-5 |
| Dropout | 0.2 |
| Embedding dim | 512 |
| Patience | 50 |
| Trainable params | 70,876,353 (100%) |
| Time/epoch | ~40 min |
| Seed | 42 |

### Training trajectory

- Best Val AUC: **0.8894** at epoch 11 (best.pt)
- Stopped manually at end of epoch 13 by user request
- Time per epoch: ~40 min (identical to R001 within ±1 %)
- Train loss saturation rate slightly different from R001: R002 0.252 at ep 13 vs R001 0.279 at ep 13 (R002 slightly more saturated; both well above the 0.1 of M10 R003 by ep 18)

| Epoch | Train Loss | Val Acc | Val AUC | Thr | LR | Note |
|------:|-----------:|--------:|--------:|----:|-----|------|
| 1 | 0.6442 | 65.7 % | 0.7083 | 0.400 | 1.0e-6 | warmup 1/5; per-rel grandparent already 90-95 % |
| 2 | 0.5863 | 66.3 % | 0.7219 | 0.350 |       |      |
| 3 | 0.5429 | 62.2 % | 0.6903 | 0.100 |       | warmup dip |
| 4 | 0.5068 | 64.7 % | 0.6913 | 0.100 |       |      |
| 5 | 0.4957 | 74.2 % | 0.7870 | 0.100 | 5.0e-6 | **unlock +0.096 at peak LR** |
| 6 | 0.4906 | 77.7 % | 0.8482 | 0.100 |       | climb |
| 7 | 0.4412 | 78.7 % | 0.8633 | 0.100 |       | new best.pt |
| 8 | 0.4102 | 81.2 % | 0.8850 | 0.100 |       | new best.pt |
| 9 | 0.3759 | 80.5 % | 0.8793 | 0.100 |       | dip |
| 10 | 0.3423 | 79.0 % | 0.8863 | 0.100 |       | new best.pt |
| 11 | 0.3043 | 80.3 % | **0.8894** | 0.100 |    | **lifetime best.pt** |
| 12 | 0.2774 | 79.7 % | 0.8825 | 0.100 |       | dip |
| 13 | 0.2521 | 79.1 % | 0.8888 | 0.100 |       | bounce, -0.0006 from peak — **manual stop** |

### Test metrics (threshold 0.500)

Stored threshold on best.pt defaulted to 0.500 because training was killed before `update_checkpoint_metadata`. Val-phase F1-optimal at ep 11 was 0.100. AUC/AP threshold-invariant.

| Metric | Value |
|--------|-------|
| **Test ROC AUC** | **0.7824** |
| Test Accuracy | 71.55 % |
| Test Balanced Acc | 70.91 % |
| Test Precision | 78.78 % |
| Test Recall | 55.58 % |
| Test F1 | 0.6518 |
| Test Avg Precision | 0.7645 |
| TAR @ FAR=0.001 | 1.32 % |
| TAR @ FAR=0.01 | 7.76 % |
| TAR @ FAR=0.1 | 45.68 % |
| Threshold (used) | 0.500 |
| **Val→test AUC gap** | **-0.107** (vs R001 -0.094 — gap widened) |

### Per-relation accuracy (FIW Track-I test)

| Relation | N | M09 R001 | **M09 R002** | Δ R002 vs R001 | M02 R031 |
|----------|--:|---------:|-------------:|---------------:|---------:|
| bb       | 860  | 64.2 % | **58.1 %** | **-6.1 pp** ⚠ | 95.5 % |
| ss       | 731  | 62.9 % | 59.9 %     | -3.0           | 94.7 % |
| sibs     | 234  | 64.5 % | 59.8 %     | -4.7           | 94.9 % |
| md       | 1038 | 53.9 % | 52.6 %     | -1.3           | 94.4 % |
| fs       | 1135 | 59.1 % | 56.7 %     | -2.4           | 95.3 % |
| ms       | 1036 | 57.3 % | 55.5 %     | -1.8           | 93.9 % |
| fd       | 918  | 63.6 % | 59.3 %     | -4.3           | 91.7 % |
| **gfgd** | 138  | 31.2 % | **50.7 %** | **+19.5 pp** 🎯 | 89.9 % |
| gfgs     | 98   | 30.6 % | 33.7 %     | +3.1           | 95.9 % |
| gmgd     | 123  | 31.7 % | 33.3 %     | +1.6           | 91.1 % |
| gmgs     | 121  | 37.2 % | 37.2 %     | =              | 88.4 % |
| non-kin  | 6993 | 84.7 % | 86.2 %     | +1.5           | (n/a)  |

### Notes

- **Balanced sampler hypothesis NOT confirmed.** Only `gfgd` improved meaningfully (+19.5 pp); the other 3 grandparent classes stayed within ±3 pp of R001. All 7 majority classes regressed by 1-6 pp. Net Test AUC was -0.0158 worse than R001.
- **Val→test gap widened** from -0.094 (R001) to -0.107 (R002). Direct evidence that oversampling rare positives can *increase* family memorization on the over-sampled families, not decrease it.
- **Per-relation training metrics worked correctly** — new feature in trainer log line. Allowed real-time diagnosis: classes oscillated heavily epoch-to-epoch (sibs swung 13.5 pp between ep 1-2), particularly grandparent classes with N<150 in val.
- Best.pt at ep 11 (not ep 8 or ep 15 like R001); R002 had its peak in a different epoch with different absolute value.
- **Per-rel during training was much higher than per-rel at test** — same pattern as R001. Val per-rel (at threshold 0.100) doesn't predict test per-rel (at threshold 0.500).

### Artifacts

- Checkpoint (epoch 11, best Val AUC = 0.8894): `output/002/checkpoints/best.pt` (852 MB) — patched with `model_config`
- Resume snapshots (ep 5, ep 10): `output/002/checkpoints/epoch_5.pt`, `epoch_10.pt` — to be pruned after commit
- Train log: `output/002/logs/train.log`
- Test log: `output/002/logs/test.log`
- Evaluate log: `output/002/logs/evaluate.log`
- Results: `output/002/results/{test_metrics_rocm.txt, metrics_rocm.json, roc_curve_rocm.png, confusion_matrix_rocm.png, attention_intensity_comparison.png, attention_analysis.json, attention_maps/}`

---

## Run 001 — 2026-05-12 — Stopped at epoch 19 (Val AUC 0.8919 peak ep 9; Test AUC 0.7982, val→test gap -0.094)

**Status:** Stopped manually at iter 0/epoch 20 after 4-epoch plateau confirmed below ep-9 peak
**Outcome:** First M09 run, multi-stage cross-attention SAI-inspired architecture validated. Best Val AUC = **0.8919** at epoch 9 (eight epochs earlier than M10 R003). **Test AUC = 0.7982** — beats M10 R003 (0.7478) by +0.050 but still below M02 R031 (0.850). Val→test gap **-0.094**, ~33% smaller than M10 R003's -0.140, confirming that multi-stage cross-attn does reduce family memorization but does not eliminate it.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=4 \
GRAD_ACCUM=8 \
USE_CLASSIFIER_HEAD=1 \
LOSS=bce \
CROSS_ATTN_STAGES=3,4 \
CROSS_ATTN_LAYERS_PER_STAGE=1 \
NUM_WORKERS=4 \
SEED=42 \
bash models/09_adaface_multistage/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw (Track-I, split_seed=42) |
| Aligned root | datasets/FIW_aligned (224 → 112 at load) |
| Backbone | AdaFace IR-101 (WebFace4M, **full fine-tune**) |
| Cross-attn placement | **After stage 3 (196 tokens × 256-d) AND after stage 4 (49 tokens × 512-d)** |
| Cross-attn layers per stage | 1 |
| Cross-attn heads | 8 |
| Positional embedding | learnable per-stage (196+49 tokens) |
| Global embedding | enabled (AdaFace pool) |
| Head | `AdaFaceMultiStageKinshipClassifier`: MLP over `[emb1, emb2, diff, product]` → logit |
| Loss | **bce** (BCE on classifier head logits) |
| Batch size | 4 (eff. 32 with grad_accum=8) |
| Grad accum | 8 |
| LR | 5e-6 peak, warmup 5, cosine, min_lr=1e-7 |
| Weight decay | 1e-5 |
| Dropout | 0.2 |
| Embedding dim | 512 |
| Patience | 50 |
| Trainable params | 70,876,353 (100%) |
| Time/epoch | ~40 min |
| Seed | 42 |

### Training trajectory

- Best Val AUC: **0.8919** at epoch 9 (best.pt)
- Tied peak at epoch 15: 0.8920 (failed `> best + 0.0001` min_delta check)
- Stopped manually at epoch 19/100 after 4 epochs of plateau-with-oscillation below peak
- Time per epoch: ~40 min (vs M10 R003's 28 min; +43% from stage-3 cross-attention overhead)

| Epoch | Train Loss | Val Acc | Val AUC | LR    | Note                          |
|------:|-----------:|--------:|--------:|-------|-------------------------------|
| 1     | 0.6569     | 65.5 %  | 0.7241  | 1.0e-6 | warmup 1/5 — already above M10 R003 ep1 |
| 2     | 0.6046     | 65.4 %  | 0.7239  | 2.0e-6 |                                |
| 3     | 0.5687     | 65.2 %  | 0.7202  | 3.0e-6 | warmup plateau                |
| 4     | 0.5475     | 64.7 %  | 0.7122  | 4.0e-6 | -0.012 slow drift             |
| 5     | 0.5152     | 78.0 %  | **0.8237** | 5.0e-6 | **unlock +0.111** at peak LR |
| 6     | 0.4990     | 79.9 %  | 0.8692  | 5.0e-6 | climb                          |
| 7     | 0.4679     | 81.7 %  | 0.8844  | 4.99e-6 | climb                         |
| 8     | 0.4339     | 81.5 %  | 0.8858  | 4.99e-6 | matches M10 R003 ep10 (best so far for M10 R003 was at ep17) |
| 9     | 0.4003     | 81.7 %  | **0.8919** | 4.98e-6 | **best.pt peak** |
| 10    | 0.3677     | 80.4 %  | 0.8777  | 4.97e-6 | dip -0.014                    |
| 11    | 0.3423     | 79.1 %  | 0.8804  | 4.95e-6 | mini bounce                   |
| 12    | 0.3113     | 80.0 %  | 0.8904  | 4.93e-6 | strong bounce +0.010          |
| 13    | 0.2785     | 80.3 %  | 0.8898  | 4.91e-6 |                                |
| 14    | 0.2538     | 78.2 %  | 0.8872  | 4.89e-6 |                                |
| 15    | 0.2327     | 80.5 %  | 0.8920  | 4.87e-6 | tied peak (below min_delta)   |
| 16    | 0.2032     | 78.7 %  | 0.8836  | 4.84e-6 | dip                            |
| 17    | 0.1789     | 79.1 %  | 0.8887  | 4.81e-6 | bounce, no new peak            |
| 18    | 0.1664     | 80.5 %  | 0.8853  | 4.78e-6 |                                |
| 19    | 0.1438     | 79.6 %  | 0.8850  | 4.74e-6 | flat — manual stop after this |

### Test metrics

Stored threshold on best.pt = 0.500 (default — training was killed before `update_checkpoint_metadata` wrote the val-optimal). Val-phase F1-optimal at ep 9 was 0.100. Numbers below use 0.500. AUC and Avg Precision are threshold-invariant.

| Metric              | Value      |
|---------------------|------------|
| **Test ROC AUC**    | **0.7982** |
| Test Accuracy       | 71.92 %    |
| Test Balanced Acc   | 71.36 %    |
| Test Precision      | 77.75 %    |
| Test Recall         | 57.98 %    |
| Test F1             | 0.6642     |
| Test Avg Precision  | 0.7742     |
| TAR @ FAR=0.001     | 1.57 %     |
| TAR @ FAR=0.01      | 10.53 %    |
| TAR @ FAR=0.1       | 44.75 %    |
| Threshold (used)    | 0.500      |
| Val→test AUC gap    | **-0.094** |

### Per-relation accuracy (FIW Track-I test)

| Relation | N    | M09 R001 Acc | M10 R003 Acc | M02 R031 Acc | Δ vs M10 R003 | Δ vs M02 R031 |
|----------|-----:|-------------:|-------------:|-------------:|--------------:|--------------:|
| bb       | 860  | 64.2 %       | 53.7 %       | 95.5 %       | **+10.5 pp**  | -31.3 pp      |
| ss       | 731  | 62.9 %       | 52.8 %       | 94.7 %       | **+10.1 pp**  | -31.8 pp      |
| sibs     | 234  | 64.5 %       | 56.0 %       | 94.9 %       | +8.5 pp       | -30.4 pp      |
| fs       | 1135 | 59.1 %       | 53.1 %       | 95.3 %       | +6.0 pp       | -36.2 pp      |
| ms       | 1036 | 57.3 %       | 52.6 %       | 93.9 %       | +4.7 pp       | -36.6 pp      |
| md       | 1038 | 53.9 %       | 50.4 %       | 94.4 %       | +3.5 pp       | -40.5 pp      |
| fd       | 918  | 63.6 %       | 57.6 %       | 91.7 %       | +6.0 pp       | -28.1 pp      |
| gfgd     | 138  | 31.2 %       | 26.8 %       | 89.9 %       | +4.4 pp       | -58.7 pp      |
| gmgd     | 123  | 31.7 %       | 21.1 %       | 91.1 %       | **+10.6 pp**  | -59.4 pp      |
| gfgs     | 98   | 30.6 %       | 28.6 %       | 95.9 %       | +2.0 pp       | -65.3 pp      |
| gmgs     | 121  | 37.2 %       | 33.9 %       | 88.4 %       | +3.3 pp       | -51.2 pp      |
| non-kin  | 6993 | 84.7 %       | 88.1 %       | (n/a)        | -3.4 pp       | —             |

Multi-stage cross-attn improves **every** kin relation over M10 R003 (+2 to +11 pp), but all classes remain catastrophically below M02 R031 (-28 to -65 pp). Grandparent classes still collapse to 30-37 %.

### Notes

- Multi-stage cross-attn (SAI/CI³Former inspired) was the hypothesis: injecting pair information after stages 3 + 4 of the IR-101 backbone, not only at the top.
- **Unlock came one epoch earlier than M10 R003** (ep 5 vs ep 6) and **0.051 higher** (0.8237 vs 0.7191). The +0.111 single-epoch jump at peak LR confirmed the architecture was holding accumulated capacity through warmup.
- **Peak Val AUC reached 9 epochs faster than M10 R003** (ep 9 vs ep 17) and slightly higher (0.8919 vs 0.8875).
- **Train loss saturation pattern qualitatively different from M10 R003.** M10 R003 was at 0.1 train loss by ep 18; M09 R001 still at 0.144 at ep 19. Suggests the multi-stage architecture distributes capacity into something other than fast train memorization — relevant for the val→test generalization question.
- Manual stop at ep 19/100 chosen because val plateaued for 5 epochs without surpassing the ep-9 peak. Patience=50 would have wasted ~20h of GPU.
- best.pt was saved at ep 9 (Val AUC 0.8919). ep 15 produced Val AUC 0.8920 (tied) but the `> best + min_delta(0.0001)` check evaluated as not greater, so it did not overwrite.

### Artifacts

- Checkpoints: `output/001/checkpoints/best.pt` (852 MB), `epoch_5.pt`, `epoch_10.pt`, `epoch_15.pt` (resume snapshots — will be pruned after test)
- Logs: `output/001/logs/train.log`
- Results: `output/001/results/{test_metrics_rocm.txt, metrics_rocm.json, *.png}` (pending test phase)

---
