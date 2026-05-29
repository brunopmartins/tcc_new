# Run Log — Model 13: Landmark Graph Kinship Transformer

Newest run on top.

---

## Run 002 — 2026-05-27 → 2026-05-28 — Stage-3 + comparison-only pooling + rel-aux 0.05 (Test AUC **0.8526**; M13 LINE RE-STOPPED)

**Status:** Trained 85/100 epochs; training crashed at ep 85 due to **disk full** during `torch.save()`. Best.pt from ep 44 (peak Val AUC 0.8952) survived intact; eval ran cleanly after manual cleanup + checkpoint patch.

**Outcome:** Test ROC AUC **0.8526** (+0.0064 vs R001's 0.8462) but **TAR@FAR=0.001 collapsed to 0.0000** (R001: 0.0403). Per [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) decision rule `AUC < 0.860 → re-stop the M13 line`. **Line stopped (again).**

**Recipe (delta vs R001):**

- `FEATURE_STAGE=stage3` (14×14×256 → Linear(256,512) projection)
- `COMPARISON_ONLY_POOLING=1` (global node excluded from `SymmetricPairPooler`, kept in graph as context)
- `RELATION_AUX_WEIGHT=0.05` balanced CE
- `UNFREEZE_LAST_STAGE=1` now means `body[43:46]` (stage-3 tail) since stage 4 is dead code in this path

**Trainable:** 9,006,645 / 70,481,653 (12.75 %, down from R001's 31.3 M / 44.40 %).

**Test metrics summary:**

| Metric | R001 | **R002** | Δ |
|---|---:|---:|---:|
| Test ROC AUC | 0.8462 | **0.8526** | +0.0064 |
| Test AP | 0.8281 | 0.8301 | +0.0020 |
| Test F1 (thr=0.1) | 0.6935 | 0.6980 | +0.0045 |
| Test Recall | 0.6037 | 0.6118 | +0.0081 |
| **TAR@FAR=0.001** | 0.0403 | **0.0000** | **−0.0403 ⚠** |
| TAR@FAR=0.01 | 0.1704 | 0.1665 | −0.0039 |
| Val→Test gap | −0.039 | −0.043 | wider |
| gfgs / gfgd (grandparent) | 38.8 / 50.0 % | **48.0 / 58.0 %** | +9.2 / +8.0 |
| bb / fs / ss | 78.0 / 73.1 / 76.2 % | 70.0 / 63.3 / 66.9 % | −5 to −10 pp |

**Net pattern:** R002 trades 5–10 pp on well-supported kin classes for 8–9 pp on the worst grandparent classes (rel-aux head doing its job). Strict-FAR collapsed (comparison-only pooling removed the global node's direct contribution to the pooled fusion). AUC modestly improved but still 0.024 below M12 R011 CV mean (0.8761 ± 0.0029).

**Mechanism diagnosis** (full version in [run-review/run-002.md](run-review/run-002.md)):

1. Stage-3 spatial gain is wasted: `LandmarkROITokenizer` does ROIAlign 3×3 then AdaptiveAvgPool to 1×1, throwing away the 4× cell density. A learned spatial encoder would be needed to use stage-3 properly.
2. Comparison-only pooling collapses strict-FAR in this small model: the global token was the strongest separator at the tails; excluding it from the pooler (while the relation_head still depends on it) creates conflicting pressures.
3. Bundle was the highest-EV combined remediation — landing at 0.8526 says the architecture cannot reach M12 territory under any combination of the levers tried.

**Disk-full incident:** 17 `epoch_{5..85}.pt` periodic snapshots × 339 MB filled the disk at ep 85. best.pt was already saved at ep 44. Recovered by deleting periodic checkpoints (~5.7 GB freed), patching `best.pt` to add `model_config` + `protocol`, then running `test.py --threshold 0.1` manually. **Action item for future runs**: drop `save_every` from 5 to 25, or move outputs to a non-system disk.

### Artifacts

- Best.pt (ep 44, patched): `output/002/checkpoints/best.pt` (339 MB)
- Train log: `output/002/logs/train.log`
- Test log (val-threshold 0.1): `output/002/logs/test_thr01.log`
- Final metrics: `output/002/results/test_metrics_rocm.txt`
- Run review: [`run-review/run-002.md`](run-review/run-002.md)

### Decision

**M13 line stopped (again).** R001 and R002 together have established that:

- Graph-transformer over 8 fixed-landmark ROIs on pre-aligned FIW cannot beat M02 R031 CV (0.8462) on aggregate AUC; it can beat M02 by +0.006 (R002 0.8526) but cannot approach M12 R006/R010/R011 territory.
- Strict-FAR is a particular weakness — both runs land at ≤0.04 TAR@FAR=0.001, vs M12 R011 CV 0.068.
- The "landmark-driven" framing reduces to "smaller fixed boxes" on pre-aligned data, where landmarks are canonical by construction. M12's larger fixed boxes + cross-region attention is empirically superior.

No R003 launches. Future revisits to this line would need a non-pre-aligned dataset (so landmarks actually vary per image) or a non-trivial spatial encoder between ROIAlign and the graph (to actually exploit stage-3 detail).

---

## Run 001 — 2026-05-26 — Manually stopped at ep 10 (peak Val AUC 0.8850 ep 5; Test AUC 0.8462 — M13 LINE STOPPED)

**Status:** Manually stopped during ep 10 training; best.pt from ep 5 (peak Val AUC 0.8850).

**Outcome:** M13 R001 reached **Test AUC 0.8462** — *identical* to M02 R031 CV mean (0.8462 ± 0.0040) and clearly below the M12 family (R006 0.8788, R010 0.8754, R011 0.8825). The architecture works (well above M12 R001's 0.7464 frozen baseline) but does not produce a ranking advantage over either M02 R031 or M12 R006-R011.

**Decision:** **M13 line stopped after R001** per EXPERIMENT_PLAN.md's "Stop the Model 13 line if: graph structure adds cost without improving AUC, low-FAR, or per-relation balance" criterion.

### Highlights

- **Project-best non-kin specificity: 87.36 %** (vs M12 R011 78.69 %, M02 R031 ~55 %).
- **3× training speedup vs M12** (~8.6 min/epoch vs ~26 min/epoch) — single backbone pass per face delivers as predicted.
- **Order-invariance by construction** — no symmetric-forward trick needed.

### Failure modes (vs M12 R011)

- **Test AUC -0.036** (0.8462 vs 0.8825) — at M02 R031 level, not M12.
- **Test Recall 60.37 %** — -20 pp vs M12 R011.
- **Catastrophic per-class collapse on small kin classes:**
  - gmgs 17.4 % (vs M12 R011 62.8 %, -45 pp)
  - gfgs 38.8 % (vs 79.6 %, -41 pp)
  - gmgd 39.8 % (vs 64.2 %, -24 pp)
  - gfgd 50.0 % (vs 74.6 %, -25 pp)

The model over-rejects positives and produces a higher non-kin accuracy at the cost of recall on rare kin classes — a trade-off that nets negative on AUC, AP, and F1.

### Launch command

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
UNFREEZE_LAST_STAGE=1 \
LEARNING_RATE=1e-5 \
BATCH_SIZE=16 GRAD_ACCUM=2 \
DROPOUT=0.2 \
RELATION_AUX_WEIGHT=0.0 \
NUM_WORKERS=4 SEED=42 \
bash models/13_landmark_graph_kinship_transformer/AMD/run_pipeline.sh
```

### Configuration

- AdaFace IR-101 (stages 1-3 frozen, stage 4 + output_layer trainable)
- LandmarkROITokenizer: 1 backbone pass + ROIAlign 3×3 over 8 canonical-landmark-derived boxes
- 2-layer × 4-head EdgeAwareGraphAttention (4 edge types: intra/homologous/global/self)
- SymmetricPairPooler ([mean, |diff|, prod] per node-pair with softmax gate)
- BCE only (no relation aux for R001 baseline)
- Batch 16 × grad-accum 2 (eff. 32), LR 1e-5 peak, cosine + warmup 5
- Trainable: 31,290,154 / 70,481,653 (44.40 %)

### Training trajectory

| Ep | Train Loss | Val AUC | M12 R010 same ep |
|---:|---:|---:|---:|
| 1 | 0.6141 | 0.8083 | 0.8218 |
| 2 | 0.4220 | 0.8645 | 0.9030 |
| 3 | 0.3022 | 0.8686 | 0.9100 |
| 4 | 0.2042 | 0.8820 | 0.9119 (R010 peak) |
| **5** | **0.1507** | **0.8850 (peak)** | 0.9052 |
| 6 | 0.1119 | 0.8787 | 0.9064 |
| 7 | 0.0860 | 0.8739 | 0.9073 |
| 8 | 0.0702 | 0.8818 | 0.9025 |
| 9 | 0.0562 | 0.8839 | 0.9032 |

### Test metrics (val-selected threshold 0.100)

| Metric | M12 R011 (HEADLINE) | M02 R031 CV mean | **M13 R001** |
|---|---:|---:|---:|
| Test ROC AUC | **0.8825** | 0.8462 ± 0.0040 | **0.8462** |
| Test Accuracy | 79.82 % | 74.66 % | 74.43 % |
| Test F1 | 0.7938 | 0.7774 | 0.6935 |
| Test Precision | 77.77 % | 67.24 % | **81.46 %** |
| Test Recall | 81.05 % | 92.25 % | 60.37 % |
| Test AP | 0.8645 | 0.8131 | 0.8281 |
| TAR@FAR=0.001 | 7.51 % | 2.19 % | 4.03 % |
| TAR@FAR=0.01 | 21.30 % | 13.49 % | 17.04 % |
| TAR@FAR=0.1 | 61.89 % | 49.64 % | 54.82 % |
| Val→Test gap | -0.021 | — | -0.039 |

### Mechanism (why M13 R001 underperformed)

1. **ROIAlign over 7×7 map is too coarse** — most landmark boxes are smaller than 16 px in 112-pixel space, so each ROI samples 1-2 grid cells. Anatomical detail is already smoothed out.
2. **No cmp-only ablation** — global node carries identity leakage like M12 gA/gB before R009.
3. **Graph attention over-smooths small-class features** — broad global edges dilute rare-class signal.
4. **"Landmark-driven" reduces to "smaller fixed boxes"** on pre-aligned FIW, where landmarks are canonical by construction. The graph + smaller boxes is empirically worse than M12's larger fixed boxes + cross-region attention.

### Files

- Checkpoint: `output/001/checkpoints/best.pt` (410 MB)
- Train log: `output/001/logs/train.log`
- Test log: `output/001/logs/test.log`
- Results: `output/001/results/test_metrics_rocm.txt`
- Run review: `run-review/run-001.md` (full analysis + decision rationale)

### What stays in the tree

The full implementation remains (model.py, AMD/train.py, AMD/test.py, AMD/run_pipeline.sh) for future researchers. R002+ runs are not planned. If revisiting:

- Stage-3 14×14 feature map (R004 in EXPERIMENT_PLAN.md) is the most plausible remediation.
- Adding the comparison-only fusion from M12 R009 may help.
- A non-pre-aligned dataset where per-image landmarks differ would justify the "landmark" framing properly.
