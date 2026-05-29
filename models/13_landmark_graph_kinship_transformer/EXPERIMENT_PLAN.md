# Experiment Plan — Model 13 LGKT-Net

This plan describes the intended experimental sequence for Model 13. It is
documentation only; no training entrypoint exists yet.

## Reference Results

Model 13 should be compared against cross-validated Model 12 numbers, not
only single-run favorable draws.

| Model | Reference role |
|---|---|
| M12 R006 CV | aggregate-AUC reference |
| M12 R010 CV | strict/mid-FAR and operating-curve reference |
| M02 R031 CV | strongest pre-M12 baseline |
| B0 AdaFace cosine | no-training low-FAR reference |

## Run Sequence

### R001 — Minimal Landmark Graph

Purpose: verify whether landmark-derived ROI tokens and graph pooling are
competitive with M12's fixed-region tokenization.

Architecture:

- AdaFace IR-101 backbone;
- stage-4 feature map;
- 8 nodes: global, left eye, right eye, nose, mouth, jaw, left cheek,
  right cheek;
- homologous cross-face edges only;
- 2-layer Graph Transformer;
- native symmetric pooling;
- BCE classifier.

Success threshold:

- Test AUC >= M12 R001;
- no catastrophic regression on grandparent classes;
- training stable without direction-specific symmetric-forward loss.

### R002 — Stage-3 features + comparison-only pooling + relation aux (REVISED)

Purpose: address the three diagnosed R001 failure modes in one combined
remediation, before deciding whether the M13 line is worth continuing.
R001 failure modes (see ``run-review/run-001.md``):

1. **ROIAlign over a 7×7 stage-4 map is too coarse** — most landmark ROIs
   sample only 1-2 grid cells.
2. **Global node identity-leakage into the classifier** — M12 R009 had
   exactly this problem and fixed it via comparison-only fusion.
3. **Per-class collapse on grandparent classes** — relation aux was the
   M12-line remedy.

Changes (single combined recipe; this is the M13 line-restart run):

- ``FEATURE_STAGE=stage3``: ROIAlign on the 14×14×256 map. A learned
  ``Linear(256, 512)`` projection brings tokens back to embedding dim.
  4× more spatial cells per ROI; new 131k params for the projection.
- ``UNFREEZE_LAST_STAGE=1`` is repurposed for stage-3: unfreezes
  ``body[43:46]`` only (the last 3 blocks of stage 3). Stage-4 blocks and
  ``output_layer`` are dead code in this path and remain frozen.
- ``COMPARISON_ONLY_POOLING=1``: global node is excluded from
  ``SymmetricPairPooler`` but kept as context inside the graph
  transformer (M12 R009 analog adapted to the graph regime).
- ``RELATION_AUX_WEIGHT=0.05`` with balanced CE on positive pairs only
  (M12 R005 setting that propagated forward).
- Everything else unchanged from R001 (2 graph layers, 4 heads, dropout
  0.2, 8 nodes, full cross-edge set, BCE, AdamW, cosine + warmup 5,
  LR 1e-5, batch 16 × grad accum 2, seed 42).

Launch wrapper: ``AMD/run_r002.sh``.

Success thresholds (gating criteria):

- ``Test AUC >= 0.870``: graduate to R003 (hard negatives + 5-fold CV).
- ``0.860 <= Test AUC < 0.870``: run an A1-only diagnostic to verify
  stage-3 carries a meaningful signal on its own before further changes.
- ``Test AUC < 0.860``: re-stop the M13 line — graph + landmark on
  pre-aligned FIW is not competitive with M12 R006/R010/R011, and
  additional changes are unlikely to close the gap.

### R003 — Role-matched hard negatives (conditional)

Only run if R002 hits the >= 0.870 graduation threshold.

Changes (R002 stack + R011's negative-sampling delta):

- ``TRAIN_NEGATIVE_STRATEGY=relation_matched``;
- ``HARD_NEGATIVE_RATIO=0.30`` (the M12 R011 mix).

Success threshold:

- repeat the M12 R010→R011 lift (+~0.007 AUC) on top of R002, OR
- meaningful low-FAR gain.

### R004 — 5-fold CV (conditional)

Only run if R002 (or R003) clearly beats the M12 R010 CV mean of 0.8739.
Minimum condition is unchanged from the original plan: single-run AUC
>= M12 R010 CV mean + 0.008, or a clear strict-FAR dominance with AUC
statistically tied.

### R005 — (folded into R004)

Originally a separate "CV candidate" entry; collapsed into R004 above
when R002 was re-scoped to bundle the stage-3, comparison-only, and
relation-aux changes into a single line-restart run.

## Metrics To Report

Primary:

- ROC AUC;
- Average Precision;
- TAR@FAR=0.001;
- TAR@FAR=0.01;
- TAR@FAR=0.1.

Secondary:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1;
- validation-to-test AUC gap.

Per-relation:

- all 11 FIW positive relation classes;
- non-kin specificity;
- separate focus on the four grandparent classes.

Interpretability:

- node attention/gate weights by relation;
- edge-type attention summaries;
- failure cases where non-kin pairs score high.

## Decision Rules

Treat changes as inconclusive when:

- Test AUC gain is below 0.008 and not replicated;
- low-FAR gains come with major non-kin or grandparent collapse;
- val-to-test gap widens toward the M09/M10/M11 pattern.

Stop the Model 13 line if:

- R001-R003 cannot beat M12 R001/R002-style numbers;
- graph structure adds cost without improving AUC, low-FAR, or per-relation
  balance;
- landmark fallback rate is high enough to make the architecture effectively
  a fixed-box model again.

Continue to CV if:

- the graph model matches M12 AUC and improves non-kin specificity;
- or it beats M12 by at least 0.008 AUC in a single controlled run;
- or it clearly dominates M12 at low-FAR with acceptable recall.

## Expected Contribution If Successful

Model 13 would support the claim that kinship verification benefits not only
from regional comparison, but from explicit anatomical structure. That is a
stronger and cleaner architectural claim than another M12 run.
