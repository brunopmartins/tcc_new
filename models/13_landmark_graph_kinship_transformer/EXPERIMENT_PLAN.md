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

### R002 — Add Relation Auxiliary Head

Purpose: test whether the relation auxiliary signal composes with graph
structure as it did with M12.

Changes:

- add 11-way relation head on positive pairs;
- balanced relation CE;
- initial weight 0.05.

Success threshold:

- better per-relation balance than R001;
- no AUC drop larger than the measured noise floor.

### R003 — Add Full Cross-Face Edge Set

Purpose: test whether structural cross-face edges improve specificity.

Changes:

- homologous edges;
- global-to-component edges;
- selected structural edges such as eye-to-nose, nose-to-mouth, jaw-to-mouth
  across faces.

Success threshold:

- improve non-kin rejection;
- improve TAR@FAR=0.01;
- maintain grandparent accuracy.

### R004 — Stage-3 Feature Map Variant

Purpose: determine whether a richer feature map helps landmark ROI tokens.

Changes:

- extract tokens from stage 3 instead of stage 4, or combine stage 3 and
  stage 4 with a lightweight projection.

Risk:

- higher memory and stronger overfitting.

Success threshold:

- AUC gain >= 0.008 over R003 or meaningful low-FAR gain;
- no large val-to-test gap expansion.

### R005 — Cross-Validation Candidate

Only run 5-fold CV if one of R001-R004 beats the M12 reference outside the
noise floor or clearly dominates a deployment metric.

Minimum condition before CV:

- single-run Test AUC >= M12 R010 CV mean + 0.008, or
- TAR@FAR=0.001 and TAR@FAR=0.01 both exceed M12 R010 CV means while AUC
  remains statistically tied.

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
