# Model 13 — Landmark Graph Kinship Transformer

**Status:** documentation scaffold only. No implementation files are included yet.

Model 13 is a proposed architectural successor to Model 12. It is not a
new run of RGCK-Net. The core change is to replace fixed region crops and
pairwise cross-attention with a landmark-driven anatomical graph extracted
from a single backbone feature map.

## Motivation

Model 12 established that regional structure, partial AdaFace fine-tuning,
relation supervision, comparison-only fusion, and symmetric processing are
strong ingredients for FIW kinship verification. Its best cross-validated
recipes reach roughly the same aggregate AUC, with R010 improving strict
and mid-FAR operating points.

The remaining limitation is architectural: Model 12 still uses fixed boxes
for face regions. Those boxes are simple and robust, but they do not adapt
to face geometry, pose, scale, expression, or age-related morphology. Model
13 tests whether a face-component graph built from landmarks can represent
kinship cues more faithfully than fixed crops.

## Architectural Difference From Model 12

Model 12:

```text
aligned 224x224 face
  -> 5 fixed crops
  -> AdaFace per crop
  -> 5 region tokens
  -> bidirectional cross-region attention
  -> regional gate + classifier
```

Model 13:

```text
aligned face
  -> one shared backbone pass
  -> landmark-conditioned ROI tokens
  -> anatomical graph per face
  -> joint two-face graph transformer
  -> order-invariant graph pooling
  -> kinship logit
```

This makes Model 13 a new model family rather than a Model 12 ablation.

## Proposed Name

Short name: **LGKT-Net**

Full name: **Landmark Graph Kinship Transformer**

Suggested directory:

```text
models/13_landmark_graph_kinship_transformer/
```

## Core Components

| Component | Proposed design |
|---|---|
| Landmark source | 5-point or dense face landmarks from pre-aligned FIW images |
| Backbone | AdaFace IR-101 or the same backbone family used by M12 |
| Token extraction | One backbone pass per face, then ROIAlign over landmark-derived regions |
| Nodes | Anatomical parts: eyes, eyebrows, nose, mouth, jaw, cheeks, forehead, global |
| Edges | Intra-face anatomical edges, homologous cross-face edges, and selected cross-component edges |
| Pair interaction | Graph Transformer over the joint graph of both faces |
| Symmetry | Native order-invariant pooling, not an optional training trick |
| Output | Binary kinship logit plus optional relation-type auxiliary head |

## Hypotheses

1. **Dynamic anatomical tokens should improve robustness.** Landmark-guided
   regions should reduce crop noise relative to fixed boxes, especially for
   faces with pose, expression, or alignment variation.

2. **Graph structure should improve non-kin rejection.** Explicit anatomical
   relations may help distinguish true kinship from broad facial similarity.

3. **Symmetry should be architectural.** The model should be invariant to
   pair order by construction, avoiding the direction-specific shortcut that
   Model 12 R006 had to remove through symmetric forward.

4. **Grandparent relations may benefit from bone-structure cues.** Jaw,
   nose bridge, eye spacing, and cheek structure can be modeled as connected
   components rather than independent region similarities.

## Non-Goals For This Scaffold

This folder intentionally does not include:

- `model.py`
- `train.py`
- `test.py`
- `evaluate.py`
- AMD/Nvidia runners
- dataset changes
- landmark extraction code
- ROIAlign implementation

Those should be added only when the architecture is ready to be implemented.

## Files

```text
13_landmark_graph_kinship_transformer/
├── README.md              # high-level proposal
├── ARCHITECTURE.md        # detailed architectural contract
├── EXPERIMENT_PLAN.md     # planned runs and success criteria
├── RUN_LOG.md             # empty run log template
├── output/                # placeholder only
└── run-review/
    └── overview.md        # planned review narrative
```

## Reference Baselines

| Reference | Role |
|---|---|
| M12 R006 | Aggregate-AUC reference |
| M12 R010 | Operating-curve and CV reference |
| M02 R031 | Strong pre-M12 supervised baseline |
| B0 AdaFace cosine | Low-FAR and no-adaptation reference |

## Success Criteria

Model 13 should only be considered successful if it beats the Model 12
family outside the measured noise floor. Single-run gains smaller than about
0.008 AUC should be treated as inconclusive unless replicated.

Target outcomes:

| Metric | Target |
|---|---:|
| Test ROC AUC | greater than M12 R010 CV mean by at least 0.008 |
| TAR@FAR=0.001 | above M12 R010 CV mean |
| TAR@FAR=0.01 | above M12 R010 CV mean |
| Non-kin accuracy | improve without collapsing kin recall |
| Grandparent relations | no regression relative to M12 R006/R010 |

The central claim to test is not merely "more AUC". It is whether an
anatomical graph representation can improve generalization and specificity
where fixed regional crops begin to saturate.
