# Overview — Model 13: Landmark Graph Kinship Transformer

**Status:** planned architecture, documentation only.

Model 13 is proposed as a true architectural successor to Model 12. It
should not be interpreted as a new M12 run unless the future implementation
keeps the defining changes listed below.

## Core Claim

Model 12 proved that anatomical regions help kinship verification, but it
uses fixed rectangular crops. Model 13 asks whether an explicit landmark
graph can improve the representation of face-component relationships.

The intended contribution is:

```text
fixed region sequence -> dynamic anatomical graph
```

## Required Differences From M12

To qualify as Model 13, a future implementation must include:

1. landmark-derived regions;
2. one backbone pass per face;
3. ROI/token extraction from feature maps;
4. graph nodes and typed edges;
5. pair interaction through a joint graph transformer;
6. native order-invariant pooling.

If the implementation keeps fixed crops and normal cross-attention, it should
be logged under Model 12 instead.

## Baseline To Beat

Use cross-validated baselines where available:

| Reference | Role |
|---|---|
| M12 R006 CV | aggregate AUC baseline |
| M12 R010 CV | operating-curve baseline |
| M02 R031 CV | pre-M12 supervised baseline |
| B0 AdaFace cosine | frozen no-adaptation low-FAR baseline |

## Open Questions

1. Do landmark-derived regions reduce crop noise enough to improve AUC?
2. Does graph message passing improve non-kin rejection?
3. Can native symmetry match or improve M12 R006 without a duplicated forward?
4. Are grandparent relations better served by structural features such as jaw,
   nose bridge, and cheek geometry?
5. Does ROIAlign on a single feature map reduce compute without losing the
   regional signal that made M12 work?

## Planned Run Table

| Run | Planned intervention | Status |
|---|---|---|
| R001 | Minimal landmark graph, homologous edges, BCE | not implemented |
| R002 | R001 + relation auxiliary head | not implemented |
| R003 | R002 + full cross-face edge set | not implemented |
| R004 | stage-3 or multi-stage ROI tokens | not implemented |
| R005 | 5-fold CV candidate | gated on R001-R004 |

## Notes For Future Reviews

The review should separate three effects:

- landmark tokenization effect;
- graph interaction effect;
- native symmetry effect.

Avoid claiming architectural progress from training-only changes unless the
graph structure itself is active.
