# Architecture — Model 13 LGKT-Net

This document defines the intended architecture for Model 13. It is a
contract for a future implementation, not executable code.

## Design Principle

Model 13 should treat a face as a structured anatomical object. Instead of
extracting a small set of fixed crops, it should build tokens from face
components located by landmarks and reason over those components as a graph.

The target distinction is:

```text
M12: fixed regional token sequence
M13: dynamic anatomical graph
```

## End-To-End Flow

```text
Input pair
  face_A, face_B
      |
      v
Landmark estimation or cached landmarks
      |
      v
Shared backbone pass per face
      |
      v
Feature map + landmark-derived ROIs
      |
      v
ROIAlign component tokens
      |
      v
Single-face anatomical graph construction
      |
      v
Joint two-face graph transformer
      |
      v
Order-invariant graph pooling
      |
      v
Kinship classifier
```

## Stage 1 — Landmarks

The model should consume landmarks either from cached preprocessing or from
an upstream detector. The first implementation should prefer cached landmarks
to keep training deterministic and avoid detector overhead inside the loop.

Minimum viable landmark set:

- left eye
- right eye
- nose tip
- left mouth corner
- right mouth corner

Preferred landmark set:

- dense 68-point or 106-point landmarks
- enough detail to define jawline, eyebrows, nose bridge, cheeks, and mouth

Failure policy:

- if dense landmarks fail, fall back to coarse landmarks;
- if all landmarks fail, fall back to M12-style fixed regions for that sample;
- log fallback rate per split.

## Stage 2 — Shared Backbone Feature Map

Model 12 runs AdaFace once per crop. Model 13 should run the backbone once
per face and extract local component tokens from an intermediate feature map.

Backbone candidate:

| Candidate | Reason |
|---|---|
| AdaFace IR-101 | continuity with Model 12 and strong facial prior |
| AdaFace stage-4 map | compact 7x7 spatial structure, low memory |
| AdaFace stage-3 map | richer 14x14 spatial structure, more memory |

The first implementation should start with stage 4 for stability and then
consider stage 3 if the graph lacks spatial detail.

## Stage 3 — Landmark-Derived ROI Tokens

Each node token should be extracted with ROIAlign or an equivalent differentiable
pooling operation over the selected feature map.

Proposed nodes:

| Node | Landmark-derived region |
|---|---|
| global | whole aligned face |
| left_eye | box around left eye landmarks |
| right_eye | box around right eye landmarks |
| eyebrows | union of eyebrow landmarks |
| nose_bridge | upper nose landmarks |
| nose_tip | lower nose landmarks |
| mouth | mouth outer contour |
| jaw | jawline landmarks |
| left_cheek | cheek region between eye, nose, and jaw |
| right_cheek | cheek region between eye, nose, and jaw |
| forehead | region above eyebrows |

The exact node set can be reduced for the first run. A good first version is:

```text
global, left_eye, right_eye, nose, mouth, jaw, left_cheek, right_cheek
```

## Stage 4 — Single-Face Graph

Each face becomes a graph:

```text
G_A = (V_A, E_A)
G_B = (V_B, E_B)
```

Intra-face edges should encode anatomical adjacency:

- eye to nose
- eye to eyebrow
- nose to mouth
- mouth to jaw
- cheek to eye
- cheek to nose
- cheek to jaw
- global to all nodes

Edges may be represented as:

- learned edge-type embeddings;
- relative landmark geometry features;
- both.

Recommended geometric features per edge:

- normalized distance;
- relative angle;
- scale ratio to inter-ocular distance;
- optional symmetry flag for left/right components.

## Stage 5 — Joint Two-Face Graph

The pair graph combines both faces:

```text
G_pair = (V_A union V_B, E_A union E_B union E_cross)
```

Cross-face edges:

1. **Homologous edges**
   - left_eye_A to left_eye_B
   - right_eye_A to right_eye_B
   - nose_A to nose_B
   - mouth_A to mouth_B
   - jaw_A to jaw_B

2. **Structural edges**
   - eye_A to nose_B
   - nose_A to mouth_B
   - jaw_A to mouth_B

3. **Global comparison edges**
   - global_A to every node_B
   - global_B to every node_A

The first implementation can begin with homologous + global edges and add
structural edges in later ablations.

## Stage 6 — Graph Transformer

The graph transformer should perform message passing over the joint graph
with edge-type awareness.

Expected block:

```text
node features
  -> edge-aware multi-head attention
  -> residual + layer norm
  -> feed-forward network
  -> residual + layer norm
```

Recommended defaults:

| Parameter | Starting value |
|---|---:|
| node dim | 512 |
| layers | 2 |
| heads | 4 or 8 |
| dropout | 0.2 |
| edge types | intra, homologous, structural, global |

## Stage 7 — Native Symmetric Pooling

Model 13 should not need a separate symmetric-forward trick. Pair order
invariance should be built into the pooled representation.

For each homologous node pair:

```text
mean_i = 0.5 * (node_A_i + node_B_i)
diff_i = abs(node_A_i - node_B_i)
prod_i = node_A_i * node_B_i
```

Then aggregate across nodes with attention pooling or gated pooling:

```text
pair_repr = pool([mean_i, diff_i, prod_i, node_type_i])
```

This keeps the representation invariant under swapping A and B.

## Stage 8 — Heads

Required head:

- binary kinship classifier.

Optional heads:

- relation-type auxiliary classifier for positive pairs;
- node-importance/gate head for interpretability;
- calibration head for low-FAR operating points.

The relation auxiliary head should follow the lesson from Model 12: useful
for per-class balance, but not sufficient by itself to improve aggregate AUC.

## What Must Differ From M12

The implementation should not simply repackage M12. To count as Model 13,
it must include all of these differences:

1. landmark-derived regions instead of fixed boxes;
2. a single backbone pass per face with ROIAlign/token extraction;
3. a graph representation with explicit node and edge types;
4. graph-level pair interaction;
5. order-invariant pooling by construction.

If any of these are missing, the result should be treated as a Model 12
variant rather than Model 13.
