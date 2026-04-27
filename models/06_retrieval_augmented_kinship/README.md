# Model 06 — Retrieval-Augmented Kinship Verification

## Overview

Model 06 adds a **non-parametric memory** to kinship verification. Instead
of forcing the decision into a single feed-forward head, the model
retrieves, at inference time, the **K positive training pairs most
similar to the current query pair** and cross-attends to them, using
their signatures and relation labels as context.

This idea — *retrieval-augmented reasoning* — is well established in
language modelling (RAG, kNN-LM, Atlas, etc.) but, to the best of our
knowledge, **has not been evaluated for facial kinship verification** in
the prior literature surveyed for this project. It aims to answer the
question: *would the model do better if, when judging a test pair, it
had direct access to a set of known kin pairs that look similar to the
query?*

## Key choices, sized for 12 GB VRAM / 32 GB RAM

1. **Backbone is frozen.** We reuse a pre-trained ViT-B/16 (or DINOv2
   ViT-B/14) as a feature extractor. No gradients flow through it, so
   training fits comfortably in 12 GB VRAM even at batch 8-16.
2. **Trainable parts are small:** the projection head (~3 M), the
   signature-to-token linear (~1 M), a 2-layer retrieval cross-attention
   (~5 M), the binary head (~0.1 M), the relation head (~0.1 M) and the
   relation embedding. Total ~10-12 M trainable params.
3. **Gallery size is bounded.** FIW has ~99k positive training pairs; we
   can keep them all. KinFaceW is much smaller. With `--store_gallery_on_cpu`
   the gallery lives in regular RAM (32 GB is plenty) and only the
   current top-K supports are moved to VRAM at each step.
4. **Retrieval is chunked.** We never materialise the full (query × gallery)
   similarity matrix; the top-K is kept via running-argmax in chunks of
   4 096 gallery rows.
5. **Gradient checkpointing** is available, but with a frozen encoder
   it is usually unnecessary.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                RETRIEVAL-AUGMENTED KINSHIP (Model 06)                    │
│                                                                          │
│    Input (A, B)  ──►  ViT encoder (FROZEN)  ──►  e_a, e_b  (512-d)       │
│                                                                          │
│    pair signature s_q = normalize([e_a+e_b, |e_a-e_b|, e_a·e_b])         │
│                                                                          │
│                          ┌──── Gallery (pos. train pairs) ────┐          │
│                          │   signatures   (~99k × 1536)       │          │
│                          │   emb_a, emb_b (~99k × 512)        │          │
│                          │   relation_idx (~99k)              │          │
│                          └─────────────────┬──────────────────┘          │
│                                            │                             │
│                   cosine top-K (chunked)  ─┘                             │
│                          │                                               │
│                          ▼                                               │
│           K support tokens = sig_to_token(sig_i) + relation_embed(r_i)   │
│                                                                          │
│    query token q = sig_to_token(s_q)                                     │
│                                                                          │
│                      ┌────────────────────────┐                          │
│                      │ Retrieval cross-attn × N                          │
│                      │   q ← Attention(q, supports)                      │
│                      └───────────────┬────────┘                          │
│                                      ▼                                   │
│         Binary head   [q_in, q_out] → logit_kin                          │
│         Relation head  [q_out]      → logits_11                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Training workflow

The script does the following, end to end:

1. Build dataloaders for the selected dataset (FIW or KinFaceW-I).
2. Instantiate the model with a frozen backbone.
3. **Build the gallery** — pass the training dataloader once through the
   encoder and store signatures / embeddings / relation indices for every
   positive pair.
4. Train: for each batch, encode → retrieve K supports → cross-attend →
   predict. BCE + (optional) contrastive + relation-CE loss.
5. Rebuild the gallery every `--gallery_refresh_every` epochs (default: 1,
   no rebuild since the encoder is frozen). If you opt to fine-tune the
   encoder, set this > 0 so gallery embeddings stay consistent.
6. After training, evaluate under the standard protocol (threshold
   selected on validation, applied as-is to test).

At test time the gallery is rebuilt from the training split, so
retrieval is always from **training** pairs only — never from val/test —
and a `train_pair_id` exclusion mask is honoured when a query itself
appears in the gallery (e.g. during validation loops inside training).

## Platforms

| File                            | Purpose                                            |
|---------------------------------|----------------------------------------------------|
| `model.py`                      | Architecture (shared)                              |
| `AMD/train.py`                  | ROCm training                                      |
| `AMD/test.py`                   | ROCm testing                                       |
| `AMD/evaluate.py`               | ROCm full analysis                                 |
| `AMD/run_pipeline.sh`           | Local end-to-end runner                            |
| `Nvidia/*.py`                   | CUDA mirror                                        |
| `docker-compose.amd.yml`        | ROCm docker                                        |
| `docker-compose.nvidia.yml`     | CUDA docker                                        |

## Hardware assumption (AMD RX 6750 XT, 12 GB VRAM, 32 GB RAM)

Defaults are OOM-safe:

| Setting                     | Default                    | Rationale                               |
|-----------------------------|----------------------------|-----------------------------------------|
| `--batch_size`              | 8                          | Encoder frozen → can go higher than M05 |
| `--gradient_accumulation`   | 4                          | Effective batch = 32                    |
| `--retrieval_k`             | 32                         | Moderate memory token cost              |
| `--freeze_backbone`         | `True`                     | Stay within VRAM                        |
| `--store_gallery_on_cpu`    | `False`                    | 99k × 1536 × fp32 ≈ 600 MB on GPU is OK |
| `--lora_rank` (future work) | —                          | Not used in M06 (backbone is frozen)    |

If you hit OOM:

1. `--batch_size 4 --gradient_accumulation 8`
2. `--retrieval_k 16`
3. `--store_gallery_on_cpu` — gallery leaves VRAM, retrieval moves only
   supports for each batch.
4. `--backbone_name vit_small_patch16_224` (half the encoder width).

## Quick start

### AMD (bare-metal)

```bash
bash models/06_retrieval_augmented_kinship/AMD/run_pipeline.sh

# KinFaceW with a DINOv2 encoder and smaller K
TRAIN_DATASET=kinface BACKBONE=vit_base_patch14_dinov2.lvd142m \
    RETRIEVAL_K=16 \
    bash models/06_retrieval_augmented_kinship/AMD/run_pipeline.sh
```

### AMD via Docker

```bash
docker-compose -f models/06_retrieval_augmented_kinship/docker-compose.amd.yml up
```

### Nvidia

```bash
docker-compose -f models/06_retrieval_augmented_kinship/docker-compose.nvidia.yml up
```

## Expected behaviour

Empirical question — no results yet. The design is motivated by:

- forcing the decision to **depend on concrete training examples**, not
  only on learned parameters;
- giving the model **relation-aware context** (the retrieved pairs'
  relation labels) that Models 02/03/05 do not receive;
- producing a **recipe transferable to other fine-grained face
  verification tasks** (identification, morph detection) where the query
  structure is pairwise.

## Contribution

1. **Non-parametric memory for kinship.** A fresh angle compared to the
   purely-parametric baselines (M02, M03, M04) and to the VLM zero-shot
   baselines.
2. **Chunked top-K retrieval on 12 GB VRAM.** The design demonstrates
   retrieval-augmented models can run on consumer AMD hardware.
3. **Relation-aware support tokens.** Concatenating signature features
   with an embedding of the retrieved pair's relation exposes
   genealogical structure to the cross-attention.

## Files

```
06_retrieval_augmented_kinship/
├── README.md
├── model.py
├── docker-compose.amd.yml
├── docker-compose.nvidia.yml
├── AMD/
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   └── run_pipeline.sh
└── Nvidia/
    ├── train.py
    ├── test.py
    └── evaluate.py
```
