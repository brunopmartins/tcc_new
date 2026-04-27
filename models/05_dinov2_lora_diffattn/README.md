# Model 05 — DINOv2 + LoRA + Differential FaCoR + Relation Head

## Overview

Model 05 integrates three techniques that — to our knowledge — have **not yet been
validated for facial kinship verification** in the prior literature surveyed for
this project:

1. **DINOv2 ViT-B/14 backbone** (Oquab et al., 2024), self-supervised on 142M
   unlabelled images. Produces strong fine-grained facial features without
   ImageNet class bias.
2. **LoRA adapters** (Hu et al., 2021) injected into the backbone's attention
   projections. The backbone stays **frozen**; only ~3-8M rank-`r` matrices plus
   the heads are trainable. This avoids the overfitting Model 03 suffered after
   full unfreeze and cuts VRAM dramatically.
3. **Differential cross-attention** (Ye et al., 2024 — arXiv:2410.05258). The
   FaCoR-style bidirectional cross-attention is reformulated as a *difference of
   two softmax attention maps*, which amplifies discriminative face-region
   correspondences and suppresses spurious ones.
4. **Auxiliary relation head**. A learnable `[REL]` query token feeds an
   11-class (FIW) or 4-class (KinFaceW) classifier, trained jointly with the
   primary binary verification loss. The multi-task signal pushes the model to
   separate fine-grained genealogical relations (e.g. `gmgs` vs `gfgd`) that
   are exactly where Model 02, Model 03, and the VLM baselines fail.

Model 05 **explicitly drops the age-synthesis branch from Model 01**, because
the SAM module was not fully active in this project and Model 01's results
were inconclusive.

### Full architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    MODEL 05 — DINOv2 + LoRA + DiffAttn                    │
│                                                                           │
│   Input: (Face A, Face B)                                                 │
│              │                                                            │
│              ▼                                                            │
│   ┌───────────────────────────────────────────┐                           │
│   │ DINOv2 ViT-B/14 (FROZEN)                  │                           │
│   │  + LoRA (rank r) on qkv + proj of blocks  │                           │
│   │  ~ 86M frozen / ~2-4M trainable           │                           │
│   └───────────────────┬───────────────────────┘                           │
│                       │ patch tokens (B, 256, 768)                        │
│                       ▼                                                   │
│   ┌───────────────────────────────────────────┐                           │
│   │ Linear + LN  → (B, 256, 512)              │                           │
│   │ Prepend [REL] query token                 │                           │
│   └───────────────────┬───────────────────────┘                           │
│                       ▼                                                   │
│   ┌───────────────────────────────────────────┐                           │
│   │ Differential bidirectional cross-attention│                           │
│   │   × 2 layers, 8 heads                     │                           │
│   │   DiffAttn = softmax(Q1K1ᵀ)-λ·softmax(Q2K2ᵀ)                          │
│   └───────────────────┬───────────────────────┘                           │
│                       ▼                                                   │
│   ┌───────────────────────────────────────────┐                           │
│   │ Mean-pool patch tokens → emb1, emb2 (512-d L2-norm)                   │
│   │ Pick [REL] token from both faces          │                           │
│   └───────────────────┬───────────────────────┘                           │
│                       │                                                   │
│         ┌─────────────┴─────────────┐                                     │
│         ▼                           ▼                                     │
│  Binary head                 Relation head (aux)                          │
│  [e1, e2, |e1-e2|, e1·e2]    [relA, relB] → K classes                     │
│         │                           │                                     │
│         ▼                           ▼                                     │
│  Kinship score                 CE loss (FIW=11, KFW=4)                    │
└───────────────────────────────────────────────────────────────────────────┘
```

### Why LoRA rather than full / partial unfreeze

Runs 004-007 of Model 03 showed:

- full unfreeze with batch 32 → OOM
- full unfreeze with batch 16 → overfitting pos-unfreeze (AUC declines)
- partial unfreeze with low LR factor → does not converge

LoRA sidesteps both problems:

- only ~2-4M params have gradients ⇒ fits comfortably in 12 GB
- pretrained backbone weights stay intact ⇒ no catastrophic forgetting of
  DINOv2's rich feature space
- implicit regularisation from the low-rank constraint

## Platforms

| File                            | Purpose                                            |
|---------------------------------|----------------------------------------------------|
| `model.py`                      | Architecture (shared across platforms)             |
| `AMD/train.py`                  | ROCm training with gradient accumulation + AMP     |
| `AMD/test.py`                   | ROCm test with validation-threshold protocol       |
| `AMD/evaluate.py`               | ROCm full analysis (ROC, per-relation, ablations)  |
| `AMD/run_pipeline.sh`           | Local pipeline (train → test → evaluate)           |
| `Nvidia/*.py`                   | CUDA mirror of the AMD scripts                     |
| `docker-compose.amd.yml`        | ROCm docker runner                                 |
| `docker-compose.nvidia.yml`     | CUDA docker runner                                 |

## Hardware assumption (AMD RX 6750 XT, 12 GB VRAM, 32 GB RAM)

The default hyperparameters target **12 GB VRAM**. All runs keep the DINOv2
backbone frozen, use gradient checkpointing on backbone + cross-attn layers,
AMP (FP16) where supported, and gradient accumulation for an effective batch
size of 32.

| Configuration        | batch_size | grad_accum | effective | peak VRAM |
|----------------------|-----------:|-----------:|----------:|----------:|
| Default (OOM-safe)   | 4          | 8          | 32        | ~7-9 GB   |
| Conservative         | 2          | 16         | 32        | ~5-6 GB   |
| Aggressive (Nvidia)  | 16         | 2          | 32        | >12 GB    |

If you still hit OOM, in order:

1. set `BATCH_SIZE=2 GRAD_ACCUM=16`
2. pass `--disable_amp` (some ROCm versions trade AMP for a smaller graph)
3. use a smaller backbone: `BACKBONE=vit_small_patch14_dinov2.lvd142m`
   (rare, but available in newer timm versions)

## Quick start

### AMD (bare-metal, no Docker)

```bash
# One-shot: train → test → evaluate on FIW
bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh

# KinFaceW-I with smaller batch (if OOM)
TRAIN_DATASET=kinface BATCH_SIZE=2 GRAD_ACCUM=16 \
    bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

### AMD via Docker

```bash
docker-compose -f models/05_dinov2_lora_diffattn/docker-compose.amd.yml up
```

### Nvidia

```bash
docker-compose -f models/05_dinov2_lora_diffattn/docker-compose.nvidia.yml up
```

## Key hyperparameters

| Flag                      | Default                              | Notes                                   |
|---------------------------|--------------------------------------|-----------------------------------------|
| `--backbone_name`         | `vit_base_patch14_dinov2.lvd142m`    | Any timm DINOv2 variant                 |
| `--img_size`              | 224                                  | Lower memory than native 518            |
| `--lora_rank`             | 8                                    | Try 16 on larger GPUs                   |
| `--lora_alpha`            | 16                                   | `alpha/r = 2` is a stable default       |
| `--embedding_dim`         | 512                                  | Projected token dim                     |
| `--cross_attn_layers`     | 2                                    | Match Model 02's FaCoR depth            |
| `--cross_attn_heads`      | 8                                    | Per-head dim = 64 (must be even)        |
| `--relation_set`          | `fiw`                                | `fiw` (11) or `kinface` (4)             |
| `--relation_loss_weight`  | 0.2                                  | λ·CE(aux) in total loss                 |
| `--loss`                  | `combined`                           | `bce`, `contrastive`, or `combined`     |
| `--batch_size`            | 4                                    | Pair with `--gradient_accumulation 8`   |
| `--gradient_accumulation` | 8                                    | Effective batch = 32                    |
| `--lr`                    | 3e-4                                 | LoRA tolerates much higher LR than full |

## Expected behaviour

We do not pre-declare numbers, as Model 05 has not been run yet. The design
aims to:

- **match or exceed Model 02/03 (AUC ≈ 0.850)** on FIW with ~20× fewer
  trainable parameters;
- **reduce the 2nd-degree relation gap** (gfgs, gmgs, gfgd, gmgd) via the
  relation head's explicit supervision;
- **train stably on 12 GB VRAM** without OOM or the overfitting seen in
  Model 03 runs 002-007.

## Contribution

1. **LoRA-based fine-tuning of a self-supervised face backbone for kinship**
   — not present in the surveyed kinship literature.
2. **Differential cross-attention for face-pair interaction** — first
   application outside language modelling.
3. **Explicit relation-classification auxiliary loss** — previous FIW papers
   focus on binary verification; the relation head directly pushes the
   model to separate the genealogical classes where the VLM baselines
   reach 0 %.

## Files

```
05_dinov2_lora_diffattn/
├── README.md                     (this file)
├── model.py                      (architecture)
├── docker-compose.amd.yml
├── docker-compose.nvidia.yml
├── AMD/
│   ├── train.py                  (ROCm training)
│   ├── test.py                   (ROCm testing)
│   ├── evaluate.py               (ROCm full analysis)
│   └── run_pipeline.sh           (local end-to-end runner)
└── Nvidia/
    ├── train.py
    ├── test.py
    └── evaluate.py
```
