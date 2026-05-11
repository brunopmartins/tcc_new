# Model 09 — AdaFace IR-101 + Multi-Stage Cross-Attention

## Motivation

Models 02 (ViT) and 10 (AdaFace) both apply cross-attention **on top** of
their backbones — the two faces are encoded independently and only interact
at the very end. Recent SOTA work (CI³Former, IEEE TCSVT 2025) shows that
**injecting** pair information **inside** the backbone — letting the two
faces query each other while features are still being extracted — yields
better kinship signal than the same cross-attention done after the fact.

M09 ports that idea to the M10 backbone:

> **Same AdaFace IR-101 (WebFace4M) as M10, but the FaCoR bidirectional
> cross-attention now runs after stage 3 (14×14×256) AND after stage 4
> (7×7×512), not only at the top.**

The hypothesis is that **where** kinship information enters matters more
than how much capacity is spent on it — and that this architectural change
is testable independently from M10's SAM age-augmentation experiment, which
is orthogonal data-side work.

## Architecture

```
   Face A (112×112)                  Face B (112×112)
        │                                  │
        ▼                                  ▼
    input_layer (Conv 3→64)          input_layer
        │                                  │
        ▼  stage 1  (body[0..2])           ▼
    (B, 64, 56, 56)                  (B, 64, 56, 56)
        │                                  │
        ▼  stage 2  (body[3..15])          ▼
    (B, 128, 28, 28)                 (B, 128, 28, 28)
        │                                  │
        ▼  stage 3  (body[16..45])         ▼
    (B, 256, 14, 14) ────┐    ┌─── (B, 256, 14, 14)
                         ▼    ▼
              ┌──────────────────────────┐
              │ Stage-3 FaCoR cross-attn │   tokens: 196 × 256
              │     (bidirectional)      │   + learnable pos-embed
              └──────────────────────────┘
                         │    │
        ┌────────────────┘    └────────────────┐
        ▼                                       ▼
    stage 4  (body[46..48])                stage 4
    (B, 512, 7, 7) ──────┐    ┌──────  (B, 512, 7, 7)
                         ▼    ▼
              ┌──────────────────────────┐
              │ Stage-4 FaCoR cross-attn │   tokens: 49 × 512
              │     (bidirectional)      │   + learnable pos-embed
              └──────────────────────────┘
                         │    │
                         ▼    ▼
                 pool & project per stage
                         │    │
                         ▼    ▼
              + AdaFace pooled (output_layer)
                         │    │
                         ▼    ▼
                Linear → LN → ReLU → Drop → Linear
                         │    │
                         ▼    ▼
                  L2-norm  emb1, emb2
                         │    │
                         ▼    ▼
              cosine contrastive loss (m=0.3)
```

### IR-101 body partitioning

The IR-101 `body` is a flat `Sequential` of 49 `BasicBlockIR` units:

| Stage | Units    | Spatial | Channels |
|-------|----------|---------|---------:|
| 1     | 0..2     | 56×56   | 64       |
| 2     | 3..15    | 28×28   | 128      |
| 3     | 16..45   | 14×14   | 256      |
| 4     | 46..48   | 7×7     | 512      |

M09 splits this into four `nn.Sequential` views (no parameter duplication)
so we can tap features at the end of each stage. By default cross-attention
is applied at stages 3 and 4. Stages 1 (3136 tokens) and 2 (784 tokens) are
omitted by default because their attention matrices are an order of
magnitude heavier than stage 3.

### Cross-attention

The per-stage block is a verbatim port of M02/M10's bidirectional FaCoR
module — Q1·K2 / Q2·K1 with shared FFN — only the dimension changes:

| Stage | Tokens | Dim | Params (1 layer) |
|-------|-------:|----:|-----------------:|
| 3     | 196    | 256 | ~530 K           |
| 4     | 49     | 512 | ~2.1 M           |

Total cross-attention overhead with the default `2 stages × 1 layer` is
~2.6 M parameters on top of AdaFace's ~65 M backbone.

### Differences vs M10

| Aspect             | M10                              | M09                               |
|--------------------|----------------------------------|-----------------------------------|
| Backbone           | AdaFace IR-101 (WebFace4M)       | Same                              |
| Where cross-attn   | After `body` (7×7) only          | After stage 3 AND stage 4         |
| Tokens per attn    | 49 × 512                         | 196 × 256  +  49 × 512            |
| Cross-attn layers  | 2 (top stack)                    | 1 per stage by default            |
| Loss               | Cosine contrastive (M02 recipe)  | Same                              |
| Input              | 112×112, [-1, 1]                 | Same                              |
| SAM age ensemble   | Optional                         | Not used (orthogonal to M09)      |

## Quick start

```bash
# 1. Weights: M09 reuses M10's AdaFace checkpoint (see weights/README.md).
ls models/09_adaface_multistage/weights/adaface_ir101_webface4m.pth

# 2. Pipeline (defaults to stages 3+4, 1 layer per stage)
bash models/09_adaface_multistage/AMD/run_pipeline.sh

# 3. Ablations
#    stage-4 only (= M10 baseline minus the extra FaCoR layer)
CROSS_ATTN_STAGES=4 bash models/09_adaface_multistage/AMD/run_pipeline.sh

#    add stage 2 too (much heavier — drop batch size)
CROSS_ATTN_STAGES=2,3,4 BATCH_SIZE=4 GRAD_ACCUM=8 \
    bash models/09_adaface_multistage/AMD/run_pipeline.sh

#    2 layers per stage instead of 1
CROSS_ATTN_LAYERS_PER_STAGE=2 bash models/09_adaface_multistage/AMD/run_pipeline.sh
```

Default hyperparameters mirror M02's tuned recipe (best so far: Test ROC
AUC = 0.850).

| Knob                            | Value                |
|---------------------------------|----------------------|
| `LEARNING_RATE`                 | 5e-6 (peak)          |
| `WARMUP_EPOCHS`                 | 5                    |
| `DROPOUT`                       | 0.2                  |
| `LOSS`                          | cosine_contrastive   |
| `TEMPERATURE`                   | 0.3                  |
| `MARGIN`                        | 0.3                  |
| `BATCH_SIZE`                    | 8                    |
| `GRAD_ACCUM`                    | 4 (effective 32)     |
| `CROSS_ATTN_STAGES`             | 3,4                  |
| `CROSS_ATTN_LAYERS_PER_STAGE`   | 1                    |

## Smoke test

```bash
python models/09_adaface_multistage/model.py
```

Expected output (random-init weights, stages 3+4 default):

```
emb1 shape:     (2, 512)
emb2 shape:     (2, 512)
attn_map (s4):  (2, 8, 49, 49)
attn stages:    [(3, (2, 8, 196, 196)), (4, (2, 8, 49, 49))]
params total/trainable: 70,351,552 / 70,351,552 (100.00%)
```

## VRAM budget (AMD RX 6750 XT, 12 GB)

| Component               | Size (with defaults) |
|-------------------------|---------------------:|
| AdaFace IR-101 params   | 65.2M                |
| Per-stage cross-attn    | ~2.6M                |
| Per-stage projection    | ~0.5M                |
| Total trainable         | ~68.3M               |
| Adam state              | 2× weights           |
| Activations (b=8)       | ~3.5 GB              |

Stage 3 attention adds ~196² × 8 heads × b × dim/heads worth of activations
on top of M10's footprint — measurable but not VRAM-breaking at b=8 +
AMP. Add stage 2 only if you have headroom.

## Reference

CI³Former: *A Cross-Image Information Interaction Network for Kinship
Verification*, IEEE TCSVT 35(10):10465–10479, 2025. M09 borrows the
"interaction-inside-the-backbone" idea but keeps M02/M10's FaCoR design as
the per-stage attention block.

## Files

```
09_adaface_multistage/
├── README.md                       # This file
├── model.py                        # M09 main model (multi-stage FaCoR + AdaFace)
├── AMD/
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   └── run_pipeline.sh
├── weights/                        # AdaFace .pth (gitignored — symlinked to M10's)
│   ├── README.md
│   └── adaface_ir101_webface4m.pth
├── data/                           # Optional per-model staging (gitignored)
└── run-review/                     # Per-run analysis notes (manual)
```

Note: M09 imports the IR-101 backbone from
`models/10_adaface_facor/adaface_iresnet.py` so there is one canonical
source of truth for AdaFace internals across both models.
