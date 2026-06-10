# Model 15 — RGCK-Net hi-res ROI-Align (no 112×112 limit)

M15 is **M12 (RGCK-Net), with the region tokenizer freed from the 112×112
constraint**. Same head, same proven recipe; only the way region tokens are
produced changes.

## The 112 problem and why it isn't a backbone limit

AdaFace IR-101 only "needs" 112×112 because of its **FC head**
(`output_layer = … Flatten → Linear(512·7·7, 512)`), which accepts a single
7×7 spatial map. The **conv body is fully convolutional and takes any
resolution** (`forward_spatial`: 112→7×7, 224→14×14, …).

The M12 line worked around this in two ways, both limited:

| Variant | Region tokens | Limitation |
|---|---|---|
| M12 R001–R012 (`FixedPartition`) | crop each box, **squash to 112×112**, re-run AdaFace 5× | crops are out-of-distribution (AdaFace was trained on whole aligned faces, not stretched eye/mouth strips) |
| M12 R013 (`ROIAlign`) | 1 backbone pass at **112 → 7×7 map**, ROI-pool regions | in-distribution, but pooled from a coarse 7×7 grid (1–3 cells/region — the M13 coarse-ROI problem) |

## What M15 does

Run the conv body at the **native 224** (→ a 14×14 feature map, 4× the spatial
detail of R013's 7×7), and ROI-Align each region from that map to a **fixed
7×7 grid** *before* the FC head. Because the ROI output is always 7×7, the FC
works at any input resolution — the 112 limit is gone and each region token is
sampled from real per-region detail. `--img_size` can push the map finer
(336→21×21, …) if VRAM allows.

All tokens still pass through `output_layer`, so they share AdaFace's embedding
space and the per-region cosine similarities the M12 head relies on stay
comparable.

```
aligned face @ 224
   │
   └─► AdaFace conv body (ONCE)  ──►  14×14×512 feature map
                                         │
        ┌────────────────────────────────┼───────────────────────────┐
   global token                     eyes / nose / mouth / jaw
   (mode=exact: genuine 112          ROI-Align each box → 7×7 → FC
    AdaFace embedding;               (in-distribution, undistorted,
    mode=roi: full-box ROI→7×7→FC)    real spatial detail)
                                         │
                            (B, 5, 512) region tokens
                                         │
                  ── M12 head verbatim ──┘
   (bidirectional cross-region attn → sigmoid gate → fusion → BCE,
    symmetric forward + comparison-only fusion + relation aux)
```

## What is shared with M12

- **Head**: `RGCKNet` is imported verbatim from `../12_rgck_net/model.py`
  (region tokens, cross-region attention, gate, fusion classifier, symmetric
  forward, comparison-only fusion, relation-aux head).
- **Backbone recipe**: stage 4 (`body[46:49]`) + `output_layer` trainable,
  stages 1–3 frozen — the proven M12 R002–R012 surface.
- **AMD harness**: copy of M12's with the builder swapped to
  `build_rgck_hires_net` and one new knob (`--global_token_mode`).

## What is new

- [`model.py`](model.py): `HiResROIAlignRegionTokenizer` (resolution-agnostic;
  decouples the feature-map size from the FC's 7×7 grid) and
  `build_rgck_hires_net(...)` (builds the M12 model, swaps in the tokenizer).
- `--global_token_mode`:
  - `exact` (default): the global token is the genuine AdaFace embedding of the
    112 face — **identical to M12 / the B0 baseline**, so M15 R001 vs M12 R011
    isolates *only* the regional-resolution change. Costs one extra cheap 112
    pass.
  - `roi`: the global token is the full-face box pooled from the hi-res map
    (single pass, marginally weaker global signal).

## Runs

- **R001** ([`AMD/run_m15_r001.sh`](AMD/run_m15_r001.sh)): the M12 R011 champion
  recipe (symmetric forward + comparison-only fusion + relation aux 0.05 +
  role-matched hard negatives 30 % + differential LR + stage-4 unfreeze) with
  the hi-res tokenizer, `global_token_mode=exact`. Outputs to `output/001/`.

The delta M15 R001 − M12 R011 isolates **hi-res in-distribution region tokens
vs 112 resize-crop tokens** (same 31.0 M trainable params, same head, same data
recipe).

## VRAM (12 GB target — AMD RX 6750 XT)

Trainable params 31.0 M (44.2 %), identical to M12 R011. The body runs once at
224 (vs M12's 5× at 112), so backbone compute is comparable. Fits 12 GB at
`BATCH_SIZE=8 GRAD_ACCUM=4`. If you hit OOM, use `BATCH_SIZE=4 GRAD_ACCUM=8`
(same effective batch 32).

## Quick start

```bash
bash models/15_rgck_hires/AMD/run_m15_r001.sh
```

## Reference

- M12 proposal/runs: `../12_rgck_net/` (R013 is the 112-bound ROI-Align baseline)
- The 112-vs-feature-map distinction: AdaFace FC head needs 7×7; conv body is
  resolution-agnostic (`../10_adaface_facor/adaface_iresnet.py`).
