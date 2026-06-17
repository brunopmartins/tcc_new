# Model 17 — PG-RGCK (Parsing-Guided Region-Guided Cross Kinship)

M17 keeps the **M15 hi-res ROI-Align head verbatim** and the proven M12 R011
recipe, and changes exactly one thing: the four anatomical region boxes (eyes,
nose, mouth, jaw) are **per-image**, derived offline by a **face-parsing
segmentation network**, instead of the fixed `DEFAULT_REGIONS_224` rectangles.

## Why this, and why it is not M13 again

The methodology names the limitation directly: the region boxes are fixed
coordinates on an aligned face — a static anatomical prior, not dynamic part
detection; if alignment shifts, the crops drift. M13 (Landmark Graph) tried to
fix this with **canonical** template landmarks, but on pre-aligned FIW the
5 landmarks land at fixed template positions by construction, so M13 was, in its
own words, *"already a fixed-box model, just with smaller boxes"* — and it
underperformed.

The 5-point similarity alignment normalises only 4 DOF (rotation, scale,
translation). What it leaves varying per face — **face shape, jaw width, brow
position, cheek geometry, mouth shape, expression, residual pose** — is exactly
what a **dense per-image parser** captures. M17 uses that: each face's region
boxes are the tight bounding boxes of *that face's* parsed masks. This is the
first model in the project where the region geometry actually adapts per image.

## Mechanism

```
aligned face @224 ──(offline, once)──► face-parsing net ──► 19-class mask
                                                              │
                          group → eyes / nose / mouth / jaw   │  tight bbox per region
                                                              ▼
   tools/parse_faces_boxes.py  ──►  cache.npz {rel-path → (4,4) xyxy boxes}
                                                              │  (loaded by the dataloader)
   ───────────────────────────── training / eval ────────────┼─────────────────
   aligned face @IMG_SIZE ─► AdaFace conv body (once) ─► feature map
                                                              │
              global token (whole face)        per-IMAGE ROI-Align(box) → 7×7 → FC
                                                              │
                                  ── M15/M12 head verbatim ───┘
   (cross-region attn → gate → comparison-only fusion → BCE, symmetric forward,
    relation aux) — UNCHANGED. Only the box *source* differs from M15.
```

- **R001 = mask → tight per-face bbox → ROI-Align → FC.** Boxes move per image,
  but tokens still pass through AdaFace's `output_layer`, so they stay in the
  AdaFace embedding space and the per-region cosine similarities the head relies
  on remain valid. **0 extra trainable params** vs M15 (the parser is offline).
- **Fixed-box fallback per region:** when a mask is empty/tiny the cache stores
  the fixed DEFAULT box for that region, so every image always has valid boxes.
  The parser logs a per-region fallback rate (M13's stop-condition lesson: if
  fallback is high, it is effectively a fixed-box model again).
- **Geometric aug off:** train-time flip/rotation would desync the cached boxes,
  so it is auto-disabled when a cache is set (R001 trades that augmentation for
  registered boxes; symmetric forward already gives order-invariance).

## What is shared / what is new

- **Head + recipe:** M15 `build_rgck_hires_net` (which imports M12 `RGCKNet`
  verbatim) — region tokens, cross-region attention, gate, fusion classifier,
  symmetric forward, comparison-only fusion, relation aux.
- **New** ([`model.py`](model.py)): `ParseGuidedHiResTokenizer` (M15 tokenizer
  whose `forward(img, boxes)` takes per-image ROIs), `PGRGCKNet` (threads boxes
  into the tokenizer; **no new params**), `build_pg_rgck_net`.
- **New** ([`../../tools/parse_faces_boxes.py`](../../tools/parse_faces_boxes.py)):
  offline face-parsing → region-box `.npz`. Backends: `bisenet` (CelebAMask-HQ)
  and `dummy` (fixed boxes — a plumbing/sanity cache that reproduces M15).
- **Shared, additive (default-off, other models unaffected):** `DataConfig`
  gains `geometric_aug` and `region_box_cache`; `KinshipPairDataset` returns
  `boxes1`/`boxes2` when a cache is set; the M17 trainer threads boxes through
  train AND eval (val + final test) so there is no train/eval geometry mismatch.

## Runs

- **R001** ([`AMD/run_m17_r001.sh`](AMD/run_m17_r001.sh)): M15 R001 recipe +
  parsing boxes, `IMG_SIZE=160` (matches M15 R001 → clean delta). Output `output/001/`.
- **R002** ([`AMD/run_m17_r002.sh`](AMD/run_m17_r002.sh)): same, `IMG_SIZE=224`
  (14×14 map) so the adaptive boxes have finer detail to act on.

The contour-aware **mask-weighted ROI** variant (weight the 7×7 ROI by the soft
parse mask before the FC) is the planned next code step — not yet wired; R001/R002
are both the bbox mechanism.

## Quick start

```bash
# 1) build the box cache once (offline). Sanity first with --backend dummy:
python tools/parse_faces_boxes.py --backend dummy \
    --aligned-root /path/FIW_aligned --out /path/fiw_region_boxes.npz
# real parser:
python tools/parse_faces_boxes.py --backend bisenet \
    --bisenet-repo /path/face-parsing.PyTorch --weights /path/79999_iter.pth \
    --aligned-root /path/FIW_aligned --out /path/fiw_region_boxes.npz

# 2) train (cache path via env):
REGION_BOX_CACHE=/path/fiw_region_boxes.npz bash models/17_rgck_parse/AMD/run_m17_r001.sh
```

## Honest prior

Two prior results temper expectations: M13 found that leaving the AdaFace
embedding space hurts, and M16 R001 suggested the ~0.876 ceiling may be
representational/data-bound. M17 stays in the embedding space (low risk) but is
not expected to leap aggregate AUC. The plausible win is **robustness / low-FAR /
hard relations (grandparents)** from geometry that tracks each face. A null
result is still informative: with a *real* per-image parser (not M13's canonical
landmarks), it would localise the ceiling to the representation/data rather than
to region placement — closing the "adaptive geometry" question for good.

A built-in control: run R001 on a `--backend dummy` cache → it must reproduce
M15 (all boxes fixed). Any M17-vs-M15 difference then comes only from the parser.

## VRAM (12 GB — AMD RX 6750 XT)

Identical to M15 (parser is offline; 0 extra trainable params). R001 @160 fits
batch 8 × accum 4; R002 @224 is heavier (drop to batch 4 × accum 8 if OOM).

## Reference

- Face parsing: BiSeNet (Yu et al., 2018) trained on CelebAMask-HQ (Lee et al.,
  2020), 19 classes.
- M15 hi-res tokenizer: `../15_rgck_hires/README.md`. M12 R011 recipe:
  `../12_rgck_net/run-review/run-011.md`. M13 (why canonical landmarks failed):
  `../13_landmark_graph_kinship_transformer/run-review/run-001.md`.
