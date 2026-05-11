# Overview — Model 10: AdaFace + FaCoR Cross-Attention

**Model:** 10 — AdaFace IR-101 (WebFace4M) + FaCoR Cross-Attention + supervised
cosine contrastive loss
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset:** FIW (FIW_aligned 224×224 → resized to 112 at load time)

This file aggregates per-run results once training has been performed.
Individual run details land in `run-NNN.md` files alongside this overview.

---

## Hypothesis recap

M10 takes M02's best-performing recipe (Test ROC-AUC 0.850 with ViT-B/16
end-to-end + bidirectional FaCoR cross-attention + cosine contrastive m=0.3)
and swaps the ImageNet ViT backbone for **AdaFace IR-101 pretrained on
WebFace4M**. The expectation is that face-discriminative pretraining +
end-to-end fine-tuning will let the FaCoR cross-attention pick up on
kinship-specific cues that ImageNet features cannot represent natively, while
avoiding the "anti-kinship trap" that frozen ArcFace exhibits in M08.

## Configuration baseline (defaults from run_pipeline.sh)

| Knob               | Default                  |
|--------------------|--------------------------|
| Backbone           | AdaFace IR-101 WebFace4M |
| Input              | 112×112, [-1, 1]         |
| Tokens             | 49 (7×7), 512-d          |
| Pos embed          | learnable (49 tokens)    |
| Global embed       | enabled (AdaFace pool)   |
| Cross-attn layers  | 2                        |
| Cross-attn heads   | 8                        |
| Dropout            | 0.2                      |
| Loss               | cosine_contrastive       |
| Temperature        | 0.3                      |
| Margin             | 0.3                      |
| Backbone trainable | yes (end-to-end)         |
| LR (peak)          | 5e-6                     |
| Scheduler          | cosine, warmup 5         |
| Min LR             | 1e-7                     |
| Batch              | 8 × grad-accum 4 (eff 32)|
| Patience           | 50                       |
| Epochs (max)       | 100                      |

## Run table

| | Run NNN |
|---|---|
| **Date** | TBD |
| **Purpose** | Replicate M02 recipe with AdaFace backbone |
| **Best Val AUC** | TBD |
| **Test ROC-AUC** | TBD |
| **Test Accuracy** | TBD |
| **Notes** | TBD |
