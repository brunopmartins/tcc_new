# Run 04 — Local / GPU / KinFaceW-I / Age Synthesis ON (SAM loaded) — FAILED

**Date:** 2026-02-19
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM)
**Status:** FAILED — OOM during validation

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | Local (`AMD/run_pipeline.sh` + `.venv`) |
| Device | GPU — cuda:0 (gfx1031, ROCm 5.7, PyTorch 2.3.1+rocm5.7) |
| Dataset | KinFaceW-I only |
| Split | 70/15/15 → 746 train / 158 val / 162 test |
| Epochs | 1 of 100 (crashed before completing epoch 1 validation) |
| Batch size | 8 |
| Learning rate | 1e-4 (cosine warmup from 2e-5) |
| Weight decay | 1e-5 |
| Backbone | ResNet-50 |
| Age synthesis | **Enabled — SAM weights loaded and functional** |
| Aggregation | Attention |
| AMP | Enabled (ROCm AMP) |
| Loss | BCE with pos_weight=1.0 |
| Training speed | ~5.5 s/batch (~8.7 min for 94 batches) |

---

## What Happened

1. SAM weights (`sam_ffhq_aging.pt`, 2.2 GB) downloaded and loaded successfully — first time age synthesis was actually active.
2. Training phase completed all 94 batches of epoch 1 without issues (~8.7 min).
3. Validation crashed with OOM when `evaluate_model()` called `model(img1, img2)` which triggered `generate_age_variants()` → `age_encoder()` → `sam_model()`.

### Error

```
torch.cuda.OutOfMemoryError: HIP out of memory. Tried to allocate 4.50 GiB.
```

**Location:** `SAM/models/stylegan2/model.py:269` — `F.conv2d(input, weight, padding=self.padding, groups=batch)`

### Root Cause

The `AgeEncoder.forward()` passed the entire validation batch through SAM's StyleGAN2 decoder at once. StyleGAN2 generates 1024x1024 images, and a single batch forward pass allocates ~4.5 GB of intermediate buffers for the conv2d operations. Combined with training optimizer states, gradients, and cached activations already in VRAM, the 12 GB GPU could not accommodate the allocation.

During training, the forward pass also ran SAM, but the training loop processes one batch at a time with gradient accumulation. Validation runs immediately after training while all optimizer state is still resident in VRAM, leaving insufficient headroom for the large SAM decoder buffers.

---

## Partial Results (epoch 1, training only)

| Metric | Value |
|--------|-------|
| Train loss | 0.6935 (near random — expected for epoch 1 with warmup LR) |
| Batches completed | 94/94 |
| Validation | CRASHED |
| Test | NOT REACHED |

---

## Fix Applied

Modified `AgeEncoder.forward()` in `model.py` to process images one at a time through SAM instead of batched:

```python
# Before (OOM):
x_sam = self._preprocess_for_sam(x, target_age)          # [B, 4, 256, 256]
result = self.sam_model(x_sam, ...)                        # allocates ~4.5 GB for batch

# After (fixed):
results = []
for i in range(x.size(0)):
    x_i = self._preprocess_for_sam(x[i:i+1], target_age)  # [1, 4, 256, 256]
    out_i = self.sam_model(x_i, ...)                        # allocates ~0.6 GB per image
    results.append(out_i)
return torch.cat(results, dim=0)
```

This limits peak VRAM to a single-image StyleGAN2 pass. Verified working in a subsequent 1-epoch test that completed train + validation + test without OOM.

---

## Lessons

- SAM's StyleGAN2 decoder at 1024x1024 is extremely VRAM-hungry — a single batch of 8 images requires ~4.5 GB just for one conv2d layer.
- Training and validation share GPU memory; validation must account for optimizer state already resident from training.
- Per-image SAM processing trades throughput for memory safety — epoch time increases from ~22s (identity fallback) to ~771s (~12.9 min) but stays within the 12 GB VRAM budget.
