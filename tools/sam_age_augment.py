#!/usr/bin/env python3
"""
SAM (Style-based Age Manipulation) age augmentation preprocessing for FIW.

For each input face, generates age-variant versions at specified target ages
(default: 8, 25, 70 — child / young / elderly) and saves them to disk.

The output preserves the input dataset's directory tree, with each image
producing additional siblings: `<stem>__age_<N>.jpg`.

Used by Model 10 (AdaFace + FaCoR) to add generational invariance: at
inference time, M10 computes embeddings for the original image plus age
variants, then takes a weighted ensemble (original has higher weight).

SAM produces 1024×1024 output regardless of input size. We downsample to
match the original image size to keep the dataset compatible with the
existing data loaders.

Usage (process all FIW_aligned_224 images):
    python tools/sam_age_augment.py \\
        --src-root /home/bruno/Desktop/tcc_new/datasets/FIW_aligned_224 \\
        --dst-root /home/bruno/Desktop/tcc_new/datasets/FIW_aligned_224_aged \\
        --ages 8,25,70 \\
        --output-size 224

Smoke test on a single image:
    python tools/sam_age_augment.py --limit 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Set ROCm env BEFORE torch is touched in any of SAM's modules.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

# Path to SAM repo bundled inside M01.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAM_ROOT = PROJECT_ROOT / "models" / "01_age_synthesis_comparison" / "SAM"
sys.path.insert(0, str(SAM_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src-root", default=str(PROJECT_ROOT / "datasets" / "FIW_aligned"),
        help="Root containing aligned FIW images (the project's canonical "
             "aligned dataset at 224x224; mirrors FIW FIDs structure)",
    )
    p.add_argument(
        "--dst-root", default=str(PROJECT_ROOT / "datasets" / "FIW_aligned_aged"),
        help="Destination for age-augmented variants (mirrors src structure)",
    )
    p.add_argument(
        "--checkpoint", default=str(SAM_ROOT / "pretrained_models" / "sam_ffhq_aging.pt"),
        help="SAM pretrained checkpoint",
    )
    p.add_argument(
        "--ages", default="8,25,70",
        help="Comma-separated target ages",
    )
    p.add_argument(
        "--output-size", type=int, default=224,
        help="Final image size after SAM (default 224 to match the source)",
    )
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Inference batch size (SAM is heavy; 1 is conservative for 12 GB)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N source images (for smoke tests)",
    )
    p.add_argument("--gpu-id", type=int, default=0)
    return p.parse_args()


def gather_jpgs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.jpg"))


def build_sam(checkpoint_path: str):
    """Construct the pSp/SAM model and load the checkpoint."""
    from argparse import Namespace
    from models.psp import pSp

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    opts = dict(ckpt["opts"])
    opts["checkpoint_path"] = checkpoint_path
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()
    return net


def make_input(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to SAM's expected 3-channel tensor (256x256, [-1,1])."""
    img = img.convert("RGB").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0,1]
    arr = (arr - 0.5) / 0.5                          # to [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, 256, 256)
    return tensor


def add_age_channel(img: torch.Tensor, target_age: int) -> torch.Tensor:
    """Append a 4th channel filled with target_age/100, matching SAM convention."""
    age_value = target_age / 100.0
    age_channel = torch.full(
        (1, img.shape[1], img.shape[2]),
        age_value, dtype=img.dtype, device=img.device,
    )
    return torch.cat([img, age_channel], dim=0)


def tensor_to_pil(t: torch.Tensor, output_size: int) -> Image.Image:
    """Convert SAM's (3, 1024, 1024) [-1,1] output to a PIL image at output_size."""
    t = t.detach().cpu().float()
    t = (t.clamp(-1, 1) + 1.0) / 2.0  # to [0,1]
    if output_size != t.shape[-1]:
        t = F.interpolate(t.unsqueeze(0), size=output_size, mode="bilinear",
                          align_corners=False).squeeze(0)
    t = (t * 255).clamp(0, 255).numpy().astype(np.uint8)
    return Image.fromarray(t.transpose(1, 2, 0))


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        sys.exit(f"src-root not found: {src_root}")
    if not Path(args.checkpoint).exists():
        sys.exit(f"SAM checkpoint not found: {args.checkpoint}")

    # Symlink track-I/ and FIW_PIDs_v2.csv if present (so the existing
    # dataset loader can read pair lists from the new root).
    for aux in ("track-I", "FIW_PIDs_v2.csv"):
        src_aux = src_root / aux
        if src_aux.exists():
            dst_aux = dst_root / aux
            if not dst_aux.exists():
                # follow the symlink so the destination owns a real link to
                # the underlying FIW data (not a chain through aligned_224).
                target = src_aux.resolve()
                dst_aux.symlink_to(target)
                print(f"  symlinked {aux} -> {target}")

    target_ages = [int(a) for a in args.ages.split(",")]
    print(f"Target ages: {target_ages}")
    print(f"Output size: {args.output_size}")
    print(f"Source:      {src_root}")
    print(f"Destination: {dst_root}")

    images = gather_jpgs(src_root)
    if args.limit:
        images = images[: args.limit]
    print(f"Images to process: {len(images)}")

    print("Loading SAM...")
    net = build_sam(args.checkpoint)
    print("SAM ready.")

    start = time.time()
    n_processed = 0
    n_skipped = 0

    for img_path in tqdm(images, desc="SAM age augment"):
        rel = img_path.relative_to(src_root)
        dst_dir = dst_root / rel.parent
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Check what's already done
        outputs_needed = []
        for age in target_ages:
            out_path = dst_dir / f"{rel.stem}__age_{age}{rel.suffix}"
            if not out_path.exists():
                outputs_needed.append((age, out_path))

        if not outputs_needed:
            n_skipped += 1
            continue

        try:
            input_img = Image.open(img_path)
            input_tensor = make_input(input_img)  # (3, 256, 256)
        except Exception as e:
            print(f"  skip {img_path}: {e}")
            continue

        with torch.no_grad():
            for age, out_path in outputs_needed:
                input_with_age = add_age_channel(input_tensor, age).unsqueeze(0).cuda()
                out = net(input_with_age, resize=False, randomize_noise=False)
                if isinstance(out, tuple):
                    out = out[0]
                pil = tensor_to_pil(out[0], args.output_size)
                pil.save(out_path, quality=92)
                n_processed += 1

    elapsed = time.time() - start
    print(f"\nDone. {n_processed} images written, {n_skipped} sources had all outputs.")
    print(f"Elapsed: {elapsed:.1f}s "
          f"({elapsed / max(n_processed, 1):.2f}s per output)")


if __name__ == "__main__":
    main()
