"""
MTCNN face alignment + 224×224 crop preprocessing for ViT-based encoders.

Project-wide utility (reusable by M09, M10, and future models). Derived from
``models/08_arcface_retrieval/preprocessing/align_faces.py``, which outputs the
canonical 112×112 ArcFace template; this version doubles the template (×2) so
the same five-point similarity transform lands on a 224×224 canvas — the
native input size of ViT-B/16, DINOv2 (ViT-B/14 patch grid post-resize), and
ImageNet ViT-L/16. No downsampling = no quality loss for high-resolution
ViT-based face encoders.

Why a separate dataset directory:
    ``datasets/FIW_aligned/`` (also 224×224) was produced by
    ``tools/align_fiw_dataset.py`` using insightface/buffalo_l. This script
    produces ``datasets/FIW_aligned_224/`` using facenet-pytorch MTCNN +
    Umeyama similarity transform onto the ArcFace 5-point template scaled to
    224 — the canonical ArcFace landmark layout, at 2× resolution.

Usage:
    python tools/align_faces_224.py \\
        --dataset fiw \\
        --src-root /home/bruno/Desktop/tcc_new/datasets/FIW \\
        --dst-root /home/bruno/Desktop/tcc_new/datasets/FIW_aligned_224

Notes:
- Uses facenet-pytorch MTCNN (BSD-licensed).
- Falls back to center-crop + resize when MTCNN fails to detect a face (rare
  on FIW, but ensures every image produces an output — important because the
  pair lists reference exact paths).
- Output JPGs (matching original FIW format so the existing dataset code in
  models/shared/dataset.py finds them via `glob("*.jpg")`).
- Resumes automatically: skips already-existing output files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ArcFace canonical 5-point template (eyes + nose + mouth corners) for 112×112.
# All target sizes are derived from this base via a uniform scale factor.
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(
        description="Align FIW/KinFaceW face crops to the ArcFace 5-point "
                    "template scaled to a configurable canvas (default 224)."
    )
    p.add_argument("--dataset", choices=["fiw", "kinface"], default="fiw")
    p.add_argument("--src-root", required=True,
                   help="Root of the original dataset (e.g. datasets/FIW)")
    p.add_argument("--dst-root", required=True,
                   help="Where to write aligned crops")
    p.add_argument("--output-size", type=int, default=224,
                   help="Output square size in pixels (default: 224). The "
                        "ArcFace 112-template is scaled by output_size/112.")
    p.add_argument("--rocm-device", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32,
                   help="MTCNN inference batch size (currently unused; "
                        "kept for compatibility with the 112 script)")
    p.add_argument("--use-cpu", action="store_true",
                   help="Run MTCNN on CPU (slower but frees VRAM if training "
                        "is happening in parallel)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N images (smoke test).")
    return p.parse_args()


def umeyama_similarity(src: np.ndarray, dst: np.ndarray):
    """Umeyama similarity transform: compute 2D similarity matrix mapping src→dst."""
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        T[:dim, :dim] = np.eye(dim)
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt
    var_src = src_demean.var(axis=0).sum()
    scale = 1.0 / var_src * (S * d).sum() if var_src > 0 else 1.0
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale
    return T


def align_face(img: Image.Image, landmarks: np.ndarray, output_size: int,
               template: np.ndarray) -> Image.Image:
    """Apply similarity transform mapping detected landmarks to ``template``.

    ``template`` must already be scaled to the target output size.
    """
    import cv2
    img_np = np.array(img)
    T = umeyama_similarity(landmarks.astype(np.float32), template)
    M = T[:2, :]
    warped = cv2.warpAffine(img_np, M, (output_size, output_size),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Image.fromarray(warped)


def fallback_center_crop(img: Image.Image, output_size: int) -> Image.Image:
    """When MTCNN fails: center-crop the square + resize. Coarse but safe."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = img.crop((left, top, left + side, top + side))
    return cropped.resize((output_size, output_size), Image.LANCZOS)


def gather_images(src_root: Path, dataset: str):
    """Yield (input_path, relative_path) tuples covering all images in the dataset."""
    if dataset == "fiw":
        # FIW layout: <root>/FIDs/<family>/<member>/*.jpg
        fids = src_root / "FIDs"
        if not fids.exists():
            # Fall back to recursive search.
            for p in src_root.rglob("*.jpg"):
                yield p, p.relative_to(src_root)
            for p in src_root.rglob("*.png"):
                yield p, p.relative_to(src_root)
            return
        for img in fids.rglob("*.jpg"):
            yield img, img.relative_to(src_root)
        for img in fids.rglob("*.png"):
            yield img, img.relative_to(src_root)
    elif dataset == "kinface":
        for sub in ["father-dau", "father-son", "mother-dau", "mother-son"]:
            sub_dir = src_root / "images" / sub
            if not sub_dir.exists():
                continue
            for img in sub_dir.rglob("*.jpg"):
                yield img, img.relative_to(src_root)
            for img in sub_dir.rglob("*.png"):
                yield img, img.relative_to(src_root)


def main():
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        sys.exit(f"src-root not found: {src_root}")

    output_size = int(args.output_size)
    if output_size < 16:
        sys.exit(f"output-size too small: {output_size}")

    # Scale the ArcFace 112 template to the requested canvas size.
    template = ARCFACE_TEMPLATE * (output_size / 112.0)

    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Output size: {output_size}×{output_size} "
          f"(template scaled by {output_size/112.0:.4f}×)")

    # Symlink auxiliary files (pair lists, metadata) so the existing dataset
    # loader in models/shared/dataset.py can find them under dst_root.
    if args.dataset == "fiw":
        for aux in ["track-I", "FIW_PIDs_v2.csv"]:
            src_aux = src_root / aux
            dst_aux = dst_root / aux
            if src_aux.exists() and not dst_aux.exists():
                dst_aux.symlink_to(src_aux)
                print(f"  symlinked {aux} -> {src_aux}")

    device = (torch.device(f"cuda:{args.rocm_device}")
              if (torch.cuda.is_available() and not args.use_cpu)
              else torch.device("cpu"))
    print(f"Device: {device}")

    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        sys.exit("facenet-pytorch is required. Install via `pip install facenet-pytorch`")

    # MTCNN configured to keep all faces ≥ 20 px, return 5-point landmarks.
    # ``image_size`` here only affects MTCNN's internal crop (which we discard —
    # we use the returned landmarks ourselves), so leaving it at 112 is fine.
    mtcnn = MTCNN(image_size=112, margin=0, min_face_size=20,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709,
                  post_process=False, select_largest=True,
                  keep_all=False, device=device)

    n_processed = 0
    n_skipped = 0
    n_fallback = 0

    images = list(gather_images(src_root, args.dataset))
    if args.limit:
        images = images[: args.limit]
    print(f"Found {len(images)} images.")

    for src_path, rel_path in tqdm(images, desc="Aligning"):
        dst_path = dst_root / rel_path.with_suffix(".jpg")
        if dst_path.exists():
            n_skipped += 1
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"  failed to open {src_path}: {e}")
            n_skipped += 1
            continue

        try:
            # detect() returns (boxes, probs, landmarks) — landmarks shape (n, 5, 2).
            _, _, landmarks = mtcnn.detect(img, landmarks=True)
        except Exception:
            landmarks = None

        if landmarks is not None and len(landmarks) > 0:
            aligned = align_face(img, landmarks[0],
                                 output_size=output_size, template=template)
        else:
            aligned = fallback_center_crop(img, output_size=output_size)
            n_fallback += 1

        aligned.save(dst_path)
        n_processed += 1

    print()
    print(f"Processed: {n_processed}")
    print(f"Skipped (already done or unreadable): {n_skipped}")
    print(f"Fallback (no face detected, center-crop): {n_fallback}  "
          f"({100*n_fallback/max(n_processed,1):.1f}%)")
    print(f"Output tree: {dst_root}")


if __name__ == "__main__":
    main()
