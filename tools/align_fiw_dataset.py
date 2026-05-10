#!/usr/bin/env python3
"""
Align FIW face crops to canonical landmarks (FaCoRNet/ArcFace style).

For each image in datasets/FIW/, runs MTCNN (via insightface) to detect a
face and its 5 landmarks, then applies a similarity transform that maps
the detected landmarks onto the canonical face-recognition layout. The
output is a 224×224 aligned face crop saved to datasets/FIW_aligned/.

Canonical landmarks at 224×224 (2× ArcFace's standard 112×112 template):
    left_eye    = (76.59, 103.39)
    right_eye   = (147.06, 103.00)
    nose        = (112.05, 143.47)
    left_mouth  = (83.10, 184.73)
    right_mouth = (141.46, 184.41)

Usage:
    python tools/align_fiw_dataset.py \\
        --input  datasets/FIW \\
        --output datasets/FIW_aligned \\
        --workers 8

Failure handling:
    - No face detected → save the original image resized to 224×224 (logged)
    - Multiple faces → use the one with highest det_score
    - Already-existing aligned crops are skipped (idempotent)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

os.environ.setdefault("ORT_LOG_LEVEL", "3")  # quiet onnxruntime

import cv2
import numpy as np


# Canonical 112×112 ArcFace landmarks, scaled to 224×224.
CANONICAL_LANDMARKS_224 = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
) * 2.0  # 112 → 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="FIW root (contains FIDs/)")
    p.add_argument("--output", required=True, help="output root for aligned crops")
    p.add_argument("--workers", type=int, default=8, help="parallel processes")
    p.add_argument("--size", type=int, default=224, help="output size (square)")
    p.add_argument("--det_size", type=int, default=320,
                   help="MTCNN detection resolution (higher = slower, more accurate)")
    p.add_argument("--min_score", type=float, default=0.3,
                   help="minimum det_score to accept a detection")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing aligned outputs")
    p.add_argument("--limit", type=int, default=0, help="process at most N images (0 = all)")
    return p.parse_args()


def _build_app(det_size: int):
    """Lazy import + initialize insightface FaceAnalysis per worker."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection"],
    )
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


# Worker-local state (initialized once per process).
_worker_app = None
_worker_args = None


def _worker_init(args_dict):
    global _worker_app, _worker_args
    _worker_args = args_dict
    _worker_app = _build_app(args_dict["det_size"])


def _align_one(task):
    """Process a single image. Returns ('ok'|'fallback'|'skip', input_path)."""
    global _worker_app, _worker_args
    if _worker_app is None:
        _worker_app = _build_app(_worker_args["det_size"])

    input_path, output_path = task
    out_path = Path(output_path)
    if out_path.exists() and not _worker_args["overwrite"]:
        return ("skip", input_path)

    out_size = _worker_args["size"]

    try:
        img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return ("error", input_path)

        faces = _worker_app.get(img_bgr)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not faces:
            # Fallback: resize original to target size, no alignment.
            resized = cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out_path), resized)
            return ("fallback", input_path)

        # Highest-confidence face.
        best = max(faces, key=lambda f: f.det_score)
        if best.det_score < _worker_args["min_score"]:
            resized = cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out_path), resized)
            return ("fallback", input_path)

        src_landmarks = best.kps.astype(np.float32)  # (5, 2)
        target_landmarks = CANONICAL_LANDMARKS_224.copy()
        if out_size != 224:
            target_landmarks = target_landmarks * (out_size / 224.0)

        # Similarity transform (rotation + uniform scale + translation), 5-point.
        M, _ = cv2.estimateAffinePartial2D(
            src_landmarks, target_landmarks,
            method=cv2.LMEDS,  # robust to landmark noise
        )
        if M is None:
            resized = cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out_path), resized)
            return ("fallback", input_path)

        aligned = cv2.warpAffine(
            img_bgr, M, (out_size, out_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        cv2.imwrite(str(out_path), aligned)
        return ("ok", input_path)

    except Exception as exc:
        return ("error", f"{input_path}: {exc!r}")


def main():
    args = parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Workers: {args.workers}, det_size: {args.det_size}, size: {args.size}")

    # Build task list: every .jpg under input_root preserves its relative path.
    print("Indexing input images...")
    tasks = []
    for p in input_root.rglob("*.jpg"):
        rel = p.relative_to(input_root)
        out = output_root / rel
        tasks.append((str(p), str(out)))

    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"  total: {len(tasks):,} images")

    if not tasks:
        print("Nothing to do.")
        return

    args_dict = {
        "det_size": args.det_size,
        "size": args.size,
        "min_score": args.min_score,
        "overwrite": args.overwrite,
    }

    counts = {"ok": 0, "fallback": 0, "skip": 0, "error": 0}
    start = time.time()
    last_log = start

    if args.workers <= 1:
        # In-process for debugging.
        global _worker_app
        _worker_app = _build_app(args.det_size)
        global _worker_args
        _worker_args = args_dict
        for i, task in enumerate(tasks):
            status, _ = _align_one(task)
            counts[status] += 1
            now = time.time()
            if now - last_log > 10:
                rate = (i + 1) / (now - start)
                eta = (len(tasks) - i - 1) / rate / 60
                print(f"  [{i+1:>6,d}/{len(tasks):,d}] ok={counts['ok']} fallback={counts['fallback']} "
                      f"err={counts['error']} skip={counts['skip']} rate={rate:.1f}/s ETA={eta:.1f}min",
                      flush=True)
                last_log = now
    else:
        with Pool(args.workers, initializer=_worker_init, initargs=(args_dict,)) as pool:
            for i, (status, _) in enumerate(pool.imap_unordered(_align_one, tasks, chunksize=16)):
                counts[status] += 1
                now = time.time()
                if now - last_log > 10:
                    rate = (i + 1) / (now - start)
                    eta = (len(tasks) - i - 1) / rate / 60
                    print(f"  [{i+1:>6,d}/{len(tasks):,d}] ok={counts['ok']} "
                          f"fallback={counts['fallback']} err={counts['error']} skip={counts['skip']} "
                          f"rate={rate:.1f}/s ETA={eta:.1f}min", flush=True)
                    last_log = now

    elapsed = time.time() - start
    total = sum(counts.values())
    print()
    print(f"Done in {elapsed/60:.1f} min ({total/elapsed:.1f} img/s)")
    print(f"  aligned:  {counts['ok']:>6,d} ({100*counts['ok']/total:.1f}%)")
    print(f"  fallback: {counts['fallback']:>6,d} ({100*counts['fallback']/total:.1f}%)")
    print(f"  skipped:  {counts['skip']:>6,d}")
    print(f"  errors:   {counts['error']:>6,d}")


if __name__ == "__main__":
    main()
