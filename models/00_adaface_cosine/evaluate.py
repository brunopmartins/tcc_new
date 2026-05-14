#!/usr/bin/env python3
"""
Baseline B0 — AdaFace Cosine Frozen.

No training. Loads AdaFace IR-101 (WebFace4M), extracts L2-normalised
embeddings for every face in val/test pairs, computes cosine similarity
per pair, selects an F1-optimal threshold on val, and reports test
metrics at that threshold.

This is the "no-adaptation reference" for the RGCK-Net comparison per
`baselines_rgck_net_tcc.md` §5. The hypothesis being tested:

> What's the kinship verification performance of an off-the-shelf face
> recognition model with no kinship-specific training?

If the RGCK-Net (and other adapted models) substantially exceeds this,
that's evidence that the kinship task requires more than face
identification.

Reports both threshold-invariant metrics (ROC AUC, Avg Precision,
TAR@FAR) and threshold-dependent metrics (Acc, F1, Precision, Recall
per relation) at the val-chosen threshold.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "AMD"))

from rocm_utils import setup_rocm_environment, clear_rocm_cache  # noqa: E402
from config import DataConfig  # noqa: E402
from dataset import KinshipPairDataset, get_transforms  # noqa: E402
from evaluation import (  # noqa: E402
    KinshipMetrics,
    compute_metrics_from_predictions,
    find_optimal_threshold,
    print_metrics,
)
from protocol import (  # noqa: E402
    apply_data_root_override,
    resolve_dataset_root,
)

sys.path.insert(0, str(Path(__file__).parent))
from adaface_iresnet import adaface_ir101  # noqa: E402


# AdaFace input convention (matches M09/M10/M11/M12 dataset configs)
ADAFACE_MEAN = [0.5, 0.5, 0.5]
ADAFACE_STD = [0.5, 0.5, 0.5]
ADAFACE_IMG_SIZE = 112


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline B0 — AdaFace Cosine Frozen")
    p.add_argument("--dataset", type=str, default="fiw",
                   choices=["kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--aligned_root", type=str, default=None,
                   help="Path to pre-aligned face crops (e.g. datasets/FIW_aligned).")
    p.add_argument("--adaface_weights", type=str,
                   default=str(Path(__file__).parent / "weights" / "adaface_ir101_webface4m.pth"))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).parent / "output"))
    p.add_argument("--rocm_device", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true",
                   help="Run on CPU (slow; useful when GPU is busy).")
    return p.parse_args()


@torch.no_grad()
def compute_pair_similarities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Return (cosine_sims, labels, relations) for every pair in the loader."""
    model.eval()
    sims, labels, relations = [], [], []

    for batch in tqdm(loader, desc=f"Extracting embeddings ({device.type})"):
        img1 = batch["img1"].to(device, non_blocking=True)
        img2 = batch["img2"].to(device, non_blocking=True)
        label_b = batch["label"]
        rel_b = batch.get("relation", [None] * len(label_b))

        emb1 = model(img1)
        emb2 = model(img2)
        if isinstance(emb1, tuple):
            emb1 = emb1[0]
        if isinstance(emb2, tuple):
            emb2 = emb2[0]
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        cos = (emb1 * emb2).sum(dim=1)
        sims.extend(cos.cpu().numpy().flatten().tolist())
        labels.extend(label_b.numpy().flatten().tolist())
        if rel_b and rel_b[0] is not None:
            relations.extend(list(rel_b))

    return (
        np.asarray(sims, dtype=float),
        np.asarray(labels, dtype=int),
        relations if relations else None,
    )


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("Baseline B0 — AdaFace Cosine Frozen (no training)")
    print("=" * 60)

    if args.cpu:
        device = torch.device("cpu")
        print("Device: CPU (slow path)")
    else:
        setup_rocm_environment(visible_devices=str(args.rocm_device))
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.rocm_device}")
        else:
            device = torch.device("cpu")
            print("WARNING: CUDA/ROCm not available; falling back to CPU.")
    print(f"Device: {device}")

    # Load AdaFace IR-101 weights
    print(f"Loading AdaFace IR-101 from {args.adaface_weights}...")
    model = adaface_ir101(weights_path=args.adaface_weights)
    model.to(device)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,} (all frozen, no training)")

    # Data configs (AdaFace 112×112 [-1, 1])
    data_config = DataConfig(
        image_size=ADAFACE_IMG_SIZE,
        normalize_mean=ADAFACE_MEAN,
        normalize_std=ADAFACE_STD,
    )
    apply_data_root_override(data_config, args.dataset, args.data_root)
    root_dir = resolve_dataset_root(data_config, args.dataset)
    print(f"Dataset: {args.dataset} from {root_dir}")
    if args.aligned_root:
        print(f"Aligned root: {args.aligned_root}")

    transform = get_transforms(data_config, train=False)

    val_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=args.dataset,
        split="val",
        transform=transform,
        split_seed=args.seed,
        negative_ratio=1.0,
        aligned_root=args.aligned_root,
    )
    test_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=args.dataset,
        split="test",
        transform=transform,
        split_seed=args.seed,
        negative_ratio=1.0,
        aligned_root=args.aligned_root,
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    print(f"\nVal pairs:  {len(val_dataset):,}")
    print(f"Test pairs: {len(test_dataset):,}")

    # Validation pass
    print("\n[1/2] Computing val similarities...")
    val_sims, val_labels, val_relations = compute_pair_similarities(model, val_loader, device)

    # Cosine similarity is in [-1, 1]; map to [0, 1] for compatibility with
    # threshold-finder and BCE-style scoring conventions used by the project.
    val_scores = (val_sims + 1.0) / 2.0

    print("\nFinding F1-optimal threshold on val...")
    threshold, val_f1 = find_optimal_threshold(val_scores, val_labels, metric="f1")
    print(f"  Val threshold (F1-optimal): {threshold:.4f}")
    print(f"  Val F1 at that threshold:   {val_f1:.4f}")

    val_metrics = compute_metrics_from_predictions(
        predictions=val_scores,
        labels=val_labels,
        threshold=threshold,
        relations=val_relations,
    )
    val_metrics["threshold"] = threshold
    print_metrics(val_metrics, prefix="Validation ")

    # Test pass
    print("\n[2/2] Computing test similarities...")
    test_sims, test_labels, test_relations = compute_pair_similarities(model, test_loader, device)
    test_scores = (test_sims + 1.0) / 2.0

    test_metrics = compute_metrics_from_predictions(
        predictions=test_scores,
        labels=test_labels,
        threshold=threshold,
        relations=test_relations,
    )
    test_metrics["threshold"] = threshold
    print("\n")
    print_metrics(test_metrics, prefix="Test ")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "test_metrics_rocm.txt"
    with open(results_path, "w") as f:
        f.write("AdaFace Cosine Frozen — Baseline B0 — Test Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Val-chosen threshold (F1-optimal on val): {threshold:.4f}\n\n")
        f.write("Validation metrics:\n")
        for k, v in val_metrics.items():
            if isinstance(v, dict):
                continue
            f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
        f.write("\nTest metrics:\n")
        for k, v in test_metrics.items():
            if isinstance(v, dict):
                continue
            f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")

        # Per-relation breakdown
        if "per_relation" in test_metrics:
            f.write("\nPer-relation test accuracy:\n")
            for rel, rel_m in sorted(test_metrics["per_relation"].items()):
                f.write(f"  {rel}: Acc={rel_m['accuracy']:.4f}, F1={rel_m['f1']:.4f}, N={rel_m['count']}\n")

    # Also save raw scores for reproducibility
    np.savez(
        out_dir / "scores.npz",
        val_sims=val_sims, val_labels=val_labels, val_relations=np.array(val_relations or []),
        test_sims=test_sims, test_labels=test_labels, test_relations=np.array(test_relations or []),
        val_threshold=threshold,
    )

    print(f"\nResults saved to {results_path}")
    print(f"Raw scores saved to {out_dir / 'scores.npz'}")

    clear_rocm_cache()


if __name__ == "__main__":
    main()
