#!/usr/bin/env python3
"""B0 5-fold CV evaluation.

The B0 baseline is no-training inference: AdaFace IR-101 (frozen) gives an
embedding per face, the pair score is cosine similarity in [-1, 1] mapped
to [0, 1], and the val-selected F1-optimal threshold is applied to the
fixed RFIW Track-I test set.

Since the model is frozen, threshold-invariant metrics (AUC, AP, TAR@FAR=*)
are deterministic across folds — only the val-selected threshold varies per
fold, which affects F1/accuracy/precision/recall. We report both faces:

- per-fold threshold-dependent metrics (5 values, mean ± std);
- threshold-invariant metrics (single deterministic value).

Optimization: we extract embeddings + compute the test cosine similarities
exactly once. For each of the 5 folds we just re-compute the val
similarities (the val pairs differ per fold) and re-pick the threshold.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
from dataset import (  # noqa: E402
    KinshipPairDataset,
    create_fiw_5fold_train_val_loaders,
    get_transforms,
)
from evaluation import (  # noqa: E402
    compute_metrics_from_predictions,
    find_optimal_threshold,
    print_metrics,
)
from protocol import apply_data_root_override, resolve_dataset_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from adaface_iresnet import adaface_ir101  # noqa: E402


@torch.no_grad()
def compute_pair_similarities(model, loader, device):
    sims, labels, relations = [], [], []
    for batch in tqdm(loader, desc="Similarities", leave=False):
        img1 = batch["img1"].to(device, non_blocking=True)
        img2 = batch["img2"].to(device, non_blocking=True)
        labs = batch["label"]
        rels = batch.get("relation", [None] * len(labs))

        emb1 = F.normalize(model(img1), dim=-1)
        emb2 = F.normalize(model(img2), dim=-1)
        sim = (emb1 * emb2).sum(dim=-1).cpu().numpy()

        sims.append(sim)
        labels.append(labs.numpy())
        if rels and rels[0] is not None:
            relations.extend(list(rels))
    return (
        np.concatenate(sims),
        np.concatenate(labels).astype(int),
        relations if relations else None,
    )


def parse_args():
    p = argparse.ArgumentParser(description="B0 5-fold CV evaluation")
    p.add_argument("--weights", type=str, default=str(
        Path(__file__).parent / "weights" / "adaface_ir101_webface4m.pth"
    ))
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--aligned_root", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str,
                   default=str(Path(__file__).parent / "output" / "cv"))
    p.add_argument("--rocm_device", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_rocm_environment(visible_devices=str(args.rocm_device))

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.rocm_device}")

    print("\n" + "=" * 60)
    print("B0 — AdaFace Cosine — 5-fold CV evaluation")
    print("=" * 60)
    print(f"Weights:      {args.weights}")
    print(f"Out dir:      {out_dir}")
    print(f"Num folds:    {args.num_folds}")

    print("\nLoading AdaFace IR-101 (frozen)...")
    model = adaface_ir101(weights_path=args.weights)
    model.to(device).eval()

    # Data config — same as evaluate.py: AdaFace expects 112×112 [-1, 1]
    data_config = DataConfig(
        image_size=112, normalize_mean=[0.5, 0.5, 0.5], normalize_std=[0.5, 0.5, 0.5],
    )
    apply_data_root_override(data_config, "fiw", args.data_root)

    # Fixed test set first — deterministic across folds.
    transform = get_transforms(data_config, train=False)
    root_dir = resolve_dataset_root(data_config, "fiw")
    test_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type="fiw",
        split="test",
        transform=transform,
        split_seed=args.seed,
        negative_ratio=1.0,
        aligned_root=args.aligned_root,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"\nFixed test set: {len(test_dataset):,} pairs")

    print("\n[setup] Computing test similarities once (deterministic)...")
    test_sims, test_labels, test_relations = compute_pair_similarities(
        model, test_loader, device,
    )
    test_scores = (test_sims + 1.0) / 2.0
    print(f"  test scores range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")

    # Per-fold loop
    per_fold_metrics: List[Dict[str, float]] = []
    per_fold_thresholds: List[float] = []

    for k in range(args.num_folds):
        print("\n" + "-" * 60)
        print(f"[fold {k}] computing val similarities and picking threshold")
        print("-" * 60)
        _, val_loader, _ = create_fiw_5fold_train_val_loaders(
            config=data_config,
            fold_k=k,
            n_folds=args.num_folds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_seed=args.seed,
            aligned_root=args.aligned_root,
        )
        print(f"  fold {k} val pairs: {len(val_loader.dataset):,}")
        val_sims, val_labels, _ = compute_pair_similarities(
            model, val_loader, device,
        )
        val_scores = (val_sims + 1.0) / 2.0
        threshold, val_f1 = find_optimal_threshold(val_scores, val_labels, metric="f1")
        per_fold_thresholds.append(float(threshold))
        print(f"  fold {k} threshold: {threshold:.4f} (val F1 {val_f1:.4f})")

        test_metrics = compute_metrics_from_predictions(
            predictions=test_scores,
            labels=test_labels,
            threshold=threshold,
            relations=test_relations,
        )
        test_metrics["threshold"] = float(threshold)
        per_fold_metrics.append(
            {k_: v for k_, v in test_metrics.items() if not isinstance(v, dict)}
        )
        print(
            f"  fold {k} test: AUC={test_metrics['roc_auc']:.4f}, "
            f"AP={test_metrics['average_precision']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, Acc={test_metrics['accuracy']:.4f}, "
            f"TAR@0.001={test_metrics['tar@far=0.001']:.4f}"
        )

    # Aggregate
    keys = [
        "roc_auc", "average_precision", "tar@far=0.001", "tar@far=0.01",
        "tar@far=0.1", "accuracy", "balanced_accuracy", "precision",
        "recall", "f1",
    ]
    print("\n" + "=" * 60)
    print("Aggregate (mean ± std, n=5)")
    print("=" * 60)
    summary: Dict[str, Tuple[float, float]] = {}
    for k_ in keys:
        vals = np.array([m.get(k_, np.nan) for m in per_fold_metrics], dtype=float)
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals, ddof=1))
        summary[k_] = (mean, std)
        marker = "  (deterministic — model frozen, test set fixed)" if std < 1e-9 else ""
        print(f"  {k_:<22} {mean:.4f} ± {std:.4f}{marker}")
    print(f"  thresholds picked: {per_fold_thresholds}")
    print(f"  threshold mean: {np.mean(per_fold_thresholds):.4f} ± "
          f"{np.std(per_fold_thresholds, ddof=1):.4f}")

    # Write
    results_path = out_dir / "cv_metrics.txt"
    with open(results_path, "w") as f:
        f.write("B0 — AdaFace Cosine — 5-fold CV evaluation\n")
        f.write("=" * 60 + "\n")
        f.write("Threshold-invariant metrics (AUC, AP, TAR@FAR) are\n")
        f.write("deterministic across folds (frozen model + fixed test set).\n")
        f.write("Threshold-dependent metrics (F1, Acc, Precision, Recall)\n")
        f.write("vary because each fold has a different val split, and the\n")
        f.write("val-selected F1-optimal threshold differs.\n\n")
        f.write("Per-fold thresholds (val-selected): ")
        f.write(", ".join(f"{t:.4f}" for t in per_fold_thresholds))
        f.write("\n\n")
        f.write("Aggregate (mean ± std, n=5):\n")
        for k_, (mean, std) in summary.items():
            marker = "  (deterministic)" if std < 1e-9 else ""
            f.write(f"  {k_}: {mean:.4f} ± {std:.4f}{marker}\n")
        f.write("\nPer-fold raw metrics:\n")
        for k, m in enumerate(per_fold_metrics):
            f.write(f"  fold {k}:\n")
            for k_, v in m.items():
                if isinstance(v, float):
                    f.write(f"    {k_}: {v:.4f}\n")

    # Probs npz
    np.savez(
        out_dir / "cv_probs.npz",
        test_scores=test_scores,
        test_labels=test_labels,
        test_relations=np.array(test_relations, dtype=object) if test_relations else None,
        per_fold_thresholds=np.array(per_fold_thresholds, dtype=float),
    )

    print(f"\nSaved metrics:  {results_path}")
    print(f"Saved probs:    {out_dir / 'cv_probs.npz'}")
    print("\nDone.")
    clear_rocm_cache()


if __name__ == "__main__":
    main()
