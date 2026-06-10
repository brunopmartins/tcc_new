#!/usr/bin/env python3
"""
CV-fold ensemble for Model 12 RGCK-Net.

Loads the 5 ``best.pt`` checkpoints from a CV run (e.g. ``output/014/`` for
R011), runs each on the canonical FIW test split, averages sigmoid
probabilities, and reports ensemble metrics. Threshold-based metrics use
the mean of the per-fold val-selected thresholds (no test-set snooping).

The 5 CV folds all evaluate on the *same* fixed RFIW Track-I test set
(13 425 pairs), so averaging predictions is well-defined.

Usage:
    python cv_ensemble.py --cv_dir /path/to/output/014 [--out_dir /path/to/output/014/ensemble]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ["MIOPEN_FIND_MODE"] = "FAST"

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "AMD"))

from rocm_utils import setup_rocm_environment, clear_rocm_cache  # noqa: E402
from config import DataConfig  # noqa: E402
from dataset import KinshipPairDataset, get_transforms  # noqa: E402
from evaluation import KinshipMetrics, print_metrics  # noqa: E402
from protocol import (  # noqa: E402
    apply_data_root_override,
    get_checkpoint_threshold,
    resolve_dataset_root,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import build_rgck_net  # noqa: E402


RGCK_MEAN = [0.5, 0.5, 0.5]
RGCK_STD = [0.5, 0.5, 0.5]
RGCK_IMG_SIZE = 224


def parse_args():
    p = argparse.ArgumentParser(description="CV-fold ensemble for M12 RGCK-Net")
    p.add_argument("--cv_dir", type=str, required=True,
                   help="Top-level CV directory containing fold_{0..N}/.")
    p.add_argument("--fold_pattern", type=str, default="fold_*",
                   help="Glob for fold subdirs (default fold_*).")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to write ensemble_metrics.txt + ensemble_probs.npz. "
                        "Defaults to <cv_dir>/ensemble/")
    p.add_argument("--dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--aligned_root", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--rocm_device", type=int, default=0)
    return p.parse_args()


def discover_folds(cv_dir: Path, pattern: str) -> List[Path]:
    folds = sorted(cv_dir.glob(pattern))
    folds = [f for f in folds if (f / "checkpoints" / "best.pt").exists()]
    if not folds:
        raise FileNotFoundError(f"No fold checkpoints under {cv_dir}/{pattern}")
    return folds


def build_from_checkpoint(ckpt_path: Path, device: torch.device):
    """Reconstruct an M12 model from a checkpoint's stored ``model_config``."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = checkpoint.get("model_config", checkpoint.get("protocol", {}).get("model_config", {}))
    model = build_rgck_net(
        adaface_weights=None,  # state_dict overrides
        embedding_dim=mc.get("embedding_dim", 512),
        cross_attn_heads=mc.get("cross_attn_heads", 4),
        cross_attn_layers=mc.get("cross_attn_layers", 1),
        gate_hidden=mc.get("gate_hidden", 128),
        classifier_hidden=mc.get("classifier_hidden", 512),
        dropout=mc.get("dropout", 0.2),
        freeze_backbone=mc.get("freeze_backbone", True),
        unfreeze_last_stage=mc.get("unfreeze_last_stage", False),
        unfreeze_extra_stage3_tail=mc.get("unfreeze_extra_stage3_tail", False),
        aux_relation_head=mc.get("aux_relation_head", False),
        num_relation_classes=mc.get("num_relation_classes", 11),
        symmetric_forward=mc.get("symmetric_forward", False),
        comparison_only_fusion=mc.get("comparison_only_fusion", False),
        roi_align_tokenizer=mc.get("roi_align_tokenizer", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    threshold = get_checkpoint_threshold(checkpoint)
    img_size = int(mc.get("img_size", RGCK_IMG_SIZE))
    return model, threshold, img_size, mc


@torch.no_grad()
def infer_probs(model, loader, device) -> tuple:
    probs, labels, relations = [], [], []
    for batch in tqdm(loader, desc="Inference", leave=False):
        img1 = batch["img1"].to(device, non_blocking=True)
        img2 = batch["img2"].to(device, non_blocking=True)
        labs = batch["label"]
        rels = batch.get("relation", [None] * len(labs))
        out = model(img1, img2)
        logit = out[0]
        p = torch.sigmoid(logit).cpu().numpy().flatten()
        probs.extend(p.tolist())
        labels.extend(labs.numpy().tolist())
        if rels and rels[0] is not None:
            relations.extend(list(rels))
    return (
        np.asarray(probs, dtype=float),
        np.asarray(labels, dtype=int),
        relations if relations else None,
    )


def main() -> None:
    args = parse_args()
    setup_rocm_environment(visible_devices=str(args.rocm_device))

    cv_dir = Path(args.cv_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else cv_dir / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.rocm_device}")

    print("\n" + "=" * 60)
    print("M12 RGCK-Net — CV-fold ensemble")
    print("=" * 60)
    print(f"CV dir:    {cv_dir}")
    print(f"Out dir:   {out_dir}")

    folds = discover_folds(cv_dir, args.fold_pattern)
    print(f"Folds:     {[f.name for f in folds]}")

    # Build the test loader once. All folds share the same FIW test split.
    # img_size is read from each ckpt; assert consistency.
    first_ckpt = folds[0] / "checkpoints" / "best.pt"
    _, _, img_size, _ = build_from_checkpoint(first_ckpt, device)
    del _  # don't hold the model in memory yet

    data_config = DataConfig(
        image_size=img_size, normalize_mean=RGCK_MEAN, normalize_std=RGCK_STD,
    )
    apply_data_root_override(data_config, args.dataset, args.data_root)

    test_dataset = KinshipPairDataset(
        root_dir=resolve_dataset_root(data_config, args.dataset),
        dataset_type=args.dataset, split="test",
        transform=get_transforms(data_config, train=False),
        split_seed=42, negative_ratio=1.0,
        aligned_root=args.aligned_root,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Test samples: {len(test_dataset)}\n")

    per_fold_probs: List[np.ndarray] = []
    per_fold_thresholds: List[float] = []
    per_fold_metrics: List[Dict[str, float]] = []
    labels_ref: np.ndarray = None
    relations_ref = None

    for k, fold_dir in enumerate(folds):
        ckpt_path = fold_dir / "checkpoints" / "best.pt"
        print(f"[fold {k}] loading {ckpt_path}")
        model, threshold, this_img_size, mc = build_from_checkpoint(ckpt_path, device)
        if this_img_size != img_size:
            raise RuntimeError(
                f"img_size mismatch across folds ({this_img_size} vs {img_size})."
            )
        if threshold is None:
            print(f"[fold {k}] no stored threshold; falling back to 0.5")
            threshold = 0.5
        print(f"[fold {k}] val-selected threshold: {threshold:.3f}")
        per_fold_thresholds.append(threshold)

        probs, labels, relations = infer_probs(model, test_loader, device)
        if labels_ref is None:
            labels_ref = labels
            relations_ref = relations
        else:
            assert np.array_equal(labels, labels_ref), \
                f"fold {k} test labels diverged from fold 0"

        # Per-fold metrics (for verification against the original test_metrics_rocm.txt)
        mc_calc = KinshipMetrics(threshold=threshold)
        mc_calc.update(predictions=probs, labels=labels, relations=relations)
        m = mc_calc.compute()
        per_fold_metrics.append({k_: v for k_, v in m.items() if not isinstance(v, dict)})
        print(f"[fold {k}] reproduced Test AUC: {m['roc_auc']:.4f}")

        per_fold_probs.append(probs)

        # Free GPU memory before loading the next fold.
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        clear_rocm_cache()

    probs_mat = np.stack(per_fold_probs, axis=0)  # (n_folds, n_test)
    ensemble_probs = probs_mat.mean(axis=0)

    ensemble_threshold = float(np.mean(per_fold_thresholds))
    print(f"\nEnsemble threshold (mean of folds): {ensemble_threshold:.3f}")

    metrics_calc = KinshipMetrics(threshold=ensemble_threshold)
    metrics_calc.update(
        predictions=ensemble_probs, labels=labels_ref, relations=relations_ref,
    )
    ens_metrics = metrics_calc.compute()

    print("\n" + "=" * 60)
    print("Ensemble metrics")
    print("=" * 60)
    print_metrics(ens_metrics, prefix="Ensemble ")

    # Also report per-fold mean ± std for direct comparison.
    keys = [
        "roc_auc", "average_precision", "tar@far=0.001", "tar@far=0.01",
        "tar@far=0.1", "accuracy", "balanced_accuracy", "precision",
        "recall", "f1",
    ]
    print("\nPer-fold reproduction (mean ± std, n=5):")
    summary = {}
    for k in keys:
        vals = np.array([m.get(k, np.nan) for m in per_fold_metrics], dtype=float)
        summary[k] = (float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)))
        print(f"  {k:<22} {summary[k][0]:.4f} ± {summary[k][1]:.4f}")

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "ensemble_probs.npz",
        probs_per_fold=probs_mat,
        ensemble_probs=ensemble_probs,
        labels=labels_ref,
        relations=np.array(relations_ref, dtype=object) if relations_ref else None,
        thresholds=np.array(per_fold_thresholds, dtype=float),
        ensemble_threshold=ensemble_threshold,
    )

    results_path = out_dir / "ensemble_metrics.txt"
    with open(results_path, "w") as f:
        f.write("M12 RGCK-Net — CV-fold ensemble metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"CV dir:              {cv_dir}\n")
        f.write(f"Folds used:          {[fp.name for fp in folds]}\n")
        f.write(f"Ensemble threshold:  {ensemble_threshold:.4f} (mean of fold thresholds)\n")
        f.write(f"Per-fold thresholds: {per_fold_thresholds}\n\n")
        f.write("Ensemble metrics (averaged sigmoid probs across folds):\n")
        for k, v in ens_metrics.items():
            if not isinstance(v, dict):
                f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
        f.write("\nPer-fold reproduction (mean ± std, n=5):\n")
        for k, (mean, std) in summary.items():
            f.write(f"  {k}: {mean:.4f} ± {std:.4f}\n")
        f.write("\nPer-fold raw metrics:\n")
        for k, m in enumerate(per_fold_metrics):
            f.write(f"  fold {k}: ")
            f.write(", ".join(
                f"{kk}={vv:.4f}"
                for kk, vv in m.items()
                if isinstance(vv, float) and kk in keys
            ))
            f.write("\n")

    print(f"\nSaved metrics:  {results_path}")
    print(f"Saved probs:    {out_dir / 'ensemble_probs.npz'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
