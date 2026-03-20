#!/usr/bin/env python3
"""
5-fold cross-validation for ViT-FaCoR Cross-Attention model.

Each fold uses 4/5 of the KinFaceW-I pairs for training and 1/5 for testing,
following the standard KinFaceW evaluation protocol.  Results are reported as
mean ± std across all folds.

Usage:
    python train_cv.py --n_folds 5
    CV_FOLDS=5 bash AMD/run_pipeline.sh
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ["MIOPEN_FIND_MODE"] = "FAST"
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "AMD"))

from rocm_utils import (
    setup_rocm_environment,
    check_rocm_availability,
    get_rocm_device,
    optimize_for_rocm,
    print_rocm_info,
    clear_rocm_cache,
)
from config import DataConfig, TrainConfig
from dataset import create_cv_fold_loaders
from evaluation import KinshipMetrics, print_metrics
from protocol import aggregate_numeric_metrics, evaluate_with_validation_threshold, save_json, set_global_seed

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import build_vit_facor_model

# Reuse loss + trainer from train.py (avoids duplication)
from train import ViTFaCoRLoss, ViTFaCoRROCmTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="5-fold CV for ViT-FaCoR (AMD ROCm)")

    # Dataset
    parser.add_argument("--train_dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"])

    # Model
    parser.add_argument("--vit_model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--cross_attn_layers", type=int, default=2)
    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_vit", action="store_true")
    parser.add_argument("--unfreeze_after_epoch", type=int, default=0)
    parser.add_argument("--unfreeze_last_vit_blocks", type=int, default=0)
    parser.add_argument("--use_classifier_head", action="store_true")

    # Training
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "step", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--negative_ratio", type=float, default=1.0)
    parser.add_argument("--eval_negative_ratio", type=float, default=1.0)
    parser.add_argument("--train_negative_strategy", type=str, default="random",
                        choices=["random", "relation_matched"])
    parser.add_argument("--eval_negative_strategy", type=str, default="random",
                        choices=["random", "relation_matched"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50)

    # Loss
    parser.add_argument("--loss", type=str, default="cosine_contrastive",
                        choices=["bce", "contrastive", "cosine_contrastive", "relation_guided"])
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=0.5)

    # ROCm
    parser.add_argument("--rocm_device", type=int, default=0)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--gfx_version", type=str, default=None)

    # Output
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Where to write cv_summary.json (defaults to checkpoint_dir)")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def evaluate_loader(model, loader, device):
    """Run cosine-similarity evaluation on a DataLoader, return metrics dict."""
    import torch.nn.functional as F

    model.eval()
    metrics_calc = KinshipMetrics(threshold=0.5)
    with torch.no_grad():
        for batch in loader:
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]
            relations = batch.get("relation", [None] * len(labels))
            emb1, emb2, _ = model(img1, img2)
            scores = (F.cosine_similarity(emb1, emb2, dim=1) + 1) / 2
            metrics_calc.update(
                predictions=scores,
                labels=labels,
                relations=relations if relations[0] is not None else None,
            )
    return metrics_calc.compute()


def run_fold(fold_k, args, device, data_config):
    """Train and evaluate one fold. Returns test metrics dict."""
    n_folds = args.n_folds
    print(f"\n{'─' * 56}")
    print(f"  Fold {fold_k + 1} / {n_folds}")
    print(f"{'─' * 56}")

    train_loader, val_loader, test_loader = create_cv_fold_loaders(
        config=data_config,
        fold_k=fold_k,
        n_folds=n_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_type=args.train_dataset,
        split_seed=args.seed,
        train_negative_ratio=args.negative_ratio,
        eval_negative_ratio=args.eval_negative_ratio,
        train_negative_sampling_strategy=args.train_negative_strategy,
        eval_negative_sampling_strategy=args.eval_negative_strategy,
    )
    print(
        f"  Train pairs: {len(train_loader.dataset)}, "
        f"Val pairs: {len(val_loader.dataset)}, "
        f"Test pairs: {len(test_loader.dataset)}"
    )

    fold_ckpt_dir = Path(args.checkpoint_dir) / f"fold_{fold_k + 1}"
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Fresh model for every fold
    set_global_seed(args.seed + fold_k)

    model = build_vit_facor_model(
        vit_model=args.vit_model,
        pretrained=True,
        embedding_dim=args.embedding_dim,
        num_cross_attn_layers=args.cross_attn_layers,
        cross_attn_heads=args.cross_attn_heads,
        dropout=args.dropout,
        freeze_vit=args.freeze_vit,
        use_classifier_head=args.use_classifier_head,
    )
    model = optimize_for_rocm(model)

    fold_config = TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler="none" if args.scheduler == "none" else args.scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        checkpoint_dir=str(fold_ckpt_dir),
        use_amp=not args.disable_amp,
        patience=args.patience,
    )
    loss_fn = ViTFaCoRLoss(args.loss, args.temperature, args.margin)

    trainer = ViTFaCoRROCmTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=fold_config,
        device=device,
        monitor_metric="roc_auc",
        unfreeze_after_epoch=args.unfreeze_after_epoch,
        unfreeze_last_vit_blocks=args.unfreeze_last_vit_blocks,
    )
    trainer.train()

    # Load best checkpoint and get final metrics
    best_ckpt = fold_ckpt_dir / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    protocol_results = evaluate_with_validation_threshold(
        model,
        val_loader,
        test_loader,
        device,
        threshold_metric=fold_config.threshold_metric,
    )
    metrics = {
        **protocol_results["test_metrics"],
        "threshold": protocol_results["threshold"],
        "validation_roc_auc": protocol_results["validation_metrics"].get("roc_auc", 0.0),
    }
    print(f"\n  Fold {fold_k + 1} result — "
          f"AUC: {metrics['roc_auc']:.4f}, "
          f"Acc: {metrics['accuracy']:.4f}, "
          f"F1: {metrics['f1']:.4f}, "
          f"Thr: {metrics['threshold']:.3f}")

    clear_rocm_cache()
    return metrics


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print(f"ViT-FaCoR {args.n_folds}-Fold Cross-Validation (AMD ROCm)")
    print("=" * 60)

    setup_rocm_environment(
        visible_devices=str(args.rocm_device),
        gfx_version=args.gfx_version,
    )
    print_rocm_info()

    set_global_seed(args.seed)

    device = get_rocm_device(args.rocm_device)
    print(f"\nDevice: {device}")
    print(f"Folds: {args.n_folds}  |  Epochs/fold: {args.epochs}  |  "
          f"LR: {args.lr}  |  Patience: {args.patience}")

    data_config = DataConfig(
        split_seed=args.seed,
        negative_ratio=args.negative_ratio,
        num_workers=args.num_workers,
    )
    fold_results = []

    for fold_k in range(args.n_folds):
        metrics = run_fold(fold_k, args, device, data_config)
        fold_results.append(metrics)

    # ── Aggregate ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"CV Summary — {args.n_folds}-fold (mean ± std)")
    print(f"{'=' * 60}")

    summary = {
        "n_folds": args.n_folds,
        "dataset": args.train_dataset,
        "config": {
            "vit_model": args.vit_model,
            "lr": args.lr,
            "temperature": args.temperature,
            "margin": args.margin,
            "loss": args.loss,
            "epochs": args.epochs,
            "patience": args.patience,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_lr,
            "negative_ratio": args.negative_ratio,
            "eval_negative_ratio": args.eval_negative_ratio,
            "train_negative_strategy": args.train_negative_strategy,
            "eval_negative_strategy": args.eval_negative_strategy,
            "dropout": args.dropout,
            "use_classifier_head": args.use_classifier_head,
        },
        "fold_results": fold_results,
    }

    aggregate = aggregate_numeric_metrics(fold_results)
    summary.update(aggregate)
    for key, value in aggregate.items():
        if key.startswith("mean_"):
            metric_name = key[len("mean_"):]
            std_key = f"std_{metric_name}"
            print(f"  {metric_name:<20} {value:.4f} ± {aggregate.get(std_key, 0.0):.4f}")

    # Per-fold AUC table
    print(f"\n  Per-fold AUC:")
    for i, r in enumerate(fold_results):
        print(f"    Fold {i + 1}: {r['roc_auc']:.4f}")

    # Save summary
    results_dir = Path(args.results_dir) if args.results_dir else Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "cv_summary.json"
    save_json(summary_path, summary)
    print(f"\nCV summary saved to {summary_path}")


if __name__ == "__main__":
    main()
