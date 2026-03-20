#!/usr/bin/env python3
"""
Training script for Age Synthesis + All-vs-All Comparison model.

Usage:
    python train.py --dataset kinface --epochs 100 --batch_size 32
    python train.py --dataset fiw --use_age_synthesis
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import DataConfig, TrainConfig, AgeSynthesisConfig, get_config
from dataset import create_dataloaders, KinshipPairDataset, get_transforms
from losses import ContrastiveLoss, CosineContrastiveLoss, get_loss
from trainer import Trainer, train_model
from evaluation import print_metrics
from protocol import (
    apply_data_root_override,
    build_protocol_metadata,
    evaluate_with_validation_threshold,
    load_best_checkpoint,
    resolve_dataset_root,
    save_json,
    set_global_seed,
    update_checkpoint_metadata,
    update_checkpoint_payload,
)

from model import AgeSynthesisComparisonModel, create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Age Synthesis Model")
    
    # Dataset
    parser.add_argument("--train_dataset", type=str, default="fiw",
                        choices=["kinface", "fiw"],
                        help="Dataset to train on")
    parser.add_argument("--test_dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"],
                        help="Dataset to test on (cross-dataset evaluation)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override default dataset path")
    
    # Model
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "arcface", "efficientnet"],
                        help="Backbone architecture")
    parser.add_argument("--embedding_dim", type=int, default=512,
                        help="Embedding dimension")
    parser.add_argument("--use_age_synthesis", action="store_true",
                        help="Enable age synthesis (requires pretrained model)")
    parser.add_argument("--aggregation", type=str, default="attention",
                        choices=["attention", "max", "mean"],
                        help="Aggregation method for age comparisons")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    
    # Loss
    parser.add_argument("--loss", type=str, default="bce",
                        choices=["bce", "contrastive", "cosine_contrastive"],
                        help="Loss function")
    
    # Misc
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()

    set_global_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Config
    train_data_config = DataConfig(split_seed=args.seed)
    test_data_config = DataConfig(split_seed=args.seed)
    apply_data_root_override(train_data_config, args.train_dataset, args.data_root)
    if args.train_dataset == args.test_dataset:
        apply_data_root_override(test_data_config, args.test_dataset, args.data_root)
    elif args.data_root:
        print("Using --data_root for the training dataset only because train/test datasets differ.")

    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        monitor_metric="roc_auc",
    )
    
    # Create training dataloaders (FIW by default)
    print(f"Loading {args.train_dataset} dataset for training...")
    train_loader, val_loader, _ = create_dataloaders(
        config=train_data_config,
        batch_size=args.batch_size,
        dataset_type=args.train_dataset,
        split_seed=args.seed,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create test dataloader (KinFaceW by default for cross-dataset eval)
    print(f"Loading {args.test_dataset} dataset for testing...")
    from torch.utils.data import DataLoader
    test_dataset = KinshipPairDataset(
        root_dir=resolve_dataset_root(test_data_config, args.test_dataset),
        dataset_type=args.test_dataset,
        split="test",
        transform=get_transforms(test_data_config, train=False),
        split_seed=args.seed,
        negative_ratio=test_data_config.negative_ratio,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("Creating model...")
    model = AgeSynthesisComparisonModel(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        use_age_synthesis=args.use_age_synthesis,
        aggregation=args.aggregation,
    )
    
    # Loss function — use class weighting to combat recall bias (issue #7)
    if args.loss == "bce":
        train_labels = torch.tensor(train_loader.dataset.labels, dtype=torch.float32)
        num_pos = train_labels.sum().item()
        num_neg = len(train_labels) - num_pos
        pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], device=device)
        print(f"BCE pos_weight: {pos_weight.item():.3f}  (pos={int(num_pos)}, neg={int(num_neg)})")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss == "contrastive":
        loss_fn = ContrastiveLoss(margin=1.0)
    else:
        loss_fn = CosineContrastiveLoss(temperature=0.07)
    
    # Custom loss wrapper for model output format
    class ModelLoss(nn.Module):
        def __init__(self, base_loss, loss_type):
            super().__init__()
            self.base_loss = base_loss
            self.loss_type = loss_type
        
        def forward(self, emb1, emb2, labels):
            if self.loss_type == "bce":
                # For BCE, we need the kinship score
                # This is handled differently in training loop
                return self.base_loss(emb1.squeeze(), labels)
            else:
                return self.base_loss(emb1, emb2, labels)
    
    wrapped_loss = ModelLoss(loss_fn, args.loss)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=wrapped_loss,
        config=train_config,
        device=device,
        monitor_metric=train_config.monitor_metric,
    )
    
    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    trainer.train()

    print("\nLoading best checkpoint for protocol evaluation...")
    load_best_checkpoint(model, args.checkpoint_dir, device)
    protocol_results = evaluate_with_validation_threshold(
        model,
        val_loader,
        test_loader,
        device,
        threshold_metric=train_config.threshold_metric,
    )
    threshold = protocol_results["threshold"]
    val_metrics = protocol_results["validation_metrics"]
    test_metrics = protocol_results["test_metrics"]

    print(f"Validation-selected threshold ({train_config.threshold_metric}): {threshold:.3f}")
    print_metrics(val_metrics, prefix="Validation ")
    print("\nFinal evaluation on test set...")
    print_metrics(test_metrics, prefix="Test ")

    model_config = {
        "backbone": args.backbone,
        "embedding_dim": args.embedding_dim,
        "use_age_synthesis": args.use_age_synthesis,
        "aggregation": args.aggregation,
    }
    protocol_metadata = build_protocol_metadata(
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        threshold=threshold,
        threshold_metric=train_config.threshold_metric,
        split_seed=args.seed,
        negative_ratio=train_data_config.negative_ratio,
        monitor_metric=train_config.monitor_metric,
        args=args,
        extra={"model_config": model_config},
    )

    for checkpoint_name in ["best.pt", "final.pt"]:
        checkpoint_path = Path(args.checkpoint_dir) / checkpoint_name
        update_checkpoint_payload(checkpoint_path, {"model_config": model_config})
        update_checkpoint_metadata(checkpoint_path, protocol_metadata)

    results_path = Path(args.checkpoint_dir) / "test_results.txt"
    with open(results_path, "w") as f:
        f.write(f"threshold: {threshold:.4f}\n")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")

    save_json(
        Path(args.checkpoint_dir) / "protocol_summary.json",
        {
            **protocol_metadata,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )
    
    print(f"\nTraining complete!")
    print(f"Trained on: {args.train_dataset}")
    print(f"Tested on: {args.test_dataset}")
    print(f"Best checkpoint saved to {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
