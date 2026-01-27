#!/usr/bin/env python3
"""
AMD ROCm Training script for Age Synthesis + All-vs-All Comparison model.

This script is optimized for AMD GPUs using the ROCm platform.

Usage:
    python train.py --dataset kinface --epochs 100 --batch_size 32
    python train.py --dataset fiw --use_age_synthesis

Requirements:
    - PyTorch with ROCm support
    - AMD GPU with ROCm drivers installed
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Setup ROCm environment before other imports
os.environ["MIOPEN_FIND_MODE"] = "FAST"
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"

# Add shared utilities to path
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
from config import DataConfig, TrainConfig, AgeSynthesisConfig, get_config
from dataset import create_dataloaders, KinshipPairDataset, get_transforms
from losses import ContrastiveLoss, CosineContrastiveLoss, get_loss
from trainer import ROCmTrainer
from evaluation import print_metrics

# Add parent directory for model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import AgeSynthesisComparisonModel, create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Age Synthesis Model (AMD ROCm)")

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

    # ROCm specific
    parser.add_argument("--rocm_device", type=int, default=0,
                        help="ROCm GPU device ID")
    parser.add_argument("--disable_amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--gfx_version", type=str, default=None,
                        help="Override GFX version (e.g., '10.3.0' for compatibility)")

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

    # Setup ROCm environment
    print("\n" + "=" * 60)
    print("AMD ROCm Training - Age Synthesis Model")
    print("=" * 60)

    setup_rocm_environment(
        visible_devices=str(args.rocm_device),
        gfx_version=args.gfx_version,
    )

    # Check ROCm availability
    is_available, status = check_rocm_availability()
    print(f"\nROCm Status: {status}")

    if not is_available:
        print("WARNING: ROCm not available, falling back to CPU")

    # Print detailed ROCm info
    print_rocm_info()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = get_rocm_device(args.rocm_device)
    print(f"\nUsing device: {device}")

    # Config
    data_config = DataConfig()
    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.disable_amp,  # ROCm AMP support
    )

    # Create training dataloaders (FIW by default)
    print(f"\nLoading {args.train_dataset} dataset for training...")
    train_loader, val_loader, _ = create_dataloaders(
        config=data_config,
        batch_size=args.batch_size,
        dataset_type=args.train_dataset,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create test dataloader (KinFaceW by default for cross-dataset eval)
    print(f"Loading {args.test_dataset} dataset for testing...")
    from torch.utils.data import DataLoader
    test_dataset = KinshipPairDataset(
        root_dir=data_config.kinface_i_root if args.test_dataset == "kinface" else data_config.fiw_root,
        dataset_type=args.test_dataset,
        split="test",
        transform=get_transforms(data_config, train=False),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test samples: {len(test_dataset)}")

    # Create model
    print("\nCreating model...")
    model = AgeSynthesisComparisonModel(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        use_age_synthesis=args.use_age_synthesis,
        aggregation=args.aggregation,
    )

    # Apply ROCm optimizations
    model = optimize_for_rocm(model)

    # Loss function
    if args.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
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
                return self.base_loss(emb1.squeeze(), labels)
            else:
                return self.base_loss(emb1, emb2, labels)

    wrapped_loss = ModelLoss(loss_fn, args.loss)

    # ROCm Trainer
    trainer = ROCmTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=wrapped_loss,
        config=train_config,
        device=device,
        rocm_device_id=args.rocm_device,
    )

    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting ROCm-optimized training...")
    history = trainer.train()

    # Clear cache before final evaluation
    clear_rocm_cache()

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    from evaluation import evaluate_model
    test_metrics = evaluate_model(model, test_loader, device)
    print_metrics(test_metrics, prefix="Test ")

    # Save test results
    results_path = Path(args.checkpoint_dir) / "test_results_rocm.txt"
    with open(results_path, "w") as f:
        f.write("AMD ROCm Training Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Platform: AMD ROCm\n")
        f.write(f"Device: {device}\n")
        f.write(f"Trained on: {args.train_dataset}\n")
        f.write(f"Tested on: {args.test_dataset}\n")
        f.write("=" * 40 + "\n")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")

    print(f"\nTraining complete!")
    print(f"Trained on: {args.train_dataset}")
    print(f"Tested on: {args.test_dataset}")
    print(f"Best checkpoint saved to {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
