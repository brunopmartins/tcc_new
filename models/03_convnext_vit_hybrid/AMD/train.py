#!/usr/bin/env python3
"""
AMD ROCm Training script for ConvNeXt + ViT Hybrid model.

This script is optimized for AMD GPUs using the ROCm platform.

Usage:
    python train.py --dataset kinface --fusion_type concat
    python train.py --dataset fiw --fusion_type attention --epochs 100
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
from config import DataConfig, TrainConfig
from dataset import create_dataloaders, KinshipPairDataset, get_transforms
from losses import CosineContrastiveLoss, ContrastiveLoss
from trainer import ROCmTrainer
from evaluation import print_metrics, evaluate_model

# Add parent directory for model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import ConvNeXtViTHybrid, AblationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-ViT Hybrid (AMD ROCm)")

    # Dataset
    parser.add_argument("--train_dataset", type=str, default="fiw",
                        choices=["kinface", "fiw"],
                        help="Dataset to train on")
    parser.add_argument("--test_dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"],
                        help="Dataset to test on (cross-dataset evaluation)")
    parser.add_argument("--data_root", type=str, default=None)

    # Model
    parser.add_argument("--convnext_model", type=str, default="convnext_base",
                        help="ConvNeXt variant")
    parser.add_argument("--vit_model", type=str, default="vit_base_patch16_224",
                        help="ViT variant")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--fusion_type", type=str, default="concat",
                        choices=["concat", "attention", "gated", "bilinear"],
                        help="Feature fusion strategy")
    parser.add_argument("--freeze_backbones", action="store_true",
                        help="Freeze pretrained backbones")

    # Ablation
    parser.add_argument("--ablation_mode", type=str, default=None,
                        choices=["convnext_only", "vit_only"],
                        help="Run ablation with single backbone")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Loss
    parser.add_argument("--loss", type=str, default="cosine_contrastive",
                        choices=["bce", "contrastive", "cosine_contrastive"])
    parser.add_argument("--temperature", type=float, default=0.07)

    # ROCm specific
    parser.add_argument("--rocm_device", type=int, default=0,
                        help="ROCm GPU device ID")
    parser.add_argument("--disable_amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--gfx_version", type=str, default=None,
                        help="Override GFX version for compatibility")

    # Misc
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


class HybridLoss(nn.Module):
    """Loss wrapper for hybrid model."""

    def __init__(self, loss_type: str, temperature: float = 0.07):
        super().__init__()
        if loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.use_similarity = True
        elif loss_type == "contrastive":
            self.loss_fn = ContrastiveLoss(margin=1.0)
            self.use_similarity = False
        else:
            self.loss_fn = CosineContrastiveLoss(temperature=temperature)
            self.use_similarity = False

    def forward(self, emb1, emb2, labels, aux=None):
        if self.use_similarity:
            similarity = torch.sum(emb1 * emb2, dim=1)
            return self.loss_fn(similarity, labels)
        return self.loss_fn(emb1, emb2, labels)


def main():
    args = parse_args()

    # Setup ROCm environment
    print("\n" + "=" * 60)
    print("AMD ROCm Training - ConvNeXt-ViT Hybrid Model")
    print("=" * 60)

    setup_rocm_environment(
        visible_devices=str(args.rocm_device),
        gfx_version=args.gfx_version,
    )

    is_available, status = check_rocm_availability()
    print(f"\nROCm Status: {status}")
    print_rocm_info()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
        use_amp=not args.disable_amp,
    )

    # Dataloaders for training
    print(f"\nLoading {args.train_dataset} dataset for training...")
    train_loader, val_loader, _ = create_dataloaders(
        config=data_config,
        batch_size=args.batch_size,
        dataset_type=args.train_dataset,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Test dataloader
    print(f"Loading {args.test_dataset} dataset for testing...")
    from torch.utils.data import DataLoader
    test_dataset = KinshipPairDataset(
        root_dir=data_config.kinface_i_root if args.test_dataset == "kinface" else data_config.fiw_root,
        dataset_type=args.test_dataset,
        split="test",
        transform=get_transforms(data_config, train=False),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test: {len(test_dataset)}")

    # Create model
    if args.ablation_mode:
        print(f"\nRunning ablation: {args.ablation_mode}")
        model = AblationModel(
            mode=args.ablation_mode,
            convnext_model=args.convnext_model,
            vit_model=args.vit_model,
            embedding_dim=args.embedding_dim,
        )
    else:
        print(f"\nCreating hybrid model with fusion: {args.fusion_type}")
        model = ConvNeXtViTHybrid(
            convnext_model=args.convnext_model,
            vit_model=args.vit_model,
            embedding_dim=args.embedding_dim,
            fusion_type=args.fusion_type,
            freeze_backbones=args.freeze_backbones,
        )

    # Apply ROCm optimizations
    model = optimize_for_rocm(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss
    loss_fn = HybridLoss(args.loss, args.temperature)

    # Trainer
    trainer = ROCmTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=train_config,
        device=device,
        rocm_device_id=args.rocm_device,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print(f"\nStarting ROCm-optimized training (fusion={args.fusion_type}, loss={args.loss})...")
    history = trainer.train()

    # Clear cache before evaluation
    clear_rocm_cache()

    # Final evaluation
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    print_metrics(test_metrics, prefix="Test ")

    # Save results
    results_path = Path(args.checkpoint_dir) / "test_results_rocm.txt"
    with open(results_path, "w") as f:
        f.write("AMD ROCm Training Results - ConvNeXt-ViT Hybrid\n")
        f.write("=" * 40 + "\n")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")

    print(f"\nTraining complete!")
    print(f"Trained on: {args.train_dataset}, Tested on: {args.test_dataset}")
    print(f"Best model: {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
