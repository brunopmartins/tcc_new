#!/usr/bin/env python3
"""
AMD ROCm Training script for ViT + FaCoR Cross-Attention model.

This script is optimized for AMD GPUs using the ROCm platform.

Usage:
    python train.py --dataset kinface --epochs 100 --loss cosine_contrastive
    python train.py --dataset fiw --vit_model vit_base_patch16_224 --freeze_vit
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
from losses import CosineContrastiveLoss, RelationGuidedContrastiveLoss, get_loss
from trainer import ROCmTrainer
from evaluation import print_metrics, evaluate_model

# Add parent directory for model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import ViTFaCoRModel, ViTFaCoRClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT-FaCoR Model (AMD ROCm)")

    # Dataset
    parser.add_argument("--train_dataset", type=str, default="fiw",
                        choices=["kinface", "fiw"],
                        help="Dataset to train on")
    parser.add_argument("--test_dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"],
                        help="Dataset to test on (cross-dataset evaluation)")
    parser.add_argument("--data_root", type=str, default=None)

    # Model
    parser.add_argument("--vit_model", type=str, default="vit_base_patch16_224",
                        help="ViT model variant")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--cross_attn_layers", type=int, default=2,
                        help="Number of cross-attention layers")
    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--freeze_vit", action="store_true",
                        help="Freeze ViT backbone")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Loss
    parser.add_argument("--loss", type=str, default="cosine_contrastive",
                        choices=["bce", "cosine_contrastive", "relation_guided"])
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for contrastive loss")

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


class ViTFaCoRLoss(nn.Module):
    """Custom loss wrapper for ViT-FaCoR model."""

    def __init__(self, loss_type: str = "cosine_contrastive", temperature: float = 0.07):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == "relation_guided":
            self.loss_fn = RelationGuidedContrastiveLoss(base_temperature=temperature)
        else:
            self.loss_fn = CosineContrastiveLoss(temperature=temperature)

    def forward(self, emb1, emb2, labels, attn_map=None):
        if self.loss_type == "bce":
            diff = emb1 - emb2
            product = emb1 * emb2
            similarity = torch.sum(emb1 * emb2, dim=1)
            return self.loss_fn(similarity, labels)
        elif self.loss_type == "relation_guided" and attn_map is not None:
            attn_weights = attn_map.mean(dim=[1, 2, 3])
            return self.loss_fn(emb1, emb2, attn_weights)
        else:
            return self.loss_fn(emb1, emb2, labels)


class ViTFaCoRROCmTrainer(ROCmTrainer):
    """Custom ROCm trainer for ViT-FaCoR model."""

    def _compute_loss(self, outputs, labels):
        """Handle ViT-FaCoR output format."""
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            emb1, emb2, attn_map = outputs[0], outputs[1], outputs[2]
            return self.loss_fn(emb1, emb2, labels, attn_map)
        return super()._compute_loss(outputs, labels)


def main():
    args = parse_args()

    # Setup ROCm environment
    print("\n" + "=" * 60)
    print("AMD ROCm Training - ViT-FaCoR Cross-Attention Model")
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

    # Create training dataloaders
    print(f"\nLoading {args.train_dataset} dataset for training...")
    train_loader, val_loader, _ = create_dataloaders(
        config=data_config,
        batch_size=args.batch_size,
        dataset_type=args.train_dataset,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Create test dataloader
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
    print("\nCreating ViT-FaCoR model...")
    model = ViTFaCoRModel(
        vit_model=args.vit_model,
        pretrained=True,
        embedding_dim=args.embedding_dim,
        num_cross_attn_layers=args.cross_attn_layers,
        cross_attn_heads=args.cross_attn_heads,
        freeze_vit=args.freeze_vit,
    )

    # Apply ROCm optimizations
    model = optimize_for_rocm(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss
    loss_fn = ViTFaCoRLoss(args.loss, args.temperature)

    # Trainer
    trainer = ViTFaCoRROCmTrainer(
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
    print("\nStarting ROCm-optimized training...")
    print(f"Loss: {args.loss}, Temperature: {args.temperature}")
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
        f.write("AMD ROCm Training Results - ViT-FaCoR\n")
        f.write("=" * 40 + "\n")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")

    print(f"\nTraining complete!")
    print(f"Trained on: {args.train_dataset}, Tested on: {args.test_dataset}")
    print(f"Best model saved to {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
