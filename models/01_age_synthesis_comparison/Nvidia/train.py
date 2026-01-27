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
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from config import DataConfig, TrainConfig, AgeSynthesisConfig, get_config
from dataset import create_dataloaders, KinshipPairDataset, get_transforms
from losses import ContrastiveLoss, CosineContrastiveLoss, get_loss
from trainer import Trainer, train_model
from evaluation import print_metrics

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
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Config
    data_config = DataConfig()
    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Create training dataloaders (FIW by default)
    print(f"Loading {args.train_dataset} dataset for training...")
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
    print("Creating model...")
    model = AgeSynthesisComparisonModel(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        use_age_synthesis=args.use_age_synthesis,
        aggregation=args.aggregation,
    )
    
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
    )
    
    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    history = trainer.train()
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    from evaluation import evaluate_model
    test_metrics = evaluate_model(model, test_loader, device)
    print_metrics(test_metrics, prefix="Test ")
    
    # Save test results
    results_path = Path(args.checkpoint_dir) / "test_results.txt"
    with open(results_path, "w") as f:
        for key, value in test_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
    
    print(f"\nTraining complete!")
    print(f"Trained on: {args.train_dataset}")
    print(f"Tested on: {args.test_dataset}")
    print(f"Best checkpoint saved to {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
