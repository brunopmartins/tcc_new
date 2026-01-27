#!/usr/bin/env python3
"""
Training script for Unified Kinship Model.

Usage:
    python train.py --dataset kinface --epochs 100
    python train.py --dataset fiw --use_age_synthesis --use_cross_attention
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from config import DataConfig, TrainConfig
from dataset import create_dataloaders, KinshipPairDataset, get_transforms
from losses import CosineContrastiveLoss, CombinedLoss
from trainer import Trainer
from evaluation import print_metrics, evaluate_model

from model import UnifiedKinshipModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train Unified Kinship Model")
    
    # Dataset
    parser.add_argument("--train_dataset", type=str, default="fiw",
                        choices=["kinface", "fiw"],
                        help="Dataset to train on")
    parser.add_argument("--test_dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"],
                        help="Dataset to test on (cross-dataset evaluation)")
    parser.add_argument("--data_root", type=str, default=None)
    
    # Model components
    parser.add_argument("--use_age_synthesis", action="store_true",
                        help="Enable age synthesis component")
    parser.add_argument("--use_cross_attention", action="store_true", default=True,
                        help="Enable cross-attention component")
    parser.add_argument("--no_cross_attention", action="store_true",
                        help="Disable cross-attention")
    
    # Architecture
    parser.add_argument("--convnext_model", type=str, default="convnext_base")
    parser.add_argument("--vit_model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--fusion_type", type=str, default="concat",
                        choices=["concat", "gated"])
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--cross_attn_layers", type=int, default=2)
    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--age_aggregation", type=str, default="attention",
                        choices=["attention", "max", "mean"])
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)  # Lower due to large model
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    
    # Loss
    parser.add_argument("--loss", type=str, default="combined",
                        choices=["bce", "contrastive", "combined"])
    parser.add_argument("--contrastive_weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.07)
    
    # Misc
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation", type=int, default=2,
                        help="Gradient accumulation steps")
    
    return parser.parse_args()


class UnifiedLoss(nn.Module):
    """Combined loss for unified model."""
    
    def __init__(
        self,
        loss_type: str = "combined",
        contrastive_weight: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.contrastive_weight = contrastive_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = CosineContrastiveLoss(temperature=temperature)
    
    def forward(self, output, labels):
        """
        Compute loss from model output dictionary.
        
        Args:
            output: Dictionary with 'logits', 'emb1', 'emb2'
            labels: Ground truth kinship labels
        """
        logits = output["logits"].squeeze()
        emb1 = output["emb1"]
        emb2 = output["emb2"]
        
        if self.loss_type == "bce":
            return self.bce_loss(logits, labels)
        elif self.loss_type == "contrastive":
            return self.contrastive_loss(emb1, emb2, labels)
        else:  # combined
            bce = self.bce_loss(logits, labels)
            contrastive = self.contrastive_loss(emb1, emb2, labels)
            return (1 - self.contrastive_weight) * bce + self.contrastive_weight * contrastive


class UnifiedTrainer(Trainer):
    """Custom trainer for unified model."""
    
    def __init__(self, *args, gradient_accumulation: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_accumulation = gradient_accumulation
        self.accumulation_step = 0
    
    def train_epoch(self) -> float:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            img1 = batch["img1"].to(self.device)
            img2 = batch["img2"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            if self.config.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    output = self.model(img1, img2)
                    loss = self.loss_fn(output, labels)
                    loss = loss / self.gradient_accumulation
                
                self.scaler.scale(loss).backward()
            else:
                output = self.model(img1, img2)
                loss = self.loss_fn(output, labels)
                loss = loss / self.gradient_accumulation
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.config.use_amp:
                    if self.config.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item() * self.gradient_accumulation:.4f}"})
        
        return total_loss / num_batches
    
    def _compute_loss(self, outputs, labels):
        """Handle unified model output format."""
        return self.loss_fn(outputs, labels)


def evaluate_unified_model(model, dataloader, device):
    """Custom evaluation for unified model."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            labels = batch["label"]
            
            output = model(img1, img2)
            preds = torch.sigmoid(output["logits"]).cpu()
            
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
    
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    binary = (preds > 0.5).astype(int)
    
    return {
        "accuracy": accuracy_score(labels, binary),
        "f1": f1_score(labels, binary),
        "roc_auc": roc_auc_score(labels, preds),
    }


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
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
    
    # Training dataloaders
    print(f"Loading {args.train_dataset} dataset for training...")
    train_loader, val_loader, _ = create_dataloaders(
        config=data_config,
        batch_size=args.batch_size,
        dataset_type=args.train_dataset,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Test dataloader (cross-dataset evaluation)
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
    use_cross_attn = args.use_cross_attention and not args.no_cross_attention
    
    print("\nCreating Unified Kinship Model...")
    print(f"  Age synthesis: {args.use_age_synthesis}")
    print(f"  Cross-attention: {use_cross_attn}")
    print(f"  Fusion type: {args.fusion_type}")
    
    model = UnifiedKinshipModel(
        use_age_synthesis=args.use_age_synthesis,
        use_cross_attention=use_cross_attn,
        convnext_model=args.convnext_model,
        vit_model=args.vit_model,
        fusion_type=args.fusion_type,
        embedding_dim=args.embedding_dim,
        num_cross_attn_layers=args.cross_attn_layers,
        cross_attn_heads=args.cross_attn_heads,
        aggregation=args.age_aggregation,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss
    loss_fn = UnifiedLoss(
        loss_type=args.loss,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
    )
    
    # Trainer
    trainer = UnifiedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=train_config,
        device=device,
        gradient_accumulation=args.gradient_accumulation,
    )
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print(f"\nStarting training...")
    print(f"Loss: {args.loss}, Gradient accumulation: {args.gradient_accumulation}")
    history = trainer.train()
    
    # Final evaluation
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate_unified_model(model, test_loader, device)
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    print("="*50)
    
    print(f"\nTraining complete!")
    print(f"Trained on: {args.train_dataset}, Tested on: {args.test_dataset}")
    print(f"Best model: {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
