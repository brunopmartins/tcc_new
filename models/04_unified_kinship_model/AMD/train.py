#!/usr/bin/env python3
"""
AMD ROCm Training script for Unified Kinship Model.

This script is optimized for AMD GPUs using the ROCm platform.

Usage:
    python train.py --dataset kinface --epochs 100
    python train.py --dataset fiw --use_age_synthesis --use_cross_attention
"""
import argparse
import gc
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Setup ROCm environment before other imports
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # RX 6700/6750 XT (gfx1031)
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
from losses import CosineContrastiveLoss, CombinedLoss
from trainer import ROCmTrainer
from evaluation import print_metrics, evaluate_model
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

# Add parent directory for model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import UnifiedKinshipModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train Unified Kinship Model (AMD ROCm)")

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
    parser.add_argument("--batch_size", type=int, default=4)  # Low to avoid OOM; effective batch = batch_size * gradient_accumulation
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Loss
    parser.add_argument("--loss", type=str, default="combined",
                        choices=["bce", "contrastive", "combined"])
    parser.add_argument("--contrastive_weight", type=float, default=0.5)
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
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")

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


class UnifiedROCmTrainer(ROCmTrainer):
    """Custom ROCm trainer for unified model with gradient accumulation."""

    def __init__(self, *args, gradient_accumulation: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_accumulation = gradient_accumulation
        self.accumulation_step = 0

    def train_epoch(self) -> float:
        """Train for one epoch with gradient accumulation and ROCm optimizations."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        # Clear cache at start of epoch
        clear_rocm_cache()

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc="Training (ROCm)")

        for batch_idx, batch in enumerate(pbar):
            img1 = batch["img1"].to(self.device, non_blocking=True)
            img2 = batch["img2"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # Forward pass
            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    output = self.model(img1, img2)
                    loss = self.loss_fn(output, labels)
                    loss = loss / self.gradient_accumulation

                _loss_val = loss.item()
                self.scaler.scale(loss).backward()
            else:
                output = self.model(img1, img2)
                loss = self.loss_fn(output, labels)
                loss = loss / self.gradient_accumulation
                _loss_val = loss.item()
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.use_amp:
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

                self.optimizer.zero_grad(set_to_none=True)

            del output, loss
            clear_rocm_cache()
            gc.collect()

            total_loss += _loss_val * self.gradient_accumulation
            num_batches += 1

            pbar.set_postfix({"loss": f"{_loss_val * self.gradient_accumulation:.4f}"})

        if num_batches % self.gradient_accumulation != 0:
            if self.use_amp:
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
            self.optimizer.zero_grad(set_to_none=True)

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
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
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

    # Setup ROCm environment
    print("\n" + "=" * 60)
    print("AMD ROCm Training - Unified Kinship Model")
    print("=" * 60)

    setup_rocm_environment(
        visible_devices=str(args.rocm_device),
        gfx_version=args.gfx_version,
    )

    is_available, status = check_rocm_availability()
    print(f"\nROCm Status: {status}")
    print_rocm_info()

    set_global_seed(args.seed)

    device = get_rocm_device(args.rocm_device)
    print(f"\nUsing device: {device}")

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
        use_amp=not args.disable_amp,
        monitor_metric="roc_auc",
    )

    # Training dataloaders
    print(f"\nLoading {args.train_dataset} dataset for training...")
    train_loader, val_loader, _ = create_dataloaders(
        config=train_data_config,
        batch_size=args.batch_size,
        dataset_type=args.train_dataset,
        split_seed=args.seed,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Test dataloader
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

    # Apply ROCm optimizations
    model = optimize_for_rocm(model)

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
    trainer = UnifiedROCmTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=train_config,
        device=device,
        rocm_device_id=args.rocm_device,
        gradient_accumulation=args.gradient_accumulation,
        monitor_metric=train_config.monitor_metric,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print(f"\nStarting ROCm-optimized training...")
    print(f"Loss: {args.loss}, Gradient accumulation: {args.gradient_accumulation}")
    trainer.train()

    # Clear cache before evaluation
    clear_rocm_cache()

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

    print("\n" + "="*50)
    print("TEST RESULTS (AMD ROCm)")
    print("="*50)
    print(f"Threshold: {threshold:.3f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    print("="*50)

    model_config = {
        "use_age_synthesis": args.use_age_synthesis,
        "use_cross_attention": use_cross_attn,
        "convnext_model": args.convnext_model,
        "vit_model": args.vit_model,
        "fusion_type": args.fusion_type,
        "embedding_dim": args.embedding_dim,
        "cross_attn_layers": args.cross_attn_layers,
        "cross_attn_heads": args.cross_attn_heads,
        "age_aggregation": args.age_aggregation,
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

    # Save results
    results_path = Path(args.checkpoint_dir) / "test_results_rocm.txt"
    with open(results_path, "w") as f:
        f.write("AMD ROCm Training Results - Unified Kinship Model\n")
        f.write("=" * 40 + "\n")
        f.write(f"Threshold: {threshold:.4f}\n")
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
    print(f"Trained on: {args.train_dataset}, Tested on: {args.test_dataset}")
    print(f"Best model: {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
