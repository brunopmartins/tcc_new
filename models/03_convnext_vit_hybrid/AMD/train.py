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
from losses import CosineContrastiveLoss, ContrastiveLoss
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
    parser.add_argument("--unfreeze_after", type=int, default=0,
                        help="Unfreeze backbones after N epochs (requires --freeze_backbones). "
                             "Backbones get a lower LR than the head (see --backbone_lr_factor).")
    parser.add_argument("--backbone_lr_factor", type=float, default=0.01,
                        help="Backbone LR = head_lr * factor (default: 0.01 = 100x lower)")
    parser.add_argument("--partial_unfreeze", action="store_true",
                        help="Only unfreeze last 2 ConvNeXt stages + last 4 ViT blocks "
                             "(keeps early universal feature layers frozen)")

    # Ablation
    parser.add_argument("--ablation_mode", type=str, default=None,
                        choices=["convnext_only", "vit_only"],
                        help="Run ablation with single backbone")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "step", "none"],
                        help="Learning-rate scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Warmup epochs before scheduler takes over")
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help="Minimum LR for cosine scheduler")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout rate for projection and fusion layers")
    parser.add_argument("--negative_ratio", type=float, default=3.0,
                        help="Negative pairs per positive pair during training")
    parser.add_argument("--eval_negative_ratio", type=float, default=1.0,
                        help="Negative pairs per positive pair for val/test")
    parser.add_argument("--train_negative_strategy", type=str, default="relation_matched",
                        choices=["random", "relation_matched"],
                        help="How to sample training negatives")
    parser.add_argument("--eval_negative_strategy", type=str, default="random",
                        choices=["random", "relation_matched"],
                        help="How to sample validation/test negatives")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without AUC improvement)")

    # Loss
    parser.add_argument("--loss", type=str, default="contrastive",
                        choices=["bce", "contrastive", "cosine_contrastive"])
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Temperature for contrastive loss")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Margin for supervised contrastive distance losses")

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

    def __init__(self, loss_type: str, temperature: float = 0.3, margin: float = 0.5):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.use_similarity = True
        elif loss_type == "contrastive":
            self.loss_fn = ContrastiveLoss(margin=margin, distance="cosine")
            self.use_similarity = False
        else:
            self.loss_fn = CosineContrastiveLoss(temperature=temperature)
            self.use_similarity = False

    def forward(self, emb1, emb2, labels, aux=None):
        if self.use_similarity:
            similarity = torch.sum(emb1 * emb2, dim=1)
            return self.loss_fn(similarity, labels)
        return self.loss_fn(emb1, emb2, labels)


class StagedUnfreezeTrainer(ROCmTrainer):
    """ROCmTrainer with staged backbone unfreezing.

    Supports full or partial unfreeze:
    - Full: all backbone parameters become trainable
    - Partial: only last 2 ConvNeXt stages + last 4 ViT blocks are unfrozen,
      keeping early universal feature layers (edges, textures) frozen
    """

    def __init__(self, *args, unfreeze_after: int = 0, backbone_lr_factor: float = 0.01,
                 partial_unfreeze: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfreeze_after = unfreeze_after
        self.backbone_lr_factor = backbone_lr_factor
        self.partial_unfreeze = partial_unfreeze
        self._unfrozen = False

    def _partial_unfreeze_backbones(self, model):
        """Unfreeze only the last 2 ConvNeXt stages and last 4 ViT blocks."""
        # ConvNeXt: stages[0-3], unfreeze stages[2] and stages[3]
        # (stage 2 = 57.9M params, stage 3 = 27.4M params)
        unfrozen_convnext_params = []
        for i, stage in enumerate(model.convnext.stages):
            if i >= 2:  # last 2 stages
                for param in stage.parameters():
                    param.requires_grad = True
                    unfrozen_convnext_params.append(param)
        # Also unfreeze convnext norm_pre and head (small layers)
        for name, param in model.convnext.named_parameters():
            if name.startswith("norm_pre") or name.startswith("head"):
                param.requires_grad = True
                unfrozen_convnext_params.append(param)

        # ViT: blocks[0-11], unfreeze blocks[8-11] (last 4)
        unfrozen_vit_params = []
        for i, block in enumerate(model.vit.blocks):
            if i >= 8:  # last 4 blocks
                for param in block.parameters():
                    param.requires_grad = True
                    unfrozen_vit_params.append(param)
        # Also unfreeze vit norm and fc_norm
        for name, param in model.vit.named_parameters():
            if name.startswith("norm") or name.startswith("fc_norm"):
                param.requires_grad = True
                unfrozen_vit_params.append(param)

        return unfrozen_convnext_params, unfrozen_vit_params

    def _full_unfreeze_backbones(self, model):
        """Unfreeze all backbone parameters."""
        for param in model.convnext.parameters():
            param.requires_grad = True
        for param in model.vit.parameters():
            param.requires_grad = True
        convnext_params = list(model.convnext.parameters())
        vit_params = list(model.vit.parameters())
        return convnext_params, vit_params

    def on_epoch_start(self, epoch: int) -> None:
        if self.unfreeze_after > 0 and epoch == self.unfreeze_after + 1 and not self._unfrozen:
            self._unfrozen = True
            mode = "partial" if self.partial_unfreeze else "full"
            print(f"\n  >>> Unfreezing backbones at epoch {epoch} ({mode}) <<<")

            model = self.model
            if hasattr(model, 'module'):
                model = model.module

            if self.partial_unfreeze:
                convnext_params, vit_params = self._partial_unfreeze_backbones(model)
                print("  >>> Partial unfreeze: ConvNeXt stages[2,3] + ViT blocks[8-11]")
            else:
                convnext_params, vit_params = self._full_unfreeze_backbones(model)

            # Enable gradient checkpointing to save VRAM
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
                print("  >>> Gradient checkpointing enabled on backbones")

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"  >>> Trainable parameters: {trainable:,}")

            # Rebuild optimizer with layerwise LR
            head_lr = self.optimizer.param_groups[0]["lr"]
            backbone_lr = head_lr * self.backbone_lr_factor

            self.optimizer = torch.optim.AdamW([
                {"params": convnext_params, "lr": backbone_lr},
                {"params": vit_params, "lr": backbone_lr},
                {"params": model.fusion.parameters(), "lr": head_lr},
                {"params": model.projection.parameters(), "lr": head_lr},
            ], weight_decay=self.config.weight_decay)

            # Rebuild scheduler for remaining epochs
            remaining = self.config.num_epochs - epoch
            if self.config.scheduler == "cosine" and remaining > 0:
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=remaining, eta_min=self.config.min_lr
                )
            print(f"  >>> Head LR: {head_lr:.2e}, Backbone LR: {backbone_lr:.2e}")
            print(f"  >>> Scheduler reset for {remaining} remaining epochs\n")


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

    set_global_seed(args.seed)

    device = get_rocm_device(args.rocm_device)
    print(f"\nUsing device: {device}")

    # Config
    train_data_config = DataConfig(
        split_seed=args.seed,
        negative_ratio=args.negative_ratio,
        num_workers=args.num_workers,
    )
    test_data_config = DataConfig(
        split_seed=args.seed,
        negative_ratio=args.negative_ratio,
        num_workers=args.num_workers,
    )
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
        scheduler="none" if args.scheduler == "none" else args.scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.disable_amp,
        patience=args.patience,
    )

    # Dataloaders for training
    print(f"\nLoading {args.train_dataset} dataset for training...")
    train_loader, val_loader, _ = create_dataloaders(
        config=train_data_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_type=args.train_dataset,
        train_negative_ratio=args.negative_ratio,
        eval_negative_ratio=args.eval_negative_ratio,
        train_negative_sampling_strategy=args.train_negative_strategy,
        eval_negative_sampling_strategy=args.eval_negative_strategy,
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
        negative_ratio=args.eval_negative_ratio,
        negative_sampling_strategy=args.eval_negative_strategy,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
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
            dropout=args.dropout,
        )

    # Apply ROCm optimizations
    model = optimize_for_rocm(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss
    loss_fn = HybridLoss(args.loss, args.temperature, args.margin)

    # Trainer
    if args.unfreeze_after > 0 and args.freeze_backbones:
        trainer = StagedUnfreezeTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            config=train_config,
            device=device,
            rocm_device_id=args.rocm_device,
            monitor_metric=train_config.monitor_metric,
            unfreeze_after=args.unfreeze_after,
            backbone_lr_factor=args.backbone_lr_factor,
            partial_unfreeze=args.partial_unfreeze,
        )
    else:
        trainer = ROCmTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            config=train_config,
            device=device,
            rocm_device_id=args.rocm_device,
            monitor_metric=train_config.monitor_metric,
        )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print(f"\nStarting ROCm-optimized training (fusion={args.fusion_type}, loss={args.loss})...")
    print(f"Loss: {args.loss}, Temperature: {args.temperature}, Margin: {args.margin}")
    print(
        f"Scheduler: {train_config.scheduler} "
        f"(warmup={train_config.warmup_epochs}, min_lr={train_config.min_lr:.1e})"
    )
    print(
        f"Negative ratio: train={args.negative_ratio:.2f}, "
        f"eval={args.eval_negative_ratio:.2f}"
    )
    print(
        f"Negative strategy: train={args.train_negative_strategy}, "
        f"eval={args.eval_negative_strategy}"
    )
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
    print("\nFinal evaluation on test set...")
    print_metrics(test_metrics, prefix="Test ")

    model_config = {
        "convnext_model": args.convnext_model,
        "vit_model": args.vit_model,
        "embedding_dim": args.embedding_dim,
        "fusion_type": args.fusion_type,
        "freeze_backbones": args.freeze_backbones,
        "ablation_mode": args.ablation_mode,
        "dropout": args.dropout,
    }
    protocol_metadata = build_protocol_metadata(
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        threshold=threshold,
        threshold_metric=train_config.threshold_metric,
        split_seed=args.seed,
        negative_ratio=args.eval_negative_ratio,
        monitor_metric=train_config.monitor_metric,
        args=args,
        extra={
            "model_config": model_config,
            "training_negative_ratio": args.negative_ratio,
            "training_negative_strategy": args.train_negative_strategy,
            "evaluation_negative_strategy": args.eval_negative_strategy,
        },
    )

    for checkpoint_name in ["best.pt", "final.pt"]:
        checkpoint_path = Path(args.checkpoint_dir) / checkpoint_name
        update_checkpoint_payload(checkpoint_path, {"model_config": model_config})
        update_checkpoint_metadata(checkpoint_path, protocol_metadata)

    # Save results
    results_path = Path(args.checkpoint_dir) / "test_results_rocm.txt"
    with open(results_path, "w") as f:
        f.write("AMD ROCm Training Results - ConvNeXt-ViT Hybrid\n")
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
