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
from losses import ContrastiveLoss, CosineContrastiveLoss, RelationGuidedContrastiveLoss, get_loss
from trainer import ROCmTrainer
from evaluation import print_metrics, evaluate_model, KinshipMetrics
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
from model import build_vit_facor_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT-FaCoR Model (AMD ROCm)")

    # Dataset
    parser.add_argument("--train_dataset", type=str, default="kinface",
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
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate used in cross-attention and projection head")
    parser.add_argument("--freeze_vit", action="store_true",
                        help="Freeze ViT backbone")
    parser.add_argument("--unfreeze_after_epoch", type=int, default=0,
                        help="Unfreeze the last ViT blocks after this many frozen epochs")
    parser.add_argument("--unfreeze_last_vit_blocks", type=int, default=0,
                        help="How many final ViT blocks to unfreeze during training")
    parser.add_argument("--use_classifier_head", action="store_true",
                        help="Use the classifier scoring head instead of cosine similarity scoring")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "step", "none"],
                        help="Learning-rate scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Warmup epochs before scheduler takes over")
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help="Minimum LR for cosine scheduler")
    parser.add_argument("--negative_ratio", type=float, default=1.0,
                        help="Negative pairs sampled per positive pair during training")
    parser.add_argument("--eval_negative_ratio", type=float, default=1.0,
                        help="Negative pairs sampled per positive pair for val/test evaluation")
    parser.add_argument("--train_negative_strategy", type=str, default="random",
                        choices=["random", "relation_matched"],
                        help="How to sample training negatives")
    parser.add_argument("--eval_negative_strategy", type=str, default="random",
                        choices=["random", "relation_matched"],
                        help="How to sample validation/test negatives")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Dataloader workers")

    # Loss
    parser.add_argument("--loss", type=str, default="cosine_contrastive",
                        choices=["bce", "contrastive", "cosine_contrastive", "relation_guided"])
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
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without AUC improvement)")

    return parser.parse_args()


class ViTFaCoRLoss(nn.Module):
    """Custom loss wrapper for ViT-FaCoR model."""

    def __init__(self, loss_type: str = "cosine_contrastive", temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == "contrastive":
            self.loss_fn = ContrastiveLoss(margin=margin, distance="cosine")
        elif loss_type == "relation_guided":
            self.loss_fn = RelationGuidedContrastiveLoss(base_temperature=temperature)
        else:
            self.loss_fn = CosineContrastiveLoss(temperature=temperature)

    def forward(self, emb1, emb2, labels, attn_map=None):
        if self.loss_type == "bce":
            if emb2 is None:
                return self.loss_fn(emb1.squeeze(), labels)
            similarity = torch.sum(emb1 * emb2, dim=1)
            return self.loss_fn(similarity, labels)
        elif self.loss_type == "relation_guided" and attn_map is not None:
            attn_weights = attn_map.mean(dim=[1, 2, 3])
            return self.loss_fn(emb1, emb2, attn_weights)
        else:
            return self.loss_fn(emb1, emb2, labels)


class ViTFaCoRROCmTrainer(ROCmTrainer):
    """Custom ROCm trainer for ViT-FaCoR model."""

    def __init__(
        self,
        *args,
        unfreeze_after_epoch: int = 0,
        unfreeze_last_vit_blocks: int = 0,
        **kwargs,
    ):
        self.unfreeze_after_epoch = max(0, unfreeze_after_epoch)
        self.unfreeze_last_vit_blocks = max(0, unfreeze_last_vit_blocks)
        self._vit_tail_unfrozen = False
        super().__init__(*args, **kwargs)

    def _compute_loss(self, outputs, labels):
        """Handle ViT-FaCoR output format."""
        if self.loss_fn.loss_type == "bce" and isinstance(outputs, tuple):
            if len(outputs) >= 4:
                logits = outputs[0]
                return self.loss_fn(logits.squeeze(-1), None, labels)
            if len(outputs) >= 2:
                emb1, emb2 = outputs[0], outputs[1]
                return self.loss_fn(emb1, emb2, labels)
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            emb1, emb2, attn_map = outputs[0], outputs[1], outputs[2]
            return self.loss_fn(emb1, emb2, labels, attn_map)
        return super()._compute_loss(outputs, labels)

    def validate(self):
        """Validate using the shared scalar-score extraction protocol."""
        return evaluate_model(self.model, self.val_loader, self.device)

    def on_epoch_start(self, epoch: int) -> None:
        if (
            not self._vit_tail_unfrozen
            and self.unfreeze_last_vit_blocks > 0
            and epoch > self.unfreeze_after_epoch
        ):
            self._unfreeze_vit_tail(epoch)

    def _unfreeze_vit_tail(self, epoch: int) -> None:
        base_model = getattr(self.model, "base_model", self.model)
        vit = getattr(base_model, "vit", None)
        if vit is None or not hasattr(vit, "blocks"):
            print("Skipping scheduled ViT unfreeze because no transformer blocks were found.")
            self._vit_tail_unfrozen = True
            return

        num_blocks = len(vit.blocks)
        last_n = min(self.unfreeze_last_vit_blocks, num_blocks)
        if last_n <= 0:
            return

        for block in vit.blocks[:-last_n]:
            for param in block.parameters():
                param.requires_grad = False
        for block in vit.blocks[-last_n:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in vit.norm.parameters():
            param.requires_grad = True

        self._vit_tail_unfrozen = True
        clear_rocm_cache()
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            f"  -> Scheduled unfreeze before epoch {epoch}: "
            f"last {last_n} ViT blocks now trainable "
            f"({trainable_params:,} trainable params total)"
        )


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

    if args.unfreeze_last_vit_blocks > 0 and not args.freeze_vit:
        print(
            "Warning: unfreeze schedule requested while --freeze_vit is disabled. "
            "The ViT backbone is already trainable from epoch 1."
        )

    # Create training dataloaders
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

    # Create test dataloader
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
    print("\nCreating ViT-FaCoR model...")
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

    # Apply ROCm optimizations
    model = optimize_for_rocm(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss
    loss_fn = ViTFaCoRLoss(args.loss, args.temperature, args.margin)

    # Trainer
    trainer = ViTFaCoRROCmTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=train_config,
        device=device,
        rocm_device_id=args.rocm_device,
        monitor_metric="roc_auc",
        unfreeze_after_epoch=args.unfreeze_after_epoch,
        unfreeze_last_vit_blocks=args.unfreeze_last_vit_blocks,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting ROCm-optimized training...")
    print(f"Loss: {args.loss}, Temperature: {args.temperature}, Margin: {args.margin}")
    print(
        "Scheduler: "
        f"{train_config.scheduler} (warmup={train_config.warmup_epochs}, min_lr={train_config.min_lr:.1e})"
    )
    print(
        f"Negative ratio: train={args.negative_ratio:.2f}, "
        f"eval={args.eval_negative_ratio:.2f}"
    )
    print(
        "Negative strategy: "
        f"train={args.train_negative_strategy}, eval={args.eval_negative_strategy}"
    )
    if args.unfreeze_last_vit_blocks > 0:
        print(
            "Scheduled ViT unfreeze: "
            f"after epoch {args.unfreeze_after_epoch}, "
            f"last {args.unfreeze_last_vit_blocks} blocks"
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
        "vit_model": args.vit_model,
        "embedding_dim": args.embedding_dim,
        "cross_attn_layers": args.cross_attn_layers,
        "cross_attn_heads": args.cross_attn_heads,
        "dropout": args.dropout,
        "freeze_vit": args.freeze_vit,
        "use_classifier_head": args.use_classifier_head,
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
        f.write("AMD ROCm Training Results - ViT-FaCoR\n")
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
    print(f"Best model saved to {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
