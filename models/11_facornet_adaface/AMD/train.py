#!/usr/bin/env python3
"""
AMD ROCm training for Model 11 — AdaFace + FaCoR Cross-Attention.

Mirrors M02's training recipe (best result so far on FIW: Test ROC AUC =
0.850 with full ViT fine-tune, cosine_contrastive m=0.3, warmup=5, lr=5e-6,
dropout=0.2). Only the backbone changes: AdaFace IR-101 (WebFace4M pretrain)
instead of timm ViT-B/16.

Notable platform-specific changes vs. M02's train.py:
- Input is 112×112 (AdaFace native) — DataConfig overrides image_size.
- Normalization is AdaFace's ([-1, 1] via mean=std=[0.5, 0.5, 0.5]).
- No `--vit_model` / `--freeze_vit` / `--unfreeze_last_vit_blocks` flags;
  replaced by `--adaface_weights` / `--freeze_backbone`.
- Gradient accumulation is supported for VRAM headroom (default 4 → eff batch 32).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Setup ROCm environment before other imports
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # RX 6700/6750 XT (gfx1031)
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")

import torch
import torch.nn as nn

# Shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "AMD"))

from rocm_utils import (  # noqa: E402
    setup_rocm_environment,
    check_rocm_availability,
    get_rocm_device,
    optimize_for_rocm,
    print_rocm_info,
    clear_rocm_cache,
)
from config import DataConfig, TrainConfig  # noqa: E402
from dataset import create_dataloaders, KinshipPairDataset, get_transforms  # noqa: E402
from losses import (  # noqa: E402
    ContrastiveLoss,
    CosineContrastiveLoss,
    RelationGuidedContrastiveLoss,
)
from trainer import ROCmTrainer  # noqa: E402
from evaluation import print_metrics  # noqa: E402
from protocol import (  # noqa: E402
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

# Model + M10-specific age-augmented dataset
# Imported AFTER the shared imports above — `age_dataset` inserts its own
# `shared/` path entry which would shadow `shared/AMD/trainer.py` for the
# `from trainer import ROCmTrainer` line if imported earlier.
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import build_adaface_facor_model  # noqa: E402
from age_dataset import (  # noqa: E402
    AgeAugmentedKinshipPairDataset,
    DEFAULT_TARGET_AGES,
    parse_target_ages,
)


# AdaFace input convention
ADAFACE_MEAN = [0.5, 0.5, 0.5]
ADAFACE_STD = [0.5, 0.5, 0.5]
ADAFACE_IMG_SIZE = 112


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Model 11 — AdaFace + FaCoR (AMD ROCm)")

    # Dataset
    p.add_argument("--train_dataset", type=str, default="fiw",
                   choices=["kinface", "fiw"])
    p.add_argument("--test_dataset", type=str, default="fiw",
                   choices=["kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--aligned_root", type=str, default=None,
                   help="Path to pre-aligned face crops (e.g. datasets/FIW_aligned). "
                        "Existing FIW_aligned is 224×224 — M10 resizes to 112 at load time.")
    p.add_argument("--num_workers", type=int, default=4)

    # Backbone
    p.add_argument("--adaface_weights", type=str,
                   default=str(Path(__file__).parent.parent / "weights" / "adaface_ir101_webface4m.pth"),
                   help="Path to AdaFace IR-101 weights "
                        "(cvlface state dict from minchul/cvlface_adaface_ir101_webface4m).")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze AdaFace backbone (M02's recipe trains end-to-end; "
                        "use this only for diagnostic runs).")
    p.add_argument("--img_size", type=int, default=ADAFACE_IMG_SIZE,
                   help="Input image size — AdaFace native is 112.")

    # FaCoR cross-attention
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--cross_attn_layers", type=int, default=2)
    p.add_argument("--cross_attn_heads", type=int, default=8)
    p.add_argument("--no_positional_embedding", action="store_true",
                   help="Disable learnable 49-token positional embedding.")
    p.add_argument("--no_global_embedding", action="store_true",
                   help="Skip AdaFace's pooled embedding (use only attended tokens).")
    p.add_argument("--dropout", type=float, default=0.2,
                   help="M02's tuned dropout for the FaCoR head is 0.2.")
    p.add_argument("--use_classifier_head", action="store_true",
                   help="Add BCE classifier on top of cosine embeddings.")

    # M11 architecture variant: top-only (default, M10-style) vs multistage (M09-style)
    p.add_argument("--use_multistage", action="store_true",
                   help="Build multi-stage cross-attention model (SAI-inspired, "
                        "stages 3 and 4 of IR-101 like M09). Default: FaCoR top-only.")
    p.add_argument("--cross_attn_stages", type=str, default="3,4",
                   help="Comma-separated stages where to inject cross-attn (multistage only).")
    p.add_argument("--cross_attn_layers_per_stage", type=int, default=1,
                   help="Layers per cross-attn block (multistage only).")

    # SAM age-augmentation (generational invariance via young/adult/elderly variants)
    p.add_argument("--age_augment_root", type=str, default=None,
                   help="Directory of SAM-aged variants mirroring aligned_root "
                        "(see tools/sam_age_augment.py). When set, every face is "
                        "loaded with N_ages variants and the model ensembles them.")
    p.add_argument("--age_target_ages", type=str, default="8,25,70",
                   help="Comma-separated target ages (must match the variants "
                        "produced by tools/sam_age_augment.py).")
    p.add_argument("--age_original_weight", type=float, default=0.5,
                   help="Weight applied to the original face in the age ensemble. "
                        "Remaining (1 - w) is split equally across aged variants. "
                        "Default 0.5: original is heaviest, three aged variants share 0.5.")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Default 8: AdaFace IR-101 full fine-tune at 112×112 fits "
                        "in 12 GB with grad accumulation. Increase if VRAM allows.")
    p.add_argument("--gradient_accumulation", type=int, default=4,
                   help="Effective batch = batch_size * grad_accum (default 8*4=32, matches M02).")
    p.add_argument("--lr", type=float, default=5e-6,
                   help="M02's tuned peak LR for full ViT fine-tune.")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "plateau", "step", "none"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Negative sampling
    p.add_argument("--negative_ratio", type=float, default=1.0)
    p.add_argument("--eval_negative_ratio", type=float, default=1.0)
    p.add_argument("--train_negative_strategy", type=str, default="random",
                   choices=["random", "relation_matched"])
    p.add_argument("--eval_negative_strategy", type=str, default="random",
                   choices=["random", "relation_matched"])

    # Loss
    p.add_argument("--loss", type=str, default="cosine_contrastive",
                   choices=["bce", "contrastive", "cosine_contrastive", "relation_guided"])
    p.add_argument("--temperature", type=float, default=0.3,
                   help="M02's tuned temperature for cosine_contrastive.")
    p.add_argument("--margin", type=float, default=0.3,
                   help="M02's tuned margin for supervised contrastive.")

    # ROCm
    p.add_argument("--rocm_device", type=int, default=0)
    p.add_argument("--disable_amp", action="store_true")
    p.add_argument("--gfx_version", type=str, default=None)

    # Misc
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=50)

    return p.parse_args()


class AdaFaceFaCoRLoss(nn.Module):
    """Loss wrapper mirroring M02's `ViTFaCoRLoss` exactly."""

    def __init__(self, loss_type: str = "cosine_contrastive",
                 temperature: float = 0.3, margin: float = 0.3):
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


class AdaFaceFaCoRROCmTrainer(ROCmTrainer):
    """ROCm trainer for M10 — output parsing + gradient accumulation."""

    def __init__(self, *args, gradient_accumulation: int = 1, **kwargs):
        self.gradient_accumulation = max(1, int(gradient_accumulation))
        super().__init__(*args, **kwargs)

    def _compute_loss(self, outputs, labels):
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


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("AMD ROCm Training — Model 11: AdaFace + FaCoR Cross-Attention")
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

    # Data configs — AdaFace 112×112 + [-1, 1] normalisation
    train_data_config = DataConfig(
        split_seed=args.seed,
        image_size=args.img_size,
        normalize_mean=ADAFACE_MEAN,
        normalize_std=ADAFACE_STD,
        negative_ratio=args.negative_ratio,
        num_workers=args.num_workers,
    )
    test_data_config = DataConfig(
        split_seed=args.seed,
        image_size=args.img_size,
        normalize_mean=ADAFACE_MEAN,
        normalize_std=ADAFACE_STD,
        negative_ratio=args.negative_ratio,
        num_workers=args.num_workers,
    )
    apply_data_root_override(train_data_config, args.train_dataset, args.data_root)
    if args.train_dataset == args.test_dataset:
        apply_data_root_override(test_data_config, args.test_dataset, args.data_root)
    elif args.data_root:
        print("Using --data_root for training only because train/test datasets differ.")

    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler="none" if args.scheduler == "none" else args.scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.disable_amp,
        patience=args.patience,
    )

    # Parse age-ensemble options
    target_ages = parse_target_ages(args.age_target_ages) if args.age_augment_root else []
    use_age_ensemble = bool(args.age_augment_root)
    if use_age_ensemble:
        num_variants = 1 + len(target_ages)
        print(f"\nSAM age-ensemble ENABLED")
        print(f"  age_augment_root: {args.age_augment_root}")
        print(f"  target_ages:      {target_ages}")
        print(f"  original_weight:  {args.age_original_weight:.3f}  "
              f"(aged each: {(1 - args.age_original_weight) / max(len(target_ages), 1):.3f})")
        print(f"  variants per face: {num_variants}  "
              f"(backbone effectively runs {num_variants}× per sample)")

    # Training data
    print(f"\nLoading {args.train_dataset} dataset for training...")
    if use_age_ensemble:
        from torch.utils.data import DataLoader
        train_dataset = AgeAugmentedKinshipPairDataset(
            root_dir=resolve_dataset_root(train_data_config, args.train_dataset),
            dataset_type=args.train_dataset,
            split="train",
            transform=get_transforms(train_data_config, train=True),
            split_seed=args.seed,
            negative_ratio=args.negative_ratio,
            negative_sampling_strategy=args.train_negative_strategy,
            aligned_root=args.aligned_root,
            age_augment_root=args.age_augment_root,
            target_ages=target_ages,
        )
        val_dataset = AgeAugmentedKinshipPairDataset(
            root_dir=resolve_dataset_root(train_data_config, args.train_dataset),
            dataset_type=args.train_dataset,
            split="val",
            transform=get_transforms(train_data_config, train=False),
            split_seed=args.seed,
            negative_ratio=args.eval_negative_ratio,
            negative_sampling_strategy=args.eval_negative_strategy,
            aligned_root=args.aligned_root,
            age_augment_root=args.age_augment_root,
            target_ages=target_ages,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
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
            aligned_root=args.aligned_root,
        )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Test data
    print(f"Loading {args.test_dataset} dataset for testing...")
    from torch.utils.data import DataLoader
    if use_age_ensemble:
        test_dataset = AgeAugmentedKinshipPairDataset(
            root_dir=resolve_dataset_root(test_data_config, args.test_dataset),
            dataset_type=args.test_dataset,
            split="test",
            transform=get_transforms(test_data_config, train=False),
            split_seed=args.seed,
            negative_ratio=args.eval_negative_ratio,
            aligned_root=args.aligned_root,
            negative_sampling_strategy=args.eval_negative_strategy,
            age_augment_root=args.age_augment_root,
            target_ages=target_ages,
        )
    else:
        test_dataset = KinshipPairDataset(
            root_dir=resolve_dataset_root(test_data_config, args.test_dataset),
            dataset_type=args.test_dataset,
            split="test",
            transform=get_transforms(test_data_config, train=False),
            split_seed=args.seed,
            negative_ratio=args.eval_negative_ratio,
            aligned_root=args.aligned_root,
            negative_sampling_strategy=args.eval_negative_strategy,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Test: {len(test_dataset)}")

    # Model
    if args.use_multistage:
        cross_attn_stages = sorted({int(s) for s in args.cross_attn_stages.split(",")})
        print("\nCreating AdaFace + Multi-Stage Cross-Attention model (M09 architecture)...")
        print(f"  AdaFace weights:    {args.adaface_weights}")
        print(f"  Cross-attn stages:  {cross_attn_stages}")
        print(f"  Layers per stage:   {args.cross_attn_layers_per_stage}")
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "09_adaface_multistage"))
        from model import build_adaface_multistage_model  # noqa: E402
        model = build_adaface_multistage_model(
            adaface_weights=args.adaface_weights,
            embedding_dim=args.embedding_dim,
            cross_attn_stages=cross_attn_stages,
            num_cross_attn_layers_per_stage=args.cross_attn_layers_per_stage,
            cross_attn_heads=args.cross_attn_heads,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
            use_positional_embedding=not args.no_positional_embedding,
            use_global_embedding=not args.no_global_embedding,
            use_classifier_head=args.use_classifier_head,
        )
    else:
        print("\nCreating AdaFace-FaCoR model (top-only, M10 architecture)...")
        print(f"  AdaFace weights: {args.adaface_weights}")
        model = build_adaface_facor_model(
            adaface_weights=args.adaface_weights,
            embedding_dim=args.embedding_dim,
            num_cross_attn_layers=args.cross_attn_layers,
            cross_attn_heads=args.cross_attn_heads,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
            use_positional_embedding=not args.no_positional_embedding,
            use_global_embedding=not args.no_global_embedding,
            use_classifier_head=args.use_classifier_head,
            original_weight=args.age_original_weight,
        )
    model = optimize_for_rocm(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")

    # Loss + trainer
    loss_fn = AdaFaceFaCoRLoss(args.loss, args.temperature, args.margin)

    trainer = AdaFaceFaCoRROCmTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=train_config,
        device=device,
        rocm_device_id=args.rocm_device,
        monitor_metric="roc_auc",
        gradient_accumulation=args.gradient_accumulation,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    print("\nStarting ROCm-optimised training...")
    print(f"Loss: {args.loss}, Temperature: {args.temperature}, Margin: {args.margin}")
    print(
        "Scheduler: "
        f"{train_config.scheduler} (warmup={train_config.warmup_epochs}, "
        f"min_lr={train_config.min_lr:.1e})"
    )
    print(f"Batch size: {args.batch_size}, Grad accum: {args.gradient_accumulation} "
          f"(effective {args.batch_size * args.gradient_accumulation})")
    print(
        f"Negative ratio: train={args.negative_ratio:.2f}, "
        f"eval={args.eval_negative_ratio:.2f}"
    )
    print(
        f"Negative strategy: train={args.train_negative_strategy}, "
        f"eval={args.eval_negative_strategy}"
    )
    trainer.train()

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
        "adaface_weights": args.adaface_weights,
        "img_size": args.img_size,
        "embedding_dim": args.embedding_dim,
        "cross_attn_layers": args.cross_attn_layers,
        "cross_attn_heads": args.cross_attn_heads,
        "dropout": args.dropout,
        "freeze_backbone": args.freeze_backbone,
        "use_positional_embedding": not args.no_positional_embedding,
        "use_global_embedding": not args.no_global_embedding,
        "use_classifier_head": args.use_classifier_head,
        "age_augment_root": args.age_augment_root,
        "age_target_ages": target_ages,
        "age_original_weight": args.age_original_weight,
        # M11 architecture flag — picked up by test.py / evaluate.py to rebuild
        "use_multistage": args.use_multistage,
        "cross_attn_stages": (
            sorted({int(s) for s in args.cross_attn_stages.split(",")})
            if args.use_multistage else None
        ),
        "cross_attn_layers_per_stage": (
            args.cross_attn_layers_per_stage if args.use_multistage else None
        ),
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

    results_path = Path(args.checkpoint_dir) / "test_results_rocm.txt"
    with open(results_path, "w") as f:
        f.write("AMD ROCm Training Results — Model 11 (AdaFace-FaCoR)\n")
        f.write("=" * 50 + "\n")
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

    print("\nTraining complete!")
    print(f"Trained on: {args.train_dataset}, Tested on: {args.test_dataset}")
    print(f"Best model saved to {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
