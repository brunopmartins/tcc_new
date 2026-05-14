#!/usr/bin/env python3
"""
AMD ROCm training for Model 12 — RGCK-Net (Region-Guided Cross Kinship).

Phase 1 recipe (per `proposta_rgck_net_kinship.md`):
- AdaFace IR-101 backbone, **frozen by default**
- Fixed-partition region tokens (5 regions: global/eyes/nose/mouth/jaw)
- Bidirectional cross-region attention (1 layer, 4 heads, 512-d)
- Sigmoid regional gating
- BCE on classifier head over [gA, gB, |diff|, prod, sims, weights, score]

Input: 224×224 aligned face (FIW_aligned native), then resized internally
to 112×112 per region for AdaFace.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "AMD"))

from rocm_utils import (  # noqa: E402
    setup_rocm_environment,
    check_rocm_availability,
    optimize_for_rocm,
    print_rocm_info,
    clear_rocm_cache,
)
from config import DataConfig, TrainConfig  # noqa: E402
from dataset import create_dataloaders, KinshipPairDataset, get_transforms  # noqa: E402
from trainer import ROCmTrainer  # noqa: E402
from evaluation import print_metrics  # noqa: E402
from protocol import (  # noqa: E402
    apply_data_root_override,
    build_protocol_metadata,
    evaluate_with_validation_threshold,
    load_best_checkpoint,
    resolve_dataset_root,
    set_global_seed,
    update_checkpoint_metadata,
    update_checkpoint_payload,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import build_rgck_net, DEFAULT_REGIONS_224  # noqa: E402


# RGCK-Net consumes the native FIW_aligned 224×224 image; cropping and
# 112×112 resize happens inside the model (one per region).
RGCK_MEAN = [0.5, 0.5, 0.5]
RGCK_STD = [0.5, 0.5, 0.5]
RGCK_IMG_SIZE = 224


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train M12 RGCK-Net (AMD ROCm)")

    # Dataset
    p.add_argument("--train_dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--test_dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--aligned_root", type=str, default=None,
                   help="Native 224×224 FIW_aligned (no resize at load).")
    p.add_argument("--num_workers", type=int, default=4)

    # Backbone
    p.add_argument("--adaface_weights", type=str,
                   default=str(Path(__file__).parent.parent / "weights" / "adaface_ir101_webface4m.pth"))
    p.add_argument("--freeze_backbone", action="store_true", default=True,
                   help="Freeze AdaFace IR-101 (Phase 1 default).")
    p.add_argument("--unfreeze_backbone", dest="freeze_backbone", action="store_false",
                   help="Allow full AdaFace fine-tune (Phase 3).")
    p.add_argument("--unfreeze_last_stage", action="store_true", default=False,
                   help="Phase 2: keep backbone frozen except for body[46:49] (stage 4) "
                        "+ output_layer. Effective only when --freeze_backbone is set.")
    p.add_argument("--img_size", type=int, default=RGCK_IMG_SIZE)

    # RGCK-Net specifics
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--cross_attn_heads", type=int, default=4)
    p.add_argument("--cross_attn_layers", type=int, default=1)
    p.add_argument("--gate_hidden", type=int, default=128)
    p.add_argument("--classifier_hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Per proposta_rgck_net section 37: 1e-4 for head with backbone frozen.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "plateau", "step", "none"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Phase 4 of proposta_rgck_net_kinship.md §28: supervised contrastive auxiliary
    p.add_argument("--supcon_weight", type=float, default=0.0,
                   help="Weight λ for the supervised-contrastive auxiliary loss. "
                        "0.0 = pure BCE (R001/R002 baseline). 0.05 = proposal Phase 4 default.")
    p.add_argument("--supcon_margin", type=float, default=0.3,
                   help="Margin for negative pairs in supcon term — cos sim above (1-margin) is penalised.")

    # Negative sampling
    p.add_argument("--negative_ratio", type=float, default=1.0)
    p.add_argument("--eval_negative_ratio", type=float, default=1.0)
    p.add_argument("--train_negative_strategy", type=str, default="random",
                   choices=["random", "relation_matched"])
    p.add_argument("--eval_negative_strategy", type=str, default="random",
                   choices=["random", "relation_matched"])

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


class RGCKBCELoss(nn.Module):
    """
    BCE-with-logits over classifier head + optional supervised-contrastive
    auxiliary over the (gA_norm, gB_norm) L2-normalised global tokens.

    Phase 4 of `proposta_rgck_net_kinship.md` §28:
      L_total = L_bce + supcon_weight * L_supcon

    SupCon term is a margin-style label-aware contrastive on cosine similarity:
      pos pairs (label=1): pull cos toward 1 → (1 - cos)^2
      neg pairs (label=0): push cos below margin → max(0, cos - (1-margin))^2

    Reduces to pure BCE when supcon_weight=0 (the default — matches R001/R002).
    """

    def __init__(self, supcon_weight: float = 0.0, supcon_margin: float = 0.3):
        super().__init__()
        self.loss_type = "bce"
        self.bce = nn.BCEWithLogitsLoss()
        self.supcon_weight = float(supcon_weight)
        self.supcon_margin = float(supcon_margin)

    @staticmethod
    def _supcon_term(gA: torch.Tensor, gB: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
        # gA, gB are already L2-normalised in M12 model.forward
        cos_sim = (gA * gB).sum(dim=1)  # (B,)
        label_f = labels.float()
        pos_loss = (1.0 - cos_sim) ** 2
        neg_loss = torch.clamp(cos_sim - (1.0 - margin), min=0.0) ** 2
        loss = label_f * pos_loss + (1.0 - label_f) * neg_loss
        return loss.mean()

    def forward(self, outputs, labels: torch.Tensor) -> torch.Tensor:
        # outputs may be a tuple (logit, weights, attn, gA, gB) from M12.forward
        if isinstance(outputs, tuple):
            logit = outputs[0]
            bce_loss = self.bce(logit.squeeze(-1).float(), labels.float())
            if self.supcon_weight > 0.0 and len(outputs) >= 5:
                gA = outputs[3]
                gB = outputs[4]
                supcon = self._supcon_term(gA, gB, labels, self.supcon_margin)
                return bce_loss + self.supcon_weight * supcon
            return bce_loss
        return self.bce(outputs.squeeze(-1).float(), labels.float())


class RGCKROCmTrainer(ROCmTrainer):
    """ROCm trainer for M12 — passes full output tuple to RGCKBCELoss."""

    def __init__(self, *args, gradient_accumulation: int = 1, **kwargs):
        self.gradient_accumulation = max(1, int(gradient_accumulation))
        super().__init__(*args, **kwargs)

    def _compute_loss(self, outputs, labels):
        # RGCKBCELoss handles the tuple unpacking itself so it can access gA, gB
        return self.loss_fn(outputs, labels)


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("AMD ROCm Training — Model 12: RGCK-Net (Region-Guided Cross Kinship)")
    print("=" * 60)

    setup_rocm_environment(visible_devices=str(args.rocm_device), gfx_version=args.gfx_version)
    is_available, status = check_rocm_availability()
    print(f"\nROCm Status: {status}")
    if not is_available:
        print("ERROR: ROCm not available")
        sys.exit(1)
    print_rocm_info()

    set_global_seed(args.seed)
    device = torch.device(f"cuda:{args.rocm_device}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Data configs
    train_data_config = DataConfig(
        image_size=args.img_size,
        normalize_mean=RGCK_MEAN,
        normalize_std=RGCK_STD,
    )
    test_data_config = DataConfig(
        image_size=args.img_size,
        normalize_mean=RGCK_MEAN,
        normalize_std=RGCK_STD,
    )
    apply_data_root_override(train_data_config, args.train_dataset, args.data_root)
    if args.test_dataset:
        apply_data_root_override(test_data_config, args.test_dataset, args.data_root)
    # Honor --aligned_root (FIW_aligned 224×224 native)
    if args.aligned_root:
        train_data_config.aligned_root = args.aligned_root  # type: ignore[attr-defined]
        test_data_config.aligned_root = args.aligned_root  # type: ignore[attr-defined]

    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        patience=args.patience,
        max_grad_norm=args.max_grad_norm,
        use_amp=not args.disable_amp,
        save_every=5,
        checkpoint_dir=args.checkpoint_dir,
        monitor_metric="roc_auc",
        threshold_metric="f1",
    )

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
        aligned_root=args.aligned_root,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    print(f"Loading {args.test_dataset} dataset for testing...")
    from torch.utils.data import DataLoader
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
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Test: {len(test_dataset)}")

    print("\nCreating RGCK-Net (Region-Guided Cross Kinship)...")
    print(f"  AdaFace weights:     {args.adaface_weights}")
    print(f"  Backbone frozen:     {args.freeze_backbone}")
    print(f"  Regions:             {[name for name, _ in DEFAULT_REGIONS_224]}")
    print(f"  Cross-attn:          {args.cross_attn_layers} layer × {args.cross_attn_heads} heads")
    print(f"  Dropout:             {args.dropout}")

    model = build_rgck_net(
        adaface_weights=args.adaface_weights,
        embedding_dim=args.embedding_dim,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_layers=args.cross_attn_layers,
        gate_hidden=args.gate_hidden,
        classifier_hidden=args.classifier_hidden,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_stage=args.unfreeze_last_stage,
    )
    model = optimize_for_rocm(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:        {total:,}")
    print(f"  Trainable params:    {trainable:,} ({100*trainable/total:.2f}%)")

    loss_fn = RGCKBCELoss(
        supcon_weight=args.supcon_weight,
        supcon_margin=args.supcon_margin,
    )
    if args.supcon_weight > 0:
        print(f"  Loss: BCE + {args.supcon_weight:.3f} × SupCon (margin={args.supcon_margin})")

    trainer = RGCKROCmTrainer(
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
    print(f"Loss: BCE on classifier head logit")
    print(f"Scheduler: {train_config.scheduler} (warmup={train_config.warmup_epochs}, min_lr={train_config.min_lr:.1e})")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.gradient_accumulation} "
          f"(effective {args.batch_size * args.gradient_accumulation})")
    print(f"Negative ratio: train={args.negative_ratio:.2f}, eval={args.eval_negative_ratio:.2f}")
    print(f"Negative strategy: train={args.train_negative_strategy}, eval={args.eval_negative_strategy}")
    trainer.train()

    clear_rocm_cache()

    print("\nLoading best checkpoint for protocol evaluation...")
    load_best_checkpoint(model, args.checkpoint_dir, device)
    protocol_results = evaluate_with_validation_threshold(
        model, val_loader, test_loader, device,
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
        "cross_attn_heads": args.cross_attn_heads,
        "cross_attn_layers": args.cross_attn_layers,
        "gate_hidden": args.gate_hidden,
        "classifier_hidden": args.classifier_hidden,
        "dropout": args.dropout,
        "freeze_backbone": args.freeze_backbone,
        "unfreeze_last_stage": args.unfreeze_last_stage,
        "supcon_weight": args.supcon_weight,
        "supcon_margin": args.supcon_margin,
        "regions": [name for name, _ in DEFAULT_REGIONS_224],
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
        f.write("AMD ROCm Training Results — Model 12 (RGCK-Net)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best epoch: {trainer.best_epoch}\n")
        f.write(f"Best {trainer.monitor_metric}: {trainer.best_metric:.4f}\n")
        f.write(f"Validation threshold (F1-optimal): {threshold:.4f}\n\n")
        f.write("Validation metrics:\n")
        for k, v in val_metrics.items():
            if not isinstance(v, dict):
                f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
        f.write("\nTest metrics:\n")
        for k, v in test_metrics.items():
            if not isinstance(v, dict):
                f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")

    print(f"\nResults saved to {results_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
