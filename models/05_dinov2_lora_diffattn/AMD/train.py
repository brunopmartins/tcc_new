#!/usr/bin/env python3
"""
Model 05 — AMD ROCm training script.

DINOv2 (frozen) + LoRA adapters + Differential cross-attention +
auxiliary relation head. Optimised for the 12 GB VRAM of the AMD
RX 6750 XT: backbone is never unfrozen, gradient checkpointing is
applied through the backbone and the cross-attention stack, and
gradient accumulation is used to reach an effective batch of 32.

Usage:
    python train.py --train_dataset fiw --epochs 80
    python train.py --train_dataset kinface --batch_size 2 --gradient_accumulation 16
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

# ROCm env MUST be set before torch is imported.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # gfx1031 (RX 6750 XT)
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared utilities.
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
from losses import CosineContrastiveLoss  # noqa: E402
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

# Model.
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import DINOv2LoRAKinship, RELATION_SETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Model 05 (AMD ROCm)")

    # Data
    p.add_argument("--train_dataset", type=str, default="fiw",
                   choices=["kinface", "fiw"])
    p.add_argument("--test_dataset", type=str, default=None,
                   choices=[None, "kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--negative_ratio", type=float, default=None,
                   help="Override training negative ratio (default: config)")
    p.add_argument("--eval_negative_ratio", type=float, default=None,
                   help="Override eval negative ratio (default: config)")
    p.add_argument("--train_negative_strategy", type=str, default=None,
                   choices=[None, "random", "relation_matched"])
    p.add_argument("--eval_negative_strategy", type=str, default=None,
                   choices=[None, "random", "relation_matched"])

    # Backbone / LoRA
    p.add_argument("--backbone_name", type=str,
                   default="vit_base_patch14_dinov2.lvd142m")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--no_pretrained", action="store_true",
                   help="Skip downloading DINOv2 weights (smoke-tests only)")

    # Head / cross-attention
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--cross_attn_layers", type=int, default=2)
    p.add_argument("--cross_attn_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)

    # Relation aux head
    p.add_argument("--relation_set", type=str, default=None,
                   choices=[None, "fiw", "kinface"],
                   help="default: matches --train_dataset")
    p.add_argument("--relation_loss_weight", type=float, default=0.2)

    # Optimisation
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Keep small — backbone activations dominate VRAM")
    p.add_argument("--gradient_accumulation", type=int, default=8,
                   help="Effective batch = batch_size * this")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="LoRA tolerates a much higher LR than full fine-tune")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "plateau", "none"])
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="Disable gradient checkpointing (uses more VRAM)")

    # Loss
    p.add_argument("--loss", type=str, default="combined",
                   choices=["bce", "contrastive", "combined"])
    p.add_argument("--contrastive_weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.1)

    # Platform
    p.add_argument("--rocm_device", type=int, default=0)
    p.add_argument("--disable_amp", action="store_true")
    p.add_argument("--gfx_version", type=str, default=None)

    # I/O
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Loss with auxiliary relation head
# ---------------------------------------------------------------------------
class KinshipCombinedLoss(nn.Module):
    """
    Primary verification loss (BCE and/or contrastive) plus a cross-entropy
    auxiliary on the relation head, applied ONLY to positive samples (kin
    pairs). Negative samples have no meaningful relation label.
    """

    def __init__(
        self,
        num_relations: int,
        loss_type: str = "combined",
        contrastive_weight: float = 0.5,
        relation_weight: float = 0.2,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.contrastive_weight = contrastive_weight
        self.relation_weight = relation_weight
        self.num_relations = num_relations

        self.bce = nn.BCEWithLogitsLoss()
        self.contrastive = CosineContrastiveLoss(temperature=temperature)
        # Reduction=none so we can mask out non-kin rows.
        self.rel_ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        output: dict,
        labels: torch.Tensor,
        relation_indices: torch.Tensor,
    ) -> torch.Tensor:
        logits = output["logits"].squeeze(-1)
        emb1 = output["emb1"]
        emb2 = output["emb2"]
        rel_logits = output["relation_logits"]

        if self.loss_type == "bce":
            loss_main = self.bce(logits, labels.float())
        elif self.loss_type == "contrastive":
            loss_main = self.contrastive(emb1, emb2, labels.float())
        else:  # combined
            bce = self.bce(logits, labels.float())
            con = self.contrastive(emb1, emb2, labels.float())
            loss_main = (1 - self.contrastive_weight) * bce + self.contrastive_weight * con

        # Relation CE only on positive pairs with a valid label.
        valid = (labels > 0.5) & (relation_indices >= 0)
        if valid.any():
            rel_loss_per = self.rel_ce(rel_logits[valid], relation_indices[valid])
            loss_rel = rel_loss_per.mean()
        else:
            loss_rel = torch.zeros((), device=logits.device, dtype=logits.dtype)

        return loss_main + self.relation_weight * loss_rel


# ---------------------------------------------------------------------------
# Trainer with gradient accumulation + relation labels in batch
# ---------------------------------------------------------------------------
class Model05Trainer(ROCmTrainer):
    def __init__(
        self,
        *args,
        gradient_accumulation: int = 1,
        relation_to_index,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gradient_accumulation = max(1, int(gradient_accumulation))
        self.relation_to_index = relation_to_index

    def _relation_indices(self, relations) -> torch.Tensor:
        indices = [self.relation_to_index(r) for r in relations]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)
        clear_rocm_cache()

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc="Training (ROCm)")

        for batch_idx, batch in enumerate(pbar):
            img1 = batch["img1"].to(self.device, non_blocking=True)
            img2 = batch["img2"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            rel_idx = self._relation_indices(batch.get("relation", ["unknown"] * labels.size(0)))

            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    output = self.model(img1, img2)
                    loss = self.loss_fn(output, labels, rel_idx)
                    loss = loss / self.gradient_accumulation
                _loss_val = loss.item()
                self.scaler.scale(loss).backward()
            else:
                output = self.model(img1, img2)
                loss = self.loss_fn(output, labels, rel_idx)
                loss = loss / self.gradient_accumulation
                _loss_val = loss.item()
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation == 0:
                self._step()

            del output, loss
            clear_rocm_cache()
            gc.collect()

            total_loss += _loss_val * self.gradient_accumulation
            n_batches += 1
            pbar.set_postfix({"loss": f"{_loss_val * self.gradient_accumulation:.4f}"})

        # Flush trailing accumulated grads.
        if n_batches % self.gradient_accumulation != 0:
            self._step()

        return total_loss / max(n_batches, 1)

    def _step(self) -> None:
        if self.use_amp:
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _compute_loss(self, outputs, labels):
        # Fallback path inside the base ROCmTrainer's eval hook.
        relations = ["unknown"] * labels.size(0)
        rel_idx = self._relation_indices(relations)
        return self.loss_fn(outputs, labels, rel_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("Model 05 — DINOv2 + LoRA + DiffAttn | AMD ROCm training")
    print("=" * 60)

    setup_rocm_environment(
        visible_devices=str(args.rocm_device),
        gfx_version=args.gfx_version,
    )
    is_available, status = check_rocm_availability()
    print(f"ROCm status: {status}")
    print_rocm_info()

    set_global_seed(args.seed)
    device = get_rocm_device(args.rocm_device)
    print(f"Device: {device}\n")

    # Data configs
    train_data_config = DataConfig(split_seed=args.seed)
    if args.negative_ratio is not None:
        train_data_config.negative_ratio = args.negative_ratio
    if args.train_negative_strategy is not None:
        train_data_config.negative_sampling_strategy = args.train_negative_strategy
    apply_data_root_override(train_data_config, args.train_dataset, args.data_root)

    test_dataset_name = args.test_dataset or args.train_dataset
    test_data_config = DataConfig(split_seed=args.seed)
    if args.eval_negative_ratio is not None:
        test_data_config.negative_ratio = args.eval_negative_ratio
    if args.eval_negative_strategy is not None:
        test_data_config.negative_sampling_strategy = args.eval_negative_strategy
    if test_dataset_name == args.train_dataset:
        apply_data_root_override(test_data_config, test_dataset_name, args.data_root)

    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        scheduler=args.scheduler,
        patience=args.patience,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.disable_amp,
        monitor_metric="roc_auc",
    )

    # Dataloaders
    print(f"Loading {args.train_dataset} for training...")
    train_loader, val_loader, _ = create_dataloaders(
        config=train_data_config,
        batch_size=args.batch_size,
        dataset_type=args.train_dataset,
        split_seed=args.seed,
        num_workers=args.num_workers,
    )
    print(f"  train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    print(f"Loading {test_dataset_name} for testing...")
    from torch.utils.data import DataLoader
    test_ds = KinshipPairDataset(
        root_dir=resolve_dataset_root(test_data_config, test_dataset_name),
        dataset_type=test_dataset_name,
        split="test",
        transform=get_transforms(test_data_config, train=False),
        split_seed=args.seed,
        negative_ratio=test_data_config.negative_ratio,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"  test: {len(test_ds)}")

    # Model
    relation_set = args.relation_set or args.train_dataset
    num_relations = len(RELATION_SETS[relation_set])

    print("\nBuilding Model 05...")
    print(f"  backbone:       {args.backbone_name}")
    print(f"  img_size:       {args.img_size}")
    print(f"  LoRA rank:      {args.lora_rank}  (alpha={args.lora_alpha})")
    print(f"  cross-attn:     {args.cross_attn_layers} layers, {args.cross_attn_heads} heads")
    print(f"  embedding_dim:  {args.embedding_dim}")
    print(f"  relation set:   {relation_set} ({num_relations} classes)")
    print(f"  grad ckpt:      {not args.no_grad_ckpt}")

    model = DINOv2LoRAKinship(
        backbone_name=args.backbone_name,
        img_size=args.img_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        backbone_pretrained=not args.no_pretrained,
        use_gradient_checkpointing=not args.no_grad_ckpt,
        embedding_dim=args.embedding_dim,
        cross_attn_layers=args.cross_attn_layers,
        cross_attn_heads=args.cross_attn_heads,
        dropout=args.dropout,
        relation_set=relation_set,
        relation_loss_weight=args.relation_loss_weight,
    )
    model = optimize_for_rocm(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.count_trainable()
    print(f"  total params:     {total_params:,}")
    print(f"  trainable params: {trainable_params:,}  "
          f"({100 * trainable_params / total_params:.2f}%)")
    print(f"  LoRA modules:     {model.backbone.n_injected}\n")

    loss_fn = KinshipCombinedLoss(
        num_relations=num_relations,
        loss_type=args.loss,
        contrastive_weight=args.contrastive_weight,
        relation_weight=args.relation_loss_weight,
        temperature=args.temperature,
    )

    trainer = Model05Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=train_config,
        device=device,
        rocm_device_id=args.rocm_device,
        gradient_accumulation=args.gradient_accumulation,
        relation_to_index=model.relation_to_index,
        monitor_metric=train_config.monitor_metric,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    print("Starting training...")
    trainer.train()
    clear_rocm_cache()

    # Protocol evaluation
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

    print("\n" + "=" * 50)
    print("TEST RESULTS (AMD ROCm — Model 05)")
    print("=" * 50)
    print(f"Threshold: {threshold:.3f}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print("=" * 50)

    model_config = {
        "backbone_name": args.backbone_name,
        "img_size": args.img_size,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "embedding_dim": args.embedding_dim,
        "cross_attn_layers": args.cross_attn_layers,
        "cross_attn_heads": args.cross_attn_heads,
        "dropout": args.dropout,
        "relation_set": relation_set,
        "relation_loss_weight": args.relation_loss_weight,
    }

    protocol_metadata = build_protocol_metadata(
        train_dataset=args.train_dataset,
        test_dataset=test_dataset_name,
        threshold=threshold,
        threshold_metric=train_config.threshold_metric,
        split_seed=args.seed,
        negative_ratio=train_data_config.negative_ratio,
        monitor_metric=train_config.monitor_metric,
        args=args,
        extra={"model_config": model_config},
    )

    for checkpoint_name in ["best.pt", "final.pt"]:
        ckpt_path = Path(args.checkpoint_dir) / checkpoint_name
        update_checkpoint_payload(ckpt_path, {"model_config": model_config})
        update_checkpoint_metadata(ckpt_path, protocol_metadata)

    results_path = Path(args.checkpoint_dir) / "test_results_rocm.txt"
    with open(results_path, "w") as f:
        f.write("AMD ROCm Training Results — Model 05\n")
        f.write("=" * 40 + "\n")
        f.write(f"Threshold: {threshold:.4f}\n")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")

    save_json(
        Path(args.checkpoint_dir) / "protocol_summary.json",
        {
            **protocol_metadata,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )

    print(f"\nTraining complete.  Best model: {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
