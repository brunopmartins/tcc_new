#!/usr/bin/env python3
"""
Model 06 — Retrieval-Augmented Kinship, AMD ROCm training.

Backbone is frozen, gallery is built from the training split and then
held in VRAM (or CPU if `--store_gallery_on_cpu`). Training uses BCE +
optional contrastive + relation-CE loss.
"""
from __future__ import annotations

import argparse
import gc
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
    setup_rocm_environment, check_rocm_availability, get_rocm_device,
    optimize_for_rocm, print_rocm_info, clear_rocm_cache,
)
from config import DataConfig, TrainConfig  # noqa: E402
from dataset import create_dataloaders, KinshipPairDataset, get_transforms  # noqa: E402
from losses import CosineContrastiveLoss  # noqa: E402
from trainer import ROCmTrainer  # noqa: E402
from evaluation import print_metrics  # noqa: E402
from protocol import (  # noqa: E402
    apply_data_root_override, build_protocol_metadata,
    evaluate_with_validation_threshold, load_best_checkpoint,
    resolve_dataset_root, save_json, set_global_seed,
    update_checkpoint_metadata, update_checkpoint_payload,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import RetrievalAugmentedKinship, RELATION_SETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Model 06 (AMD ROCm)")

    # Data
    p.add_argument("--train_dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--test_dataset", type=str, default=None, choices=[None, "kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=4)

    # Encoder
    p.add_argument("--backbone_name", type=str, default="vit_base_patch16_224")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--freeze_backbone", action="store_true", default=True,
                   help="Keep the encoder frozen (default)")
    p.add_argument("--no_freeze_backbone", action="store_true",
                   help="Unfreeze the encoder (careful: OOM risk on 12 GB)")
    p.add_argument("--no_pretrained", action="store_true")

    # Retrieval
    p.add_argument("--retrieval_k", type=int, default=32)
    p.add_argument("--retrieval_attn_layers", type=int, default=2)
    p.add_argument("--retrieval_attn_heads", type=int, default=4)
    p.add_argument("--max_gallery", type=int, default=200_000)
    p.add_argument("--store_gallery_on_cpu", action="store_true")
    p.add_argument("--gallery_refresh_every", type=int, default=0,
                   help="Rebuild gallery every N epochs (0 = only once at start)")

    # Head
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # Relation head
    p.add_argument("--relation_set", type=str, default=None, choices=[None, "fiw", "kinface"])
    p.add_argument("--relation_loss_weight", type=float, default=0.15)

    # Optimisation
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "plateau", "none"])
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--no_grad_ckpt", action="store_true")

    # Loss
    p.add_argument("--loss", type=str, default="combined",
                   choices=["bce", "contrastive", "combined"])
    p.add_argument("--contrastive_weight", type=float, default=0.3)
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


class RetrievalLoss(nn.Module):
    def __init__(self, num_relations: int, loss_type: str, contrastive_weight: float,
                 relation_weight: float, temperature: float):
        super().__init__()
        self.loss_type = loss_type
        self.contrastive_weight = contrastive_weight
        self.relation_weight = relation_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.contrastive = CosineContrastiveLoss(temperature=temperature)
        self.rel_ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, output, labels, relation_indices):
        logits = output["logits"].squeeze(-1)
        if self.loss_type == "bce":
            loss_main = self.bce(logits, labels.float())
        elif self.loss_type == "contrastive":
            loss_main = self.contrastive(output["emb1"], output["emb2"], labels.float())
        else:
            bce = self.bce(logits, labels.float())
            con = self.contrastive(output["emb1"], output["emb2"], labels.float())
            loss_main = (1 - self.contrastive_weight) * bce + self.contrastive_weight * con

        valid = (labels > 0.5) & (relation_indices >= 0)
        if valid.any():
            loss_rel = self.rel_ce(output["relation_logits"][valid], relation_indices[valid]).mean()
        else:
            loss_rel = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return loss_main + self.relation_weight * loss_rel


class RetrievalROCmTrainer(ROCmTrainer):
    def __init__(self, *args, gradient_accumulation: int = 1, relation_to_index=None,
                 gallery_builder=None, gallery_refresh_every: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_accumulation = max(1, int(gradient_accumulation))
        self.relation_to_index = relation_to_index
        self.gallery_builder = gallery_builder
        self.gallery_refresh_every = max(0, int(gallery_refresh_every))
        self._epoch_counter = 0

    def _relation_indices(self, relations):
        return torch.tensor(
            [self.relation_to_index(r) for r in relations],
            dtype=torch.long, device=self.device,
        )

    def train_epoch(self) -> float:
        if self.gallery_refresh_every and self.gallery_builder is not None:
            if self._epoch_counter > 0 and self._epoch_counter % self.gallery_refresh_every == 0:
                print(f"  [gallery] refreshing at epoch {self._epoch_counter}")
                self.gallery_builder()
                clear_rocm_cache()
        self._epoch_counter += 1

        self.model.train()
        total, n = 0.0, 0
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
                    out = self.model(img1, img2)
                    loss = self.loss_fn(out, labels, rel_idx) / self.gradient_accumulation
                _loss_val = loss.item()
                self.scaler.scale(loss).backward()
            else:
                out = self.model(img1, img2)
                loss = self.loss_fn(out, labels, rel_idx) / self.gradient_accumulation
                _loss_val = loss.item()
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation == 0:
                self._step()

            del out, loss
            clear_rocm_cache()
            gc.collect()
            total += _loss_val * self.gradient_accumulation
            n += 1
            pbar.set_postfix({"loss": f"{_loss_val * self.gradient_accumulation:.4f}"})

        if n % self.gradient_accumulation != 0:
            self._step()
        return total / max(n, 1)

    def _step(self):
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
        rel_idx = self._relation_indices(["unknown"] * labels.size(0))
        return self.loss_fn(outputs, labels, rel_idx)


def main() -> None:
    args = parse_args()
    print("\n" + "=" * 60)
    print("Model 06 — Retrieval-Augmented Kinship | AMD ROCm")
    print("=" * 60)

    setup_rocm_environment(visible_devices=str(args.rocm_device), gfx_version=args.gfx_version)
    is_available, status = check_rocm_availability()
    print(f"ROCm: {status}")
    print_rocm_info()

    set_global_seed(args.seed)
    device = get_rocm_device(args.rocm_device)
    print(f"Device: {device}\n")

    freeze = not args.no_freeze_backbone and args.freeze_backbone

    train_data_config = DataConfig(split_seed=args.seed)
    apply_data_root_override(train_data_config, args.train_dataset, args.data_root)
    test_dataset_name = args.test_dataset or args.train_dataset
    test_data_config = DataConfig(split_seed=args.seed)
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

    train_loader, val_loader, _ = create_dataloaders(
        config=train_data_config,
        batch_size=args.batch_size,
        dataset_type=args.train_dataset,
        split_seed=args.seed,
        num_workers=args.num_workers,
    )
    print(f"  train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    from torch.utils.data import DataLoader
    test_ds = KinshipPairDataset(
        root_dir=resolve_dataset_root(test_data_config, test_dataset_name),
        dataset_type=test_dataset_name,
        split="test",
        transform=get_transforms(test_data_config, train=False),
        split_seed=args.seed,
        negative_ratio=test_data_config.negative_ratio,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"  test:  {len(test_ds)}")

    relation_set = args.relation_set or args.train_dataset

    print("\nBuilding Model 06...")
    print(f"  backbone:   {args.backbone_name} (frozen={freeze})")
    print(f"  K retrievals: {args.retrieval_k}")
    print(f"  relation set: {relation_set}")

    model = RetrievalAugmentedKinship(
        backbone_name=args.backbone_name,
        img_size=args.img_size,
        freeze_backbone=freeze,
        backbone_pretrained=not args.no_pretrained,
        embedding_dim=args.embedding_dim,
        retrieval_k=args.retrieval_k,
        retrieval_attn_layers=args.retrieval_attn_layers,
        retrieval_attn_heads=args.retrieval_attn_heads,
        dropout=args.dropout,
        relation_set=relation_set,
        relation_loss_weight=args.relation_loss_weight,
        max_gallery=args.max_gallery,
        store_gallery_on_cpu=args.store_gallery_on_cpu,
        use_gradient_checkpointing=not args.no_grad_ckpt,
    )
    model = optimize_for_rocm(model)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = model.count_trainable()
    print(f"  total:     {total_params:,}")
    print(f"  trainable: {trainable:,} ({100*trainable/total_params:.2f}%)")

    # Dedicated gallery builder loader (no shuffle, no negatives → training
    # positives only).
    gallery_ds = KinshipPairDataset(
        root_dir=resolve_dataset_root(train_data_config, args.train_dataset),
        dataset_type=args.train_dataset,
        split="train",
        transform=get_transforms(train_data_config, train=False),
        split_seed=args.seed,
        negative_ratio=0.0,  # positives only
    )
    gallery_loader = DataLoader(
        gallery_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    def build_gallery():
        n = model.build_gallery(gallery_loader, device=device, positive_only=True)
        print(f"  [gallery] {n} positive pairs stored.")

    print("\nBuilding initial retrieval gallery...")
    build_gallery()
    clear_rocm_cache()

    loss_fn = RetrievalLoss(
        num_relations=len(RELATION_SETS[relation_set]),
        loss_type=args.loss,
        contrastive_weight=args.contrastive_weight,
        relation_weight=args.relation_loss_weight,
        temperature=args.temperature,
    )

    trainer = RetrievalROCmTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, config=train_config, device=device,
        rocm_device_id=args.rocm_device,
        gradient_accumulation=args.gradient_accumulation,
        relation_to_index=model.relation_to_index,
        gallery_builder=build_gallery if args.gallery_refresh_every > 0 else None,
        gallery_refresh_every=args.gallery_refresh_every,
        monitor_metric=train_config.monitor_metric,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    print("\nStarting training...")
    trainer.train()
    clear_rocm_cache()

    # Ensure gallery is fresh before final evaluation.
    print("\nRefreshing gallery for final evaluation...")
    build_gallery()

    print("\nLoading best checkpoint and running validation-threshold protocol...")
    load_best_checkpoint(model, args.checkpoint_dir, device)
    protocol_results = evaluate_with_validation_threshold(
        model, val_loader, test_loader, device,
        threshold_metric=train_config.threshold_metric,
    )
    threshold = protocol_results["threshold"]
    val_metrics = protocol_results["validation_metrics"]
    test_metrics = protocol_results["test_metrics"]
    print_metrics(val_metrics, prefix="Validation ")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

    model_config = {
        "backbone_name": args.backbone_name,
        "img_size": args.img_size,
        "freeze_backbone": freeze,
        "embedding_dim": args.embedding_dim,
        "retrieval_k": args.retrieval_k,
        "retrieval_attn_layers": args.retrieval_attn_layers,
        "retrieval_attn_heads": args.retrieval_attn_heads,
        "dropout": args.dropout,
        "relation_set": relation_set,
        "relation_loss_weight": args.relation_loss_weight,
        "max_gallery": args.max_gallery,
        "store_gallery_on_cpu": args.store_gallery_on_cpu,
    }

    metadata = build_protocol_metadata(
        train_dataset=args.train_dataset, test_dataset=test_dataset_name,
        threshold=threshold, threshold_metric=train_config.threshold_metric,
        split_seed=args.seed, negative_ratio=train_data_config.negative_ratio,
        monitor_metric=train_config.monitor_metric, args=args,
        extra={"model_config": model_config},
    )
    for name in ["best.pt", "final.pt"]:
        p = Path(args.checkpoint_dir) / name
        update_checkpoint_payload(p, {"model_config": model_config})
        update_checkpoint_metadata(p, metadata)
    save_json(Path(args.checkpoint_dir) / "protocol_summary.json",
              {**metadata, "validation_metrics": val_metrics, "test_metrics": test_metrics})

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
