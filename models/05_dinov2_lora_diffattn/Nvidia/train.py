#!/usr/bin/env python3
"""
Model 05 — Nvidia CUDA training script (mirrors AMD/train.py).

The architecture, loss and training loop are identical to the AMD version;
only the device handling and env setup differ. The AMD version is the
reference implementation — prefer it on the 12 GB AMD RX 6750 XT.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import DataConfig, TrainConfig  # noqa: E402
from dataset import create_dataloaders, KinshipPairDataset, get_transforms  # noqa: E402
from losses import CosineContrastiveLoss  # noqa: E402
from trainer import Trainer  # noqa: E402
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import DINOv2LoRAKinship, RELATION_SETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Model 05 (Nvidia CUDA)")
    p.add_argument("--train_dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--test_dataset", type=str, default=None, choices=[None, "kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--backbone_name", type=str,
                   default="vit_base_patch14_dinov2.lvd142m")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--no_pretrained", action="store_true")

    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--cross_attn_layers", type=int, default=2)
    p.add_argument("--cross_attn_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--relation_set", type=str, default=None, choices=[None, "fiw", "kinface"])
    p.add_argument("--relation_loss_weight", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "plateau", "none"])
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--no_grad_ckpt", action="store_true")

    p.add_argument("--loss", type=str, default="combined",
                   choices=["bce", "contrastive", "combined"])
    p.add_argument("--contrastive_weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.1)

    p.add_argument("--disable_amp", action="store_true")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class KinshipCombinedLoss(nn.Module):
    def __init__(self, loss_type: str, contrastive_weight: float,
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


class Model05CUDATrainer(Trainer):
    def __init__(self, *args, gradient_accumulation=1, relation_to_index=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_accumulation = max(1, int(gradient_accumulation))
        self.relation_to_index = relation_to_index

    def _relation_indices(self, relations):
        return torch.tensor(
            [self.relation_to_index(r) for r in relations],
            dtype=torch.long, device=self.device,
        )

    def train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc="Training (CUDA)")
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
    print("Model 05 — DINOv2 + LoRA + DiffAttn | Nvidia CUDA training")
    print("=" * 60)
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    from torch.utils.data import DataLoader
    test_ds = KinshipPairDataset(
        root_dir=resolve_dataset_root(test_data_config, test_dataset_name),
        dataset_type=test_dataset_name,
        split="test",
        transform=get_transforms(test_data_config, train=False),
        split_seed=args.seed,
        negative_ratio=test_data_config.negative_ratio,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    relation_set = args.relation_set or args.train_dataset
    num_relations = len(RELATION_SETS[relation_set])
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
    total = sum(p.numel() for p in model.parameters())
    trainable = model.count_trainable()
    print(f"Total params: {total:,}  |  Trainable: {trainable:,} "
          f"({100*trainable/total:.2f}%)")

    loss_fn = KinshipCombinedLoss(args.loss, args.contrastive_weight,
                                  args.relation_loss_weight, args.temperature)

    trainer = Model05CUDATrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, config=train_config, device=device,
        gradient_accumulation=args.gradient_accumulation,
        relation_to_index=model.relation_to_index,
        monitor_metric=train_config.monitor_metric,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()

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
        "backbone_name": args.backbone_name, "img_size": args.img_size,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "embedding_dim": args.embedding_dim,
        "cross_attn_layers": args.cross_attn_layers,
        "cross_attn_heads": args.cross_attn_heads, "dropout": args.dropout,
        "relation_set": relation_set,
        "relation_loss_weight": args.relation_loss_weight,
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
              {**metadata, "validation_metrics": val_metrics,
               "test_metrics": test_metrics})


if __name__ == "__main__":
    main()
