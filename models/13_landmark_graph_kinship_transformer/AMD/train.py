#!/usr/bin/env python3
"""
AMD ROCm training for Model 13 — LGKT-Net (Landmark Graph Kinship Transformer).

R001 recipe (per ``EXPERIMENT_PLAN.md``):
- AdaFace IR-101 backbone (stages 1-3 frozen by default, stage 4 + output_layer
  optionally trainable via --unfreeze_last_stage, mirroring M12 Phase 2)
- ONE backbone pass per face, ROIAlign over 8 landmark-derived component boxes
- 2-layer edge-aware graph transformer over the joint pair graph
- Native symmetric pooling: [mean, |diff|, prod] per homologous node-pair
- BCE classifier head, optional 11-way relation aux head

Input: 224×224 aligned face (FIW_aligned native), internally resized to 112×112
once per face for the AdaFace backbone.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

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
from dataset import (  # noqa: E402
    create_dataloaders,
    create_fiw_5fold_train_val_loaders,
    KinshipPairDataset,
    get_transforms,
    FIW_RELATION_TO_IDX,
    FIW_NUM_RELATIONS,
)
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
from model import build_lgkt_model, NODE_NAMES  # noqa: E402


LGKT_MEAN = [0.5, 0.5, 0.5]
LGKT_STD = [0.5, 0.5, 0.5]
LGKT_IMG_SIZE = 224


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train M13 LGKT-Net (AMD ROCm)")

    # Dataset
    p.add_argument("--train_dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--test_dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--aligned_root", type=str, default=None,
                   help="Native 224×224 FIW_aligned (no resize at load).")
    p.add_argument("--num_workers", type=int, default=4)

    # Backbone
    p.add_argument("--adaface_weights", type=str,
                   default=str(Path(__file__).parent.parent.parent / "12_rgck_net"
                               / "weights" / "adaface_ir101_webface4m.pth"))
    p.add_argument("--freeze_backbone", action="store_true", default=True)
    p.add_argument("--unfreeze_backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--unfreeze_last_stage", action="store_true", default=False,
                   help="Keep backbone frozen except body[46:49] + output_layer.")
    p.add_argument("--img_size", type=int, default=LGKT_IMG_SIZE)

    # LGKT specifics
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_graph_layers", type=int, default=2)
    p.add_argument("--gate_hidden", type=int, default=128)
    p.add_argument("--classifier_hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--roi_output_size", type=int, default=3,
                   help="ROIAlign output spatial size (3×3 default).")
    p.add_argument("--feature_stage", type=str, default="stage4",
                   choices=["stage3", "stage4"],
                   help="Backbone feature stage for ROIAlign. stage4=7×7×512 "
                        "(R001); stage3=14×14×256 (R002).")
    p.add_argument("--comparison_only_pooling", action="store_true", default=False,
                   help="Exclude the global node from the symmetric pooler "
                        "while keeping it in the graph as context (M12 R009 analog).")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "plateau", "step", "none"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Relation aux head (M12 lesson — improves per-class balance)
    p.add_argument("--relation_aux_weight", type=float, default=0.0,
                   help="λ for relation-type CE on positive pairs. 0 = off (R001 default).")
    p.add_argument("--relation_aux_balanced", action="store_true", default=True)
    p.add_argument("--relation_aux_unbalanced", dest="relation_aux_balanced",
                   action="store_false")

    # Differential LR (3 groups: backbone stage 4 / output_layer / head)
    p.add_argument("--differential_lr", action="store_true", default=False)
    p.add_argument("--lr_stage4", type=float, default=5e-6)
    p.add_argument("--lr_output_layer", type=float, default=5e-6)
    p.add_argument("--lr_head", type=float, default=2e-5)

    # Negative sampling
    p.add_argument("--negative_ratio", type=float, default=1.0)
    p.add_argument("--eval_negative_ratio", type=float, default=1.0)
    p.add_argument("--train_negative_strategy", type=str, default="random",
                   choices=["random", "relation_matched"])
    p.add_argument("--eval_negative_strategy", type=str, default="random",
                   choices=["random", "relation_matched"])
    p.add_argument("--hard_negative_ratio", type=float, default=0.0,
                   help="Fraction of role-matched negatives when "
                        "train_negative_strategy=relation_matched.")

    # ROCm
    p.add_argument("--rocm_device", type=int, default=0)
    p.add_argument("--disable_amp", action="store_true")
    p.add_argument("--gfx_version", type=str, default=None)

    # Misc
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=50)

    # K-fold CV
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--num_folds", type=int, default=5)

    return p.parse_args()


class LGKTLoss(nn.Module):
    """BCE on the kinship logit + optional masked CE on the relation aux head.

    LGKT-Net's forward returns a dict with keys: ``logit``, ``gate``,
    ``tokens_a``, ``tokens_b``, and optionally ``relation_logits``.
    """

    def __init__(
        self,
        relation_weight: float = 0.0,
        relation_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.relation_weight = float(relation_weight)
        if relation_class_weights is not None:
            self.register_buffer(
                "relation_class_weights", relation_class_weights.float()
            )
        else:
            self.relation_class_weights = None

    def _masked_relation_ce(
        self,
        rel_logits: torch.Tensor,
        relation_idx: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        mask = (relation_idx >= 0) & (labels.long() == 1)
        if not mask.any():
            return None
        return nn.functional.cross_entropy(
            rel_logits[mask].float(),
            relation_idx[mask],
            weight=self.relation_class_weights,
        )

    def forward(
        self,
        outputs,
        labels: torch.Tensor,
        relation_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(outputs, dict):
            # Trainer may pass a raw tensor (no relation head wired).
            return self.bce(outputs.squeeze(-1).float(), labels.float())

        logits = outputs["logits"]
        total = self.bce(logits.squeeze(-1).float(), labels.float())

        if (
            self.relation_weight > 0.0
            and relation_idx is not None
            and outputs.get("relation_logits") is not None
        ):
            ce = self._masked_relation_ce(outputs["relation_logits"], relation_idx, labels)
            if ce is not None:
                total = total + self.relation_weight * ce

        return total


def compute_fiw_relation_class_weights(train_dataset, num_classes: int = FIW_NUM_RELATIONS):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for (_, _, rel), label in zip(train_dataset.pairs, train_dataset.labels):
        if int(label) != 1:
            continue
        idx = FIW_RELATION_TO_IDX.get(rel, -1)
        if 0 <= idx < num_classes:
            counts[idx] += 1
    safe = counts.clamp(min=1).float()
    weights = 1.0 / safe
    weights = weights * (float(num_classes) / weights.sum())
    return weights, counts


def build_differential_lr_optimizer_and_scheduler(
    model: nn.Module,
    lr_stage4: float,
    lr_output_layer: float,
    lr_head: float,
    weight_decay: float,
    warmup_epochs: int,
    num_epochs: int,
    min_lr: float,
):
    backbone = model.tokenizer.backbone
    feature_stage = getattr(model, "feature_stage", "stage4")

    if feature_stage == "stage4":
        backbone_blocks = [p for p in backbone.body[46:49].parameters() if p.requires_grad]
        output_layer_params = [
            p for p in backbone.output_layer.parameters() if p.requires_grad
        ]
        backbone_block_name = "backbone_stage4"
    else:  # stage3 — last 3 blocks of stage 3; output_layer is not used.
        backbone_blocks = [p for p in backbone.body[43:46].parameters() if p.requires_grad]
        output_layer_params = []
        backbone_block_name = "backbone_stage3_tail"

    # Channel projection (stage3 only) goes with the head — it sits between
    # backbone features and the graph and is initialised fresh.
    head_modules = [
        model.tokenizer.channel_projection,
        model.node_type_embed,
        model.graph,
        model.pooler,
        model.classifier,
    ]
    if getattr(model, "relation_head", None) is not None:
        head_modules.append(model.relation_head)
    head_params: list = []
    for module in head_modules:
        head_params.extend(p for p in module.parameters() if p.requires_grad)

    param_groups = [
        {"params": backbone_blocks, "lr": lr_stage4, "name": backbone_block_name},
    ]
    if output_layer_params:
        param_groups.append(
            {"params": output_layer_params, "lr": lr_output_layer,
             "name": "backbone_output_layer"}
        )
    param_groups.append({"params": head_params, "lr": lr_head, "name": "head"})
    optimizer = AdamW(param_groups, weight_decay=weight_decay, eps=1e-8)

    warmup = LinearLR(
        optimizer, start_factor=1.0 / max(warmup_epochs, 1), end_factor=1.0,
        total_iters=max(warmup_epochs, 1),
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(num_epochs - warmup_epochs, 1), eta_min=min_lr,
    )
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    group_summary = [
        {"name": pg["name"], "lr": pg["lr"],
         "num_params": sum(p.numel() for p in pg["params"]),
         "num_tensors": len(pg["params"])}
        for pg in param_groups
    ]
    return optimizer, scheduler, group_summary


class LGKTROCmTrainer(ROCmTrainer):
    """ROCm trainer that threads ``relation_idx`` to the loss and accepts the
    dict-shaped LGKT output."""

    def __init__(self, *args, gradient_accumulation: int = 1, **kwargs):
        self.gradient_accumulation = max(1, int(gradient_accumulation))
        super().__init__(*args, **kwargs)

    def _compute_loss(self, outputs, labels):
        return self.loss_fn(outputs, labels)

    def train_epoch(self) -> float:
        from tqdm import tqdm

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        clear_rocm_cache()

        pbar = tqdm(self.train_loader, desc="Training (ROCm)")
        for batch in pbar:
            img1 = batch["img1"].to(self.device, non_blocking=True)
            img2 = batch["img2"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            relation_idx = batch.get("relation_idx")
            if relation_idx is not None:
                relation_idx = relation_idx.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = self.model(img1, img2)
                    loss = self.loss_fn(outputs, labels, relation_idx)
                self.scaler.scale(loss).backward()
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(img1, img2)
                loss = self.loss_fn(outputs, labels, relation_idx)
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm,
                    )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("AMD ROCm Training — Model 13: LGKT-Net (Landmark Graph Kinship Transformer)")
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

    train_data_config = DataConfig(
        image_size=args.img_size, normalize_mean=LGKT_MEAN, normalize_std=LGKT_STD,
    )
    test_data_config = DataConfig(
        image_size=args.img_size, normalize_mean=LGKT_MEAN, normalize_std=LGKT_STD,
    )
    apply_data_root_override(train_data_config, args.train_dataset, args.data_root)
    if args.test_dataset:
        apply_data_root_override(test_data_config, args.test_dataset, args.data_root)
    if args.aligned_root:
        train_data_config.aligned_root = args.aligned_root  # type: ignore[attr-defined]
        test_data_config.aligned_root = args.aligned_root   # type: ignore[attr-defined]

    train_config = TrainConfig(
        batch_size=args.batch_size, num_epochs=args.epochs,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        scheduler=args.scheduler, warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr, patience=args.patience,
        max_grad_norm=args.max_grad_norm, use_amp=not args.disable_amp,
        save_every=5, checkpoint_dir=args.checkpoint_dir,
        monitor_metric="roc_auc", threshold_metric="f1",
    )

    print(f"\nLoading {args.train_dataset} dataset for training...")
    if args.fold is not None:
        if args.train_dataset != "fiw":
            raise ValueError(f"--fold is only supported for FIW; got {args.train_dataset}")
        print(f"  K-fold CV active: fold {args.fold}/{args.num_folds}")
        train_loader, val_loader, _ = create_fiw_5fold_train_val_loaders(
            config=train_data_config, fold_k=args.fold, n_folds=args.num_folds,
            batch_size=args.batch_size, num_workers=args.num_workers,
            train_negative_ratio=args.negative_ratio,
            eval_negative_ratio=args.eval_negative_ratio,
            train_negative_sampling_strategy=args.train_negative_strategy,
            eval_negative_sampling_strategy=args.eval_negative_strategy,
            split_seed=args.seed, aligned_root=args.aligned_root,
            train_hard_negative_ratio=args.hard_negative_ratio,
            eval_hard_negative_ratio=0.0,
        )
    else:
        train_loader, val_loader, _ = create_dataloaders(
            config=train_data_config, batch_size=args.batch_size,
            num_workers=args.num_workers, dataset_type=args.train_dataset,
            train_negative_ratio=args.negative_ratio,
            eval_negative_ratio=args.eval_negative_ratio,
            train_negative_sampling_strategy=args.train_negative_strategy,
            eval_negative_sampling_strategy=args.eval_negative_strategy,
            split_seed=args.seed, aligned_root=args.aligned_root,
            train_hard_negative_ratio=args.hard_negative_ratio,
            eval_hard_negative_ratio=0.0,
        )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    print(f"Loading {args.test_dataset} dataset for testing...")
    from torch.utils.data import DataLoader
    test_dataset = KinshipPairDataset(
        root_dir=resolve_dataset_root(test_data_config, args.test_dataset),
        dataset_type=args.test_dataset, split="test",
        transform=get_transforms(test_data_config, train=False),
        split_seed=args.seed, negative_ratio=args.eval_negative_ratio,
        aligned_root=args.aligned_root,
        negative_sampling_strategy=args.eval_negative_strategy,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Test: {len(test_dataset)}")

    print("\nCreating LGKT-Net (Landmark Graph Kinship Transformer)...")
    print(f"  AdaFace weights:     {args.adaface_weights}")
    print(f"  Backbone frozen:     {args.freeze_backbone}")
    print(f"  Feature stage:       {args.feature_stage}")
    print(f"  Comparison-only pool: {args.comparison_only_pooling}")
    print(f"  Nodes:               {NODE_NAMES}")
    print(f"  Graph layers:        {args.num_graph_layers} × {args.num_heads} heads")
    print(f"  Dropout:             {args.dropout}")

    aux_relation_head_enabled = bool(args.relation_aux_weight > 0)
    model = build_lgkt_model(
        adaface_weights_path=args.adaface_weights,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_graph_layers=args.num_graph_layers,
        gate_hidden=args.gate_hidden,
        classifier_hidden=args.classifier_hidden,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_stage=args.unfreeze_last_stage,
        aux_relation_head=aux_relation_head_enabled,
        num_relation_classes=FIW_NUM_RELATIONS,
        roi_output_size=args.roi_output_size,
        feature_stage=args.feature_stage,
        comparison_only_pooling=args.comparison_only_pooling,
    )
    model = optimize_for_rocm(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:        {total:,}")
    print(f"  Trainable params:    {trainable:,} ({100*trainable/total:.2f}%)")

    relation_class_weights = None
    if aux_relation_head_enabled:
        if args.relation_aux_balanced:
            relation_class_weights, counts = compute_fiw_relation_class_weights(train_loader.dataset)
            print(f"  Relation CE class weights (balanced, mean=1.0):")
            print(f"    counts:  {counts.tolist()}")
            print(f"    weights: {[f'{w:.3f}' for w in relation_class_weights.tolist()]}")
        else:
            print("  Relation CE class weights: uniform")

    loss_fn = LGKTLoss(
        relation_weight=args.relation_aux_weight,
        relation_class_weights=relation_class_weights,
    )
    if args.relation_aux_weight > 0:
        print(f"  Loss: BCE + {args.relation_aux_weight:.3f} × CE_rel (positives only)")
    else:
        print(f"  Loss: BCE")

    trainer = LGKTROCmTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, config=train_config, device=device,
        rocm_device_id=args.rocm_device, monitor_metric="roc_auc",
        gradient_accumulation=args.gradient_accumulation,
    )
    loss_fn.to(device)

    if args.differential_lr:
        new_optimizer, new_scheduler, group_summary = build_differential_lr_optimizer_and_scheduler(
            model=model, lr_stage4=args.lr_stage4, lr_output_layer=args.lr_output_layer,
            lr_head=args.lr_head, weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs, num_epochs=args.epochs, min_lr=args.min_lr,
        )
        trainer.optimizer = new_optimizer
        trainer.scheduler = new_scheduler
        trainer.config.warmup_epochs = 0
        print("\n  Differential LR enabled:")
        for grp in group_summary:
            print(f"    {grp['name']:<22} lr={grp['lr']:.1e}  "
                  f"({grp['num_tensors']} tensors, {grp['num_params']:,} params)")

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    print("\nStarting ROCm-optimised training...")
    print(f"Scheduler: {train_config.scheduler} (warmup={train_config.warmup_epochs}, min_lr={train_config.min_lr:.1e})")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.gradient_accumulation} "
          f"(effective {args.batch_size * args.gradient_accumulation})")
    print(f"Negative ratio: train={args.negative_ratio:.2f}, eval={args.eval_negative_ratio:.2f}")
    print(f"Negative strategy: train={args.train_negative_strategy}, "
          f"eval={args.eval_negative_strategy} (hard ratio: {args.hard_negative_ratio})")
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
        "num_heads": args.num_heads,
        "num_graph_layers": args.num_graph_layers,
        "gate_hidden": args.gate_hidden,
        "classifier_hidden": args.classifier_hidden,
        "dropout": args.dropout,
        "freeze_backbone": args.freeze_backbone,
        "unfreeze_last_stage": args.unfreeze_last_stage,
        "aux_relation_head": aux_relation_head_enabled,
        "relation_aux_weight": args.relation_aux_weight,
        "relation_aux_balanced": bool(args.relation_aux_balanced),
        "num_relation_classes": FIW_NUM_RELATIONS,
        "roi_output_size": args.roi_output_size,
        "feature_stage": args.feature_stage,
        "comparison_only_pooling": bool(args.comparison_only_pooling),
        "differential_lr": bool(args.differential_lr),
        "lr_stage4": args.lr_stage4 if args.differential_lr else None,
        "lr_output_layer": args.lr_output_layer if args.differential_lr else None,
        "lr_head": args.lr_head if args.differential_lr else None,
        "nodes": NODE_NAMES,
    }
    protocol_metadata = build_protocol_metadata(
        train_dataset=args.train_dataset, test_dataset=args.test_dataset,
        threshold=threshold, threshold_metric=train_config.threshold_metric,
        split_seed=args.seed, negative_ratio=args.eval_negative_ratio,
        monitor_metric=train_config.monitor_metric, args=args,
        extra={
            "model_config": model_config,
            "training_negative_ratio": args.negative_ratio,
            "training_negative_strategy": args.train_negative_strategy,
            "evaluation_negative_strategy": args.eval_negative_strategy,
            "training_hard_negative_ratio": args.hard_negative_ratio,
        },
    )

    for checkpoint_name in ["best.pt", "final.pt"]:
        checkpoint_path = Path(args.checkpoint_dir) / checkpoint_name
        update_checkpoint_payload(checkpoint_path, {"model_config": model_config})
        update_checkpoint_metadata(checkpoint_path, protocol_metadata)

    results_path = Path(args.checkpoint_dir) / "test_results_rocm.txt"
    with open(results_path, "w") as f:
        f.write("AMD ROCm Training Results — Model 13 (LGKT-Net)\n")
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
