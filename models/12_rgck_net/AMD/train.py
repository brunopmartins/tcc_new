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
from dataset import create_dataloaders, KinshipPairDataset, get_transforms, FIW_RELATION_TO_IDX, FIW_NUM_RELATIONS  # noqa: E402
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

    # Phase 5 of proposta_rgck_net_kinship.md §38: relation-type auxiliary head
    p.add_argument("--relation_aux_weight", type=float, default=0.0,
                   help="Weight λ for the relation-type CE auxiliary loss applied to positive pairs only. "
                        "0.0 = disabled (R002 baseline). 0.05 = R005 default.")
    p.add_argument("--relation_aux_balanced", action="store_true", default=True,
                   help="Use inverse-frequency class weights computed from train positives.")
    p.add_argument("--relation_aux_unbalanced", dest="relation_aux_balanced",
                   action="store_false",
                   help="Disable class-balancing weights (uniform CE).")

    # R006: symmetric forward — process each pair as both (A,B) and (B,A) and
    # combine the two directions in the loss. Same parameter count, ~+15-25%
    # training time (tokenizer reused, only head runs twice).
    p.add_argument("--symmetric_forward", action="store_true", default=False,
                   help="Run the post-tokenizer head in both (A,B) and (B,A) orders "
                        "and apply Option-B BCE / CE_rel on each direction "
                        "(R006 default).")

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
    BCE-with-logits over classifier head + optional auxiliary losses.

    Phase 4 (`proposta_rgck_net_kinship.md` §28) — supervised contrastive on
    (gA_norm, gB_norm):
      pos pairs (label=1): pull cos toward 1 → (1 - cos)^2
      neg pairs (label=0): push cos below margin → max(0, cos - (1-margin))^2

    Phase 5 (`proposta_rgck_net_kinship.md` §38) — relation-type CE on the
    11-way relation_head logits, applied only to positive pairs (non-kin
    masked out via relation_idx = -1). Class-balanced via inverse-frequency
    weights computed from train positives.

    L_total = L_bce
            + supcon_weight   * L_supcon
            + relation_weight * L_ce_relation(mask=positive)

    Reduces to pure BCE when both aux weights are 0 (matches R002).
    """

    def __init__(
        self,
        supcon_weight: float = 0.0,
        supcon_margin: float = 0.3,
        relation_weight: float = 0.0,
        relation_class_weights: "Optional[torch.Tensor]" = None,
    ):
        super().__init__()
        self.loss_type = "bce"
        self.bce = nn.BCEWithLogitsLoss()
        self.supcon_weight = float(supcon_weight)
        self.supcon_margin = float(supcon_margin)
        self.relation_weight = float(relation_weight)
        if relation_class_weights is not None:
            self.register_buffer(
                "relation_class_weights", relation_class_weights.float()
            )
        else:
            self.relation_class_weights = None

    @staticmethod
    def _supcon_term(gA: torch.Tensor, gB: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
        # gA, gB are already L2-normalised in M12 model.forward
        cos_sim = (gA * gB).sum(dim=1)  # (B,)
        label_f = labels.float()
        pos_loss = (1.0 - cos_sim) ** 2
        neg_loss = torch.clamp(cos_sim - (1.0 - margin), min=0.0) ** 2
        loss = label_f * pos_loss + (1.0 - label_f) * neg_loss
        return loss.mean()

    def _masked_relation_ce(
        self,
        rel_logits: torch.Tensor,
        relation_idx: torch.Tensor,
        labels: torch.Tensor,
    ) -> "Optional[torch.Tensor]":
        """CE over rel_logits restricted to positive pairs with a valid
        relation index. Returns None when the mask is empty (so the caller
        can skip the term)."""
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
        relation_idx: "Optional[torch.Tensor]" = None,
    ) -> torch.Tensor:
        # outputs may be a 6-tuple (logit, weights, attn, gA, gB, rel_logits)
        # or a 7-tuple ending with sym_extras when M12 is in symmetric_forward
        # mode. rel_logits is None when aux_relation_head is off.
        if not isinstance(outputs, tuple):
            return self.bce(outputs.squeeze(-1).float(), labels.float())

        logit = outputs[0]
        sym_extras = outputs[6] if len(outputs) >= 7 else None
        is_symmetric = isinstance(sym_extras, dict)

        if is_symmetric:
            # Option-B BCE: penalise each direction individually so both heads
            # are forced to be correct, then average. Equivalent to BCE on
            # logits stacked along a virtual "direction" axis.
            logit_ab = sym_extras["logit_ab"]
            logit_ba = sym_extras["logit_ba"]
            bce_ab = self.bce(logit_ab.squeeze(-1).float(), labels.float())
            bce_ba = self.bce(logit_ba.squeeze(-1).float(), labels.float())
            total = 0.5 * (bce_ab + bce_ba)
        else:
            total = self.bce(logit.squeeze(-1).float(), labels.float())

        # SupCon aux: in symmetric mode average the two directions, otherwise
        # use the AB tokens.
        if self.supcon_weight > 0.0 and len(outputs) >= 5 and outputs[3] is not None:
            gA_ab = outputs[3]
            gB_ab = outputs[4]
            supcon_ab = self._supcon_term(gA_ab, gB_ab, labels, self.supcon_margin)
            if is_symmetric:
                gA_ba = sym_extras["gA_norm_ba"]
                gB_ba = sym_extras["gB_norm_ba"]
                supcon_ba = self._supcon_term(gA_ba, gB_ba, labels, self.supcon_margin)
                supcon = 0.5 * (supcon_ab + supcon_ba)
            else:
                supcon = supcon_ab
            total = total + self.supcon_weight * supcon

        # Phase 5 CE_rel aux: positives-only, masked CE on the relation head.
        if self.relation_weight > 0.0 and relation_idx is not None:
            if is_symmetric and sym_extras["rel_logits_ab"] is not None:
                ce_ab = self._masked_relation_ce(
                    sym_extras["rel_logits_ab"], relation_idx, labels
                )
                ce_ba = self._masked_relation_ce(
                    sym_extras["rel_logits_ba"], relation_idx, labels
                )
                if ce_ab is not None and ce_ba is not None:
                    total = total + self.relation_weight * 0.5 * (ce_ab + ce_ba)
                elif ce_ab is not None:
                    total = total + self.relation_weight * ce_ab
                elif ce_ba is not None:
                    total = total + self.relation_weight * ce_ba
            elif (
                not is_symmetric
                and len(outputs) >= 6
                and outputs[5] is not None
            ):
                ce = self._masked_relation_ce(outputs[5], relation_idx, labels)
                if ce is not None:
                    total = total + self.relation_weight * ce

        return total


def compute_fiw_relation_class_weights(
    train_dataset, num_classes: int = FIW_NUM_RELATIONS
) -> torch.Tensor:
    """Inverse-frequency class weights for the Phase 5 CE aux loss.

    Counts only positive (kin) pairs. Returns a (num_classes,) tensor that
    can be passed to F.cross_entropy as `weight`. Weights are rescaled so
    that their mean is 1.0, which keeps the loss magnitude comparable to
    an unweighted CE for stable λ tuning.
    """
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


class RGCKROCmTrainer(ROCmTrainer):
    """ROCm trainer for M12 — passes full output tuple to RGCKBCELoss and
    threads ``relation_idx`` through to support the Phase 5 aux CE loss."""

    def __init__(self, *args, gradient_accumulation: int = 1, **kwargs):
        self.gradient_accumulation = max(1, int(gradient_accumulation))
        super().__init__(*args, **kwargs)

    def _compute_loss(self, outputs, labels):
        return self.loss_fn(outputs, labels)

    def train_epoch(self) -> float:
        """Train for one epoch. Mirrors ROCmTrainer.train_epoch but also pulls
        ``relation_idx`` from each batch and forwards it to the loss so the
        Phase 5 relation-type CE head can be supervised."""
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

    aux_relation_head_enabled = bool(args.relation_aux_weight > 0)
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
        aux_relation_head=aux_relation_head_enabled,
        num_relation_classes=FIW_NUM_RELATIONS,
        symmetric_forward=args.symmetric_forward,
    )
    model = optimize_for_rocm(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:        {total:,}")
    print(f"  Trainable params:    {trainable:,} ({100*trainable/total:.2f}%)")

    relation_class_weights = None
    relation_class_counts = None
    if aux_relation_head_enabled:
        if args.relation_aux_balanced:
            relation_class_weights, relation_class_counts = (
                compute_fiw_relation_class_weights(train_loader.dataset)
            )
            print(
                f"  Relation CE class weights (balanced, mean=1.0):\n"
                f"    counts:  {relation_class_counts.tolist()}\n"
                f"    weights: {[f'{w:.3f}' for w in relation_class_weights.tolist()]}"
            )
        else:
            print("  Relation CE class weights: uniform (--relation_aux_unbalanced)")

    loss_fn = RGCKBCELoss(
        supcon_weight=args.supcon_weight,
        supcon_margin=args.supcon_margin,
        relation_weight=args.relation_aux_weight,
        relation_class_weights=relation_class_weights,
    )
    if args.supcon_weight > 0:
        print(f"  Loss: BCE + {args.supcon_weight:.3f} × SupCon (margin={args.supcon_margin})")
    if args.relation_aux_weight > 0:
        print(
            f"  Loss: + {args.relation_aux_weight:.3f} × CE_rel "
            f"(positives only, {FIW_NUM_RELATIONS} classes, "
            f"{'balanced' if args.relation_aux_balanced else 'uniform'})"
        )

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
    # Ensure loss-side buffers (e.g. Phase 5 relation_class_weights) live on the
    # same device as the model.
    loss_fn.to(device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    print("\nStarting ROCm-optimised training...")
    if args.symmetric_forward:
        bce_term = "0.5·(BCE_AB + BCE_BA)"
    else:
        bce_term = "BCE(classifier_logit)"
    loss_components = [bce_term]
    if args.supcon_weight > 0:
        if args.symmetric_forward:
            loss_components.append(f"{args.supcon_weight:.3f}·avg(SupCon_AB, SupCon_BA)")
        else:
            loss_components.append(f"{args.supcon_weight:.3f}·SupCon(gA,gB)")
    if args.relation_aux_weight > 0:
        if args.symmetric_forward:
            loss_components.append(f"{args.relation_aux_weight:.3f}·avg(CE_rel_AB, CE_rel_BA)|pos")
        else:
            loss_components.append(f"{args.relation_aux_weight:.3f}·CE_rel(rel_logits|pos)")
    print(f"Loss: {' + '.join(loss_components)}")
    if args.symmetric_forward:
        print("Symmetric forward: each pair processed in both (A,B) and (B,A) orders")
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
        "aux_relation_head": aux_relation_head_enabled,
        "relation_aux_weight": args.relation_aux_weight,
        "relation_aux_balanced": bool(args.relation_aux_balanced),
        "num_relation_classes": FIW_NUM_RELATIONS,
        "symmetric_forward": bool(args.symmetric_forward),
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
