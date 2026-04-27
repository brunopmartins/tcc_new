#!/usr/bin/env python3
"""
Model 05 — AMD ROCm test script.

Loads a checkpoint trained by `train.py`, re-applies the validation-selected
threshold stored in the checkpoint, and reports kinship metrics plus
per-relation accuracy on the chosen dataset.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "AMD"))

from rocm_utils import setup_rocm_environment, get_rocm_device, clear_rocm_cache  # noqa: E402
from config import DataConfig  # noqa: E402
from dataset import KinshipPairDataset, get_transforms  # noqa: E402
from evaluation import KinshipMetrics, print_metrics  # noqa: E402
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import DINOv2LoRAKinship  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Model 05 (AMD ROCm)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override stored validation threshold (debug only)")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--rocm_device", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def build_model_from_checkpoint(checkpoint: dict) -> DINOv2LoRAKinship:
    cfg = checkpoint.get("model_config", checkpoint.get("protocol", {}).get("model_config", {}))
    return DINOv2LoRAKinship(
        backbone_name=cfg.get("backbone_name", "vit_base_patch14_dinov2.lvd142m"),
        img_size=cfg.get("img_size", 224),
        lora_rank=cfg.get("lora_rank", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        backbone_pretrained=True,  # safe even if no internet; weights overwritten below
        use_gradient_checkpointing=False,  # inference — no need
        embedding_dim=cfg.get("embedding_dim", 512),
        cross_attn_layers=cfg.get("cross_attn_layers", 2),
        cross_attn_heads=cfg.get("cross_attn_heads", 8),
        dropout=cfg.get("dropout", 0.1),
        relation_set=cfg.get("relation_set", "fiw"),
        relation_loss_weight=cfg.get("relation_loss_weight", 0.2),
    )


def main() -> None:
    args = parse_args()
    print("\n" + "=" * 60)
    print("Model 05 — AMD ROCm test")
    print("=" * 60)

    setup_rocm_environment(visible_devices=str(args.rocm_device))
    device = get_rocm_device(args.rocm_device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    print(f"  total params:     {sum(p.numel() for p in model.parameters()):,}")
    print(f"  trainable params: {model.count_trainable():,}")

    data_config = DataConfig()
    apply_data_root_override(data_config, args.dataset, args.data_root)
    root_dir = resolve_dataset_root(data_config, args.dataset)

    test_ds = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=args.dataset,
        split="test",
        transform=get_transforms(data_config, train=False),
        split_seed=checkpoint.get("protocol", {}).get("split_seed", data_config.split_seed),
        negative_ratio=checkpoint.get("protocol", {}).get("negative_ratio", data_config.negative_ratio),
    )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_ds)}")
    clear_rocm_cache()

    threshold = (
        args.threshold
        if args.threshold is not None
        else get_checkpoint_threshold(checkpoint, default=0.5)
    )
    print(f"Threshold: {threshold:.3f}\n")

    all_preds, all_labels, all_rels = [], [], []
    all_rel_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing (ROCm)"):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]
            relations = batch.get("relation", ["unknown"] * labels.size(0))

            out = model(img1, img2)
            probs = torch.sigmoid(out["logits"]).cpu().numpy().flatten()
            rel_probs = torch.softmax(out["relation_logits"], dim=-1).cpu().numpy()

            all_preds.extend(probs.tolist())
            all_labels.extend(labels.numpy().flatten().tolist())
            all_rels.extend(relations)
            all_rel_preds.extend(rel_probs.tolist())

    preds = np.asarray(all_preds)
    labels = np.asarray(all_labels)

    metrics = KinshipMetrics(threshold=threshold)
    metrics.all_predictions = list(preds)
    metrics.all_labels = list(labels)
    metrics.all_relations = all_rels
    results = metrics.compute()
    print_metrics(results, prefix="Test ")

    # Persist
    with open(out_dir / "test_metrics_rocm.json", "w") as f:
        serial = {k: v for k, v in results.items() if isinstance(v, (int, float, str))}
        serial["platform"] = "AMD ROCm"
        serial["threshold"] = float(threshold)
        json.dump(serial, f, indent=2)

    # Relation-head top-1 accuracy (positive pairs only)
    relation_classes = list(model.relation_classes)
    rel_pred_indices = np.asarray(all_rel_preds).argmax(axis=1)
    rel_names = [relation_classes[i] for i in rel_pred_indices]
    kin_mask = labels > 0.5
    if kin_mask.any():
        correct = sum(
            1
            for i, is_kin in enumerate(kin_mask)
            if is_kin and rel_names[i] == all_rels[i]
        )
        acc_rel = correct / kin_mask.sum()
        print(f"\nRelation head top-1 accuracy (kin only): {acc_rel:.4f}")
        with open(out_dir / "relation_head_accuracy.json", "w") as f:
            json.dump({"accuracy": float(acc_rel), "n_kin": int(kin_mask.sum())}, f, indent=2)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
