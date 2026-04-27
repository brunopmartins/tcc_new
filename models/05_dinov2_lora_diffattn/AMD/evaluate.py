#!/usr/bin/env python3
"""
Model 05 — AMD ROCm comprehensive evaluation.

Goes beyond `test.py` with:
  - ROC curve and confusion matrix figures
  - per-relation breakdown (accuracy + F1)
  - relation-head confusion matrix (positive pairs only)
  - optional LoRA-only vs base-only comparison (ablation)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "AMD"))

from rocm_utils import setup_rocm_environment, get_rocm_device, clear_rocm_cache  # noqa: E402
from config import DataConfig  # noqa: E402
from dataset import KinshipPairDataset, get_transforms  # noqa: E402
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import DINOv2LoRAKinship  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Model 05 (AMD ROCm)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="evaluation_results")
    p.add_argument("--full_analysis", action="store_true")
    p.add_argument("--rocm_device", type=int, default=0)
    return p.parse_args()


def build_model_from_checkpoint(checkpoint: dict) -> DINOv2LoRAKinship:
    cfg = checkpoint.get("model_config", checkpoint.get("protocol", {}).get("model_config", {}))
    return DINOv2LoRAKinship(
        backbone_name=cfg.get("backbone_name", "vit_base_patch14_dinov2.lvd142m"),
        img_size=cfg.get("img_size", 224),
        lora_rank=cfg.get("lora_rank", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        backbone_pretrained=True,
        use_gradient_checkpointing=False,
        embedding_dim=cfg.get("embedding_dim", 512),
        cross_attn_layers=cfg.get("cross_attn_layers", 2),
        cross_attn_heads=cfg.get("cross_attn_heads", 8),
        dropout=cfg.get("dropout", 0.1),
        relation_set=cfg.get("relation_set", "fiw"),
        relation_loss_weight=cfg.get("relation_loss_weight", 0.2),
    )


def collect_predictions(model: DINOv2LoRAKinship, loader: DataLoader, device: torch.device):
    model.eval()
    probs, labels, relations, rel_probs = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting predictions (ROCm)"):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            out = model(img1, img2)
            probs.extend(torch.sigmoid(out["logits"]).cpu().numpy().flatten().tolist())
            labels.extend(batch["label"].numpy().flatten().tolist())
            relations.extend(batch.get("relation", ["unknown"] * batch["label"].size(0)))
            rel_probs.extend(torch.softmax(out["relation_logits"], dim=-1).cpu().numpy().tolist())
    return (
        np.asarray(probs),
        np.asarray(labels),
        relations,
        np.asarray(rel_probs),
    )


def plot_roc(labels, probs, auc, out_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "r--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Model 05 (AMD ROCm)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(labels, binary_preds, out_path):
    cm = confusion_matrix(labels, binary_preds)
    plt.figure(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Non-Kin", "Kin"]).plot(cmap="Blues")
    plt.title("Confusion Matrix — Model 05 (AMD ROCm)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def per_relation(labels, probs, relations, threshold):
    results = {}
    rels = sorted(set(r for r, l in zip(relations, labels) if l > 0.5))
    for rel in rels:
        mask = np.array([(r == rel) and (l > 0.5) for r, l in zip(relations, labels)])
        if not mask.any():
            continue
        # Accuracy here is "fraction of kin pairs of this relation correctly
        # predicted as kin at the global threshold".
        correct = (probs[mask] > threshold).sum()
        results[rel] = {
            "n": int(mask.sum()),
            "accuracy": float(correct / mask.sum()),
        }
    return results


def plot_relation_bar(per_rel, out_path):
    rels = list(per_rel.keys())
    accs = [per_rel[r]["accuracy"] for r in rels]
    ns = [per_rel[r]["n"] for r in rels]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(rels, accs, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-relation accuracy — Model 05")
    for bar, n in zip(bars, ns):
        ax.annotate(f"n={n}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    print("\n" + "=" * 60)
    print("Model 05 — AMD ROCm evaluation")
    print("=" * 60)

    setup_rocm_environment(visible_devices=str(args.rocm_device))
    device = get_rocm_device(args.rocm_device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()

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
    clear_rocm_cache()

    threshold = get_checkpoint_threshold(checkpoint, default=0.5)
    print(f"Threshold: {threshold:.3f}")

    probs, labels, relations, rel_probs = collect_predictions(model, loader, device)
    binary_preds = (probs > threshold).astype(int)

    summary = {
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "f1": f1_score(labels, binary_preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
        "threshold": float(threshold),
        "platform": "AMD ROCm",
    }
    with open(out_dir / "metrics_rocm.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.full_analysis:
        plot_roc(labels, probs, summary["roc_auc"], out_dir / "roc_curve_rocm.png")
        plot_confusion(labels, binary_preds, out_dir / "confusion_matrix_rocm.png")

        per_rel = per_relation(labels, probs, relations, threshold)
        with open(out_dir / "per_relation.json", "w") as f:
            json.dump(per_rel, f, indent=2)
        if per_rel:
            plot_relation_bar(per_rel, out_dir / "per_relation_rocm.png")

        # Relation head confusion on positive pairs
        kin_mask = labels > 0.5
        if kin_mask.any():
            classes = list(model.relation_classes)
            true_idx = np.array([classes.index(r) if r in classes else -1 for r in relations])
            pred_idx = rel_probs.argmax(axis=1)
            valid = kin_mask & (true_idx >= 0)
            if valid.any():
                cm = confusion_matrix(true_idx[valid], pred_idx[valid],
                                      labels=list(range(len(classes))))
                plt.figure(figsize=(8, 7))
                ConfusionMatrixDisplay(cm, display_labels=classes).plot(
                    cmap="Blues", xticks_rotation=45
                )
                plt.title("Relation head — confusion (kin only)")
                plt.tight_layout()
                plt.savefig(out_dir / "relation_confusion_rocm.png", dpi=150)
                plt.close()

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY (Model 05, AMD ROCm)")
    print("=" * 50)
    for k in ["accuracy", "f1", "roc_auc", "precision", "recall", "threshold"]:
        print(f"{k:>10}: {summary[k]:.4f}")
    print("=" * 50)
    print(f"\nArtifacts: {out_dir}")


if __name__ == "__main__":
    main()
