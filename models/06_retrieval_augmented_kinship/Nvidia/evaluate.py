#!/usr/bin/env python3
"""Model 06 — Nvidia CUDA full analysis (mirror of AMD/evaluate.py)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from config import DataConfig  # noqa: E402
from dataset import KinshipPairDataset, get_transforms  # noqa: E402
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import RetrievalAugmentedKinship  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Model 06 (Nvidia CUDA)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="evaluation_results")
    p.add_argument("--full_analysis", action="store_true")
    return p.parse_args()


def build_from_checkpoint(ckpt) -> RetrievalAugmentedKinship:
    cfg = ckpt.get("model_config", ckpt.get("protocol", {}).get("model_config", {}))
    return RetrievalAugmentedKinship(
        backbone_name=cfg.get("backbone_name", "vit_base_patch16_224"),
        img_size=cfg.get("img_size", 224),
        freeze_backbone=cfg.get("freeze_backbone", True),
        backbone_pretrained=True,
        embedding_dim=cfg.get("embedding_dim", 512),
        retrieval_k=cfg.get("retrieval_k", 32),
        retrieval_attn_layers=cfg.get("retrieval_attn_layers", 2),
        retrieval_attn_heads=cfg.get("retrieval_attn_heads", 4),
        dropout=cfg.get("dropout", 0.1),
        relation_set=cfg.get("relation_set", "fiw"),
        relation_loss_weight=cfg.get("relation_loss_weight", 0.15),
        max_gallery=cfg.get("max_gallery", 200_000),
        store_gallery_on_cpu=cfg.get("store_gallery_on_cpu", False),
        use_gradient_checkpointing=False,
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()

    data_config = DataConfig()
    apply_data_root_override(data_config, args.dataset, args.data_root)
    split_seed = ckpt.get("protocol", {}).get("split_seed", data_config.split_seed)

    gal_ds = KinshipPairDataset(
        root_dir=resolve_dataset_root(data_config, args.dataset),
        dataset_type=args.dataset, split="train",
        transform=get_transforms(data_config, train=False),
        split_seed=split_seed, negative_ratio=0.0,
    )
    gal_loader = DataLoader(gal_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    n_gal = model.build_gallery(gal_loader, device=device, positive_only=True)

    test_ds = KinshipPairDataset(
        root_dir=resolve_dataset_root(data_config, args.dataset),
        dataset_type=args.dataset, split="test",
        transform=get_transforms(data_config, train=False),
        split_seed=split_seed,
        negative_ratio=ckpt.get("protocol", {}).get("negative_ratio", data_config.negative_ratio),
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    threshold = get_checkpoint_threshold(ckpt, 0.5)

    probs, labels, relations = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting (CUDA)"):
            out = model(batch["img1"].to(device), batch["img2"].to(device))
            probs.extend(torch.sigmoid(out["logits"]).cpu().numpy().flatten())
            labels.extend(batch["label"].numpy().flatten())
            relations.extend(batch.get("relation", ["unknown"] * batch["label"].size(0)))
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    binary = (probs > threshold).astype(int)

    summary = {
        "accuracy": accuracy_score(labels, binary),
        "precision": precision_score(labels, binary, zero_division=0),
        "recall": recall_score(labels, binary, zero_division=0),
        "f1": f1_score(labels, binary, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
        "threshold": float(threshold),
        "gallery_size": int(n_gal),
        "platform": "Nvidia CUDA",
    }
    with open(out_dir / "metrics_cuda.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.full_analysis:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {summary['roc_auc']:.4f}")
        plt.plot([0, 1], [0, 1], "r--", lw=1)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC — Model 06 (CUDA)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(out_dir / "roc_curve_cuda.png", dpi=150); plt.close()

        cm = confusion_matrix(labels, binary)
        plt.figure(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=["Non-Kin", "Kin"]).plot(cmap="Blues")
        plt.title("Confusion — Model 06 (CUDA)"); plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix_cuda.png", dpi=150); plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
