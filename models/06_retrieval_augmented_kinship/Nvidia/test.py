#!/usr/bin/env python3
"""Model 06 — Nvidia CUDA test script."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from config import DataConfig  # noqa: E402
from dataset import KinshipPairDataset, get_transforms  # noqa: E402
from evaluation import KinshipMetrics, print_metrics  # noqa: E402
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import RetrievalAugmentedKinship  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Model 06 (Nvidia CUDA)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--output_dir", type=str, default="results")
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

    gallery_ds = KinshipPairDataset(
        root_dir=resolve_dataset_root(data_config, args.dataset),
        dataset_type=args.dataset, split="train",
        transform=get_transforms(data_config, train=False),
        split_seed=split_seed, negative_ratio=0.0,
    )
    gal_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    n = model.build_gallery(gal_loader, device=device, positive_only=True)
    print(f"Gallery: {n}")

    test_ds = KinshipPairDataset(
        root_dir=resolve_dataset_root(data_config, args.dataset),
        dataset_type=args.dataset, split="test",
        transform=get_transforms(data_config, train=False),
        split_seed=split_seed,
        negative_ratio=ckpt.get("protocol", {}).get("negative_ratio", data_config.negative_ratio),
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    threshold = args.threshold if args.threshold is not None else get_checkpoint_threshold(ckpt, 0.5)

    probs, labels, relations = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing (CUDA)"):
            out = model(batch["img1"].to(device), batch["img2"].to(device))
            probs.extend(torch.sigmoid(out["logits"]).cpu().numpy().flatten())
            labels.extend(batch["label"].numpy().flatten())
            relations.extend(batch.get("relation", ["unknown"] * batch["label"].size(0)))

    preds = np.asarray(probs)
    labels = np.asarray(labels)
    metrics = KinshipMetrics(threshold=threshold)
    metrics.all_predictions = list(preds)
    metrics.all_labels = list(labels)
    metrics.all_relations = relations
    results = metrics.compute()
    print_metrics(results, prefix="Test ")

    with open(out_dir / "test_metrics_cuda.json", "w") as f:
        serial = {k: v for k, v in results.items() if isinstance(v, (int, float, str))}
        serial["platform"] = "Nvidia CUDA"
        serial["threshold"] = float(threshold)
        serial["gallery_size"] = int(n)
        json.dump(serial, f, indent=2)


if __name__ == "__main__":
    main()
