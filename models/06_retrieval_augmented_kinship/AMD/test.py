#!/usr/bin/env python3
"""Model 06 — AMD ROCm test script."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "AMD"))

from rocm_utils import setup_rocm_environment, get_rocm_device, clear_rocm_cache  # noqa: E402
from config import DataConfig  # noqa: E402
from dataset import KinshipPairDataset, get_transforms  # noqa: E402
from evaluation import KinshipMetrics, print_metrics  # noqa: E402
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import RetrievalAugmentedKinship  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Model 06 (AMD ROCm)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--rocm_device", type=int, default=0)
    p.add_argument("--skip_gallery_rebuild", action="store_true",
                   help="Skip rebuilding the gallery; only safe if checkpoint "
                        "was saved with gallery buffers (it isn't by default).")
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
    setup_rocm_environment(visible_devices=str(args.rocm_device))
    device = get_rocm_device(args.rocm_device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    clear_rocm_cache()

    dataset_type = args.dataset
    data_config = DataConfig()
    apply_data_root_override(data_config, dataset_type, args.data_root)

    # Build the retrieval gallery from the training split used during training.
    if not args.skip_gallery_rebuild:
        split_seed = ckpt.get("protocol", {}).get("split_seed", data_config.split_seed)
        gallery_ds = KinshipPairDataset(
            root_dir=resolve_dataset_root(data_config, dataset_type),
            dataset_type=dataset_type,
            split="train",
            transform=get_transforms(data_config, train=False),
            split_seed=split_seed,
            negative_ratio=0.0,
        )
        gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
        n = model.build_gallery(gallery_loader, device=device, positive_only=True)
        print(f"Gallery: {n} positive pairs stored.")

    test_ds = KinshipPairDataset(
        root_dir=resolve_dataset_root(data_config, dataset_type),
        dataset_type=dataset_type, split="test",
        transform=get_transforms(data_config, train=False),
        split_seed=ckpt.get("protocol", {}).get("split_seed", data_config.split_seed),
        negative_ratio=ckpt.get("protocol", {}).get("negative_ratio", data_config.negative_ratio),
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    threshold = args.threshold if args.threshold is not None else get_checkpoint_threshold(ckpt, 0.5)
    print(f"Threshold: {threshold:.3f}")

    probs, labels, relations = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing (ROCm)"):
            out = model(batch["img1"].to(device, non_blocking=True),
                        batch["img2"].to(device, non_blocking=True))
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

    with open(out_dir / "test_metrics_rocm.json", "w") as f:
        serial = {k: v for k, v in results.items() if isinstance(v, (int, float, str))}
        serial["platform"] = "AMD ROCm"
        serial["threshold"] = float(threshold)
        serial["gallery_size"] = len(model.gallery)
        json.dump(serial, f, indent=2)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
