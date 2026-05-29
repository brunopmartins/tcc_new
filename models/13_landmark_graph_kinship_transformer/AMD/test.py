#!/usr/bin/env python3
"""AMD ROCm test script for Model 13 — LGKT-Net."""
import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ["MIOPEN_FIND_MODE"] = "FAST"

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "AMD"))

from rocm_utils import setup_rocm_environment, clear_rocm_cache  # noqa: E402
from config import DataConfig  # noqa: E402
from dataset import KinshipPairDataset, get_transforms  # noqa: E402
from evaluation import KinshipMetrics, print_metrics  # noqa: E402
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import build_lgkt_model  # noqa: E402


LGKT_MEAN = [0.5, 0.5, 0.5]
LGKT_STD = [0.5, 0.5, 0.5]
LGKT_IMG_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser(description="Test Model 13 (AMD ROCm)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="fiw", choices=["kinface", "fiw"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--rocm_device", type=int, default=0)
    parser.add_argument("--aligned_root", type=str, default=None)
    parser.add_argument("--adaface_weights", type=str,
                        default=str(Path(__file__).parent.parent.parent / "12_rgck_net"
                                    / "weights" / "adaface_ir101_webface4m.pth"))
    return parser.parse_args()


def main():
    args = parse_args()
    setup_rocm_environment(visible_devices=str(args.rocm_device))

    print("\n" + "=" * 60)
    print("AMD ROCm Testing — Model 13: LGKT-Net")
    print("=" * 60)

    device = torch.device(f"cuda:{args.rocm_device}")

    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", checkpoint.get("protocol", {}).get("model_config", {}))
    img_size = int(model_config.get("img_size", LGKT_IMG_SIZE))

    model = build_lgkt_model(
        adaface_weights_path=args.adaface_weights,
        embedding_dim=model_config.get("embedding_dim", 512),
        num_heads=model_config.get("num_heads", 4),
        num_graph_layers=model_config.get("num_graph_layers", 2),
        gate_hidden=model_config.get("gate_hidden", 128),
        classifier_hidden=model_config.get("classifier_hidden", 512),
        dropout=model_config.get("dropout", 0.2),
        freeze_backbone=model_config.get("freeze_backbone", True),
        unfreeze_last_stage=model_config.get("unfreeze_last_stage", False),
        aux_relation_head=model_config.get("aux_relation_head", False),
        num_relation_classes=model_config.get("num_relation_classes", 11),
        roi_output_size=model_config.get("roi_output_size", 3),
        feature_stage=model_config.get("feature_stage", "stage4"),
        comparison_only_pooling=model_config.get("comparison_only_pooling", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    data_config = DataConfig(image_size=img_size, normalize_mean=LGKT_MEAN, normalize_std=LGKT_STD)
    apply_data_root_override(data_config, args.dataset, args.data_root)

    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using provided threshold: {threshold}")
    else:
        threshold = get_checkpoint_threshold(checkpoint) or 0.5
        print(f"Using stored validation threshold: {threshold:.3f}")

    test_dataset = KinshipPairDataset(
        root_dir=resolve_dataset_root(data_config, args.dataset),
        dataset_type=args.dataset, split="test",
        transform=get_transforms(data_config, train=False),
        split_seed=42, negative_ratio=1.0,
        aligned_root=args.aligned_root,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    print(f"Test samples: {len(test_dataset)}")

    metrics_calc = KinshipMetrics(threshold=threshold)

    all_preds, all_labels, all_relations = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing (ROCm)"):
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            labels = batch["label"]
            relations = batch.get("relation", [None] * len(labels))

            outputs = model(img1, img2)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            if relations and relations[0] is not None:
                all_relations.extend(list(relations))

    preds = np.asarray(all_preds, dtype=float)
    labels = np.asarray(all_labels, dtype=int)
    relations = all_relations if all_relations else None

    metrics_calc.update(predictions=preds, labels=labels, relations=relations)
    metrics = metrics_calc.compute()

    print()
    print_metrics(metrics, prefix="Test ")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "test_metrics_rocm.txt"
    with open(results_path, "w") as f:
        f.write("AMD ROCm Test Results — Model 13 (LGKT-Net)\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            if not isinstance(v, dict):
                f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")
    print(f"\nResults saved to {results_path}")

    clear_rocm_cache()


if __name__ == "__main__":
    main()
