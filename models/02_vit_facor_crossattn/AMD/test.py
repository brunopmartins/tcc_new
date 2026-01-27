#!/usr/bin/env python3
"""
AMD ROCm Testing script for ViT + FaCoR Cross-Attention model.

This script is optimized for AMD GPUs using the ROCm platform.

Usage:
    python test.py --checkpoint checkpoints/best.pt --dataset kinface
    python test.py --checkpoint checkpoints/best.pt --visualize_attention
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setup ROCm environment
os.environ["MIOPEN_FIND_MODE"] = "FAST"

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "AMD"))

from rocm_utils import (
    setup_rocm_environment,
    get_rocm_device,
    clear_rocm_cache,
)
from config import DataConfig
from dataset import KinshipPairDataset, get_transforms
from evaluation import KinshipMetrics, print_metrics, find_optimal_threshold
from torch.utils.data import DataLoader

# Add parent directory for model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import ViTFaCoRModel


def parse_args():
    parser = argparse.ArgumentParser(description="Test ViT-FaCoR Model (AMD ROCm)")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--visualize_attention", action="store_true",
                        help="Visualize cross-attention maps")
    parser.add_argument("--num_visualizations", type=int, default=10)

    # ROCm specific
    parser.add_argument("--rocm_device", type=int, default=0,
                        help="ROCm GPU device ID")

    return parser.parse_args()


def visualize_attention(
    img1: torch.Tensor,
    img2: torch.Tensor,
    attn_map: torch.Tensor,
    save_path: str,
    label: int,
    prediction: float,
):
    """Visualize cross-attention between two faces."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img1 = img1.cpu() * std + mean
    img2 = img2.cpu() * std + mean

    img1 = img1.permute(1, 2, 0).numpy().clip(0, 1)
    img2 = img2.permute(1, 2, 0).numpy().clip(0, 1)

    attn = attn_map.cpu().mean(dim=0).numpy()
    patch_size = int(np.sqrt(attn.shape[0]))
    attn_spatial = attn.mean(axis=1).reshape(patch_size, patch_size)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img1)
    axes[0].set_title("Face 1")
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title("Face 2")
    axes[1].axis("off")

    axes[2].imshow(attn_spatial, cmap='hot', interpolation='nearest')
    axes[2].set_title("Cross-Attention")
    axes[2].axis("off")

    axes[3].imshow(img1)
    attn_resized = np.array(
        plt.cm.get_cmap('jet')(
            np.interp(attn_spatial,
                     (attn_spatial.min(), attn_spatial.max()),
                     (0, 1))
        )
    )
    attn_resized = np.kron(attn_resized, np.ones((16, 16, 1)))[:224, :224]
    axes[3].imshow(attn_resized, alpha=0.5)
    axes[3].set_title("Attention Overlay")
    axes[3].axis("off")

    kin_str = "Kin" if label == 1 else "Non-Kin"
    pred_str = "Kin" if prediction > 0.5 else "Non-Kin"
    correct = "V" if (label == 1) == (prediction > 0.5) else "X"

    plt.suptitle(f"Ground Truth: {kin_str} | Predicted: {pred_str} ({prediction:.3f}) {correct}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("AMD ROCm Testing - ViT-FaCoR Model")
    print("=" * 60)

    setup_rocm_environment(visible_devices=str(args.rocm_device))
    device = get_rocm_device(args.rocm_device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = ViTFaCoRModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Dataset
    data_config = DataConfig()
    root_dir = data_config.kinface_i_root if args.dataset == "kinface" else data_config.fiw_root
    if args.data_root:
        root_dir = args.data_root

    test_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=args.dataset,
        split="test",
        transform=get_transforms(data_config, train=False),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    # Clear cache
    clear_rocm_cache()

    # Collect predictions
    all_preds, all_labels, all_relations = [], [], []
    attention_samples = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing (ROCm)"):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]
            relations = batch.get("relation", ["unknown"] * len(labels))

            emb1, emb2, attn_map = model(img1, img2)
            similarities = F.cosine_similarity(emb1, emb2, dim=1)
            predictions = (similarities + 1) / 2

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_relations.extend(relations)

            if args.visualize_attention and len(attention_samples) < args.num_visualizations:
                for i in range(min(len(img1), args.num_visualizations - len(attention_samples))):
                    attention_samples.append({
                        "img1": img1[i],
                        "img2": img2[i],
                        "attn": attn_map[i],
                        "label": labels[i].item(),
                        "pred": predictions[i].item(),
                    })

    predictions = np.array(all_preds)
    labels = np.array(all_labels)

    # Find optimal threshold
    if args.threshold is None:
        threshold, best_f1 = find_optimal_threshold(predictions, labels, "f1")
        print(f"Optimal threshold: {threshold:.3f} (F1: {best_f1:.4f})")
    else:
        threshold = args.threshold

    # Compute metrics
    metrics = KinshipMetrics(threshold=threshold)
    metrics.all_predictions = list(predictions)
    metrics.all_labels = list(labels)
    metrics.all_relations = all_relations
    results = metrics.compute()

    print_metrics(results, prefix="Test ")

    # Visualize attention
    if args.visualize_attention and attention_samples:
        print(f"\nGenerating {len(attention_samples)} attention visualizations...")
        attn_dir = output_dir / "attention_maps_rocm"
        attn_dir.mkdir(exist_ok=True)

        for i, sample in enumerate(attention_samples):
            visualize_attention(
                sample["img1"],
                sample["img2"],
                sample["attn"],
                str(attn_dir / f"attention_{i:03d}.png"),
                sample["label"],
                sample["pred"],
            )
        print(f"Attention maps saved to {attn_dir}")

    # Save results
    with open(output_dir / "test_metrics_rocm.txt", "w") as f:
        f.write("AMD ROCm Test Results - ViT-FaCoR\n")
        f.write("=" * 40 + "\n")
        for k, v in results.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
