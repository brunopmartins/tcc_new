#!/usr/bin/env python3
"""
AMD ROCm Testing script for Unified Kinship Model.

This script is optimized for AMD GPUs using the ROCm platform.

Usage:
    python test.py --checkpoint checkpoints/best.pt
    python test.py --checkpoint checkpoints/best.pt --full_analysis
"""
import argparse
import os
import sys
from pathlib import Path
import json

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
from model import UnifiedKinshipModel


def parse_args():
    parser = argparse.ArgumentParser(description="Test Unified Kinship Model (AMD ROCm)")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--full_analysis", action="store_true",
                        help="Run comprehensive analysis")

    # ROCm specific
    parser.add_argument("--rocm_device", type=int, default=0,
                        help="ROCm GPU device ID")

    return parser.parse_args()


def component_ablation(model, dataloader, device, output_dir):
    """Analyze contribution of each component."""
    model.eval()

    orig_age = model.use_age_synthesis
    orig_cross = model.use_cross_attention

    results = {}

    configurations = [
        ("full", True, True),
        ("no_age", False, True),
        ("no_cross_attn", True, False),
        ("baseline", False, False),
    ]

    for name, use_age, use_cross in configurations:
        model.use_age_synthesis = use_age if orig_age else False
        model.use_cross_attention = use_cross

        # Clear cache between runs
        clear_rocm_cache()

        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Testing {name} (ROCm)"):
                img1 = batch["img1"].to(device, non_blocking=True)
                img2 = batch["img2"].to(device, non_blocking=True)
                labels = batch["label"]

                output = model(img1, img2)
                preds = torch.sigmoid(output["logits"]).cpu()

                all_preds.extend(preds.numpy().flatten())
                all_labels.extend(labels.numpy().flatten())

        preds = np.array(all_preds)
        labels = np.array(all_labels)
        binary = (preds > 0.5).astype(int)

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        results[name] = {
            "accuracy": accuracy_score(labels, binary),
            "f1": f1_score(labels, binary),
            "roc_auc": roc_auc_score(labels, preds),
            "use_age": use_age,
            "use_cross_attn": use_cross,
        }

        print(f"\n{name}:")
        print(f"  Accuracy: {results[name]['accuracy']:.4f}")
        print(f"  F1: {results[name]['f1']:.4f}")
        print(f"  AUC: {results[name]['roc_auc']:.4f}")

    model.use_age_synthesis = orig_age
    model.use_cross_attention = orig_cross

    with open(output_dir / "ablation_results_rocm.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    configs = list(results.keys())
    accuracies = [results[c]["accuracy"] for c in configs]
    f1s = [results[c]["f1"] for c in configs]
    aucs = [results[c]["roc_auc"] for c in configs]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(configs))
    width = 0.25

    ax.bar(x - width, accuracies, width, label='Accuracy', color='steelblue')
    ax.bar(x, f1s, width, label='F1', color='forestgreen')
    ax.bar(x + width, aucs, width, label='AUC', color='coral')

    ax.set_ylabel('Score')
    ax.set_xlabel('Configuration')
    ax.set_title('Component Ablation Study (AMD ROCm)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_plot_rocm.png", dpi=150)
    plt.close()

    return results


def analyze_age_comparisons(model, dataloader, device, output_dir):
    """Analyze which age comparisons contribute most."""
    if not model.use_age_synthesis:
        print("Age synthesis not enabled, skipping age analysis")
        return

    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing age comparisons (ROCm)"):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]

            output = model(img1, img2)

            if "comparison_scores" in output:
                all_scores.append(output["comparison_scores"].cpu())
                all_labels.extend(labels.numpy())

    if not all_scores:
        return

    scores = torch.cat(all_scores, dim=0).numpy()
    labels = np.array(all_labels)

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_mean = scores[pos_mask].mean(axis=0)
    neg_mean = scores[neg_mask].mean(axis=0)

    num_ages = int(np.sqrt(len(pos_mean)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im1 = axes[0].imshow(pos_mean.reshape(num_ages, num_ages), cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_title('Kin Pairs')
    axes[0].set_xlabel('Person 2 Age')
    axes[0].set_ylabel('Person 1 Age')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(neg_mean.reshape(num_ages, num_ages), cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_title('Non-Kin Pairs')
    axes[1].set_xlabel('Person 2 Age')
    plt.colorbar(im2, ax=axes[1])

    diff = pos_mean - neg_mean
    im3 = axes[2].imshow(diff.reshape(num_ages, num_ages), cmap='coolwarm', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Difference (Kin - Non-Kin)')
    axes[2].set_xlabel('Person 2 Age')
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle('Age Comparison Analysis (AMD ROCm)')
    plt.tight_layout()
    plt.savefig(output_dir / "age_analysis_rocm.png", dpi=150)
    plt.close()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("AMD ROCm Testing - Unified Kinship Model")
    print("=" * 60)

    setup_rocm_environment(visible_devices=str(args.rocm_device))
    device = get_rocm_device(args.rocm_device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = UnifiedKinshipModel(use_age_synthesis=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Check platform
    platform = checkpoint.get("platform", "Unknown")
    print(f"Model trained on platform: {platform}")

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

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing (ROCm)"):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]
            relations = batch.get("relation", ["unknown"] * len(labels))

            output = model(img1, img2)
            preds = torch.sigmoid(output["logits"]).cpu()

            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            all_relations.extend(relations)

    predictions = np.array(all_preds)
    labels = np.array(all_labels)

    # Find threshold
    if args.threshold is None:
        threshold, _ = find_optimal_threshold(predictions, labels)
        print(f"Optimal threshold: {threshold:.3f}")
    else:
        threshold = args.threshold

    # Compute metrics
    metrics = KinshipMetrics(threshold=threshold)
    metrics.all_predictions = list(predictions)
    metrics.all_labels = list(labels)
    metrics.all_relations = all_relations
    results = metrics.compute()

    print_metrics(results, prefix="Test ")

    # Full analysis
    if args.full_analysis:
        print("\n" + "="*50)
        print("RUNNING FULL ANALYSIS (AMD ROCm)")
        print("="*50)

        print("\n1. Component Ablation Study...")
        ablation_results = component_ablation(model, test_loader, device, output_dir)

        print("\n2. Age Comparison Analysis...")
        analyze_age_comparisons(model, test_loader, device, output_dir)

    # Save results
    with open(output_dir / "test_metrics_rocm.json", "w") as f:
        serializable = {k: v for k, v in results.items() if isinstance(v, (int, float, str))}
        serializable["platform"] = "AMD ROCm"
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
