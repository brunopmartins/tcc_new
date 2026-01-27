#!/usr/bin/env python3
"""
AMD ROCm Comprehensive evaluation script for Age Synthesis model.
Performs detailed analysis including per-relation, cross-age, and ablation studies.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --full_analysis
    python evaluate.py --checkpoint checkpoints/best.pt --ablation
"""
import argparse
import os
import sys
from pathlib import Path
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
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
from evaluation import KinshipMetrics, evaluate_model
from torch.utils.data import DataLoader

# Add parent directory for model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import AgeSynthesisComparisonModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Age Synthesis Model (AMD ROCm)")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--full_analysis", action="store_true",
                        help="Run full analysis with visualizations")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study on aggregation methods")

    # ROCm specific
    parser.add_argument("--rocm_device", type=int, default=0,
                        help="ROCm GPU device ID")

    return parser.parse_args()


def plot_roc_curve(predictions, labels, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, predictions)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Age Synthesis Model (AMD ROCm)')
    plt.grid(True, alpha=0.3)

    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    plt.annotate(f'AUC = {roc_auc:.4f}', xy=(0.6, 0.2), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(predictions, labels, threshold, output_path):
    """Plot and save confusion matrix."""
    binary_preds = (predictions > threshold).astype(int)
    cm = confusion_matrix(labels, binary_preds)

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Kin', 'Kin'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Age Synthesis Model (AMD ROCm)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_per_relation_accuracy(per_relation_metrics, output_path):
    """Plot per-relation accuracy bar chart."""
    relations = list(per_relation_metrics.keys())
    accuracies = [per_relation_metrics[r]['accuracy'] for r in relations]
    counts = [per_relation_metrics[r]['count'] for r in relations]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(relations))
    width = 0.6

    bars = ax1.bar(x, accuracies, width, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Relation Type')
    ax1.set_ylabel('Accuracy', color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(relations, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, 1)

    for bar, count in zip(bars, counts):
        ax1.annotate(f'n={count}',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=8)

    plt.title('Per-Relation Accuracy - Age Synthesis Model (AMD ROCm)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_comparison_matrix_analysis(model, dataloader, device, output_path):
    """Analyze which age comparisons contribute most."""
    model.eval()

    all_comparison_matrices = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing comparisons (ROCm)"):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]

            _, comparison_matrix = model(img1, img2)

            all_comparison_matrices.append(comparison_matrix.cpu().numpy())
            all_labels.extend(labels.numpy())

    comparisons = np.vstack(all_comparison_matrices)
    labels = np.array(all_labels)

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_mean = comparisons[pos_mask].mean(axis=0) if pos_mask.sum() > 0 else np.zeros(9)
    neg_mean = comparisons[neg_mask].mean(axis=0) if neg_mask.sum() > 0 else np.zeros(9)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im1 = axes[0].imshow(pos_mean.reshape(3, 3), cmap='RdYlGn', vmin=-2, vmax=2)
    axes[0].set_title('Positive Pairs (Kin)')
    axes[0].set_xlabel('Person 2 Age')
    axes[0].set_ylabel('Person 1 Age')
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_xticklabels(['Young', 'Mid', 'Old'])
    axes[0].set_yticklabels(['Young', 'Mid', 'Old'])
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(neg_mean.reshape(3, 3), cmap='RdYlGn', vmin=-2, vmax=2)
    axes[1].set_title('Negative Pairs (Non-Kin)')
    axes[1].set_xlabel('Person 2 Age')
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_xticklabels(['Young', 'Mid', 'Old'])
    axes[1].set_yticklabels(['Young', 'Mid', 'Old'])
    plt.colorbar(im2, ax=axes[1])

    diff = pos_mean - neg_mean
    im3 = axes[2].imshow(diff.reshape(3, 3), cmap='coolwarm', vmin=-1, vmax=1)
    axes[2].set_title('Difference (Kin - Non-Kin)')
    axes[2].set_xlabel('Person 2 Age')
    axes[2].set_xticks([0, 1, 2])
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_xticklabels(['Young', 'Mid', 'Old'])
    axes[2].set_yticklabels(['Young', 'Mid', 'Old'])
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle('Age Comparison Matrix Analysis (AMD ROCm)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_ablation_study(model, dataloader, device, output_dir):
    """Run ablation study on different aggregation methods."""
    results = {}
    aggregation_methods = ["attention", "max", "mean"]

    for method in aggregation_methods:
        print(f"\nTesting aggregation: {method}")

        original_aggregation = model.age_aggregator.aggregation
        model.age_aggregator.aggregation = method

        # Clear cache between runs
        clear_rocm_cache()

        metrics = evaluate_model(model, dataloader, device)
        results[method] = {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics.get("roc_auc", 0),
        }

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics.get('roc_auc', 0):.4f}")

        model.age_aggregator.aggregation = original_aggregation

    ablation_path = output_dir / "ablation_results_rocm.json"
    with open(ablation_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    methods = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in methods]
    f1_scores = [results[m]["f1"] for m in methods]
    aucs = [results[m]["roc_auc"] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.25

    ax.bar(x - width, accuracies, width, label='Accuracy', color='steelblue')
    ax.bar(x, f1_scores, width, label='F1 Score', color='forestgreen')
    ax.bar(x + width, aucs, width, label='ROC-AUC', color='coral')

    ax.set_xlabel('Aggregation Method')
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Aggregation Methods (AMD ROCm)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_plot_rocm.png", dpi=150)
    plt.close()

    return results


def main():
    args = parse_args()

    # Setup ROCm
    print("\n" + "=" * 60)
    print("AMD ROCm Evaluation - Age Synthesis Model")
    print("=" * 60)

    setup_rocm_environment(visible_devices=str(args.rocm_device))
    device = get_rocm_device(args.rocm_device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = AgeSynthesisComparisonModel(use_age_synthesis=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load dataset
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
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Evaluating on {len(test_dataset)} samples")

    # Clear cache before evaluation
    clear_rocm_cache()

    # Basic evaluation
    metrics = evaluate_model(model, test_loader, device)

    # Save basic metrics
    with open(output_dir / "metrics_rocm.json", "w") as f:
        serializable = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
        serializable["platform"] = "AMD ROCm"
        json.dump(serializable, f, indent=2)

    if args.full_analysis:
        print("\nRunning full analysis...")

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                img1 = batch["img1"].to(device, non_blocking=True)
                img2 = batch["img2"].to(device, non_blocking=True)
                score, _ = model(img1, img2)
                all_preds.extend(torch.sigmoid(score).cpu().numpy().flatten())
                all_labels.extend(batch["label"].numpy().flatten())

        predictions = np.array(all_preds)
        labels = np.array(all_labels)

        print("Generating ROC curve...")
        plot_roc_curve(predictions, labels, output_dir / "roc_curve_rocm.png")

        print("Generating confusion matrix...")
        plot_confusion_matrix(predictions, labels, 0.5, output_dir / "confusion_matrix_rocm.png")

        if "per_relation" in metrics:
            print("Generating per-relation analysis...")
            plot_per_relation_accuracy(metrics["per_relation"], output_dir / "per_relation_rocm.png")

        print("Analyzing comparison matrix...")
        plot_comparison_matrix_analysis(model, test_loader, device, output_dir / "comparison_analysis_rocm.png")

    if args.ablation:
        print("\nRunning ablation study...")
        run_ablation_study(model, test_loader, device, output_dir)

    print(f"\nResults saved to {output_dir}")

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY (AMD ROCm)")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
