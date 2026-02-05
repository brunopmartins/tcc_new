#!/usr/bin/env python3
"""
NVIDIA CUDA Comprehensive evaluation script for Unified Kinship Model.
Performs detailed analysis including component ablation and multi-age analysis.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --full_analysis
    python evaluate.py --checkpoint checkpoints/best.pt --component_ablation
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
from sklearn.metrics import (
    roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)
from tqdm import tqdm

# Setup CUDA environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "AMD"))

from cuda_utils import (
    setup_cuda_environment,
    get_cuda_device,
    clear_cuda_cache,
)
from config import DataConfig
from dataset import KinshipPairDataset, get_transforms
from evaluation import KinshipMetrics, print_metrics, find_optimal_threshold
from torch.utils.data import DataLoader

# Add parent directory for model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import UnifiedKinshipModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Unified Kinship Model (NVIDIA CUDA)")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--full_analysis", action="store_true")
    parser.add_argument("--component_ablation", action="store_true",
                        help="Run ablation study on model components")
    parser.add_argument("--cross_dataset", action="store_true",
                        help="Evaluate on both datasets")

    # CUDA specific
    parser.add_argument("--cuda_device", type=int, default=0)

    return parser.parse_args()


def evaluate_with_config(model, dataloader, device, use_age=True, use_cross_attn=True):
    """Evaluate model with specific component configuration."""
    model.eval()

    # Store original settings
    orig_age = model.use_age_synthesis
    orig_cross = model.use_cross_attention

    # Set configuration
    model.use_age_synthesis = use_age if orig_age else False
    model.use_cross_attention = use_cross_attn

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]

            output = model(img1, img2)
            preds = torch.sigmoid(output["logits"]).cpu()

            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    # Restore settings
    model.use_age_synthesis = orig_age
    model.use_cross_attention = orig_cross

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    binary = (preds > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, binary),
        "precision": precision_score(labels, binary, zero_division=0),
        "recall": recall_score(labels, binary, zero_division=0),
        "f1": f1_score(labels, binary, zero_division=0),
        "roc_auc": roc_auc_score(labels, preds),
        "predictions": preds,
        "labels": labels,
    }


def run_component_ablation(model, dataloader, device, output_dir):
    """Run ablation study on model components."""
    print("\nRunning component ablation study...")

    configurations = [
        ("Full Model", True, True),
        ("No Age Synthesis", False, True),
        ("No Cross-Attention", True, False),
        ("Baseline (Neither)", False, False),
    ]

    results = {}

    for name, use_age, use_cross in configurations:
        print(f"\n  Testing: {name}")
        clear_cuda_cache()

        metrics = evaluate_with_config(model, dataloader, device, use_age, use_cross)
        results[name] = {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "use_age_synthesis": use_age,
            "use_cross_attention": use_cross,
        }

        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    AUC: {metrics['roc_auc']:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    configs = list(results.keys())
    x = np.arange(len(configs))
    width = 0.25

    # Metrics comparison
    ax = axes[0]
    accs = [results[c]["accuracy"] for c in configs]
    f1s = [results[c]["f1"] for c in configs]
    aucs = [results[c]["roc_auc"] for c in configs]

    ax.bar(x - width, accs, width, label='Accuracy', color='steelblue')
    ax.bar(x, f1s, width, label='F1', color='forestgreen')
    ax.bar(x + width, aucs, width, label='AUC', color='coral')

    ax.set_ylabel('Score')
    ax.set_xlabel('Configuration')
    ax.set_title('Component Ablation Study')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)

    # Component contribution
    ax = axes[1]
    full_acc = results["Full Model"]["accuracy"]
    contributions = {
        "Age Synthesis": full_acc - results["No Age Synthesis"]["accuracy"],
        "Cross-Attention": full_acc - results["No Cross-Attention"]["accuracy"],
        "Combined Effect": full_acc - results["Baseline (Neither)"]["accuracy"],
    }

    colors = ['forestgreen' if v > 0 else 'coral' for v in contributions.values()]
    bars = ax.bar(contributions.keys(), contributions.values(), color=colors)
    ax.set_ylabel('Accuracy Improvement')
    ax.set_title('Component Contribution to Accuracy')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:+.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    ha='center', va='bottom' if height > 0 else 'top')

    plt.suptitle('Unified Model Component Analysis (NVIDIA CUDA)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "component_ablation_cuda.png", dpi=150)
    plt.close()

    return results


def analyze_age_comparisons(model, dataloader, device, output_dir):
    """Analyze which age comparisons contribute most to predictions."""
    if not model.use_age_synthesis:
        print("Age synthesis not enabled, skipping age analysis")
        return None

    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing age comparisons (CUDA)"):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]

            output = model(img1, img2)

            if "comparison_scores" in output:
                all_scores.append(output["comparison_scores"].cpu())
                all_labels.extend(labels.numpy())

    if not all_scores:
        print("No comparison scores available")
        return None

    scores = torch.cat(all_scores, dim=0).numpy()
    labels = np.array(all_labels)

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_mean = scores[pos_mask].mean(axis=0) if pos_mask.sum() > 0 else np.zeros(scores.shape[1])
    neg_mean = scores[neg_mask].mean(axis=0) if neg_mask.sum() > 0 else np.zeros(scores.shape[1])

    num_ages = int(np.sqrt(len(pos_mean)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Kin pairs
    im1 = axes[0].imshow(pos_mean.reshape(num_ages, num_ages), cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_title('Kin Pairs')
    axes[0].set_xlabel('Person 2 Age')
    axes[0].set_ylabel('Person 1 Age')
    age_labels = ['Young', 'Mid', 'Old'][:num_ages]
    axes[0].set_xticks(range(num_ages))
    axes[0].set_yticks(range(num_ages))
    axes[0].set_xticklabels(age_labels)
    axes[0].set_yticklabels(age_labels)
    plt.colorbar(im1, ax=axes[0])

    # Non-kin pairs
    im2 = axes[1].imshow(neg_mean.reshape(num_ages, num_ages), cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_title('Non-Kin Pairs')
    axes[1].set_xlabel('Person 2 Age')
    axes[1].set_xticks(range(num_ages))
    axes[1].set_yticks(range(num_ages))
    axes[1].set_xticklabels(age_labels)
    axes[1].set_yticklabels(age_labels)
    plt.colorbar(im2, ax=axes[1])

    # Difference
    diff = pos_mean - neg_mean
    im3 = axes[2].imshow(diff.reshape(num_ages, num_ages), cmap='coolwarm', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Difference (Kin - Non-Kin)')
    axes[2].set_xlabel('Person 2 Age')
    axes[2].set_xticks(range(num_ages))
    axes[2].set_yticks(range(num_ages))
    axes[2].set_xticklabels(age_labels)
    axes[2].set_yticklabels(age_labels)
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle('Age Comparison Analysis (NVIDIA CUDA)')
    plt.tight_layout()
    plt.savefig(output_dir / "age_analysis_cuda.png", dpi=150)
    plt.close()

    return {
        "kin_mean_scores": pos_mean.tolist(),
        "nonkin_mean_scores": neg_mean.tolist(),
        "difference": diff.tolist(),
    }


def cross_dataset_evaluation(model, device, data_config, batch_size, output_dir):
    """Evaluate on both KinFaceW and FIW datasets."""
    results = {}

    datasets = [
        ("kinface", data_config.kinface_i_root),
        ("fiw", data_config.fiw_root),
    ]

    for dataset_name, root_dir in datasets:
        print(f"\nEvaluating on {dataset_name}...")

        try:
            test_dataset = KinshipPairDataset(
                root_dir=root_dir,
                dataset_type=dataset_name,
                split="test",
                transform=get_transforms(data_config, train=False),
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )

            metrics = evaluate_with_config(model, test_loader, device)
            results[dataset_name] = {
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "samples": len(test_dataset),
            }

            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  AUC: {metrics['roc_auc']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            results[dataset_name] = {"error": str(e)}

    # Visualization
    if all("accuracy" in results[d] for d in results):
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(results))
        width = 0.25

        datasets = list(results.keys())
        accs = [results[d]["accuracy"] for d in datasets]
        f1s = [results[d]["f1"] for d in datasets]
        aucs = [results[d]["roc_auc"] for d in datasets]

        ax.bar(x - width, accs, width, label='Accuracy', color='steelblue')
        ax.bar(x, f1s, width, label='F1', color='forestgreen')
        ax.bar(x + width, aucs, width, label='AUC', color='coral')

        ax.set_ylabel('Score')
        ax.set_xlabel('Dataset')
        ax.set_title('Cross-Dataset Evaluation (NVIDIA CUDA)')
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets])
        ax.legend()
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "cross_dataset_cuda.png", dpi=150)
        plt.close()

    return results


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("NVIDIA CUDA Evaluation - Unified Kinship Model")
    print("=" * 60)

    setup_cuda_environment(visible_devices=str(args.cuda_device))
    device = get_cuda_device(args.cuda_device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = UnifiedKinshipModel(use_age_synthesis=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Age synthesis: {model.use_age_synthesis}")
    print(f"Cross-attention: {model.use_cross_attention}")

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
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    clear_cuda_cache()

    # Basic evaluation
    print("\nRunning evaluation...")
    metrics = evaluate_with_config(model, test_loader, device)

    predictions = metrics["predictions"]
    labels = metrics["labels"]

    # Optimal threshold
    threshold, best_f1 = find_optimal_threshold(predictions, labels)
    print(f"Optimal threshold: {threshold:.3f} (F1: {best_f1:.4f})")

    # Recompute with optimal threshold
    binary_preds = (predictions > threshold).astype(int)
    final_results = {
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "f1": f1_score(labels, binary_preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, predictions),
        "threshold": threshold,
        "platform": "NVIDIA CUDA",
    }

    # Save metrics
    with open(output_dir / "metrics_cuda.json", "w") as f:
        json.dump(final_results, f, indent=2)

    if args.component_ablation:
        ablation_results = run_component_ablation(model, test_loader, device, output_dir)
        with open(output_dir / "ablation_results.json", "w") as f:
            json.dump(ablation_results, f, indent=2)

    if args.full_analysis:
        print("\nRunning full analysis...")

        # ROC curve
        print("Generating ROC curve...")
        fpr, tpr, _ = roc_curve(labels, predictions)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Unified Model (NVIDIA CUDA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve_cuda.png", dpi=150)
        plt.close()

        # Confusion matrix
        print("Generating confusion matrix...")
        cm = confusion_matrix(labels, binary_preds)
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Kin', 'Kin'])
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix - Unified Model (NVIDIA CUDA)')
        plt.savefig(output_dir / "confusion_matrix_cuda.png", dpi=150)
        plt.close()

        # Age analysis
        print("Analyzing age comparisons...")
        age_results = analyze_age_comparisons(model, test_loader, device, output_dir)
        if age_results:
            with open(output_dir / "age_analysis.json", "w") as f:
                json.dump(age_results, f, indent=2)

    if args.cross_dataset:
        print("\nRunning cross-dataset evaluation...")
        cross_results = cross_dataset_evaluation(model, device, data_config, args.batch_size, output_dir)
        with open(output_dir / "cross_dataset_results.json", "w") as f:
            json.dump(cross_results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY (NVIDIA CUDA)")
    print("=" * 50)
    print(f"Accuracy:  {final_results['accuracy']:.4f}")
    print(f"F1 Score:  {final_results['f1']:.4f}")
    print(f"ROC-AUC:   {final_results['roc_auc']:.4f}")
    print(f"Precision: {final_results['precision']:.4f}")
    print(f"Recall:    {final_results['recall']:.4f}")
    print(f"Threshold: {final_results['threshold']:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
