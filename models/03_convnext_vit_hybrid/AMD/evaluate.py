#!/usr/bin/env python3
"""
AMD ROCm Comprehensive evaluation script for ConvNeXt + ViT Hybrid model.
Performs detailed analysis including backbone contribution analysis and ablation studies.

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
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, auc
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
from evaluation import KinshipMetrics, print_metrics
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root
from torch.utils.data import DataLoader

# Add parent directory for model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import ConvNeXtViTHybrid, AblationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt-ViT Hybrid (AMD ROCm)")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--full_analysis", action="store_true")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study comparing backbones")
    parser.add_argument("--analyze_features", action="store_true",
                        help="Detailed feature contribution analysis")

    # ROCm specific
    parser.add_argument("--rocm_device", type=int, default=0)

    return parser.parse_args()


def analyze_backbone_contributions(model, dataloader, device, output_dir):
    """Analyze how ConvNeXt and ViT features contribute to predictions."""
    model.eval()

    conv_similarities = []
    vit_similarities = []
    fused_similarities = []
    labels_list = []
    relations_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing backbones (ROCm)"):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]
            relations = batch.get("relation", ["unknown"] * len(labels))

            emb1, emb2, aux = model(img1, img2)

            # Extract individual backbone features
            conv_sim = F.cosine_similarity(aux["conv1"], aux["conv2"], dim=1)
            vit_sim = F.cosine_similarity(aux["vit1"], aux["vit2"], dim=1)
            fused_sim = F.cosine_similarity(emb1, emb2, dim=1)

            conv_similarities.extend(conv_sim.cpu().numpy())
            vit_similarities.extend(vit_sim.cpu().numpy())
            fused_similarities.extend(fused_sim.cpu().numpy())
            labels_list.extend(labels.numpy())
            relations_list.extend(relations)

    # Convert to arrays
    conv_sim = np.array(conv_similarities)
    vit_sim = np.array(vit_similarities)
    fused_sim = np.array(fused_similarities)
    labels = np.array(labels_list)

    pos_mask = labels == 1
    neg_mask = labels == 0

    # Separability analysis
    def compute_separability(pos_vals, neg_vals):
        pooled_std = np.sqrt((np.var(pos_vals) + np.var(neg_vals)) / 2)
        return (np.mean(pos_vals) - np.mean(neg_vals)) / (pooled_std + 1e-8)

    results = {
        "convnext_separability": compute_separability(conv_sim[pos_mask], conv_sim[neg_mask]),
        "vit_separability": compute_separability(vit_sim[pos_mask], vit_sim[neg_mask]),
        "fused_separability": compute_separability(fused_sim[pos_mask], fused_sim[neg_mask]),
        "convnext_kin_mean": float(conv_sim[pos_mask].mean()),
        "convnext_nonkin_mean": float(conv_sim[neg_mask].mean()),
        "vit_kin_mean": float(vit_sim[pos_mask].mean()),
        "vit_nonkin_mean": float(vit_sim[neg_mask].mean()),
        "fused_kin_mean": float(fused_sim[pos_mask].mean()),
        "fused_nonkin_mean": float(fused_sim[neg_mask].mean()),
    }

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Distribution plots
    ax = axes[0, 0]
    ax.hist(conv_sim[pos_mask], bins=30, alpha=0.5, label='ConvNeXt (Kin)', color='blue')
    ax.hist(conv_sim[neg_mask], bins=30, alpha=0.5, label='ConvNeXt (Non-Kin)', color='red')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('ConvNeXt Feature Similarities')
    ax.legend()

    ax = axes[0, 1]
    ax.hist(vit_sim[pos_mask], bins=30, alpha=0.5, label='ViT (Kin)', color='blue')
    ax.hist(vit_sim[neg_mask], bins=30, alpha=0.5, label='ViT (Non-Kin)', color='red')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('ViT Feature Similarities')
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(conv_sim[pos_mask], vit_sim[pos_mask], alpha=0.5, label='Kin', c='blue', s=20)
    ax.scatter(conv_sim[neg_mask], vit_sim[neg_mask], alpha=0.5, label='Non-Kin', c='red', s=20)
    ax.set_xlabel('ConvNeXt Similarity')
    ax.set_ylabel('ViT Similarity')
    ax.set_title('ConvNeXt vs ViT Feature Space')
    ax.legend()
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    ax = axes[1, 1]
    separabilities = [results["convnext_separability"], results["vit_separability"], results["fused_separability"]]
    bars = ax.bar(["ConvNeXt", "ViT", "Fused"], separabilities, color=['steelblue', 'forestgreen', 'coral'])
    ax.set_ylabel('Separability (d\')')
    ax.set_title('Feature Separability Comparison')
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, separabilities):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom')

    plt.suptitle('Backbone Contribution Analysis (AMD ROCm)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "backbone_analysis_rocm.png", dpi=150)
    plt.close()

    # Per-relation analysis
    unique_relations = list(set(relations_list))
    if len(unique_relations) > 1:
        relation_results = {}
        for rel in unique_relations:
            rel_mask = np.array(relations_list) == rel
            if rel_mask.sum() > 10:
                rel_fused = fused_sim[rel_mask]
                rel_labels = labels[rel_mask]
                rel_pos = rel_labels == 1
                rel_neg = rel_labels == 0
                if rel_pos.sum() > 0 and rel_neg.sum() > 0:
                    relation_results[rel] = {
                        "separability": compute_separability(rel_fused[rel_pos], rel_fused[rel_neg]),
                        "count": int(rel_mask.sum()),
                    }
        results["per_relation"] = relation_results

    return results


def run_fusion_ablation(model, dataloader, device, output_dir, threshold: float):
    """Compare different fusion strategies."""
    results = {}
    fusion_types = ["concat", "attention", "gated"]

    original_fusion = model.fusion.fusion_type if hasattr(model.fusion, 'fusion_type') else "concat"

    for fusion_type in fusion_types:
        print(f"\nTesting fusion: {fusion_type}")

        # Create new model with different fusion (simplified - in practice would need model recreation)
        try:
            test_model = ConvNeXtViTHybrid(fusion_type=fusion_type)
            # Copy weights where possible
            test_model.load_state_dict(model.state_dict(), strict=False)
            test_model.to(device)
            test_model.eval()

            clear_rocm_cache()

            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Testing {fusion_type}"):
                    img1 = batch["img1"].to(device, non_blocking=True)
                    img2 = batch["img2"].to(device, non_blocking=True)
                    labels = batch["label"]

                    emb1, emb2, _ = test_model(img1, img2)
                    preds = (F.cosine_similarity(emb1, emb2, dim=1) + 1) / 2

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            preds = np.array(all_preds)
            labels = np.array(all_labels)
            binary = (preds > threshold).astype(int)

            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            results[fusion_type] = {
                "accuracy": accuracy_score(labels, binary),
                "f1": f1_score(labels, binary),
                "roc_auc": roc_auc_score(labels, preds),
            }
            print(f"  Accuracy: {results[fusion_type]['accuracy']:.4f}")

        except Exception as e:
            print(f"  Skipped {fusion_type}: {e}")
            continue

    if results:
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        fusions = list(results.keys())
        x = np.arange(len(fusions))
        width = 0.25

        accs = [results[f]["accuracy"] for f in fusions]
        f1s = [results[f]["f1"] for f in fusions]
        aucs = [results[f]["roc_auc"] for f in fusions]

        ax.bar(x - width, accs, width, label='Accuracy', color='steelblue')
        ax.bar(x, f1s, width, label='F1', color='forestgreen')
        ax.bar(x + width, aucs, width, label='AUC', color='coral')

        ax.set_xlabel('Fusion Type')
        ax.set_ylabel('Score')
        ax.set_title('Fusion Strategy Comparison (AMD ROCm)')
        ax.set_xticks(x)
        ax.set_xticklabels(fusions)
        ax.legend()
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "fusion_ablation_rocm.png", dpi=150)
        plt.close()

    return results


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("AMD ROCm Evaluation - ConvNeXt-ViT Hybrid Model")
    print("=" * 60)

    setup_rocm_environment(visible_devices=str(args.rocm_device))
    device = get_rocm_device(args.rocm_device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_config = checkpoint.get("model_config", checkpoint.get("protocol", {}).get("model_config", {}))
    model = ConvNeXtViTHybrid(
        convnext_model=model_config.get("convnext_model", "convnext_base"),
        vit_model=model_config.get("vit_model", "vit_base_patch16_224"),
        embedding_dim=model_config.get("embedding_dim", 512),
        fusion_type=model_config.get("fusion_type", "concat"),
        freeze_backbones=model_config.get("freeze_backbones", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    data_config = DataConfig()
    apply_data_root_override(data_config, args.dataset, args.data_root)
    root_dir = resolve_dataset_root(data_config, args.dataset)

    test_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=args.dataset,
        split="test",
        transform=get_transforms(data_config, train=False),
        split_seed=checkpoint.get("protocol", {}).get("split_seed", data_config.split_seed),
        negative_ratio=checkpoint.get("protocol", {}).get("negative_ratio", data_config.negative_ratio),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    clear_rocm_cache()

    # Basic evaluation
    all_preds, all_labels, all_relations = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating (ROCm)"):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]
            relations = batch.get("relation", ["unknown"] * len(labels))

            emb1, emb2, _ = model(img1, img2)
            predictions = (F.cosine_similarity(emb1, emb2, dim=1) + 1) / 2

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_relations.extend(relations)

    predictions = np.array(all_preds)
    labels = np.array(all_labels)

    threshold = get_checkpoint_threshold(checkpoint, default=0.5)
    print(f"Using stored validation threshold: {threshold:.3f}")

    # Metrics
    metrics = KinshipMetrics(threshold=threshold)
    metrics.all_predictions = list(predictions)
    metrics.all_labels = list(labels)
    metrics.all_relations = all_relations
    results = metrics.compute()

    print_metrics(results, prefix="Test ")

    # Save basic metrics
    with open(output_dir / "metrics_rocm.json", "w") as f:
        serializable = {k: v for k, v in results.items() if isinstance(v, (int, float, str))}
        serializable["platform"] = "AMD ROCm"
        serializable["threshold"] = threshold
        json.dump(serializable, f, indent=2)

    if args.full_analysis or args.analyze_features:
        print("\nAnalyzing backbone contributions...")
        backbone_results = analyze_backbone_contributions(model, test_loader, device, output_dir)

        with open(output_dir / "backbone_analysis.json", "w") as f:
            # Filter non-serializable items
            serializable = {k: v for k, v in backbone_results.items()
                          if isinstance(v, (int, float, str, dict))}
            json.dump(serializable, f, indent=2)

        print(f"\nBackbone Separability:")
        print(f"  ConvNeXt: {backbone_results['convnext_separability']:.4f}")
        print(f"  ViT:      {backbone_results['vit_separability']:.4f}")
        print(f"  Fused:    {backbone_results['fused_separability']:.4f}")

    if args.full_analysis:
        print("\nGenerating ROC curve...")
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - ConvNeXt-ViT Hybrid (AMD ROCm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve_rocm.png", dpi=150)
        plt.close()

        print("Generating confusion matrix...")
        binary_preds = (predictions > threshold).astype(int)
        cm = confusion_matrix(labels, binary_preds)
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Kin', 'Kin'])
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix - ConvNeXt-ViT Hybrid (AMD ROCm)')
        plt.savefig(output_dir / "confusion_matrix_rocm.png", dpi=150)
        plt.close()

    if args.ablation:
        print("\nRunning fusion ablation study...")
        ablation_results = run_fusion_ablation(model, test_loader, device, output_dir, threshold)
        with open(output_dir / "fusion_ablation.json", "w") as f:
            json.dump(ablation_results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY (AMD ROCm)")
    print("=" * 50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"ROC-AUC:   {results.get('roc_auc', 0):.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
