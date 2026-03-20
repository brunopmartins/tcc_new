#!/usr/bin/env python3
"""
NVIDIA CUDA Comprehensive evaluation script for ViT + FaCoR Cross-Attention model.
Performs detailed analysis including attention visualization and cross-attention analysis.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --full_analysis
    python evaluate.py --checkpoint checkpoints/best.pt --visualize_attention
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
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
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
from evaluation import KinshipMetrics, print_metrics
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root
from torch.utils.data import DataLoader

# Add parent directory for model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import ViTFaCoRModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViT-FaCoR Model (NVIDIA CUDA)")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--full_analysis", action="store_true")
    parser.add_argument("--visualize_attention", action="store_true")
    parser.add_argument("--num_visualizations", type=int, default=10)

    # CUDA specific
    parser.add_argument("--cuda_device", type=int, default=0)

    return parser.parse_args()


def visualize_cross_attention(model, dataloader, device, output_dir, num_samples=10):
    """Visualize cross-attention maps between face pairs."""
    model.eval()
    output_dir = Path(output_dir) / "attention_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break

            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]
            relations = batch.get("relation", ["unknown"] * len(labels))

            emb1, emb2, attn_maps = model(img1, img2)

            # Process each sample in batch
            for i in range(min(len(labels), num_samples - sample_count)):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Original images (denormalize)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

                img1_vis = img1[i].cpu() * std + mean
                img2_vis = img2[i].cpu() * std + mean

                axes[0, 0].imshow(img1_vis.permute(1, 2, 0).clamp(0, 1))
                axes[0, 0].set_title("Person 1")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(img2_vis.permute(1, 2, 0).clamp(0, 1))
                axes[0, 1].set_title("Person 2")
                axes[0, 1].axis("off")

                # Attention map visualization
                if attn_maps is not None and len(attn_maps.shape) >= 3:
                    attn = attn_maps[i].cpu().numpy()
                    if len(attn.shape) == 3:
                        attn = attn.mean(axis=0)  # Average over heads

                    im = axes[0, 2].imshow(attn, cmap='hot', aspect='auto')
                    axes[0, 2].set_title("Cross-Attention Map")
                    plt.colorbar(im, ax=axes[0, 2])

                # Similarity and prediction
                similarity = F.cosine_similarity(emb1[i:i+1], emb2[i:i+1]).item()
                label = labels[i].item()
                relation = relations[i]

                axes[1, 0].bar(["Similarity"], [similarity], color='steelblue')
                axes[1, 0].set_ylim(-1, 1)
                axes[1, 0].set_title(f"Cosine Similarity: {similarity:.3f}")
                axes[1, 0].axhline(y=0.5, color='r', linestyle='--', label='Threshold')

                # Embedding visualization (t-SNE style projection)
                emb_concat = torch.cat([emb1[i], emb2[i]]).cpu().numpy()
                axes[1, 1].bar(range(0, len(emb_concat), 10), emb_concat[::10])
                axes[1, 1].set_title("Embedding Sample (every 10th dim)")

                # Info text
                info_text = f"Label: {'Kin' if label == 1 else 'Non-Kin'}\n"
                info_text += f"Relation: {relation}\n"
                info_text += f"Prediction: {'Kin' if similarity > 0.5 else 'Non-Kin'}\n"
                info_text += f"Correct: {'Yes' if (similarity > 0.5) == label else 'No'}"
                axes[1, 2].text(0.5, 0.5, info_text, fontsize=14,
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].axis("off")

                plt.suptitle(f"ViT-FaCoR Cross-Attention Analysis (NVIDIA CUDA)")
                plt.tight_layout()
                plt.savefig(output_dir / f"attention_sample_{sample_count}.png", dpi=150)
                plt.close()

                sample_count += 1

    print(f"Saved {sample_count} attention visualizations to {output_dir}")


def analyze_attention_patterns(model, dataloader, device, output_dir):
    """Analyze attention patterns for kin vs non-kin pairs."""
    model.eval()

    kin_attentions = []
    nonkin_attentions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing attention (CUDA)"):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"].numpy()

            _, _, attn_maps = model(img1, img2)

            if attn_maps is not None:
                attn_flat = attn_maps.cpu().numpy()
                if len(attn_flat.shape) > 2:
                    attn_flat = attn_flat.mean(axis=tuple(range(1, len(attn_flat.shape))))

                for i, label in enumerate(labels):
                    if label == 1:
                        kin_attentions.append(attn_flat[i] if len(attn_flat.shape) > 1 else attn_flat)
                    else:
                        nonkin_attentions.append(attn_flat[i] if len(attn_flat.shape) > 1 else attn_flat)

    if kin_attentions and nonkin_attentions:
        kin_mean = np.mean(kin_attentions)
        nonkin_mean = np.mean(nonkin_attentions)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(["Kin Pairs", "Non-Kin Pairs"], [kin_mean, nonkin_mean],
               color=['forestgreen', 'coral'])
        ax.set_ylabel("Mean Attention Intensity")
        ax.set_title("Attention Intensity: Kin vs Non-Kin (NVIDIA CUDA)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "attention_intensity_comparison.png", dpi=150)
        plt.close()

        return {"kin_mean_attention": kin_mean, "nonkin_mean_attention": nonkin_mean}

    return {}


def plot_roc_curve(predictions, labels, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(labels, predictions)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ViT-FaCoR Model (NVIDIA CUDA)')
    plt.grid(True, alpha=0.3)

    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    plt.annotate(f'AUC = {roc_auc:.4f}', xy=(0.6, 0.2), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("NVIDIA CUDA Evaluation - ViT-FaCoR Cross-Attention Model")
    print("=" * 60)

    setup_cuda_environment(visible_devices=str(args.cuda_device))
    device = get_cuda_device(args.cuda_device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_config = checkpoint.get("model_config", checkpoint.get("protocol", {}).get("model_config", {}))
    model = ViTFaCoRModel(
        vit_model=model_config.get("vit_model", "vit_base_patch16_224"),
        pretrained=True,
        embedding_dim=model_config.get("embedding_dim", 512),
        num_cross_attn_layers=model_config.get("cross_attn_layers", 2),
        cross_attn_heads=model_config.get("cross_attn_heads", 8),
        freeze_vit=model_config.get("freeze_vit", False),
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

    # Clear cache
    clear_cuda_cache()

    # Collect predictions
    all_preds, all_labels, all_relations = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating (CUDA)"):
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

    # Compute metrics
    metrics = KinshipMetrics(threshold=threshold)
    metrics.all_predictions = list(predictions)
    metrics.all_labels = list(labels)
    metrics.all_relations = all_relations
    results = metrics.compute()

    print_metrics(results, prefix="Test ")

    # Save metrics
    with open(output_dir / "metrics_cuda.json", "w") as f:
        serializable = {k: v for k, v in results.items() if isinstance(v, (int, float, str))}
        serializable["platform"] = "NVIDIA CUDA"
        serializable["threshold"] = threshold
        json.dump(serializable, f, indent=2)

    if args.full_analysis:
        print("\nRunning full analysis...")

        print("Generating ROC curve...")
        plot_roc_curve(predictions, labels, output_dir / "roc_curve_cuda.png")

        print("Generating confusion matrix...")
        binary_preds = (predictions > threshold).astype(int)
        cm = confusion_matrix(labels, binary_preds)
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Kin', 'Kin'])
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix - ViT-FaCoR (NVIDIA CUDA)')
        plt.savefig(output_dir / "confusion_matrix_cuda.png", dpi=150)
        plt.close()

        print("Analyzing attention patterns...")
        attn_results = analyze_attention_patterns(model, test_loader, device, output_dir)
        if attn_results:
            with open(output_dir / "attention_analysis.json", "w") as f:
                json.dump(attn_results, f, indent=2)

    if args.visualize_attention:
        print(f"\nVisualizing attention maps ({args.num_visualizations} samples)...")
        visualize_cross_attention(model, test_loader, device, output_dir, args.num_visualizations)

    print(f"\nResults saved to {output_dir}")

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY (NVIDIA CUDA)")
    print("=" * 50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"ROC-AUC:   {results.get('roc_auc', 0):.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
