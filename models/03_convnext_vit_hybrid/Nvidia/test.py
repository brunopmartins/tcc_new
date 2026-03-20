#!/usr/bin/env python3
"""
Testing script for ConvNeXt + ViT Hybrid model.

Usage:
    python test.py --checkpoint checkpoints/best.pt
    python test.py --checkpoint checkpoints/best.pt --analyze_features
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import DataConfig
from dataset import KinshipPairDataset, get_transforms
from evaluation import KinshipMetrics, print_metrics
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root
from torch.utils.data import DataLoader

from model import ConvNeXtViTHybrid


def parse_args():
    parser = argparse.ArgumentParser(description="Test ConvNeXt-ViT Hybrid")
    
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--analyze_features", action="store_true",
                        help="Analyze contribution of each backbone")
    
    return parser.parse_args()


def analyze_feature_contributions(model, dataloader, device, output_dir):
    """Analyze how ConvNeXt and ViT features contribute to predictions."""
    model.eval()
    
    conv_similarities = []
    vit_similarities = []
    fused_similarities = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing features"):
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            labels = batch["label"]
            
            emb1, emb2, aux = model(img1, img2)
            
            # Compute similarities at different levels
            conv_sim = F.cosine_similarity(aux["conv1"], aux["conv2"], dim=1)
            vit_sim = F.cosine_similarity(aux["vit1"], aux["vit2"], dim=1)
            fused_sim = F.cosine_similarity(emb1, emb2, dim=1)
            
            conv_similarities.extend(conv_sim.cpu().numpy())
            vit_similarities.extend(vit_sim.cpu().numpy())
            fused_similarities.extend(fused_sim.cpu().numpy())
            labels_list.extend(labels.numpy())
    
    conv_similarities = np.array(conv_similarities)
    vit_similarities = np.array(vit_similarities)
    fused_similarities = np.array(fused_similarities)
    labels_arr = np.array(labels_list)
    
    # Separate positive and negative pairs
    pos_mask = labels_arr == 1
    neg_mask = labels_arr == 0
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distribution comparison
    ax = axes[0, 0]
    ax.hist(conv_similarities[pos_mask], bins=30, alpha=0.5, label='ConvNeXt (Kin)', color='blue')
    ax.hist(conv_similarities[neg_mask], bins=30, alpha=0.5, label='ConvNeXt (Non-Kin)', color='red')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('ConvNeXt Feature Similarities')
    ax.legend()
    
    ax = axes[0, 1]
    ax.hist(vit_similarities[pos_mask], bins=30, alpha=0.5, label='ViT (Kin)', color='blue')
    ax.hist(vit_similarities[neg_mask], bins=30, alpha=0.5, label='ViT (Non-Kin)', color='red')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('ViT Feature Similarities')
    ax.legend()
    
    # 2. Scatter: ConvNeXt vs ViT
    ax = axes[1, 0]
    ax.scatter(conv_similarities[pos_mask], vit_similarities[pos_mask], 
               alpha=0.5, label='Kin', c='blue', s=20)
    ax.scatter(conv_similarities[neg_mask], vit_similarities[neg_mask], 
               alpha=0.5, label='Non-Kin', c='red', s=20)
    ax.set_xlabel('ConvNeXt Similarity')
    ax.set_ylabel('ViT Similarity')
    ax.set_title('ConvNeXt vs ViT Similarities')
    ax.legend()
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # 3. Fused distribution
    ax = axes[1, 1]
    ax.hist(fused_similarities[pos_mask], bins=30, alpha=0.5, label='Fused (Kin)', color='blue')
    ax.hist(fused_similarities[neg_mask], bins=30, alpha=0.5, label='Fused (Non-Kin)', color='red')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Fused Feature Similarities')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_analysis.png", dpi=150)
    plt.close()
    
    # Compute and print statistics
    print("\n" + "="*50)
    print("FEATURE CONTRIBUTION ANALYSIS")
    print("="*50)
    
    def compute_separability(pos_vals, neg_vals):
        """Compute separability as difference in means / pooled std."""
        pooled_std = np.sqrt((np.var(pos_vals) + np.var(neg_vals)) / 2)
        return (np.mean(pos_vals) - np.mean(neg_vals)) / (pooled_std + 1e-8)
    
    conv_sep = compute_separability(conv_similarities[pos_mask], conv_similarities[neg_mask])
    vit_sep = compute_separability(vit_similarities[pos_mask], vit_similarities[neg_mask])
    fused_sep = compute_separability(fused_similarities[pos_mask], fused_similarities[neg_mask])
    
    print(f"\nSeparability (higher is better):")
    print(f"  ConvNeXt: {conv_sep:.4f}")
    print(f"  ViT:      {vit_sep:.4f}")
    print(f"  Fused:    {fused_sep:.4f}")
    
    print(f"\nMean Similarities:")
    print(f"  ConvNeXt - Kin: {conv_similarities[pos_mask].mean():.4f}, Non-Kin: {conv_similarities[neg_mask].mean():.4f}")
    print(f"  ViT      - Kin: {vit_similarities[pos_mask].mean():.4f}, Non-Kin: {vit_similarities[neg_mask].mean():.4f}")
    print(f"  Fused    - Kin: {fused_similarities[pos_mask].mean():.4f}, Non-Kin: {fused_similarities[neg_mask].mean():.4f}")
    
    return {
        "conv_separability": conv_sep,
        "vit_separability": vit_sep,
        "fused_separability": fused_sep,
    }


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
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
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    # Collect predictions
    all_preds, all_labels, all_relations = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            labels = batch["label"]
            relations = batch.get("relation", ["unknown"] * len(labels))
            
            emb1, emb2, _ = model(img1, img2)
            predictions = (F.cosine_similarity(emb1, emb2, dim=1) + 1) / 2
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_relations.extend(relations)
    
    predictions = np.array(all_preds)
    labels = np.array(all_labels)
    
    if args.threshold is None:
        threshold = get_checkpoint_threshold(checkpoint, default=0.5)
        print(f"Using stored validation threshold: {threshold:.3f}")
    else:
        threshold = args.threshold
    
    # Compute metrics
    metrics = KinshipMetrics(threshold=threshold)
    metrics.all_predictions = list(predictions)
    metrics.all_labels = list(labels)
    metrics.all_relations = all_relations
    results = metrics.compute()
    
    print_metrics(results, prefix="Test ")
    
    # Analyze features
    if args.analyze_features:
        print("\nAnalyzing feature contributions...")
        analyze_feature_contributions(model, test_loader, device, output_dir)
    
    # Save results
    with open(output_dir / "test_metrics.txt", "w") as f:
        for k, v in results.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
