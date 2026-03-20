#!/usr/bin/env python3
"""
Testing script for Age Synthesis + All-vs-All Comparison model.

Usage:
    python test.py --checkpoint checkpoints/best.pt --dataset kinface
    python test.py --checkpoint checkpoints/best.pt --dataset fiw --save_predictions
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import DataConfig
from dataset import KinshipPairDataset, get_transforms
from evaluation import KinshipMetrics, print_metrics
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root
from torch.utils.data import DataLoader

from model import AgeSynthesisComparisonModel


def parse_args():
    parser = argparse.ArgumentParser(description="Test Age Synthesis Model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="kinface",
                        choices=["kinface", "fiw"],
                        help="Dataset to test on")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override default dataset path")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Classification threshold (auto if not specified)")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save predictions to CSV")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get("model_config", checkpoint.get("protocol", {}).get("model_config", {}))
    model = AgeSynthesisComparisonModel(
        backbone=model_config.get("backbone", "resnet50"),
        embedding_dim=model_config.get("embedding_dim", 512),
        use_age_synthesis=model_config.get("use_age_synthesis", False),
        aggregation=model_config.get("aggregation", "attention"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def test_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple:
    """
    Test model and return metrics + predictions.
    """
    model.eval()
    metrics = KinshipMetrics(threshold=threshold)
    
    all_predictions = []
    all_labels = []
    all_relations = []
    all_img_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            labels = batch["label"]
            relations = batch.get("relation", ["unknown"] * len(labels))
            
            # Forward pass
            kinship_score, comparison_matrix = model(img1, img2)
            
            # Apply sigmoid
            predictions = torch.sigmoid(kinship_score).cpu()
            
            # Store predictions
            all_predictions.extend(predictions.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            all_relations.extend(relations)
            
            # Update metrics
            metrics.update(predictions, labels, relations)
    
    results = metrics.compute()
    
    return results, {
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "relations": all_relations,
    }


def main():
    args = parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, device)
    train_metrics = checkpoint.get("metrics", {})
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if train_metrics:
        print(f"Training metrics: Acc={train_metrics.get('accuracy', 'N/A')}")
    
    # Dataset config
    data_config = DataConfig()
    apply_data_root_override(data_config, args.dataset, args.data_root)
    
    # Create test dataset
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
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    if args.threshold is None:
        threshold = get_checkpoint_threshold(checkpoint, default=0.5)
        print(f"Using stored validation threshold: {threshold:.3f}")
    else:
        threshold = args.threshold
    
    # Final test with optimal threshold
    print(f"\nTesting with threshold={threshold:.3f}")
    results, pred_data = test_model(model, test_loader, device, threshold=threshold)
    
    # Print results
    print_metrics(results, prefix="Test ")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = output_dir / "test_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Threshold source: {'manual' if args.threshold is not None else 'checkpoint protocol'}\n")
        f.write(f"Samples: {len(test_dataset)}\n")
        f.write("-" * 40 + "\n")
        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
    
    print(f"\nMetrics saved to {metrics_path}")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_df = pd.DataFrame({
            "prediction": pred_data["predictions"],
            "label": pred_data["labels"],
            "relation": pred_data["relations"],
            "predicted_class": (pred_data["predictions"] > threshold).astype(int),
            "correct": ((pred_data["predictions"] > threshold) == pred_data["labels"]).astype(int),
        })
        
        predictions_path = output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")
    
    return results


if __name__ == "__main__":
    main()
