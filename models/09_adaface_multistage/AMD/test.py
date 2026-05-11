#!/usr/bin/env python3
"""
AMD ROCm test script for Model 09 — AdaFace + Multi-Stage Cross-Attention.

Same surface as M02/M10 test scripts. Image size + normalisation come from
the checkpoint metadata; default to AdaFace 112×112 + [-1, 1].
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ["MIOPEN_FIND_MODE"] = "FAST"

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "AMD"))

from rocm_utils import (  # noqa: E402
    setup_rocm_environment,
    get_rocm_device,
    clear_rocm_cache,
)
from config import DataConfig  # noqa: E402
from dataset import KinshipPairDataset, get_transforms  # noqa: E402
from evaluation import KinshipMetrics, print_metrics  # noqa: E402
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import build_adaface_multistage_model, parse_model_outputs  # noqa: E402


ADAFACE_MEAN = [0.5, 0.5, 0.5]
ADAFACE_STD = [0.5, 0.5, 0.5]
ADAFACE_IMG_SIZE = 112


def parse_args():
    parser = argparse.ArgumentParser(description="Test Model 09 (AMD ROCm)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="fiw",
                        choices=["kinface", "fiw"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--visualize_attention", action="store_true")
    parser.add_argument("--num_visualizations", type=int, default=10)
    parser.add_argument("--rocm_device", type=int, default=0)
    parser.add_argument("--aligned_root", type=str, default=None)
    return parser.parse_args()


def visualize_attention(
    img1: torch.Tensor,
    img2: torch.Tensor,
    attn_map: torch.Tensor,
    save_path: str,
    label: int,
    prediction: float,
):
    """Visualise the 7×7 cross-attention. Inputs are AdaFace-normalised [-1, 1]."""
    img1 = (img1.cpu() + 1) / 2  # [-1, 1] -> [0, 1]
    img2 = (img2.cpu() + 1) / 2
    img1 = img1.permute(1, 2, 0).numpy().clip(0, 1)
    img2 = img2.permute(1, 2, 0).numpy().clip(0, 1)

    attn = attn_map.cpu().mean(dim=0).numpy()  # average heads
    patch_size = int(np.sqrt(attn.shape[0]))   # 7
    attn_spatial = attn.mean(axis=1).reshape(patch_size, patch_size)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img1)
    axes[0].set_title("Face 1")
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title("Face 2")
    axes[1].axis("off")

    axes[2].imshow(attn_spatial, cmap='hot', interpolation='nearest')
    axes[2].set_title("Cross-Attention (7×7)")
    axes[2].axis("off")

    axes[3].imshow(img1)
    scale = img1.shape[0] // patch_size
    attn_resized = np.interp(
        attn_spatial,
        (attn_spatial.min(), attn_spatial.max() + 1e-9),
        (0, 1),
    )
    attn_resized = np.kron(attn_resized, np.ones((scale, scale)))
    attn_resized = attn_resized[: img1.shape[0], : img1.shape[1]]
    axes[3].imshow(attn_resized, cmap='jet', alpha=0.5)
    axes[3].set_title("Attention Overlay")
    axes[3].axis("off")

    kin_str = "Kin" if label == 1 else "Non-Kin"
    pred_str = "Kin" if prediction > 0.5 else "Non-Kin"
    correct = "V" if (label == 1) == (prediction > 0.5) else "X"
    plt.suptitle(f"GT: {kin_str} | Pred: {pred_str} ({prediction:.3f}) {correct}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("AMD ROCm Testing — Model 09: AdaFace + Multi-Stage Cross-Attention")
    print("=" * 60)

    setup_rocm_environment(visible_devices=str(args.rocm_device))
    device = get_rocm_device(args.rocm_device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config", checkpoint.get("protocol", {}).get("model_config", {}))
    img_size = int(model_config.get("img_size", ADAFACE_IMG_SIZE))

    # Build model — skip AdaFace pretrained load (we'll load checkpoint state below)
    model = build_adaface_multistage_model(
        adaface_weights=None,  # ckpt overrides this
        embedding_dim=model_config.get("embedding_dim", 512),
        cross_attn_stages=model_config.get("cross_attn_stages", [3, 4]),
        num_cross_attn_layers_per_stage=model_config.get("cross_attn_layers_per_stage", 1),
        cross_attn_heads=model_config.get("cross_attn_heads", 8),
        dropout=model_config.get("dropout", 0.2),
        freeze_backbone=model_config.get("freeze_backbone", False),
        use_positional_embedding=model_config.get("use_positional_embedding", True),
        use_global_embedding=model_config.get("use_global_embedding", True),
        use_classifier_head=model_config.get("use_classifier_head", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Dataset — AdaFace normalisation
    data_config = DataConfig(
        image_size=img_size,
        normalize_mean=ADAFACE_MEAN,
        normalize_std=ADAFACE_STD,
    )
    apply_data_root_override(data_config, args.dataset, args.data_root)
    root_dir = resolve_dataset_root(data_config, args.dataset)

    test_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=args.dataset,
        split="test",
        transform=get_transforms(data_config, train=False),
        split_seed=checkpoint.get("protocol", {}).get("split_seed", data_config.split_seed),
        negative_ratio=checkpoint.get("protocol", {}).get("negative_ratio", data_config.negative_ratio),
        aligned_root=args.aligned_root,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    clear_rocm_cache()

    all_preds, all_labels, all_relations = [], [], []
    attention_samples = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing (ROCm)"):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"]
            relations = batch.get("relation", ["unknown"] * len(labels))

            parsed = parse_model_outputs(model(img1, img2))
            predictions = parsed["scores"]
            attn_map = parsed["attn_map"]

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_relations.extend(relations)

            if args.visualize_attention and len(attention_samples) < args.num_visualizations:
                for i in range(min(len(img1), args.num_visualizations - len(attention_samples))):
                    attention_samples.append({
                        "img1": img1[i],
                        "img2": img2[i],
                        "attn": attn_map[i] if attn_map is not None else torch.zeros(1, 1, 1, device=device),
                        "label": labels[i].item(),
                        "pred": predictions[i].item(),
                    })

    predictions = np.array(all_preds)
    labels = np.array(all_labels)

    if args.threshold is None:
        threshold = get_checkpoint_threshold(checkpoint, default=0.5)
        print(f"Using stored validation threshold: {threshold:.3f}")
    else:
        threshold = args.threshold

    metrics = KinshipMetrics(threshold=threshold)
    metrics.all_predictions = list(predictions)
    metrics.all_labels = list(labels)
    metrics.all_relations = all_relations
    results = metrics.compute()

    print_metrics(results, prefix="Test ")

    if args.visualize_attention and attention_samples:
        attn_dir = output_dir / "attention_maps_rocm"
        attn_dir.mkdir(exist_ok=True)
        print(f"\nGenerating {len(attention_samples)} attention visualisations...")
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

    with open(output_dir / "test_metrics_rocm.txt", "w") as f:
        f.write("AMD ROCm Test Results — Model 09 (AdaFace + Multi-Stage Cross-Attention)\n")
        f.write("=" * 50 + "\n")
        for k, v in results.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
