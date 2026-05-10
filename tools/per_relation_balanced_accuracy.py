#!/usr/bin/env python3
"""
Per-relation balanced accuracy (FaCoRNet-style metric).

Given a trained kinship checkpoint, compute the accuracy metric used by
FaCoRNet (Su et al. 2023, arXiv:2304.04546) and the RFIW Track-I protocol:

  For each of the 11 FIW kin relations r:
      pos_acc_r = accuracy on positive pairs labeled with relation r
      bal_acc_r = (pos_acc_r + nonkin_acc) / 2
  Average across the 11 relations to get the headline number.

This is the metric the literature reports (e.g. FaCoRNet's 82.0% on FIW).
Our previous `Test Accuracy` was overall binary acc with a single global
threshold, not directly comparable to literature numbers.

Negatives are pooled (`relation == "non-kin"` in our dataset) — we use the
same negative pool for every relation's balanced calculation.

Usage:
    python tools/per_relation_balanced_accuracy.py \\
        --model 05 \\
        --checkpoint models/05_dinov2_lora_diffattn/output/001/checkpoints/best.pt \\
        --output_dir models/05_dinov2_lora_diffattn/output/001/results

Compatible with M02, M03, M05, M06 checkpoints (auto-detects architecture).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


FIW_RELATIONS = ("bb", "ss", "sibs", "fd", "fs", "md", "ms", "gfgd", "gfgs", "gmgd", "gmgs")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["02", "03", "05", "06"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="fiw", choices=["fiw", "kinface"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--rocm_device", type=int, default=0)
    p.add_argument("--output_dir", required=True,
                   help="Where to save the JSON output. Filename: per_relation_balanced_accuracy.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override global threshold (default: select on val by F1).")
    return p.parse_args()


def load_model(model_id: str, checkpoint_path: str, device):
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "models"))
    sys.path.insert(0, str(project_root / "models" / "shared"))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    cfg = ckpt.get("model_config", {})

    if model_id == "05":
        sys.path.insert(0, str(project_root / "models" / "05_dinov2_lora_diffattn"))
        from model import DINOv2LoRAKinship, DINOv2HybridKinship  # type: ignore

        is_hybrid = any(k.startswith("face_vit.") or k.startswith("intra_face_layers.")
                        for k in state.keys())
        if is_hybrid:
            n_intra = sum(1 for k in state.keys()
                          if k.startswith("intra_face_layers.") and k.endswith(".attn.q_proj.weight"))
            model = DINOv2HybridKinship(
                dinov2_name=cfg.get("backbone_name", "vit_base_patch14_dinov2.lvd142m"),
                face_backbone_name=cfg.get("face_backbone_name", "vit_base_patch16_224"),
                face_backbone_checkpoint=None,
                face_backbone_state_prefix=cfg.get("face_backbone_state_prefix", "vit."),
                img_size=cfg.get("img_size", 224),
                embedding_dim=cfg.get("embedding_dim", 512),
                intra_face_attn_layers=cfg.get("intra_face_attn_layers", n_intra or 1),
                cross_attn_layers=cfg.get("cross_attn_layers", 2),
                cross_attn_heads=cfg.get("cross_attn_heads", 8),
                dropout=cfg.get("dropout", 0.1),
                relation_set=cfg.get("relation_set", "fiw"),
                relation_loss_weight=cfg.get("relation_loss_weight", 0.2),
                backbone_pretrained=True,
                use_gradient_checkpointing=False,
            )
        else:
            model = DINOv2LoRAKinship(
                backbone_name=cfg.get("backbone_name", "vit_base_patch14_dinov2.lvd142m"),
                img_size=cfg.get("img_size", 224),
                lora_rank=cfg.get("lora_rank", 8),
                lora_alpha=cfg.get("lora_alpha", 16),
                lora_dropout=cfg.get("lora_dropout", 0.0),
                backbone_pretrained=True,
                use_gradient_checkpointing=False,
                embedding_dim=cfg.get("embedding_dim", 512),
                cross_attn_layers=cfg.get("cross_attn_layers", 2),
                cross_attn_heads=cfg.get("cross_attn_heads", 8),
                dropout=cfg.get("dropout", 0.1),
                relation_set=cfg.get("relation_set", "fiw"),
                relation_loss_weight=cfg.get("relation_loss_weight", 0.2),
            )
    elif model_id == "06":
        sys.path.insert(0, str(project_root / "models" / "06_retrieval_augmented_kinship"))
        from model import RetrievalAugmentedKinship  # type: ignore
        model = RetrievalAugmentedKinship(**cfg) if cfg else RetrievalAugmentedKinship()
    elif model_id == "02":
        sys.path.insert(0, str(project_root / "models" / "02_vit_facor_crossattn"))
        from model import ViTFaCoRModel  # type: ignore
        model = ViTFaCoRModel(**cfg) if cfg else ViTFaCoRModel()
    elif model_id == "03":
        sys.path.insert(0, str(project_root / "models" / "03_convnext_vit_hybrid"))
        from model import ConvNeXtViTHybrid  # type: ignore
        model = ConvNeXtViTHybrid(**cfg) if cfg else ConvNeXtViTHybrid()
    else:
        raise ValueError(f"unknown model {model_id}")

    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def collect_scores(model, loader, device):
    scores, labels, relations = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            lab = batch["label"].cpu().numpy()
            rel = batch.get("relation", ["unknown"] * len(lab))
            if isinstance(rel, torch.Tensor):
                rel = rel.cpu().tolist()

            output = model(img1, img2)
            if isinstance(output, dict):
                if "logits" in output:
                    s = torch.sigmoid(output["logits"].squeeze(-1))
                else:
                    emb1 = output.get("emb1", output.get("embedding1"))
                    emb2 = output.get("emb2", output.get("embedding2"))
                    s = (F.cosine_similarity(emb1, emb2, dim=-1) + 1) / 2
            elif isinstance(output, (tuple, list)):
                emb1, emb2 = output[0], output[1]
                s = (F.cosine_similarity(emb1, emb2, dim=-1) + 1) / 2
            else:
                s = torch.sigmoid(output.squeeze(-1))

            scores.extend(s.cpu().numpy().tolist())
            labels.extend(lab.tolist())
            relations.extend(list(rel))

            if i % 100 == 0:
                print(f"    batch {i}/{len(loader)}", flush=True)

    return np.array(scores), np.array(labels, dtype=int), np.array(relations)


def fit_threshold_f1(scores, labels, grid=None):
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.01)
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        pred = (scores >= thr).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.rocm_device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model {args.model} from {args.checkpoint}...")
    model = load_model(args.model, args.checkpoint, device)

    sys.path.insert(0, str(project_root / "models" / "shared"))
    from dataset import KinshipPairDataset, get_transforms  # type: ignore
    from config import DataConfig  # type: ignore
    from protocol import resolve_dataset_root  # type: ignore

    cfg = DataConfig(split_seed=args.seed)
    data_root = resolve_dataset_root(cfg, args.dataset)

    val_ds = KinshipPairDataset(
        root_dir=data_root, dataset_type=args.dataset, split="val",
        transform=get_transforms(cfg, train=False),
        split_seed=args.seed, negative_ratio=cfg.negative_ratio,
    )
    test_ds = KinshipPairDataset(
        root_dir=data_root, dataset_type=args.dataset, split="test",
        transform=get_transforms(cfg, train=False),
        split_seed=args.seed, negative_ratio=cfg.negative_ratio,
    )
    print(f"  val:  {len(val_ds)} pairs")
    print(f"  test: {len(test_ds)} pairs")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print("\nCollecting validation scores...")
    val_scores, val_labels, val_rels = collect_scores(model, val_loader, device)
    print(f"  collected {len(val_scores)} val pairs")

    print("\nCollecting test scores...")
    test_scores, test_labels, test_rels = collect_scores(model, test_loader, device)
    print(f"  collected {len(test_scores)} test pairs")

    # Pick threshold on val (F1-maximized) — same global threshold used per relation.
    if args.threshold is not None:
        thr = float(args.threshold)
        print(f"\n[Threshold] forced to {thr}")
    else:
        thr, val_f1 = fit_threshold_f1(val_scores, val_labels)
        print(f"\n[Threshold] val-selected by F1: {thr:.3f}  (val F1={val_f1:.4f})")

    test_pred = (test_scores >= thr).astype(int)

    # Overall metrics (for reference vs old methodology)
    overall_acc = accuracy_score(test_labels, test_pred)
    overall_bal_acc = balanced_accuracy_score(test_labels, test_pred)
    overall_f1 = f1_score(test_labels, test_pred, zero_division=0)
    print(f"\n[Overall (binary, single threshold)]")
    print(f"  accuracy:         {overall_acc:.4f}")
    print(f"  balanced acc:     {overall_bal_acc:.4f}")
    print(f"  f1:               {overall_f1:.4f}")

    # FaCoRNet-style per-relation balanced accuracy
    # For each relation r:
    #   pos_acc_r = acc on positive pairs labeled r
    #   bal_acc_r = (pos_acc_r + nonkin_acc) / 2
    # Mean of 11 relations = headline number

    # Get non-kin set (negatives, label=0)
    nonkin_idx = np.where(test_labels == 0)[0]
    nonkin_pred = test_pred[nonkin_idx]
    nonkin_lab = test_labels[nonkin_idx]
    nonkin_acc = float(accuracy_score(nonkin_lab, nonkin_pred)) if len(nonkin_idx) else 0.0

    per_relation = {}
    bal_accs = []
    for r in FIW_RELATIONS:
        pos_idx = np.where((test_labels == 1) & (test_rels == r))[0]
        if len(pos_idx) == 0:
            continue
        pos_pred = test_pred[pos_idx]
        pos_lab = test_labels[pos_idx]
        pos_acc = float(accuracy_score(pos_lab, pos_pred))
        bal_acc = (pos_acc + nonkin_acc) / 2.0
        per_relation[r] = {
            "n_pos": int(len(pos_idx)),
            "pos_acc": pos_acc,
            "balanced_acc": bal_acc,
        }
        bal_accs.append(bal_acc)

    mean_balanced = float(np.mean(bal_accs)) if bal_accs else 0.0
    mean_pos_acc = float(np.mean([per_relation[r]["pos_acc"] for r in per_relation]))

    print(f"\n[Per-relation balanced accuracy (FaCoRNet methodology)]")
    print(f"  non-kin accuracy: {nonkin_acc:.4f}  (n={len(nonkin_idx)})")
    print(f"  {'Relation':<8s} {'N':>6s} {'PosAcc':>8s} {'BalAcc':>8s}")
    for r in FIW_RELATIONS:
        if r not in per_relation:
            continue
        d = per_relation[r]
        print(f"  {r:<8s} {d['n_pos']:>6d} {d['pos_acc']:>8.4f} {d['balanced_acc']:>8.4f}")
    print(f"\n  Mean positive accuracy:       {mean_pos_acc:.4f}")
    print(f"  Mean balanced accuracy (SOTA-comparable): {mean_balanced:.4f}  ← FaCoRNet protocol")
    print(f"  Reference: FaCoRNet AdaFace = 0.8200 on FIW Track-I\n")

    output = {
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "dataset": args.dataset,
        "threshold": float(thr),
        "overall": {
            "accuracy": overall_acc,
            "balanced_accuracy": overall_bal_acc,
            "f1": overall_f1,
        },
        "nonkin": {
            "accuracy": nonkin_acc,
            "n": int(len(nonkin_idx)),
        },
        "per_relation": per_relation,
        "headline": {
            "mean_pos_acc": mean_pos_acc,
            "mean_balanced_acc": mean_balanced,
        },
    }

    out_file = output_dir / "per_relation_balanced_accuracy.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
