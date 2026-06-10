#!/usr/bin/env python3
"""Diagnostic: where does M12's decision signal come from?

Runs the M12 checkpoint over the fixed RFIW Track-I test set and decomposes
the signal:
  - AUC of the final classifier prob (the model)
  - AUC of the contextualised GLOBAL cosine alone (gA_norm . gB_norm)
  - mean per-region gate weights (are the anatomical regions used at all?)

If AUC(global cosine) ~= AUC(model), the regional + classifier apparatus adds
little over a plain global AdaFace comparison, and the ceiling is the backbone.

Read-only; writes nothing except a printed report.
"""
import os, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
for p in ["models/shared", "models/shared/AMD", "models/12_rgck_net"]:
    sys.path.insert(0, str(ROOT / p))

from rocm_utils import setup_rocm_environment  # noqa
from config import DataConfig  # noqa
from dataset import KinshipPairDataset, get_transforms  # noqa
from protocol import apply_data_root_override, get_checkpoint_threshold, resolve_dataset_root  # noqa
from model import build_rgck_net  # noqa

CKPT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models/12_rgck_net/output/016/checkpoints/best.pt")
setup_rocm_environment(visible_devices="0")
dev = torch.device("cuda:0")

ck = torch.load(CKPT, map_location=dev, weights_only=False)
mc = ck.get("model_config", {})
model = build_rgck_net(
    adaface_weights=None, embedding_dim=mc.get("embedding_dim", 512),
    cross_attn_heads=mc.get("cross_attn_heads", 4), cross_attn_layers=mc.get("cross_attn_layers", 1),
    gate_hidden=mc.get("gate_hidden", 128), classifier_hidden=mc.get("classifier_hidden", 512),
    dropout=mc.get("dropout", 0.2), freeze_backbone=mc.get("freeze_backbone", True),
    unfreeze_last_stage=mc.get("unfreeze_last_stage", False),
    unfreeze_extra_stage3_tail=mc.get("unfreeze_extra_stage3_tail", False),
    aux_relation_head=mc.get("aux_relation_head", False),
    num_relation_classes=mc.get("num_relation_classes", 11),
    symmetric_forward=mc.get("symmetric_forward", False),
    comparison_only_fusion=mc.get("comparison_only_fusion", False),
)
model.load_state_dict(ck["model_state_dict"]); model.to(dev).eval()

dc = DataConfig(image_size=int(mc.get("img_size", 224)), normalize_mean=[0.5]*3, normalize_std=[0.5]*3)
apply_data_root_override(dc, "fiw", str(ROOT / "datasets/FIW"))
ds = KinshipPairDataset(root_dir=resolve_dataset_root(dc, "fiw"), dataset_type="fiw", split="test",
                        transform=get_transforms(dc, train=False), split_seed=42, negative_ratio=1.0,
                        aligned_root=str(ROOT / "datasets/FIW_aligned"))
ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
print(f"test pairs: {len(ds)} | regions: {model.region_names}")

probs, gcos, labels, wts = [], [], [], []
with torch.no_grad():
    for b in tqdm(ld, desc="diag"):
        out = model(b["img1"].to(dev), b["img2"].to(dev))
        logit, weights, _, gA_n, gB_n = out[0], out[1], out[2], out[3], out[4]
        probs.extend(torch.sigmoid(logit).cpu().numpy().flatten())
        gcos.extend((gA_n * gB_n).sum(-1).cpu().numpy().flatten())
        wts.append(weights.cpu().numpy())
        labels.extend(b["label"].numpy().tolist())

probs = np.array(probs); gcos = np.array(gcos); labels = np.array(labels, int)
wts = np.concatenate(wts, 0)  # (N, K)

print("\n=== SIGNAL DECOMPOSITION ===")
print(f"AUC(model prob)            : {roc_auc_score(labels, probs):.4f}")
print(f"AUC(contextualised global cosine): {roc_auc_score(labels, gcos):.4f}")
print(f"  -> head/regional lift over raw global cosine: "
      f"{roc_auc_score(labels, probs) - roc_auc_score(labels, gcos):+.4f}")
print("\n=== MEAN GATE WEIGHT PER REGION (0..1; high = region used) ===")
for i, name in enumerate(model.region_names):
    print(f"  {name:7s}: {wts[:, i].mean():.3f}  (std {wts[:, i].std():.3f})")
