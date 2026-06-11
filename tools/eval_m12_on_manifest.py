#!/usr/bin/env python3
"""Evaluate Model 12 (RGCK-Net) on an EXPLICIT pair manifest.

Unlike ``models/12_rgck_net/AMD/test.py`` (which builds its own RFIW
Track-I test split via ``KinshipPairDataset(split="test")``), this script
runs the exact same M12 checkpoint over a fixed list of pairs supplied by
a manifest JSON. It exists so that M12 can be scored on *exactly* the same
6000 pairs already evaluated by the VLM runs (Codex / Claude Sonnet),
making M12-vs-VLM a like-for-like comparison.

It does NOT touch the TCC text, does NOT re-run any VLM, and reuses the
same preprocessing / model / checkpoint / threshold as the M12 protocol.

Manifest format (list of dicts), as produced by the VLM pipelines:
    {
      "label": "kin" | "non_kin",
      "relation": "fd", ...,
      "p1_rel": "F0367/MID1/P03887_face2.jpg",
      "p2_rel": "F0367/MID3/P03895_face0.jpg",
      "p1_abs": "/abs/.../datasets/FIW/FIDs/F0367/MID1/P03887_face2.jpg",
      "p2_abs": "...",
      ... (other VLM-specific fields are ignored)
    }

Preprocessing parity with M12:
    Resize((224,224)) -> ToTensor -> Normalize(mean=.5, std=.5)
    Images are loaded from the ALIGNED tree (datasets/FIW_aligned), mapping
    each absolute raw-FIW path datasets/FIW/<rel> -> datasets/FIW_aligned/<rel>.

Usage:
    python3 tools/eval_m12_on_manifest.py \
        --manifest data/codex_vlm_fiw_binary_6000_medium_combined/manifest.json \
        --checkpoint models/12_rgck_net/output/016/checkpoints/best.pt \
        --output-dir data/m12_r012_on_vlm_6000_pairs \
        --fiw-root datasets/FIW \
        --aligned-root datasets/FIW_aligned
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
M12_ROOT = PROJECT_ROOT / "models" / "12_rgck_net"
SHARED = PROJECT_ROOT / "models" / "shared"

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(SHARED / "AMD"))
sys.path.insert(0, str(M12_ROOT))

import torchvision.transforms as T  # noqa: E402
from evaluation import KinshipMetrics  # noqa: E402
from protocol import get_checkpoint_threshold  # noqa: E402
from model import build_rgck_net  # noqa: E402

# M12 preprocessing constants (mirror AMD/test.py).
RGCK_MEAN = [0.5, 0.5, 0.5]
RGCK_STD = [0.5, 0.5, 0.5]
RGCK_IMG_SIZE = 224

LABEL_TO_INT = {"kin": 1, "non_kin": 0}


def normalize_label(raw):
    """Accept 'kin'/'non_kin' strings or 1/0 ints; return ('kin'|'non_kin', int)."""
    if isinstance(raw, str):
        s = raw.strip().lower().replace("-", "_")
        if s in ("kin", "1", "positive", "pos"):
            return "kin", 1
        return "non_kin", 0
    return ("kin", 1) if int(raw) == 1 else ("non_kin", 0)


def derive_rel(abs_path, fiw_root):
    """Path relative to the FIW root (e.g. 'F0367/MID1/P03887_face2.jpg'),
    stripping a leading 'FIDs/' so it matches the codex p1_rel convention."""
    p = Path(abs_path)
    try:
        rel = p.relative_to(fiw_root)
    except ValueError:
        parts = p.parts
        rel = Path(*parts[parts.index("FIW") + 1:]) if "FIW" in parts else Path(p.name)
    parts = rel.parts
    if parts and parts[0] == "FIDs":
        rel = Path(*parts[1:])
    return str(rel)


def normalize_items(raw_items, fiw_root):
    """Canonicalise a VLM manifest (codex or claude schema) into uniform records.

    Output record keys: label, label_int, relation, p1_rel, p2_rel,
    p1_abs, p2_abs, relation_name, sample_id.

    - codex: label='kin'/'non_kin', has 'relation' and 'p1_rel'/'p2_rel'.
    - claude: label=1/0, 'ptype' is the stratum ('non-kin' for negatives,
      the relation for positives), no p1_rel/p2_rel (derived from abs paths).
    """
    out = []
    for i, it in enumerate(raw_items):
        label_str, label_int = normalize_label(it["label"])
        relation = it.get("relation")
        if relation is None:
            relation = it.get("ptype") or it.get("stratum") or "unknown"
        relation = str(relation).replace("-", "_") if relation == "non-kin" else str(relation)
        p1_abs = it["p1_abs"]
        p2_abs = it["p2_abs"]
        p1_rel = it.get("p1_rel") or derive_rel(p1_abs, fiw_root)
        p2_rel = it.get("p2_rel") or derive_rel(p2_abs, fiw_root)
        sid = it.get("combined_sample_id") or it.get("sample_id") or f"P{i:05d}"
        out.append({
            "label": label_str,
            "label_int": label_int,
            "relation": relation,
            "relation_name": it.get("relation_name", ""),
            "p1_rel": p1_rel,
            "p2_rel": p2_rel,
            "p1_abs": p1_abs,
            "p2_abs": p2_abs,
            "sample_id": sid,
        })
    return out


def eval_transform(img_size: int) -> T.Compose:
    """Exactly the M12 eval transform (dataset.get_transforms train=False)."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=RGCK_MEAN, std=RGCK_STD),
    ])


def remap_aligned(abs_path: str, fiw_root: Path, aligned_root: Path) -> Path:
    """Map a raw-FIW absolute path to its aligned-tree counterpart.

    Mirrors KinshipPairDataset._maybe_remap_aligned: rel = path relative to
    the FIW root; aligned = aligned_root / rel.
    """
    p = Path(abs_path)
    try:
        rel = p.relative_to(fiw_root)
    except ValueError:
        # abs_path not under fiw_root (e.g. a differently-rooted manifest):
        # fall back to splitting on the dataset name.
        parts = p.parts
        if "FIW" in parts:
            rel = Path(*parts[parts.index("FIW") + 1:])
        else:
            rel = Path(p.name)
    return aligned_root / rel


class ManifestPairDataset(Dataset):
    def __init__(self, items, fiw_root: Path, aligned_root: Path, img_size: int):
        self.items = items
        self.fiw_root = fiw_root
        self.aligned_root = aligned_root
        self.transform = eval_transform(img_size)
        self.missing = []  # aligned files that were absent (fell back to raw)

    def __len__(self):
        return len(self.items)

    def _load(self, abs_path: str):
        aligned = remap_aligned(abs_path, self.fiw_root, self.aligned_root)
        if aligned.exists():
            path = aligned
        else:
            self.missing.append(abs_path)
            path = Path(abs_path)
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        it = self.items[idx]
        img1 = self._load(it["p1_abs"])
        img2 = self._load(it["p2_abs"])
        return {
            "img1": img1,
            "img2": img2,
            "label": it["label_int"],
            "idx": idx,
        }


def validate_manifest(items):
    """Enforce the task's mandatory manifest validations on canonical records.

    Hard requirements (raise on violation): exactly 6000 records, kin==3000,
    non_kin==3000, unique sample_ids (no duplicate SAMPLES), required fields
    present. Duplicate underlying image-pairs are REPORTED, not failed —
    the combined VLM set legitimately scores a handful of pairs as separate
    samples, and M12 must score the identical sample list to be comparable.
    """
    facts = {}
    facts["total_pairs"] = len(items)
    assert len(items) == 6000, f"expected 6000 pairs, got {len(items)}"

    label_counts = defaultdict(int)
    for it in items:
        label_counts[it["label"]] += 1
    facts["label_counts"] = dict(label_counts)
    assert label_counts["kin"] == 3000, f"kin={label_counts['kin']} != 3000"
    assert label_counts["non_kin"] == 3000, f"non_kin={label_counts['non_kin']} != 3000"

    # Unique SAMPLES (the real no-duplicate guarantee).
    sids = [it["sample_id"] for it in items]
    facts["unique_sample_ids"] = len(set(sids))
    assert len(set(sids)) == len(sids), (
        f"duplicate sample_ids: {len(sids) - len(set(sids))}")

    # Duplicate underlying pairs reported (informational).
    pair_keys = [(it["p1_rel"], it["p2_rel"], it["label"]) for it in items]
    facts["unique_pair_keys"] = len(set(pair_keys))
    facts["duplicate_pair_rows"] = len(pair_keys) - len(set(pair_keys))

    required = ["label", "relation", "p1_rel", "p2_rel", "p1_abs", "p2_abs", "sample_id"]
    for i, it in enumerate(items):
        for r in required:
            assert it.get(r) not in (None, ""), f"item {i} missing {r}"
    facts["required_fields_ok"] = True
    return facts


def cross_check_predictions_csv(items, csv_path: Path):
    """Confirm M12 scores the same pair set the VLM scored by matching the
    manifest's (p1_rel,p2_rel,label) multiset against the predictions.csv.
    Schema-tolerant: derives p1_rel/relation/label if the CSV uses raw paths.
    """
    if not csv_path.exists():
        return {"checked": False, "reason": f"{csv_path} not found"}
    csv_keys = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for row in reader:
            ls, _ = normalize_label(row.get("label"))
            if "p1_rel" in cols:
                p1, p2 = row["p1_rel"], row["p2_rel"]
            else:
                p1 = derive_rel(row.get("p1_abs", ""), Path("."))
                p2 = derive_rel(row.get("p2_abs", ""), Path("."))
            csv_keys.append((p1, p2, ls))
    man_keys = [(it["p1_rel"], it["p2_rel"], it["label"]) for it in items]
    return {
        "checked": True,
        "csv_rows": len(csv_keys),
        "manifest_rows": len(man_keys),
        "identical_pair_multiset": sorted(csv_keys) == sorted(man_keys),
        "identical_pair_set": set(csv_keys) == set(man_keys),
        "only_in_manifest": len(set(man_keys) - set(csv_keys)),
        "only_in_csv": len(set(csv_keys) - set(man_keys)),
    }


def compute_supplementary(preds, labels, relations, threshold):
    """Specificity, confusion matrix, per-label and per-relation in the same
    schema the VLM metrics.json uses, so the two are directly comparable."""
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=int)
    binary = (preds > threshold).astype(int)

    tp = int(((binary == 1) & (labels == 1)).sum())
    fn = int(((binary == 0) & (labels == 1)).sum())
    fp = int(((binary == 1) & (labels == 0)).sum())
    tn = int(((binary == 0) & (labels == 0)).sum())

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    confusion = {
        "kin": {"kin": tp, "non_kin": fn},        # true kin row
        "non_kin": {"kin": fp, "non_kin": tn},    # true non_kin row
    }

    per_label = {
        "kin": {"count": int((labels == 1).sum()),
                "correct": tp,
                "accuracy": tp / int((labels == 1).sum()) if (labels == 1).sum() else 0.0},
        "non_kin": {"count": int((labels == 0).sum()),
                    "correct": tn,
                    "accuracy": tn / int((labels == 0).sum()) if (labels == 0).sum() else 0.0},
    }

    per_relation = {}
    rel = np.asarray(relations)
    for r in sorted(set(relations)):
        m = rel == r
        rl = labels[m]
        rb = binary[m]
        correct = int((rb == rl).sum())
        per_relation[r] = {
            "count": int(m.sum()),
            "correct": correct,
            "accuracy": correct / int(m.sum()) if m.sum() else 0.0,
            "kin_count": int((rl == 1).sum()),
            "non_kin_count": int((rl == 0).sum()),
        }

    return {
        "specificity": specificity,
        "confusion_matrix": confusion,
        "per_label": per_label,
        "per_relation": per_relation,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate M12 on an explicit pair manifest")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fiw-root", default=str(PROJECT_ROOT / "datasets" / "FIW"))
    ap.add_argument("--aligned-root", default=str(PROJECT_ROOT / "datasets" / "FIW_aligned"))
    ap.add_argument("--predictions-csv", default=None,
                    help="VLM predictions.csv to cross-check the pair set against")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override; default = checkpoint stored threshold")
    ap.add_argument("--rocm-device", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    fiw_root = Path(args.fiw_root).resolve()
    aligned_root = Path(args.aligned_root).resolve()

    log("=" * 64)
    log("Evaluate M12 (RGCK-Net) on explicit VLM pair manifest")
    log("=" * 64)
    log(f"Manifest:    {args.manifest}")
    log(f"Checkpoint:  {args.checkpoint}")
    log(f"FIW root:    {fiw_root}")
    log(f"Aligned root:{aligned_root}")

    raw_items = json.load(open(args.manifest))
    assert isinstance(raw_items, list), "manifest must be a list of pair dicts"
    items = normalize_items(raw_items, fiw_root)

    # ---- Mandatory validations on the manifest ------------------------
    facts = validate_manifest(items)
    log("\n[VALIDATION] manifest")
    log(f"  total_pairs        : {facts['total_pairs']} (== 6000 ✓)")
    log(f"  label_counts       : {facts['label_counts']} (kin==3000, non_kin==3000 ✓)")
    log(f"  unique_sample_ids  : {facts['unique_sample_ids']} (no duplicate samples ✓)")
    log(f"  unique_pair_keys   : {facts['unique_pair_keys']} "
        f"(duplicate pair rows: {facts['duplicate_pair_rows']} — kept, real repeated VLM samples)")
    log(f"  required_fields_ok : {facts['required_fields_ok']}")

    csv_path = Path(args.predictions_csv) if args.predictions_csv else \
        Path(args.manifest).parent / "predictions.csv"
    xcheck = cross_check_predictions_csv(items, csv_path)
    log("\n[VALIDATION] M12 pairs == VLM pairs (label/relation/p1_rel/p2_rel)")
    if xcheck["checked"]:
        log(f"  identical_pair_set : {xcheck['identical_pair_set']} "
            f"(manifest={xcheck['manifest_rows']}, csv={xcheck['csv_rows']}, "
            f"only_in_manifest={xcheck['only_in_manifest']}, only_in_csv={xcheck['only_in_csv']})")
    else:
        log(f"  SKIPPED: {xcheck['reason']}")

    # ---- Model ---------------------------------------------------------
    from rocm_utils import setup_rocm_environment, clear_rocm_cache  # noqa: E402
    setup_rocm_environment(visible_devices=str(args.rocm_device))
    device = torch.device(f"cuda:{args.rocm_device}")

    log(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mc = ckpt.get("model_config", ckpt.get("protocol", {}).get("model_config", {}))
    img_size = int(mc.get("img_size", RGCK_IMG_SIZE))

    model = build_rgck_net(
        adaface_weights=None,
        embedding_dim=mc.get("embedding_dim", 512),
        cross_attn_heads=mc.get("cross_attn_heads", 4),
        cross_attn_layers=mc.get("cross_attn_layers", 1),
        gate_hidden=mc.get("gate_hidden", 128),
        classifier_hidden=mc.get("classifier_hidden", 512),
        dropout=mc.get("dropout", 0.2),
        freeze_backbone=mc.get("freeze_backbone", True),
        unfreeze_last_stage=mc.get("unfreeze_last_stage", False),
        unfreeze_extra_stage3_tail=mc.get("unfreeze_extra_stage3_tail", False),
        aux_relation_head=mc.get("aux_relation_head", False),
        num_relation_classes=mc.get("num_relation_classes", 11),
        symmetric_forward=mc.get("symmetric_forward", False),
        comparison_only_fusion=mc.get("comparison_only_fusion", False),
        roi_align_tokenizer=mc.get("roi_align_tokenizer", False),
        backbone_input_size=mc.get("backbone_input_size", 112),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    if args.threshold is not None:
        threshold = args.threshold
        log(f"Threshold (override): {threshold}")
    else:
        threshold = get_checkpoint_threshold(ckpt) or 0.5
        log(f"Threshold (checkpoint stored): {threshold:.4f}")
    log(f"Best epoch: {ckpt.get('epoch', '?')}  |  "
        f"unfreeze_extra_stage3_tail={mc.get('unfreeze_extra_stage3_tail', False)}  |  "
        f"consistency-trained={mc.get('consistency_weight', 'n/a')}")

    # ---- Inference -----------------------------------------------------
    ds = ManifestPairDataset(items, fiw_root, aligned_root, img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    probs = np.zeros(len(items), dtype=float)
    with torch.no_grad():
        for batch in tqdm(loader, desc="M12 inference"):
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            idxs = batch["idx"].numpy()
            out = model(img1, img2)
            logit = out[0]
            p = torch.sigmoid(logit).cpu().numpy().flatten()
            probs[idxs] = p

    if ds.missing:
        log(f"\n[WARN] {len(ds.missing)} image(s) missing in aligned tree, "
            f"fell back to raw FIW. First few: {ds.missing[:3]}")
    else:
        log("\n[OK] all images resolved from the aligned tree (no raw fallback).")

    labels = np.array([it["label_int"] for it in items], dtype=int)
    relations = [it["relation"] for it in items]

    # ---- Metrics (reuse M12 protocol's KinshipMetrics) -----------------
    km = KinshipMetrics(threshold=threshold)
    km.update(predictions=probs, labels=labels, relations=relations)
    core = km.compute()
    supp = compute_supplementary(probs, labels, relations, threshold)

    binary = (probs > threshold).astype(int)
    metrics = {
        "task": "binary_kinship_verification",
        "model": "M12_RGCK-Net_R012",
        "checkpoint": str(args.checkpoint),
        "evaluated_on": "VLM_6000_pairs (NOT the 13425-pair RFIW Track-I test split)",
        "manifest": str(args.manifest),
        "threshold": float(threshold),
        "total_pairs": int(len(items)),
        "total_images": int(len(items) * 2),
        "accuracy": float(core["accuracy"]),
        "balanced_accuracy": float(core["balanced_accuracy"]),
        "precision": float(core["precision"]),
        "recall": float(core["recall"]),
        "specificity": float(supp["specificity"]),
        "f1": float(core["f1"]),
        "roc_auc": float(core["roc_auc"]),
        "average_precision": float(core["average_precision"]),
        "tar@far=0.001": float(core["tar@far=0.001"]),
        "tar@far=0.01": float(core["tar@far=0.01"]),
        "tar@far=0.1": float(core["tar@far=0.1"]),
        "per_label": supp["per_label"],
        "per_relation": supp["per_relation"],
        "confusion_matrix": supp["confusion_matrix"],
        "validation": {
            "manifest": facts,
            "pairset_cross_check": xcheck,
            "n_predictions": int(len(probs)),
            "n_predictions_equals_6000": bool(len(probs) == 6000),
            "aligned_missing": len(ds.missing),
        },
    }
    # per-relation ROC-AUC (M12-rich) kept separately for reference.
    metrics["per_relation_roc_auc"] = {
        r: float(v["roc_auc"]) for r, v in core.get("per_relation", {}).items()
    }

    # ---- Write artifacts ----------------------------------------------
    # 1. manifest.json (the exact pair set M12 scored, copied verbatim)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    # 2. predictions.csv
    pred_label_str = ["kin" if b == 1 else "non_kin" for b in binary]
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "label", "relation", "relation_name",
                    "p1_rel", "p2_rel", "m12_prob_kin", "predicted_label", "correct"])
        for i, it in enumerate(items):
            sid = it["sample_id"]
            correct = int(pred_label_str[i] == it["label"])
            w.writerow([sid, it["label"], it["relation"], it.get("relation_name", ""),
                        it["p1_rel"], it["p2_rel"], f"{probs[i]:.6f}",
                        pred_label_str[i], correct])

    # 3. metrics.json
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---- Final validation summary -------------------------------------
    log("\n[VALIDATION] metrics computed over exactly "
        f"{metrics['validation']['n_predictions']} predictions "
        f"(== 6000: {metrics['validation']['n_predictions_equals_6000']})")
    log("\n=== M12 R012 on the VLM 6000-pair set ===")
    for k in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity",
              "f1", "roc_auc", "average_precision",
              "tar@far=0.001", "tar@far=0.01", "tar@far=0.1"]:
        log(f"  {k:20s}: {metrics[k]:.4f}")
    log(f"  confusion (TP/FN/FP/TN): "
        f"{supp['confusion_matrix']['kin']['kin']}/"
        f"{supp['confusion_matrix']['kin']['non_kin']}/"
        f"{supp['confusion_matrix']['non_kin']['kin']}/"
        f"{supp['confusion_matrix']['non_kin']['non_kin']}")

    with open(out_dir / "run.log", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"\nArtifacts written to {out_dir}/")
    clear_rocm_cache()


if __name__ == "__main__":
    main()
