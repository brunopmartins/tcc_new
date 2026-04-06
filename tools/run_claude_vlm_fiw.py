#!/usr/bin/env python3
"""
Zero-shot Claude VLM kinship-relation baseline on FIW test pairs.

Samples valid positive FIW pairs (one per relation, balanced), encodes each
pair as a side-by-side image, and asks claude-haiku-4-5 to classify the
kinship relation in a closed-set zero-shot setting.

Outputs (saved to --output-dir):
  manifest.json    – sampled pairs with metadata
  config.json      – run configuration
  predictions.csv  – per-pair prediction + ground truth
  metrics.json     – overall + per-relation metrics + confusion matrix
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import anthropic
from PIL import Image, ImageDraw, ImageOps


RELATIONS = [
    "bb", "fd", "fs", "md", "ms", "ss",
    "sibs", "gfgd", "gfgs", "gmgd", "gmgs",
]

RELATION_NAMES = {
    "bb":   "brother-brother",
    "fd":   "father-daughter",
    "fs":   "father-son",
    "md":   "mother-daughter",
    "ms":   "mother-son",
    "ss":   "sister-sister",
    "sibs": "brother-sister (mixed)",
    "gfgd": "grandfather-granddaughter",
    "gfgs": "grandfather-grandson",
    "gmgd": "grandmother-granddaughter",
    "gmgs": "grandmother-grandson",
}

SYSTEM_PROMPT = (
    "You are an expert in facial kinship analysis. "
    "Given a side-by-side image of two faces, classify their kinship relation "
    "from the following closed set: "
    + ", ".join(f"{r} ({RELATION_NAMES[r]})" for r in RELATIONS)
    + ". Respond ONLY with a JSON object: "
    '{"predicted_relation": "<code>", "confidence": <0-1 float>}. '
    "No other text."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path("/home/bruno/Desktop/tcc_new"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/home/bruno/Desktop/tcc_new/data/claude_vlm_fiw_150"))
    p.add_argument("--total-images", type=int, default=150,
                   help="Total individual images (must be even, pairs=total/2).")
    p.add_argument("--seed", type=int, default=20260406)
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="Claude model to use.")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Seconds between API calls to respect rate limits.")
    p.add_argument("--reuse-manifest", type=Path, default=None,
                   help="Reuse an existing manifest.json for exact reproducibility.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def load_valid_positive_pairs(root: Path) -> Dict[str, List[dict]]:
    csv_path = root / "datasets" / "FIW" / "track-I" / "test-pairs.csv"
    fiw_root = root / "datasets" / "FIW" / "FIDs"
    by_relation: Dict[str, List[dict]] = defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["labels"] != "1":
                continue
            rel = row["ptype"]
            if rel not in RELATIONS:
                continue
            p1 = fiw_root / row["p1"]
            p2 = fiw_root / row["p2"]
            if not p1.exists() or not p2.exists():
                continue
            by_relation[rel].append({
                "relation": rel,
                "relation_name": RELATION_NAMES[rel],
                "p1_rel": row["p1"],
                "p2_rel": row["p2"],
                "p1_abs": str(p1),
                "p2_abs": str(p2),
            })

    return by_relation


def build_allocation(by_relation: Dict[str, List[dict]], total_pairs: int) -> Dict[str, int]:
    base = total_pairs // len(RELATIONS)
    remainder = total_pairs % len(RELATIONS)
    alloc = {r: base for r in RELATIONS}
    for r in sorted(RELATIONS, key=lambda r: len(by_relation[r]), reverse=True)[:remainder]:
        alloc[r] += 1
    return alloc


def sample_pairs(by_relation, allocation, seed) -> List[dict]:
    rng = random.Random(seed)
    sampled: List[dict] = []
    idx = 1
    for rel in RELATIONS:
        candidates = list(by_relation[rel])
        rng.shuffle(candidates)
        for rec in candidates[:allocation[rel]]:
            sampled.append({**rec, "sample_id": f"S{idx:03d}"})
            idx += 1
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def make_pair_sheet_bytes(p1_abs: str, p2_abs: str) -> bytes:
    """Return a JPEG side-by-side pair sheet as raw bytes."""
    sz = (256, 256)
    img1 = ImageOps.fit(Image.open(p1_abs).convert("RGB"), sz)
    img2 = ImageOps.fit(Image.open(p2_abs).convert("RGB"), sz)
    canvas = Image.new("RGB", (sz[0] * 2 + 20, sz[1] + 20), "white")
    canvas.paste(img1, (5, 10))
    canvas.paste(img2, (sz[0] + 15, 10))
    buf = BytesIO()
    canvas.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def image_to_base64(raw: bytes) -> str:
    return base64.standard_b64encode(raw).decode("utf-8")


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def classify_pair(client: anthropic.Anthropic, model: str, img_bytes: bytes) -> dict:
    """Call Claude with a pair-sheet image and return {predicted_relation, confidence}."""
    b64 = image_to_base64(img_bytes)
    message = client.messages.create(
        model=model,
        max_tokens=64,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Classify the kinship relation shown in this image.",
                    },
                ],
            }
        ],
    )
    raw = message.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw)
    pred = parsed.get("predicted_relation", "")
    if pred not in RELATIONS:
        pred = RELATIONS[0]
    conf = float(parsed.get("confidence", 0.5))
    return {"predicted_relation": pred, "confidence": conf}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(records: List[dict]) -> dict:
    confusion: Dict[str, Dict[str, int]] = {
        r: {p: 0 for p in RELATIONS} for r in RELATIONS
    }
    for rec in records:
        confusion[rec["relation"]][rec["predicted_relation"]] += 1

    per_relation = {}
    metric_rows = []
    for rel in RELATIONS:
        rel_recs = [r for r in records if r["relation"] == rel]
        tp = confusion[rel][rel]
        fp = sum(confusion[o][rel] for o in RELATIONS if o != rel)
        fn = sum(confusion[rel][o] for o in RELATIONS if o != rel)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec_  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
        acc  = tp / len(rel_recs) if rel_recs else 0.0
        per_relation[rel] = {
            "count": len(rel_recs),
            "correct": tp,
            "accuracy": acc,
            "precision": prec,
            "recall": rec_,
            "f1": f1,
        }
        metric_rows.append((prec, rec_, f1))

    total = len(records)
    correct = sum(1 for r in records if r["predicted_relation"] == r["relation"])
    confs = [r["confidence"] for r in records]
    c_confs = [r["confidence"] for r in records if r["predicted_relation"] == r["relation"]]
    w_confs = [r["confidence"] for r in records if r["predicted_relation"] != r["relation"]]

    return {
        "task": "closed_set_relation_classification",
        "model": records[0].get("model", "unknown") if records else "unknown",
        "total_pairs": total,
        "total_images": total * 2,
        "overall_accuracy": correct / total if total else 0.0,
        "macro_precision": sum(r[0] for r in metric_rows) / len(metric_rows),
        "macro_recall": sum(r[1] for r in metric_rows) / len(metric_rows),
        "macro_f1": sum(r[2] for r in metric_rows) / len(metric_rows),
        "mean_confidence": sum(confs) / len(confs) if confs else 0.0,
        "mean_confidence_correct": sum(c_confs) / len(c_confs) if c_confs else 0.0,
        "mean_confidence_incorrect": sum(w_confs) / len(w_confs) if w_confs else 0.0,
        "per_relation": per_relation,
        "confusion_matrix": confusion,
    }


def write_predictions_csv(path: Path, records: List[dict]) -> None:
    fields = [
        "sample_id", "relation", "relation_name",
        "predicted_relation", "predicted_relation_name",
        "confidence", "correct", "p1_rel", "p2_rel",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({
                "sample_id":               r["sample_id"],
                "relation":                r["relation"],
                "relation_name":           r["relation_name"],
                "predicted_relation":      r["predicted_relation"],
                "predicted_relation_name": RELATION_NAMES[r["predicted_relation"]],
                "confidence":              r["confidence"],
                "correct":                 int(r["predicted_relation"] == r["relation"]),
                "p1_rel":                  r["p1_rel"],
                "p2_rel":                  r["p2_rel"],
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.total_images % 2 != 0:
        raise ValueError("--total-images must be even.")

    total_pairs = args.total_images // 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- sampling ----
    if args.reuse_manifest:
        print(f"Reusing manifest from {args.reuse_manifest}")
        sampled = json.loads(args.reuse_manifest.read_text(encoding="utf-8"))
        # Rebuild abs paths relative to this machine's root
        fiw_root = args.root / "datasets" / "FIW" / "FIDs"
        for rec in sampled:
            rec["p1_abs"] = str(fiw_root / rec["p1_rel"])
            rec["p2_abs"] = str(fiw_root / rec["p2_rel"])
        allocation = {r: sum(1 for s in sampled if s["relation"] == r) for r in RELATIONS}
    else:
        by_relation = load_valid_positive_pairs(args.root)
        allocation = build_allocation(by_relation, total_pairs)
        sampled = sample_pairs(by_relation, allocation, args.seed)

    # Save manifest + config
    manifest_out = [{k: v for k, v in r.items() if k not in ("p1_abs", "p2_abs")}
                    for r in sampled]
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8"
    )
    config = {
        "model": args.model,
        "seed": args.seed,
        "total_images": args.total_images,
        "total_pairs": total_pairs,
        "allocation": allocation,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # ---- Claude inference ----
    client = anthropic.Anthropic()
    print(f"Running {total_pairs} pairs with {args.model} …")

    for i, rec in enumerate(sampled, start=1):
        img_bytes = make_pair_sheet_bytes(rec["p1_abs"], rec["p2_abs"])
        try:
            result = classify_pair(client, args.model, img_bytes)
        except Exception as exc:
            print(f"  [{i}/{total_pairs}] ERROR for {rec['sample_id']}: {exc}")
            result = {"predicted_relation": RELATIONS[0], "confidence": 0.0}

        rec["predicted_relation"] = result["predicted_relation"]
        rec["confidence"] = result["confidence"]
        rec["model"] = args.model
        correct = rec["predicted_relation"] == rec["relation"]
        print(
            f"  [{i:3d}/{total_pairs}] {rec['sample_id']} "
            f"gt={rec['relation']:5s} "
            f"pred={rec['predicted_relation']:5s} "
            f"conf={rec['confidence']:.2f} "
            f"{'✓' if correct else '✗'}"
        )
        if args.delay > 0 and i < total_pairs:
            time.sleep(args.delay)

    # ---- save outputs ----
    metrics = compute_metrics(sampled)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_predictions_csv(args.output_dir / "predictions.csv", sampled)

    print("\n" + "=" * 50)
    print(f"Model:            {args.model}")
    print(f"Pairs evaluated:  {metrics['total_pairs']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Macro precision:  {metrics['macro_precision']:.4f}")
    print(f"Macro recall:     {metrics['macro_recall']:.4f}")
    print(f"Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
