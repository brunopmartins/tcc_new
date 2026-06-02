#!/usr/bin/env python3
"""
Claude Sonnet VLM — BINARY kinship verification on FIW.

Companion to ``run_claude_vlm_fiw.py`` (which does 11-class relation
classification on positive-only pairs). This version asks Claude the
exact same binary question the supervised models answer:

    given a pair of faces, are these two people biologically related?

so the result is directly comparable to M02 / M05 / M06 / M12 / B0 on the
same metric pack (AUC, AP, TAR@FAR, F1, Acc, Precision, Recall).

Pair source: ``KinshipPairDataset(split="test", negative_ratio=1.0,
split_seed=42)`` — the same sampler the supervised models train against,
which means the test pool, the non-kin construction, and the family
disjointness assumptions are identical to the rest of the project.

Score mapping (for AUC/AP/TAR@FAR):

    is_related=True  →  score = 0.5 + 0.5 * confidence  ∈ [0.5, 1.0]
    is_related=False →  score = 0.5 - 0.5 * confidence  ∈ [0.0, 0.5]

This produces a continuous score where higher means more kin-confidence.

Outputs (saved to --output-dir):
  manifest.json     — sampled (label, p1, p2, ptype) per pair
  config.json       — run configuration
  predictions.csv   — per-pair: pair_id, label, ptype, predicted, confidence, score, raw_response
  metrics.json      — AUC, AP, TAR@FAR, F1, Acc, Precision, Recall, per-relation
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
import numpy as np
from PIL import Image, ImageOps


# Make the shared dataset module importable.
PROJECT_ROOT = Path("/home/bruno/Desktop/tcc_new")
sys.path.insert(0, str(PROJECT_ROOT / "models" / "shared"))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "shared" / "AMD"))

from config import DataConfig  # noqa: E402
from dataset import KinshipPairDataset, get_transforms  # noqa: E402
from evaluation import KinshipMetrics, print_metrics  # noqa: E402


SYSTEM_PROMPT = (
    "You are an expert in facial kinship analysis. "
    "Given a side-by-side image of two faces, decide whether the two people "
    "are biologically related — that is, whether ANY direct kin relationship "
    "exists between them (parent–child, sibling, grandparent–grandchild). "
    "Base your decision only on the visual cues in the image. "
    "Respond ONLY with a JSON object: "
    '{"is_related": true|false, "confidence": <0-1 float>}. '
    "No other text. The confidence reflects how certain you are about your "
    "is_related answer, where 0 means 'no information' and 1 means 'fully "
    "certain'."
)


# ---------------------------------------------------------------------------
# Image helpers (kept identical to run_claude_vlm_fiw.py)
# ---------------------------------------------------------------------------

def make_pair_sheet_bytes(p1_abs: str, p2_abs: str) -> bytes:
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
# Sampling — pull (label, p1, p2, ptype) from KinshipPairDataset
# ---------------------------------------------------------------------------

def sample_pairs(
    num_positives: int,
    num_negatives: int,
    seed: int,
    aligned_root: Optional[str] = None,
) -> List[Dict]:
    """Return a balanced manifest of dicts with keys:
    sample_id, label (0/1), p1_abs, p2_abs, ptype (or '' for negatives).

    The dataset is built once with the same params B0/M02/M12 use, then we
    subsample positives and negatives to the requested sizes.
    """
    data_config = DataConfig(
        image_size=112,
        normalize_mean=[0.5, 0.5, 0.5],
        normalize_std=[0.5, 0.5, 0.5],
        fiw_root=str(PROJECT_ROOT / "datasets" / "FIW"),
    )
    dataset = KinshipPairDataset(
        root_dir=str(PROJECT_ROOT / "datasets" / "FIW"),
        dataset_type="fiw",
        split="test",
        transform=get_transforms(data_config, train=False),
        split_seed=seed,
        negative_ratio=1.0,
        aligned_root=aligned_root,
    )

    # KinshipPairDataset stores (img1, img2, rel) in .pairs and 0/1 in .labels.
    # We just want absolute file paths + labels + ptype.
    positives = []
    negatives = []
    for (p1, p2, rel), lab in zip(dataset.pairs, dataset.labels):
        # p1, p2 are either absolute paths or paths relative to root_dir.
        p1_abs = p1 if Path(p1).is_absolute() else str(PROJECT_ROOT / "datasets" / "FIW" / p1)
        p2_abs = p2 if Path(p2).is_absolute() else str(PROJECT_ROOT / "datasets" / "FIW" / p2)
        rec = {"label": int(lab), "p1_abs": p1_abs, "p2_abs": p2_abs, "ptype": rel or ""}
        if int(lab) == 1:
            positives.append(rec)
        else:
            negatives.append(rec)

    print(f"  pool: {len(positives)} positives, {len(negatives)} negatives")

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if num_positives > len(positives):
        raise ValueError(f"requested {num_positives} positives, only {len(positives)} available")
    if num_negatives > len(negatives):
        raise ValueError(f"requested {num_negatives} negatives, only {len(negatives)} available")

    sampled = positives[:num_positives] + negatives[:num_negatives]
    rng.shuffle(sampled)
    for i, rec in enumerate(sampled, start=1):
        rec["sample_id"] = f"S{i:04d}"
    return sampled


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def call_claude_binary(client, model: str, system_prompt: str, image_b64: str) -> Dict:
    """Send a binary kinship-verification request; parse JSON response."""
    response = client.messages.create(
        model=model,
        max_tokens=128,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Are these two people biologically related? "
                                "Respond ONLY with the JSON object.",
                    },
                ],
            },
        ],
    )
    raw_text = response.content[0].text.strip() if response.content else ""
    # Try to find a JSON object in the response.
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    parsed = None
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        # Last-ditch fallback: treat any "true"/"yes" in the text as positive.
        lower = raw_text.lower()
        if "true" in lower or '"is_related": true' in lower:
            parsed = {"is_related": True, "confidence": 0.5}
        else:
            parsed = {"is_related": False, "confidence": 0.5}

    is_related = bool(parsed.get("is_related", False))
    confidence = float(parsed.get("confidence", 0.5) or 0.5)
    confidence = max(0.0, min(1.0, confidence))
    return {
        "is_related": is_related,
        "confidence": confidence,
        "raw_text": raw_text,
    }


def score_from_decision(is_related: bool, confidence: float) -> float:
    """Continuous score in [0, 1] for AUC/AP/TAR@FAR.

    is_related=True  → score = 0.5 + 0.5 * confidence  (∈ [0.5, 1.0])
    is_related=False → score = 0.5 - 0.5 * confidence  (∈ [0.0, 0.5])
    """
    if is_related:
        return 0.5 + 0.5 * confidence
    return 0.5 - 0.5 * confidence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-positives", type=int, default=750)
    p.add_argument("--num-negatives", type=int, default=750)
    p.add_argument("--seed", type=int, default=20260406)
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Seconds between API calls.")
    p.add_argument("--reuse-manifest", type=Path, default=None,
                   help="Reuse an existing manifest.json from a prior binary run.")
    p.add_argument("--aligned-root", type=str, default=None,
                   help="Optional FIW_aligned root for remapped face paths.")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-backoff", type=float, default=4.0)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Claude VLM — BINARY kinship verification on FIW")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Output:      {out}")
    print(f"Pos / Neg:   {args.num_positives} / {args.num_negatives}")

    if args.reuse_manifest:
        print(f"Reusing manifest from {args.reuse_manifest}")
        sampled = json.loads(args.reuse_manifest.read_text(encoding="utf-8"))
    else:
        print("\nSampling pairs from KinshipPairDataset(test, seed=42)...")
        sampled = sample_pairs(
            num_positives=args.num_positives,
            num_negatives=args.num_negatives,
            seed=args.seed,
            aligned_root=args.aligned_root,
        )
        (out / "manifest.json").write_text(
            json.dumps(sampled, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"  wrote manifest with {len(sampled)} pairs")

    # Config
    cfg = {
        "model": args.model,
        "task": "binary_kin_verification",
        "seed": args.seed,
        "num_positives": args.num_positives,
        "num_negatives": args.num_negatives,
        "total_pairs": len(sampled),
        "delay": args.delay,
        "aligned_root": args.aligned_root,
        "prompt_system": SYSTEM_PROMPT,
        "score_mapping": "is_related=True→0.5+0.5*conf;False→0.5-0.5*conf",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    client = anthropic.Anthropic()
    pred_path = out / "predictions.csv"
    print(f"\nWriting predictions live to {pred_path}")

    # Live-writing CSV in case the run dies partway.
    fieldnames = [
        "sample_id", "label", "ptype",
        "predicted", "confidence", "score",
        "raw_text",
    ]
    n_done = 0
    with open(pred_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for i, rec in enumerate(sampled, start=1):
            img_bytes = make_pair_sheet_bytes(rec["p1_abs"], rec["p2_abs"])
            img_b64 = image_to_base64(img_bytes)

            result = None
            for attempt in range(args.max_retries):
                try:
                    result = call_claude_binary(
                        client, args.model, SYSTEM_PROMPT, img_b64,
                    )
                    break
                except (anthropic.APIError, anthropic.APIStatusError) as e:
                    wait = args.retry_backoff * (attempt + 1)
                    print(f"  [retry {attempt+1}/{args.max_retries}] {type(e).__name__}: {e}; "
                          f"sleeping {wait}s")
                    time.sleep(wait)
                except Exception as e:  # noqa: BLE001
                    wait = args.retry_backoff * (attempt + 1)
                    print(f"  [retry {attempt+1}/{args.max_retries}] {type(e).__name__}: {e}; "
                          f"sleeping {wait}s")
                    time.sleep(wait)

            if result is None:
                print(f"  WARN: pair {rec['sample_id']} failed after {args.max_retries} retries; "
                      "writing default neg.")
                result = {"is_related": False, "confidence": 0.0, "raw_text": "ERROR"}

            score = score_from_decision(result["is_related"], result["confidence"])
            writer.writerow({
                "sample_id": rec["sample_id"],
                "label": rec["label"],
                "ptype": rec["ptype"],
                "predicted": int(result["is_related"]),
                "confidence": f"{result['confidence']:.4f}",
                "score": f"{score:.4f}",
                "raw_text": result["raw_text"].replace("\n", " "),
            })
            fh.flush()
            n_done += 1

            if i % 25 == 0 or i == len(sampled):
                print(f"  [{i}/{len(sampled)}] last sample {rec['sample_id']}: "
                      f"label={rec['label']} pred={int(result['is_related'])} "
                      f"conf={result['confidence']:.3f} score={score:.3f}")

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"\n[done] {n_done} predictions written.")

    # Aggregate metrics
    print("\nComputing metrics...")
    rows = list(csv.DictReader(open(pred_path, encoding="utf-8")))
    scores = np.array([float(r["score"]) for r in rows], dtype=float)
    labels = np.array([int(r["label"]) for r in rows], dtype=int)
    relations = [r["ptype"] if r["ptype"] else "non-kin" for r in rows]

    # The supervised models use a val-selected threshold. For VLM with no
    # val data, the natural default is 0.5 (the boundary between
    # is_related=True and is_related=False under our score mapping).
    metrics_calc = KinshipMetrics(threshold=0.5)
    metrics_calc.update(predictions=scores, labels=labels, relations=relations)
    metrics = metrics_calc.compute()

    print()
    print_metrics(metrics, prefix="Test (VLM binary) ")

    metrics_path = out / "metrics.json"
    serialisable = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            serialisable[k] = {kk: {kkk: float(vvv) if hasattr(vvv, "__float__") else vvv
                                    for kkk, vvv in vv.items()}
                                for kk, vv in v.items()}
        elif isinstance(v, (int, float)):
            serialisable[k] = float(v)
        else:
            serialisable[k] = v
    metrics_path.write_text(json.dumps(serialisable, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"\nSaved: {metrics_path}")


if __name__ == "__main__":
    main()
