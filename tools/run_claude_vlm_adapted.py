#!/usr/bin/env python3
"""
Domain-adapted Claude VLM kinship evaluation — FIW test pairs.

Three evaluation modes:
  zero_shot      — replicates the baseline (simple closed-set prompt)
  structured     — structured chain-of-thought prompt (no examples)
  few_shot       — structured + text-based few-shot demonstrations
  calibrated     — few_shot predictions + label-frequency calibration
                   (fitted on the validation split)

Usage — build validation + test splits:
  python run_claude_vlm_adapted.py sample --val-per-rel 3 --test-per-rel 30

Usage — run evaluation:
  python run_claude_vlm_adapted.py evaluate --mode structured --split test
  python run_claude_vlm_adapted.py evaluate --mode calibrated --split test

All outputs go to:
  data/claude_vlm_adapted/
    val/          manifest.json, pair_sheets/, predictions_<mode>.csv, metrics_<mode>.json
    test/         manifest.json, pair_sheets/, predictions_<mode>.csv, metrics_<mode>.json
    config.json
    calibration.json   (fitted from val/predictions_few_shot.csv)
    prompts/           all prompts used, serialised for reproducibility
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
from typing import Dict, List, Optional, Tuple

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELATIONS = ["bb", "fd", "fs", "md", "ms", "ss", "sibs", "gfgd", "gfgs", "gmgd", "gmgs"]

RELATION_NAMES = {
    "bb":   "brother-brother",
    "fd":   "father-daughter",
    "fs":   "father-son",
    "md":   "mother-daughter",
    "ms":   "mother-son",
    "ss":   "sister-sister",
    "sibs": "brother-sister",
    "gfgd": "grandfather-granddaughter",
    "gfgs": "grandfather-grandson",
    "gmgd": "grandmother-granddaughter",
    "gmgs": "grandmother-grandson",
}

# Generation-gap groupings — used in structured prompt and calibration
GEN_GAP_GROUPS = {
    "same":  ["bb", "ss", "sibs"],
    "one":   ["fd", "fs", "md", "ms"],
    "two":   ["gfgd", "gfgs", "gmgd", "gmgs"],
}

# Decision table: (generation_gap, older_gender, younger_gender) → relation
DECISION_TABLE = {
    ("same",  "male",   "male"):   "bb",
    ("same",  "female", "female"): "ss",
    ("same",  "male",   "female"): "sibs",
    ("same",  "female", "male"):   "sibs",
    ("one",   "male",   "female"): "fd",
    ("one",   "male",   "male"):   "fs",
    ("one",   "female", "female"): "md",
    ("one",   "female", "male"):   "ms",
    ("two",   "male",   "female"): "gfgd",
    ("two",   "male",   "male"):   "gfgs",
    ("two",   "female", "female"): "gmgd",
    ("two",   "female", "male"):   "gmgs",
}

ROOT = Path("/home/bruno/Desktop/tcc_new")
ADAPTED_DIR = ROOT / "data" / "claude_vlm_adapted"
FIW_CSV = ROOT / "datasets" / "FIW" / "track-I" / "test-pairs.csv"
FIW_ROOT = ROOT / "datasets" / "FIW" / "FIDs"
BASELINE_MANIFEST = ROOT / "data" / "claude_vlm_fiw_150" / "manifest.json"

SEED = 20260413

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ZERO_SHOT_SYSTEM = (
    "You are an expert in facial kinship analysis. "
    "Given a side-by-side image of two faces (left face and right face), "
    "classify their kinship relation from the following closed set: "
    + ", ".join(f"{r} ({RELATION_NAMES[r]})" for r in RELATIONS)
    + ". Respond ONLY with a JSON object: "
    '{"predicted_relation": "<code>", "confidence": <0.0-1.0>}. '
    "No other text."
)

STRUCTURED_SYSTEM = """\
You are an expert in facial kinship analysis. For the side-by-side kinship pair image \
(left face | right face), reason through the following steps and then classify:

STEP 1 — GENERATION GAP
Estimate the apparent age difference:
- "same" → both people appear to be in the same generation (ages within ~15 years)
- "one"  → one person appears clearly older (one generation apart, ~20-35 year gap)
- "two"  → one person appears much older (two generations apart, ~40-60 year gap)

STEP 2 — GENDER
Identify the apparent gender of each face: "male" or "female".
Left face gender: ?
Right face gender: ?

STEP 3 — DECISION LOGIC
Apply the following decision table based on generation gap and genders:
Same generation:
  male   + male   → bb (brother-brother)
  female + female → ss (sister-sister)
  mixed           → sibs (brother-sister)

One generation gap (older → younger):
  male   + female → fd (father-daughter)
  male   + male   → fs (father-son)
  female + female → md (mother-daughter)
  female + male   → ms (mother-son)

Two generation gap (older → younger):
  male   + female → gfgd (grandfather-granddaughter)
  male   + male   → gfgs (grandfather-grandson)
  female + female → gmgd (grandmother-granddaughter)
  female + male   → gmgs (grandmother-grandson)

STEP 4 — VISUAL SIMILARITY CHECK
Rate overall facial similarity (0.0 = no resemblance, 1.0 = near-identical).
Very high similarity (>0.7) with same-generation assessment reinforces sibling relations.
Low similarity with large age gap reinforces grandparent relations.

Return ONLY a valid JSON object with exactly these fields:
{
  "generation_gap": "same|one|two",
  "left_gender": "male|female",
  "right_gender": "male|female",
  "similarity": <0.0-1.0>,
  "predicted_relation": "<one of the 11 codes>",
  "confidence": <0.0-1.0>
}
No other text. The predicted_relation must be one of: """ + ", ".join(RELATIONS) + "."

# Text-based few-shot demonstrations embedded in the system prompt.
# These are representative descriptions (not images) that show the reasoning chain.
FEW_SHOT_EXAMPLES = """
EXAMPLES OF CORRECT REASONING:

Example 1 — gfgs (grandfather-grandson):
  Left face: elderly male, deeply wrinkled, grey hair, ~75 years old
  Right face: young male, smooth skin, ~12 years old
  Step 1: generation_gap = "two" (60-year gap)
  Step 2: left=male, right=male
  Step 3: two + male + male → gfgs
  Step 4: moderate similarity (shared jawline shape)
  → {"generation_gap":"two","left_gender":"male","right_gender":"male",
     "similarity":0.35,"predicted_relation":"gfgs","confidence":0.82}

Example 2 — gmgd (grandmother-granddaughter):
  Left face: elderly female, grey hair, ~70 years old
  Right face: young female, ~10 years old
  Step 1: generation_gap = "two"
  Step 2: left=female, right=female
  Step 3: two + female + female → gmgd
  Step 4: slight facial structure similarity
  → {"generation_gap":"two","left_gender":"female","right_gender":"female",
     "similarity":0.28,"predicted_relation":"gmgd","confidence":0.78}

Example 3 — bb (brother-brother):
  Left face: young male, ~22 years old
  Right face: young male, ~20 years old
  Step 1: generation_gap = "same"
  Step 2: left=male, right=male
  Step 3: same + male + male → bb
  Step 4: strong facial similarity (nose, eye shape)
  → {"generation_gap":"same","left_gender":"male","right_gender":"male",
     "similarity":0.72,"predicted_relation":"bb","confidence":0.80}

Example 4 — fd (father-daughter):
  Left face: adult male, ~45 years old
  Right face: young female, ~18 years old
  Step 1: generation_gap = "one" (~27-year gap)
  Step 2: left=male, right=female
  Step 3: one + male + female → fd
  Step 4: moderate similarity
  → {"generation_gap":"one","left_gender":"male","right_gender":"female",
     "similarity":0.45,"predicted_relation":"fd","confidence":0.88}

Example 5 — ms (mother-son):
  Left face: adult female, ~40 years old
  Right face: teenage male, ~16 years old
  Step 1: generation_gap = "one"
  Step 2: left=female, right=male
  Step 3: one + female + male → ms
  Step 4: facial structure similarity moderate
  → {"generation_gap":"one","left_gender":"female","right_gender":"male",
     "similarity":0.5,"predicted_relation":"ms","confidence":0.82}

KEY DISTINCTION — grandparent vs parent:
  If the age gap looks like 20-35 years → one generation (parent-child)
  If the age gap looks like 40-60 years → two generations (grandparent-grandchild)
  When uncertain between "one" and "two": examine whether the older person looks
  50+/60+ (grandparent) vs 30-50 (parent).

"""

FEW_SHOT_SYSTEM = STRUCTURED_SYSTEM.replace(
    "Return ONLY a valid JSON object",
    FEW_SHOT_EXAMPLES + "\nReturn ONLY a valid JSON object"
)


def get_system_prompt(mode: str) -> str:
    if mode == "zero_shot":
        return ZERO_SHOT_SYSTEM
    elif mode == "structured":
        return STRUCTURED_SYSTEM
    else:  # few_shot or calibrated
        return FEW_SHOT_SYSTEM


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def load_used_pairs() -> set:
    """Load pairs already used in the 75-pair baseline test set."""
    if not BASELINE_MANIFEST.exists():
        return set()
    manifest = json.loads(BASELINE_MANIFEST.read_text())
    return {(r["p1_rel"], r["p2_rel"]) for r in manifest}


def load_available_pairs(exclude: set) -> Dict[str, List[dict]]:
    by_rel: Dict[str, List[dict]] = defaultdict(list)
    with FIW_CSV.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["labels"] != "1":
                continue
            rel = row["ptype"]
            if rel not in RELATIONS:
                continue
            p1, p2 = row["p1"], row["p2"]
            if (p1, p2) in exclude:
                continue
            if not (FIW_ROOT / p1).exists() or not (FIW_ROOT / p2).exists():
                continue
            by_rel[rel].append({"relation": rel, "p1_rel": p1, "p2_rel": p2})
    return by_rel


def sample_split(
    by_rel: Dict[str, List[dict]],
    per_rel: int,
    seed: int,
    exclude_pairs: Optional[set] = None,
) -> Tuple[List[dict], set]:
    """Sample `per_rel` pairs per relation. Returns (sampled, used_pairs_set)."""
    rng = random.Random(seed)
    sampled: List[dict] = []
    used: set = set()
    for rel in RELATIONS:
        pool = [
            r for r in by_rel[rel]
            if exclude_pairs is None or (r["p1_rel"], r["p2_rel"]) not in exclude_pairs
        ]
        rng.shuffle(pool)
        chosen = pool[:per_rel]
        sampled.extend(chosen)
        for r in chosen:
            used.add((r["p1_rel"], r["p2_rel"]))
    # Assign sample_ids
    rng2 = random.Random(seed + 1)
    rng2.shuffle(sampled)
    for i, rec in enumerate(sampled, start=1):
        rec["sample_id"] = f"S{i:03d}"
    return sampled, used


# ---------------------------------------------------------------------------
# Pair-sheet creation
# ---------------------------------------------------------------------------

def make_pair_sheet(p1_abs: str, p2_abs: str, out_path: Path) -> None:
    sz = (256, 256)
    img1 = ImageOps.fit(Image.open(p1_abs).convert("RGB"), sz)
    img2 = ImageOps.fit(Image.open(p2_abs).convert("RGB"), sz)
    canvas = Image.new("RGB", (sz[0] * 2 + 20, sz[1] + 20), "white")
    canvas.paste(img1, (5, 10))
    canvas.paste(img2, (sz[0] + 15, 10))
    canvas.save(str(out_path), format="JPEG", quality=90)


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


# ---------------------------------------------------------------------------
# API call (requires anthropic SDK + API key)
# ---------------------------------------------------------------------------

def classify_via_api(
    client,
    model: str,
    img_bytes: bytes,
    system_prompt: str,
    mode: str,
) -> dict:
    b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
    user_text = (
        "Classify the kinship relation shown in this image."
        if mode == "zero_shot"
        else "Apply the structured reasoning steps to classify the kinship relation in this image."
    )
    msg = client.messages.create(
        model=model,
        max_tokens=256,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": user_text},
            ],
        }],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw)
    pred = parsed.get("predicted_relation", "")
    if pred not in RELATIONS:
        pred = RELATIONS[0]
    return {
        "predicted_relation": pred,
        "confidence": float(parsed.get("confidence", 0.5)),
        "generation_gap": parsed.get("generation_gap", ""),
        "left_gender": parsed.get("left_gender", ""),
        "right_gender": parsed.get("right_gender", ""),
        "similarity": float(parsed.get("similarity", 0.5)),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def fit_calibration(val_predictions_path: Path) -> dict:
    """
    Fit per-class calibration weights from validation predictions.

    Strategy: compute per-class accuracy on validation. Use the ratio
    (val_accuracy[c] / mean_val_accuracy) as a weight for each class.
    During test-time calibration, multiply raw confidence by the class
    weight (for the predicted class), then re-normalise across relations.

    Also computes empirical label priors from validation to detect
    systematic over/under-prediction.
    """
    records = []
    with val_predictions_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            records.append(row)

    # Per-class accuracy
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    per_class_predicted = defaultdict(int)  # how often model predicted this class
    for r in records:
        gt = r["relation"]
        pred = r["predicted_relation"]
        per_class_total[gt] += 1
        per_class_predicted[pred] += 1
        if gt == pred:
            per_class_correct[gt] += 1

    accuracy = {
        rel: per_class_correct[rel] / per_class_total[rel]
        if per_class_total[rel] > 0 else 0.0
        for rel in RELATIONS
    }
    mean_acc = sum(accuracy.values()) / len(RELATIONS)
    if mean_acc == 0:
        mean_acc = 1e-6

    # Calibration weight: boost classes the model under-predicts; penalise over-predicted
    total_pairs = len(records)
    expected_per_class = total_pairs / len(RELATIONS)
    pred_rate = {
        rel: per_class_predicted.get(rel, 0) / total_pairs for rel in RELATIONS
    }
    expected_rate = 1.0 / len(RELATIONS)

    # Combine accuracy signal + prediction rate signal
    weights: Dict[str, float] = {}
    for rel in RELATIONS:
        acc_weight = accuracy[rel] / mean_acc  # >1 if class is easier, <1 if harder
        # Under-predicted classes get a boost
        rate_weight = expected_rate / max(pred_rate[rel], 0.001)
        # Blend: 60% accuracy signal, 40% prediction-rate signal
        weights[rel] = 0.6 * acc_weight + 0.4 * rate_weight

    return {
        "per_class_accuracy": accuracy,
        "mean_accuracy": mean_acc,
        "per_class_pred_rate": pred_rate,
        "calibration_weights": weights,
    }


def apply_calibration(
    predictions: List[dict],
    calibration: dict,
) -> List[dict]:
    """
    Adjust predictions using calibration weights.
    For each prediction, reweight the confidence and possibly flip the
    predicted relation if calibration strongly favours a different class.
    """
    weights = calibration["calibration_weights"]
    calibrated = []
    for rec in predictions:
        pred = rec["predicted_relation"]
        raw_conf = rec["confidence"]
        gen_gap = rec.get("generation_gap", "")
        left_g = rec.get("left_gender", "")
        right_g = rec.get("right_gender", "")

        # Get candidate relations in the same generation-gap group
        group = None
        for g, rels in GEN_GAP_GROUPS.items():
            if pred in rels:
                group = g
                break
        if group is None:
            group = gen_gap  # fallback to model's own assessment

        # If model provided generation gap and gender, use decision table
        # to get the "correct" relation for that assessment, then compare
        # with calibration weights
        dt_key_older = (group, left_g if left_g else "male", right_g if right_g else "male")
        dt_pred = DECISION_TABLE.get(dt_key_older, pred)

        # Score current pred and decision-table pred after calibration
        score_pred = raw_conf * weights.get(pred, 1.0)
        score_dt = raw_conf * weights.get(dt_pred, 1.0) * 0.9  # slight discount

        if gen_gap and left_g and right_g and score_dt > score_pred * 1.2:
            # Calibration strongly favours decision-table prediction
            final_pred = dt_pred
            final_conf = min(raw_conf * weights.get(dt_pred, 1.0), 1.0)
        else:
            final_pred = pred
            final_conf = min(raw_conf * weights.get(pred, 1.0), 1.0)

        calibrated.append({**rec, "predicted_relation": final_pred, "confidence": final_conf})
    return calibrated


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(records: List[dict], model_label: str) -> dict:
    confusion = {r: {p: 0 for p in RELATIONS} for r in RELATIONS}
    for rec in records:
        gt = rec["relation"]
        pred = rec.get("predicted_relation", RELATIONS[0])
        if gt in RELATIONS and pred in RELATIONS:
            confusion[gt][pred] += 1

    per_relation = {}
    macro_rows = []
    for rel in RELATIONS:
        rel_recs = [r for r in records if r["relation"] == rel]
        tp = confusion[rel][rel]
        fp = sum(confusion[o][rel] for o in RELATIONS if o != rel)
        fn = sum(confusion[rel][o] for o in RELATIONS if o != rel)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
        acc = tp / len(rel_recs) if rel_recs else 0.0
        per_relation[rel] = {
            "count": len(rel_recs), "correct": tp,
            "accuracy": acc, "precision": prec, "recall": rec_, "f1": f1,
        }
        macro_rows.append((prec, rec_, f1))

    total = len(records)
    correct = sum(1 for r in records if r.get("predicted_relation") == r["relation"])
    confs = [float(r.get("confidence", 0.5)) for r in records]
    c_confs = [float(r.get("confidence", 0.5)) for r in records if r.get("predicted_relation") == r["relation"]]
    w_confs = [float(r.get("confidence", 0.5)) for r in records if r.get("predicted_relation") != r["relation"]]

    return {
        "model": model_label,
        "total_pairs": total,
        "overall_accuracy": correct / total if total else 0.0,
        "macro_precision": sum(r[0] for r in macro_rows) / len(macro_rows),
        "macro_recall": sum(r[1] for r in macro_rows) / len(macro_rows),
        "macro_f1": sum(r[2] for r in macro_rows) / len(macro_rows),
        "mean_confidence": sum(confs) / len(confs) if confs else 0.0,
        "mean_confidence_correct": sum(c_confs) / len(c_confs) if c_confs else 0.0,
        "mean_confidence_incorrect": sum(w_confs) / len(w_confs) if w_confs else 0.0,
        "per_relation": per_relation,
        "confusion_matrix": confusion,
    }


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "sample_id", "relation", "relation_name",
    "predicted_relation", "predicted_relation_name",
    "confidence", "correct",
    "generation_gap", "left_gender", "right_gender", "similarity",
    "p1_rel", "p2_rel",
]


def write_predictions_csv(path: Path, records: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({
                "sample_id": r.get("sample_id", ""),
                "relation": r["relation"],
                "relation_name": RELATION_NAMES[r["relation"]],
                "predicted_relation": r.get("predicted_relation", ""),
                "predicted_relation_name": RELATION_NAMES.get(r.get("predicted_relation", ""), ""),
                "confidence": r.get("confidence", 0.5),
                "correct": int(r.get("predicted_relation") == r["relation"]),
                "generation_gap": r.get("generation_gap", ""),
                "left_gender": r.get("left_gender", ""),
                "right_gender": r.get("right_gender", ""),
                "similarity": r.get("similarity", ""),
                "p1_rel": r.get("p1_rel", ""),
                "p2_rel": r.get("p2_rel", ""),
            })


def load_predictions_csv(path: Path) -> List[dict]:
    records = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            records.append(row)
    return records


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_sample(args):
    """Sample validation and test splits, create pair_sheets."""
    ADAPTED_DIR.mkdir(parents=True, exist_ok=True)
    val_dir = ADAPTED_DIR / "val"
    test_dir = ADAPTED_DIR / "test"
    for d in [val_dir, test_dir, val_dir / "pair_sheets", test_dir / "pair_sheets"]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading used pairs from baseline …")
    baseline_used = load_used_pairs()
    print(f"  Excluding {len(baseline_used)} pairs from baseline test set")

    print("Loading available pairs …")
    by_rel = load_available_pairs(exclude=baseline_used)
    for rel in RELATIONS:
        print(f"  {rel}: {len(by_rel[rel])} available")

    # Sample validation split
    print(f"\nSampling validation split ({args.val_per_rel} per relation) …")
    val_pairs, val_used = sample_split(by_rel, args.val_per_rel, SEED, exclude_pairs=baseline_used)
    print(f"  Validation: {len(val_pairs)} pairs")

    # Sample test split (exclude validation)
    all_excluded = baseline_used | val_used
    print(f"\nSampling test split ({args.test_per_rel} per relation) …")
    test_pairs, _ = sample_split(by_rel, args.test_per_rel, SEED + 100, exclude_pairs=all_excluded)
    print(f"  Test: {len(test_pairs)} pairs")

    # Save manifests
    val_manifest = [{k: v for k, v in r.items()} for r in val_pairs]
    (val_dir / "manifest.json").write_text(json.dumps(val_manifest, indent=2), encoding="utf-8")
    test_manifest = [{k: v for k, v in r.items()} for r in test_pairs]
    (test_dir / "manifest.json").write_text(json.dumps(test_manifest, indent=2), encoding="utf-8")

    # Create pair_sheets
    print("\nCreating validation pair_sheets …")
    for rec in val_pairs:
        p1 = str(FIW_ROOT / rec["p1_rel"])
        p2 = str(FIW_ROOT / rec["p2_rel"])
        out = val_dir / "pair_sheets" / f"{rec['sample_id']}.jpg"
        make_pair_sheet(p1, p2, out)
    print(f"  Saved {len(val_pairs)} pair_sheets to {val_dir}/pair_sheets/")

    print("Creating test pair_sheets …")
    for rec in test_pairs:
        p1 = str(FIW_ROOT / rec["p1_rel"])
        p2 = str(FIW_ROOT / rec["p2_rel"])
        out = test_dir / "pair_sheets" / f"{rec['sample_id']}.jpg"
        make_pair_sheet(p1, p2, out)
    print(f"  Saved {len(test_pairs)} pair_sheets to {test_dir}/pair_sheets/")

    # Save config + prompts
    config = {
        "seed": SEED,
        "val_per_rel": args.val_per_rel,
        "test_per_rel": args.test_per_rel,
        "total_val_pairs": len(val_pairs),
        "total_test_pairs": len(test_pairs),
        "baseline_excluded": len(baseline_used),
        "relations": RELATIONS,
    }
    (ADAPTED_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    prompts_dir = ADAPTED_DIR / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    (prompts_dir / "zero_shot.txt").write_text(ZERO_SHOT_SYSTEM, encoding="utf-8")
    (prompts_dir / "structured.txt").write_text(STRUCTURED_SYSTEM, encoding="utf-8")
    (prompts_dir / "few_shot.txt").write_text(FEW_SHOT_SYSTEM, encoding="utf-8")

    print(f"\nConfig saved to {ADAPTED_DIR}/config.json")
    print(f"Prompts saved to {ADAPTED_DIR}/prompts/")
    print("\nDone. Next steps:")
    print("  python run_claude_vlm_adapted.py evaluate --mode structured --split val")
    print("  python run_claude_vlm_adapted.py evaluate --mode few_shot --split test")
    print("  python run_claude_vlm_adapted.py calibrate")


def cmd_evaluate(args):
    """Run Claude API evaluation for a given mode and split."""
    if not HAS_ANTHROPIC:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return

    split_dir = ADAPTED_DIR / args.split
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run 'sample' first.")
        return

    records = json.loads(manifest_path.read_text())
    system_prompt = get_system_prompt(args.mode)
    client = anthropic.Anthropic()
    model = args.model

    out_csv = split_dir / f"predictions_{args.mode}.csv"
    out_metrics = split_dir / f"metrics_{args.mode}.json"

    print(f"Evaluating {len(records)} pairs — mode={args.mode}, split={args.split}")

    results = []
    for i, rec in enumerate(records, start=1):
        p1 = str(FIW_ROOT / rec["p1_rel"])
        p2 = str(FIW_ROOT / rec["p2_rel"])
        img_bytes = make_pair_sheet_bytes(p1, p2)
        try:
            result = classify_via_api(client, model, img_bytes, system_prompt, args.mode)
        except Exception as exc:
            print(f"  [{i}/{len(records)}] ERROR {rec['sample_id']}: {exc}")
            result = {"predicted_relation": RELATIONS[0], "confidence": 0.0,
                      "generation_gap": "", "left_gender": "", "right_gender": "", "similarity": 0.5}

        rec.update(result)
        correct = rec["predicted_relation"] == rec["relation"]
        print(
            f"  [{i:3d}/{len(records)}] {rec['sample_id']} "
            f"gt={rec['relation']:5s} pred={rec['predicted_relation']:5s} "
            f"gap={rec.get('generation_gap','?'):4s} conf={rec['confidence']:.2f} "
            f"{'✓' if correct else '✗'}"
        )
        results.append(rec)
        if args.delay > 0 and i < len(records):
            time.sleep(args.delay)

    write_predictions_csv(out_csv, results)
    metrics = compute_metrics(results, f"claude-sonnet-4-6/{args.mode}")
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nOverall accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"Saved: {out_csv}")


def cmd_calibrate(args):
    """Fit calibration on val/few_shot predictions and apply to test/few_shot."""
    val_csv = ADAPTED_DIR / "val" / "predictions_few_shot.csv"
    test_csv = ADAPTED_DIR / "test" / "predictions_few_shot.csv"

    if not val_csv.exists():
        print(f"ERROR: {val_csv} not found. Run evaluate --mode few_shot --split val first.")
        return
    if not test_csv.exists():
        print(f"ERROR: {test_csv} not found. Run evaluate --mode few_shot --split test first.")
        return

    print("Fitting calibration from validation predictions …")
    calibration = fit_calibration(val_csv)
    (ADAPTED_DIR / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    print("  Calibration weights:")
    for rel, w in calibration["calibration_weights"].items():
        acc = calibration["per_class_accuracy"][rel]
        print(f"    {rel:5s}: acc={acc:.2f}, weight={w:.3f}")

    print("\nApplying calibration to test predictions …")
    test_records = load_predictions_csv(test_csv)
    # Convert back to proper types
    for r in test_records:
        r["confidence"] = float(r.get("confidence", 0.5))

    calibrated = apply_calibration(test_records, calibration)
    out_csv = ADAPTED_DIR / "test" / "predictions_calibrated.csv"
    out_metrics = ADAPTED_DIR / "test" / "metrics_calibrated.json"
    write_predictions_csv(out_csv, calibrated)
    metrics = compute_metrics(calibrated, "claude-sonnet-4-6/calibrated")
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nCalibrated accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Calibrated macro F1: {metrics['macro_f1']:.4f}")
    print(f"Saved: {out_csv}")


def cmd_summary(args):
    """Print comparison table across all three conditions."""
    paths = {
        "zero_shot (baseline, 75 pairs)": ROOT / "data/claude_vlm_fiw_150/metrics.json",
        "structured (330 pairs)": ADAPTED_DIR / "test/metrics_structured.json",
        "few_shot (330 pairs)": ADAPTED_DIR / "test/metrics_few_shot.json",
        "calibrated (330 pairs)": ADAPTED_DIR / "test/metrics_calibrated.json",
    }
    print(f"\n{'Mode':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 65)
    for label, path in paths.items():
        if path.exists():
            m = json.loads(path.read_text())
            print(
                f"{label:<35} "
                f"{m['overall_accuracy']:>6.3f} "
                f"{m['macro_precision']:>6.3f} "
                f"{m['macro_recall']:>6.3f} "
                f"{m['macro_f1']:>6.3f}"
            )
        else:
            print(f"{label:<35} {'—':>6} {'—':>6} {'—':>6} {'—':>6}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Claude VLM domain adaptation study — FIW kinship")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="Sample val/test splits and create pair_sheets")
    s.add_argument("--val-per-rel", type=int, default=3)
    s.add_argument("--test-per-rel", type=int, default=30)

    e = sub.add_parser("evaluate", help="Run Claude evaluation via API")
    e.add_argument("--mode", choices=["zero_shot", "structured", "few_shot"], required=True)
    e.add_argument("--split", choices=["val", "test"], required=True)
    e.add_argument("--model", default="claude-sonnet-4-6")
    e.add_argument("--delay", type=float, default=0.3)

    sub.add_parser("calibrate", help="Fit calibration on val and apply to test")
    sub.add_parser("summary", help="Print comparison table")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "sample":
        cmd_sample(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
    elif args.cmd == "calibrate":
        cmd_calibrate(args)
    elif args.cmd == "summary":
        cmd_summary(args)
