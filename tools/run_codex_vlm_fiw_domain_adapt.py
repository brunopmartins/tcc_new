#!/usr/bin/env python3
"""
Run inference-time domain adaptation experiments for the Codex VLM on FIW.

This script keeps the original zero-shot baseline intact and adds a separate
prompt-based adaptation path built from:
  1. few-shot in-context examples
  2. a structured decision prompt
  3. light validation-based calibration

Validation selection uses the official FIW `val-pairs.csv` pool, while held-out
evaluation reuses the zero-shot test manifest (by default, the 750-pair sample
already used for the baseline).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image, ImageDraw, ImageOps


RELATIONS = [
    "bb",
    "fd",
    "fs",
    "md",
    "ms",
    "ss",
    "sibs",
    "gfgd",
    "gfgs",
    "gmgd",
    "gmgs",
]

RELATION_NAMES = {
    "bb": "brother-brother",
    "fd": "father-daughter",
    "fs": "father-son",
    "md": "mother-daughter",
    "ms": "mother-son",
    "ss": "sister-sister",
    "sibs": "brother-sister",
    "gfgd": "grandfather-granddaughter",
    "gfgs": "grandfather-grandson",
    "gmgd": "grandmother-granddaughter",
    "gmgs": "grandmother-grandson",
}

GAP_ENUM = [
    "same_generation",
    "one_generation_apart",
    "two_generation_apart",
    "uncertain",
]

FACE_SIDE_ENUM = ["left", "right", "similar_age", "uncertain"]
GENDER_ENUM = ["male", "female", "uncertain"]

RELATION_GROUP = {
    "bb": "same_generation",
    "ss": "same_generation",
    "sibs": "same_generation",
    "fd": "one_generation_apart",
    "fs": "one_generation_apart",
    "md": "one_generation_apart",
    "ms": "one_generation_apart",
    "gfgd": "two_generation_apart",
    "gfgs": "two_generation_apart",
    "gmgd": "two_generation_apart",
    "gmgs": "two_generation_apart",
}

SEVEN_SHOT_RELATIONS = ["bb", "ss", "sibs", "fd", "ms", "gfgd", "gmgs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new"),
        help="Project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new/data/codex_vlm_fiw_domain_adapt"),
        help="Directory for all artifacts.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new/data/codex_vlm_fiw_1500"),
        help="Existing zero-shot baseline directory used as the held-out test manifest/metrics.",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=None,
        help="Optional explicit held-out test manifest. Defaults to <baseline-dir>/manifest.json.",
    )
    parser.add_argument(
        "--validation-per-relation",
        type=int,
        default=10,
        help="Number of validation query pairs per relation used for prompt selection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of query pair sheets per Codex call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260413,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="Codex model slug.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        help="Codex reasoning effort override.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated pair sheets in the output directory.",
    )
    return parser.parse_args()


def create_pair_sheet(record: dict, destination: Path) -> None:
    img1 = Image.open(record["p1_abs"]).convert("RGB")
    img2 = Image.open(record["p2_abs"]).convert("RGB")

    face_size = (256, 256)
    img1 = ImageOps.fit(img1, face_size)
    img2 = ImageOps.fit(img2, face_size)

    canvas = Image.new("RGB", (face_size[0] * 2 + 30, face_size[1] + 60), "white")
    canvas.paste(img1, (10, 40))
    canvas.paste(img2, (face_size[0] + 20, 40))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"{record['sample_id']} | left face", fill="black")
    draw.text((face_size[0] + 20, 10), "right face", fill="black")
    canvas.save(destination, quality=90)


def compute_metrics(records: List[dict], prediction_key: str = "predicted_relation") -> dict:
    total = len(records)
    correct = sum(1 for record in records if record[prediction_key] == record["relation"])
    per_relation = {}
    confusion: Dict[str, Dict[str, int]] = {
        rel: {pred: 0 for pred in RELATIONS} for rel in RELATIONS
    }

    for relation in RELATIONS:
        rel_records = [record for record in records if record["relation"] == relation]
        rel_total = len(rel_records)
        rel_correct = sum(1 for record in rel_records if record[prediction_key] == relation)
        per_relation[relation] = {
            "count": rel_total,
            "correct": rel_correct,
            "accuracy": rel_correct / rel_total if rel_total else 0.0,
        }

    for record in records:
        confusion[record["relation"]][record[prediction_key]] += 1

    metric_rows = []
    for relation in RELATIONS:
        tp = confusion[relation][relation]
        fp = sum(confusion[other][relation] for other in RELATIONS if other != relation)
        fn = sum(confusion[relation][other] for other in RELATIONS if other != relation)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metric_rows.append((precision, recall, f1))
        per_relation[relation].update(
            {"precision": precision, "recall": recall, "f1": f1}
        )

    confidences = [float(record["confidence"]) for record in records]
    correct_confidences = [
        float(record["confidence"]) for record in records if record[prediction_key] == record["relation"]
    ]
    incorrect_confidences = [
        float(record["confidence"]) for record in records if record[prediction_key] != record["relation"]
    ]

    return {
        "task": "closed_set_relation_classification",
        "prediction_key": prediction_key,
        "total_pairs": total,
        "total_images": total * 2,
        "overall_accuracy": correct / total if total else 0.0,
        "macro_precision": sum(row[0] for row in metric_rows) / len(metric_rows),
        "macro_recall": sum(row[1] for row in metric_rows) / len(metric_rows),
        "macro_f1": sum(row[2] for row in metric_rows) / len(metric_rows),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "mean_confidence_correct": (
            sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0
        ),
        "mean_confidence_incorrect": (
            sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0.0
        ),
        "per_relation": per_relation,
        "confusion_matrix": confusion,
    }


def write_predictions_csv(path: Path, records: List[dict]) -> None:
    fieldnames = [
        "sample_id",
        "relation",
        "relation_name",
        "predicted_relation",
        "predicted_relation_name",
        "calibrated_relation",
        "calibrated_relation_name",
        "runner_up_relation",
        "confidence",
        "generation_gap",
        "older_face_side",
        "left_gender_guess",
        "right_gender_guess",
        "derived_relation",
        "correct_raw",
        "correct_calibrated",
        "p1_rel",
        "p2_rel",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            calibrated = record.get("calibrated_relation", record["predicted_relation"])
            derived = record.get("derived_relation", "")
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "relation": record["relation"],
                    "relation_name": record["relation_name"],
                    "predicted_relation": record["predicted_relation"],
                    "predicted_relation_name": RELATION_NAMES[record["predicted_relation"]],
                    "calibrated_relation": calibrated,
                    "calibrated_relation_name": RELATION_NAMES[calibrated],
                    "runner_up_relation": record.get("runner_up_relation", ""),
                    "confidence": record["confidence"],
                    "generation_gap": record.get("generation_gap", ""),
                    "older_face_side": record.get("older_face_side", ""),
                    "left_gender_guess": record.get("left_gender_guess", ""),
                    "right_gender_guess": record.get("right_gender_guess", ""),
                    "derived_relation": derived,
                    "correct_raw": int(record["predicted_relation"] == record["relation"]),
                    "correct_calibrated": int(calibrated == record["relation"]),
                    "p1_rel": record["p1_rel"],
                    "p2_rel": record["p2_rel"],
                }
            )


def chunked(records: List[dict], batch_size: int) -> List[List[dict]]:
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


def load_baseline_test_manifest(root: Path, manifest_path: Path) -> List[dict]:
    fiw_root = root / "datasets" / "FIW" / "FIDs"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized = []
    for record in records:
        normalized.append(
            {
                "sample_id": record["sample_id"],
                "relation": record["relation"],
                "relation_name": RELATION_NAMES[record["relation"]],
                "p1_rel": record["p1_rel"],
                "p2_rel": record["p2_rel"],
                "p1_abs": str(fiw_root / record["p1_rel"]),
                "p2_abs": str(fiw_root / record["p2_rel"]),
            }
        )
    return normalized


def load_validation_face_pairs(root: Path) -> Dict[str, List[dict]]:
    csv_path = root / "datasets" / "FIW" / "track-I" / "val-pairs.csv"
    fiw_root = root / "datasets" / "FIW" / "FIDs"
    by_relation: Dict[str, List[dict]] = defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relation = (row.get("old_tags") or row.get("ptype") or "").strip()
            if relation == "sib":
                relation = "sibs"
            if relation not in RELATIONS:
                continue

            try:
                face_pairs = ast.literal_eval(row["face_pairs"])
            except (SyntaxError, ValueError):
                continue

            for p1_rel, p2_rel in face_pairs:
                p1_abs = fiw_root / p1_rel
                p2_abs = fiw_root / p2_rel
                if not p1_abs.exists() or not p2_abs.exists():
                    continue
                by_relation[relation].append(
                    {
                        "relation": relation,
                        "relation_name": RELATION_NAMES[relation],
                        "p1_rel": p1_rel,
                        "p2_rel": p2_rel,
                        "p1_abs": str(p1_abs),
                        "p2_abs": str(p2_abs),
                        "source_row_pair": row.get("pp", ""),
                    }
                )
    return by_relation


def select_validation_assets(
    by_relation: Dict[str, List[dict]],
    validation_per_relation: int,
    seed: int,
) -> tuple[Dict[str, dict], List[dict]]:
    rng = random.Random(seed)
    demos: Dict[str, dict] = {}
    validation_records: List[dict] = []
    demo_index = 1
    val_index = 1

    for relation in RELATIONS:
        candidates = list(by_relation[relation])
        rng.shuffle(candidates)
        needed = 1 + validation_per_relation
        if len(candidates) < needed:
            raise ValueError(
                f"Validation pool for relation {relation} has {len(candidates)} examples, "
                f"but {needed} are required."
            )

        demo_record = {**candidates[0], "sample_id": f"D{demo_index:03d}"}
        demos[relation] = demo_record
        demo_index += 1

        for record in candidates[1 : 1 + validation_per_relation]:
            validation_records.append({**record, "sample_id": f"V{val_index:03d}"})
            val_index += 1

    rng.shuffle(validation_records)
    return demos, validation_records


def create_pair_sheets(records: Iterable[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        pair_path = output_dir / f"{record['sample_id']}.jpg"
        record["pair_sheet_path"] = str(pair_path)
        create_pair_sheet(record, pair_path)


def write_schema(path: Path, batch_size: int) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "minItems": batch_size,
                "maxItems": batch_size,
                "items": {
                    "type": "object",
                    "properties": {
                        "predicted_relation": {"type": "string", "enum": RELATIONS},
                        "runner_up_relation": {"type": "string", "enum": RELATIONS},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "generation_gap": {"type": "string", "enum": GAP_ENUM},
                        "older_face_side": {"type": "string", "enum": FACE_SIDE_ENUM},
                        "left_gender_guess": {"type": "string", "enum": GENDER_ENUM},
                        "right_gender_guess": {"type": "string", "enum": GENDER_ENUM},
                    },
                    "required": [
                        "predicted_relation",
                        "runner_up_relation",
                        "confidence",
                        "generation_gap",
                        "older_face_side",
                        "left_gender_guess",
                        "right_gender_guess",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }
    path.write_text(json.dumps(schema), encoding="utf-8")


def relation_glossary() -> str:
    return (
        "Relation codes:\n"
        "- bb = brother-brother\n"
        "- ss = sister-sister\n"
        "- sibs = mixed-gender siblings (brother-sister)\n"
        "- fd = father-daughter\n"
        "- fs = father-son\n"
        "- md = mother-daughter\n"
        "- ms = mother-son\n"
        "- gfgd = grandfather-granddaughter\n"
        "- gfgs = grandfather-grandson\n"
        "- gmgd = grandmother-granddaughter\n"
        "- gmgs = grandmother-grandson\n"
    )


def build_prompt(version: str, demos: List[dict], batch_records: List[dict]) -> str:
    demo_lines = []
    for index, demo in enumerate(demos, start=1):
        demo_lines.append(
            f"- Attachment {index}: labeled example, relation = {demo['relation']} "
            f"({RELATION_NAMES[demo['relation']]})"
        )

    first_query_attachment = len(demos) + 1
    last_query_attachment = len(demos) + len(batch_records)

    shared_header = (
        f"You will receive {len(demos) + len(batch_records)} attached images.\n"
        "The first attachments are labeled few-shot examples. The remaining attachments "
        "are unlabeled FIW query pairs that you must classify independently.\n\n"
        + relation_glossary()
        + "\n"
        "Labeled examples:\n"
        + "\n".join(demo_lines)
        + "\n\n"
        f"Query attachments are {first_query_attachment} through {last_query_attachment}. "
        f"Return a JSON object with a predictions array containing exactly {len(batch_records)} "
        "items in the same order as the query attachments only.\n\n"
        "For each query pair, analyze only the visual evidence in the image and provide:\n"
        "- predicted_relation\n"
        "- runner_up_relation\n"
        "- confidence\n"
        "- generation_gap: same_generation, one_generation_apart, two_generation_apart, or uncertain\n"
        "- older_face_side: left, right, similar_age, or uncertain\n"
        "- left_gender_guess: male, female, or uncertain\n"
        "- right_gender_guess: male, female, or uncertain\n\n"
        "Do not use shell commands. Do not inspect files. Return JSON only.\n"
    )

    if version == "v1":
        return (
            shared_header
            + "\nUse this decision process internally before choosing the final label:\n"
            "1. Estimate whether the pair looks like same-generation siblings, parent-child, or grandparent-grandchild.\n"
            "2. Estimate the likely gender of each face.\n"
            "3. Use facial resemblance cues (eyes, nose, mouth, jawline, face shape) to break ties within the valid relation subset.\n"
            "4. Prefer relation labels that stay consistent with the estimated generation gap and gender pattern unless the visual evidence strongly contradicts them.\n"
        )

    if version == "v2":
        return (
            shared_header
            + "\nUse a structured kinship decision table internally:\n"
            "- If the pair appears same generation: only bb, ss, sibs should be primary candidates.\n"
            "- If the pair appears one generation apart: only fd, fs, md, ms should be primary candidates.\n"
            "- If the pair appears two generations apart: only gfgd, gfgs, gmgd, gmgs should be primary candidates.\n"
            "- Use the likely older face and the genders of older/younger faces to decide among labels inside that generation bucket.\n"
            "- Use fine-grained facial similarity only after narrowing the generation bucket.\n"
            "- If uncertain, still choose the single best label, but keep the auxiliary fields honest.\n"
        )

    raise ValueError(f"Unknown prompt version: {version}")


def run_batch(
    batch_records: List[dict],
    demo_records: List[dict],
    model: str,
    reasoning_effort: str,
    schema_path: Path,
    output_json_path: Path,
    prompt_text: str,
) -> List[dict]:
    command = [
        "codex",
        "exec",
        "-m",
        model,
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "-C",
        "/tmp",
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_json_path),
    ]

    for record in demo_records:
        command.extend(["--image", str(record["pair_sheet_path"])])
    for record in batch_records:
        command.extend(["--image", str(record["pair_sheet_path"])])
    command.append("-")

    result = subprocess.run(
        command,
        input=prompt_text,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Codex batch failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    if not output_json_path.exists():
        raise RuntimeError("Codex finished without writing the expected JSON output file.")

    payload = json.loads(output_json_path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(batch_records):
        raise RuntimeError(f"Unexpected Codex output payload: {payload}")
    return predictions


def prepare_records(records: List[dict]) -> None:
    for record in records:
        record["predicted_relation"] = ""
        record["runner_up_relation"] = ""
        record["confidence"] = 0.0
        record["generation_gap"] = "uncertain"
        record["older_face_side"] = "uncertain"
        record["left_gender_guess"] = "uncertain"
        record["right_gender_guess"] = "uncertain"
        record["derived_relation"] = ""
        record["calibrated_relation"] = ""


def derive_relation_from_cues(record: dict) -> Optional[str]:
    gap = record.get("generation_gap", "uncertain")
    left_gender = record.get("left_gender_guess", "uncertain")
    right_gender = record.get("right_gender_guess", "uncertain")
    older_side = record.get("older_face_side", "uncertain")

    if gap == "same_generation":
        if left_gender == right_gender == "male":
            return "bb"
        if left_gender == right_gender == "female":
            return "ss"
        if {left_gender, right_gender} == {"male", "female"}:
            return "sibs"
        return None

    if gap not in {"one_generation_apart", "two_generation_apart"}:
        return None
    if older_side not in {"left", "right"}:
        return None
    if left_gender == "uncertain" or right_gender == "uncertain":
        return None

    older_gender = left_gender if older_side == "left" else right_gender
    younger_gender = right_gender if older_side == "left" else left_gender

    if gap == "one_generation_apart":
        mapping = {
            ("male", "female"): "fd",
            ("male", "male"): "fs",
            ("female", "female"): "md",
            ("female", "male"): "ms",
        }
        return mapping.get((older_gender, younger_gender))

    mapping = {
        ("male", "female"): "gfgd",
        ("male", "male"): "gfgs",
        ("female", "female"): "gmgd",
        ("female", "male"): "gmgs",
    }
    return mapping.get((older_gender, younger_gender))


def apply_cue_derivation(records: List[dict]) -> None:
    for record in records:
        record["derived_relation"] = derive_relation_from_cues(record) or ""


def apply_calibration_policy(records: List[dict], policy_name: str, threshold: float) -> List[dict]:
    calibrated = []
    for original in records:
        record = dict(original)
        predicted = record["predicted_relation"]
        runner_up = record.get("runner_up_relation", "")
        derived = record.get("derived_relation", "")
        gap = record.get("generation_gap", "uncertain")
        expected_group = gap if gap in RELATION_GROUP.values() else None

        calibrated_label = predicted

        if policy_name != "none" and expected_group and RELATION_GROUP[predicted] != expected_group:
            if runner_up in RELATIONS and RELATION_GROUP[runner_up] == expected_group:
                calibrated_label = runner_up
            elif derived and RELATION_GROUP[derived] == expected_group:
                calibrated_label = derived

        if (
            policy_name == "gap_plus_low_conf"
            and derived
            and calibrated_label == predicted
            and predicted != derived
            and float(record["confidence"]) < threshold
        ):
            calibrated_label = derived

        record["calibrated_relation"] = calibrated_label
        calibrated.append(record)
    return calibrated


def summarize_config_result(metrics: dict) -> dict:
    return {
        "overall_accuracy": metrics["overall_accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
    }


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_manifest(path: Path, records: List[dict]) -> None:
    slim = []
    for record in records:
        slim.append(
            {
                "sample_id": record["sample_id"],
                "relation": record["relation"],
                "relation_name": record["relation_name"],
                "p1_rel": record["p1_rel"],
                "p2_rel": record["p2_rel"],
            }
        )
    write_json(path, slim)


def run_config(
    records: List[dict],
    demo_records: List[dict],
    config_name: str,
    prompt_version: str,
    model: str,
    reasoning_effort: str,
    batch_size: int,
    output_dir: Path,
) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "config.json",
        {
            "config_name": config_name,
            "prompt_version": prompt_version,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "batch_size": batch_size,
            "demo_relations": [record["relation"] for record in demo_records],
            "demo_sample_ids": [record["sample_id"] for record in demo_records],
        },
    )
    save_manifest(output_dir / "manifest.json", records)
    save_manifest(output_dir / "demo_manifest.json", demo_records)

    prompt_text = build_prompt(prompt_version, demo_records, records[:batch_size])
    (output_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")

    working_records = [dict(record) for record in records]
    prepare_records(working_records)
    record_by_id = {record["sample_id"]: record for record in working_records}

    batches = chunked(working_records, batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        schema_path = output_dir / f"schema_{batch_index:02d}.json"
        batch_output_path = output_dir / f"batch_{batch_index:02d}.json"
        write_schema(schema_path, len(batch))
        prompt_text = build_prompt(prompt_version, demo_records, batch)
        predictions = None
        if batch_output_path.exists():
            try:
                payload = json.loads(batch_output_path.read_text(encoding="utf-8"))
                candidate_predictions = payload.get("predictions")
                if isinstance(candidate_predictions, list) and len(candidate_predictions) == len(batch):
                    predictions = candidate_predictions
                    print(
                        f"[{config_name}] reusing batch {batch_index}/{len(batches)} "
                        f"from {batch_output_path.name}."
                    )
            except json.JSONDecodeError:
                predictions = None

        if predictions is None:
            predictions = run_batch(
                batch_records=batch,
                demo_records=demo_records,
                model=model,
                reasoning_effort=reasoning_effort,
                schema_path=schema_path,
                output_json_path=batch_output_path,
                prompt_text=prompt_text,
            )
        for record, prediction in zip(batch, predictions):
            normalized = record_by_id[record["sample_id"]]
            normalized["predicted_relation"] = prediction["predicted_relation"]
            normalized["runner_up_relation"] = prediction["runner_up_relation"]
            normalized["confidence"] = float(prediction["confidence"])
            normalized["generation_gap"] = prediction["generation_gap"]
            normalized["older_face_side"] = prediction["older_face_side"]
            normalized["left_gender_guess"] = prediction["left_gender_guess"]
            normalized["right_gender_guess"] = prediction["right_gender_guess"]
        print(
            f"[{config_name}] completed batch {batch_index}/{len(batches)} "
            f"with {len(batch)} query pairs."
        )

    ordered_records = [record_by_id[record["sample_id"]] for record in records]
    apply_cue_derivation(ordered_records)
    for record in ordered_records:
        record["calibrated_relation"] = record["predicted_relation"]

    metrics = compute_metrics(ordered_records, prediction_key="predicted_relation")
    write_json(output_dir / "metrics_raw.json", metrics)
    write_predictions_csv(output_dir / "predictions.csv", ordered_records)
    return ordered_records


def main() -> None:
    args = parse_args()
    if args.test_manifest is None:
        args.test_manifest = args.baseline_dir / "manifest.json"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics_path = args.baseline_dir / "metrics.json"
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    test_records = load_baseline_test_manifest(args.root, args.test_manifest)

    validation_by_relation = load_validation_face_pairs(args.root)
    demos_by_relation, validation_records = select_validation_assets(
        validation_by_relation,
        validation_per_relation=args.validation_per_relation,
        seed=args.seed,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="codex_vlm_fiw_domain_adapt_"))
    pair_sheet_root = temp_dir / "pair_sheets"
    validation_sheet_dir = pair_sheet_root / "validation"
    demo_sheet_dir = pair_sheet_root / "demos"
    test_sheet_dir = pair_sheet_root / "test"

    try:
        create_pair_sheets(demos_by_relation.values(), demo_sheet_dir)
        create_pair_sheets(validation_records, validation_sheet_dir)
        create_pair_sheets(test_records, test_sheet_dir)

        all_demo_records = [demos_by_relation[relation] for relation in RELATIONS]
        seven_shot_demos = [demos_by_relation[relation] for relation in SEVEN_SHOT_RELATIONS]

        write_json(
            args.output_dir / "selection_setup.json",
            {
                "seed": args.seed,
                "validation_per_relation": args.validation_per_relation,
                "seven_shot_relations": SEVEN_SHOT_RELATIONS,
                "baseline_dir": str(args.baseline_dir),
                "test_manifest": str(args.test_manifest),
            },
        )
        save_manifest(args.output_dir / "validation_manifest.json", validation_records)
        save_manifest(args.output_dir / "demo_manifest_full.json", all_demo_records)
        save_manifest(args.output_dir / "demo_manifest_seven_shot.json", seven_shot_demos)

        candidate_configs = [
            {"config_name": "seven_shot_v1", "prompt_version": "v1", "demos": seven_shot_demos},
            {"config_name": "seven_shot_v2", "prompt_version": "v2", "demos": seven_shot_demos},
            {"config_name": "eleven_shot_v1", "prompt_version": "v1", "demos": all_demo_records},
            {"config_name": "eleven_shot_v2", "prompt_version": "v2", "demos": all_demo_records},
        ]

        validation_results = []
        validation_records_by_config = {}
        for config in candidate_configs:
            config_dir = args.output_dir / "validation" / config["config_name"]
            result_records = run_config(
                records=validation_records,
                demo_records=config["demos"],
                config_name=config["config_name"],
                prompt_version=config["prompt_version"],
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                batch_size=args.batch_size,
                output_dir=config_dir,
            )
            metrics = compute_metrics(result_records, prediction_key="predicted_relation")
            validation_results.append(
                {
                    "config_name": config["config_name"],
                    "prompt_version": config["prompt_version"],
                    "shots": len(config["demos"]),
                    **summarize_config_result(metrics),
                }
            )
            validation_records_by_config[config["config_name"]] = result_records

        validation_results.sort(
            key=lambda row: (row["macro_f1"], row["overall_accuracy"], row["macro_recall"]),
            reverse=True,
        )
        best_config_name = validation_results[0]["config_name"]
        best_config = next(cfg for cfg in candidate_configs if cfg["config_name"] == best_config_name)
        best_validation_records = validation_records_by_config[best_config_name]

        calibration_candidates = [
            {"policy_name": "none", "threshold": 0.0},
            {"policy_name": "gap_only", "threshold": 0.0},
            {"policy_name": "gap_plus_low_conf", "threshold": 0.55},
            {"policy_name": "gap_plus_low_conf", "threshold": 0.65},
            {"policy_name": "gap_plus_low_conf", "threshold": 0.75},
        ]

        calibration_results = []
        best_calibration = None
        best_calibration_key = None
        for candidate in calibration_candidates:
            calibrated_validation = apply_calibration_policy(
                best_validation_records,
                policy_name=candidate["policy_name"],
                threshold=candidate["threshold"],
            )
            metrics = compute_metrics(calibrated_validation, prediction_key="calibrated_relation")
            summary = {
                "policy_name": candidate["policy_name"],
                "threshold": candidate["threshold"],
                **summarize_config_result(metrics),
            }
            calibration_results.append(summary)
            key = (summary["macro_f1"], summary["overall_accuracy"], summary["macro_recall"])
            if best_calibration_key is None or key > best_calibration_key:
                best_calibration_key = key
                best_calibration = candidate

        write_json(args.output_dir / "validation_results.json", validation_results)
        write_json(args.output_dir / "calibration_results.json", calibration_results)
        write_json(
            args.output_dir / "selected_configuration.json",
            {
                "best_prompt_config": best_config_name,
                "best_prompt_prompt_version": best_config["prompt_version"],
                "best_prompt_shots": len(best_config["demos"]),
                "best_calibration": best_calibration,
                "baseline_metrics_summary": summarize_config_result(baseline_metrics),
            },
        )

        test_run_dir = args.output_dir / "test" / best_config_name
        test_result_records = run_config(
            records=test_records,
            demo_records=best_config["demos"],
            config_name=best_config_name,
            prompt_version=best_config["prompt_version"],
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            batch_size=args.batch_size,
            output_dir=test_run_dir,
        )

        raw_test_metrics = compute_metrics(test_result_records, prediction_key="predicted_relation")
        calibrated_test_records = apply_calibration_policy(
            test_result_records,
            policy_name=best_calibration["policy_name"],
            threshold=best_calibration["threshold"],
        )
        calibrated_test_metrics = compute_metrics(
            calibrated_test_records, prediction_key="calibrated_relation"
        )

        write_json(test_run_dir / "metrics_raw.json", raw_test_metrics)
        write_json(test_run_dir / "metrics_calibrated.json", calibrated_test_metrics)
        write_predictions_csv(test_run_dir / "predictions.csv", calibrated_test_records)

        comparison_summary = {
            "baseline_zero_shot": summarize_config_result(baseline_metrics),
            "adapted_prompt_raw": summarize_config_result(raw_test_metrics),
            "adapted_prompt_calibrated": summarize_config_result(calibrated_test_metrics),
            "selected_prompt_config": best_config_name,
            "selected_calibration": best_calibration,
        }
        write_json(args.output_dir / "comparison_summary.json", comparison_summary)

        print("\nDomain adaptation run complete")
        print(f"Output directory: {args.output_dir}")
        print(f"Best prompt config: {best_config_name}")
        print(f"Best calibration: {best_calibration}")
        print(
            "Test macro F1 | baseline: "
            f"{baseline_metrics['macro_f1']:.4f} | "
            f"adapted raw: {raw_test_metrics['macro_f1']:.4f} | "
            f"adapted calibrated: {calibrated_test_metrics['macro_f1']:.4f}"
        )
    finally:
        if args.keep_temp:
            kept_dir = args.output_dir / "pair_sheets"
            if kept_dir.exists():
                shutil.rmtree(kept_dir)
            shutil.copytree(pair_sheet_root, kept_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
