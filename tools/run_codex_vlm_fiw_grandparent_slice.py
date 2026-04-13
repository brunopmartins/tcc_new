#!/usr/bin/env python3
"""
Run a focused grandparent-only Codex VLM comparison on FIW.

This script reuses the existing 750-pair held-out FIW manifest, samples a small
deterministic slice with 3 pairs for each grandparent relation, and compares:
  1. the original zero-shot prompt
  2. the selected prompt-adapted configuration (seven-shot v1)

The goal is diagnostic, not to replace the full held-out evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

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

GRANDPARENT_RELATIONS = ["gfgd", "gfgs", "gmgd", "gmgs"]

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

GAP_ENUM = [
    "same_generation",
    "one_generation_apart",
    "two_generation_apart",
    "uncertain",
]
FACE_SIDE_ENUM = ["left", "right", "similar_age", "uncertain"]
GENDER_ENUM = ["male", "female", "uncertain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new"),
        help="Project root.",
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new/data/codex_vlm_fiw_1500/manifest.json"),
        help="Held-out manifest used by the main zero-shot baseline.",
    )
    parser.add_argument(
        "--adaptation-dir",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new/data/codex_vlm_fiw_domain_adapt_1500"),
        help="Directory with the selected adapted prompt configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new/data/codex_vlm_fiw_grandparent_slice_12"),
        help="Directory for focused-slice artifacts.",
    )
    parser.add_argument(
        "--pairs-per-relation",
        type=int,
        default=3,
        help="Number of held-out pairs sampled for each grandparent relation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Number of query pairs per Codex call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260413,
        help="Sampling seed for the focused slice.",
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
        help="Keep generated pair sheets after the run.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_records_from_manifest(root: Path, manifest_path: Path) -> List[dict]:
    fiw_root = root / "datasets" / "FIW" / "FIDs"
    raw_records = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for record in raw_records:
        records.append(
            {
                "sample_id": record["sample_id"],
                "relation": record["relation"],
                "relation_name": record.get("relation_name", RELATION_NAMES[record["relation"]]),
                "p1_rel": record["p1_rel"],
                "p2_rel": record["p2_rel"],
                "p1_abs": str(fiw_root / record["p1_rel"]),
                "p2_abs": str(fiw_root / record["p2_rel"]),
            }
        )
    return records


def select_grandparent_slice(records: List[dict], pairs_per_relation: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    selected = []
    chosen_ids: List[str] = []

    for relation in GRANDPARENT_RELATIONS:
        candidates = [record for record in records if record["relation"] == relation]
        if len(candidates) < pairs_per_relation:
            raise ValueError(
                f"Relation {relation} only has {len(candidates)} records in the baseline manifest."
            )
        rng.shuffle(candidates)
        picks = [dict(record) for record in candidates[:pairs_per_relation]]
        selected.extend(picks)
        chosen_ids.extend(record["sample_id"] for record in picks)

    rng.shuffle(selected)
    for index, record in enumerate(selected, start=1):
        record["slice_order"] = index

    print(
        "Selected grandparent slice sample IDs:",
        ", ".join(chosen_ids),
    )
    return selected


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


def create_pair_sheets(records: Iterable[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        pair_path = output_dir / f"{record['sample_id']}.jpg"
        create_pair_sheet(record, pair_path)
        record["pair_sheet_path"] = str(pair_path)


def write_zero_shot_schema(path: Path, batch_size: int) -> None:
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
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["predicted_relation", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }
    write_json(path, schema)


def write_structured_schema(path: Path, batch_size: int) -> None:
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
    write_json(path, schema)


def chunked(records: List[dict], batch_size: int) -> List[List[dict]]:
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


def run_codex_batch(
    image_paths: List[Path],
    prompt_text: str,
    model: str,
    reasoning_effort: str,
    schema_path: Path,
    output_json_path: Path,
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

    for image_path in image_paths:
        command.extend(["--image", str(image_path)])
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
    if not isinstance(predictions, list):
        raise RuntimeError(f"Unexpected Codex output payload: {payload}")
    return predictions


def build_zero_shot_prompt(batch_size: int) -> str:
    return (
        f"You will receive {batch_size} attached images. "
        "Each attached image is one FIW kinship pair sheet containing the left and right face "
        "for a single positive pair. "
        "Classify each attached pair independently using only the visual evidence in that "
        "attached image. Do not run shell commands. Do not inspect files. "
        f"Return a JSON object with a predictions array of exactly {batch_size} items, "
        "in the same order as the attachments. "
        "For each item, choose exactly one label from: "
        + ", ".join(RELATIONS)
        + "."
    )


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


def build_structured_prompt(demos: List[dict], batch_records: List[dict]) -> str:
    demo_lines = []
    for index, demo in enumerate(demos, start=1):
        demo_lines.append(
            f"- Attachment {index}: labeled example, relation = {demo['relation']} "
            f"({RELATION_NAMES[demo['relation']]})"
        )

    first_query_attachment = len(demos) + 1
    last_query_attachment = len(demos) + len(batch_records)

    return (
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
        "\nUse this decision process internally before choosing the final label:\n"
        "1. Estimate whether the pair looks like same-generation siblings, parent-child, or grandparent-grandchild.\n"
        "2. Estimate the likely gender of each face.\n"
        "3. Use facial resemblance cues (eyes, nose, mouth, jawline, face shape) to break ties within the valid relation subset.\n"
        "4. Prefer relation labels that stay consistent with the estimated generation gap and gender pattern unless the visual evidence strongly contradicts them.\n"
    )


def derive_relation_from_cues(record: dict) -> str:
    gap = record.get("generation_gap", "uncertain")
    left_gender = record.get("left_gender_guess", "uncertain")
    right_gender = record.get("right_gender_guess", "uncertain")
    older_side = record.get("older_face_side", "uncertain")

    if gap not in {"one_generation_apart", "two_generation_apart"}:
        return ""
    if older_side not in {"left", "right"}:
        return ""
    if left_gender == "uncertain" or right_gender == "uncertain":
        return ""

    older_gender = left_gender if older_side == "left" else right_gender
    younger_gender = right_gender if older_side == "left" else left_gender

    if gap == "one_generation_apart":
        mapping = {
            ("male", "female"): "fd",
            ("male", "male"): "fs",
            ("female", "female"): "md",
            ("female", "male"): "ms",
        }
        return mapping.get((older_gender, younger_gender), "")

    mapping = {
        ("male", "female"): "gfgd",
        ("male", "male"): "gfgs",
        ("female", "female"): "gmgd",
        ("female", "male"): "gmgs",
    }
    return mapping.get((older_gender, younger_gender), "")


def run_zero_shot(
    records: List[dict],
    model: str,
    reasoning_effort: str,
    batch_size: int,
    output_dir: Path,
) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(output_dir / "manifest.json", records)
    write_json(
        output_dir / "config.json",
        {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "batch_size": batch_size,
            "mode": "zero_shot",
        },
    )

    working_records = [dict(record) for record in records]
    batches = chunked(working_records, batch_size)

    for batch_index, batch in enumerate(batches, start=1):
        prompt_text = build_zero_shot_prompt(len(batch))
        schema_path = output_dir / f"schema_{batch_index:02d}.json"
        batch_output_path = output_dir / f"batch_{batch_index:02d}.json"
        write_zero_shot_schema(schema_path, len(batch))
        (output_dir / f"prompt_{batch_index:02d}.txt").write_text(prompt_text, encoding="utf-8")

        predictions = None
        if batch_output_path.exists():
            try:
                payload = json.loads(batch_output_path.read_text(encoding="utf-8"))
                candidate_predictions = payload.get("predictions")
                if isinstance(candidate_predictions, list) and len(candidate_predictions) == len(batch):
                    predictions = candidate_predictions
            except json.JSONDecodeError:
                predictions = None

        if predictions is None:
            predictions = run_codex_batch(
                image_paths=[Path(record["pair_sheet_path"]) for record in batch],
                prompt_text=prompt_text,
                model=model,
                reasoning_effort=reasoning_effort,
                schema_path=schema_path,
                output_json_path=batch_output_path,
            )

        for record, prediction in zip(batch, predictions):
            record["predicted_relation"] = prediction["predicted_relation"]
            record["confidence"] = float(prediction["confidence"])

    metrics = compute_slice_metrics(working_records)
    write_json(output_dir / "metrics.json", metrics)
    write_zero_shot_predictions_csv(output_dir / "predictions.csv", working_records)
    return working_records


def run_adapted(
    records: List[dict],
    demo_records: List[dict],
    model: str,
    reasoning_effort: str,
    batch_size: int,
    output_dir: Path,
) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(output_dir / "manifest.json", records)
    save_manifest(output_dir / "demo_manifest.json", demo_records)
    write_json(
        output_dir / "config.json",
        {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "batch_size": batch_size,
            "mode": "seven_shot_v1",
            "prompt_version": "v1",
            "demo_relations": [record["relation"] for record in demo_records],
            "demo_sample_ids": [record["sample_id"] for record in demo_records],
        },
    )

    working_records = [dict(record) for record in records]
    for record in working_records:
        record["predicted_relation"] = ""
        record["runner_up_relation"] = ""
        record["confidence"] = 0.0
        record["generation_gap"] = "uncertain"
        record["older_face_side"] = "uncertain"
        record["left_gender_guess"] = "uncertain"
        record["right_gender_guess"] = "uncertain"
        record["derived_relation"] = ""

    batches = chunked(working_records, batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        prompt_text = build_structured_prompt(demo_records, batch)
        schema_path = output_dir / f"schema_{batch_index:02d}.json"
        batch_output_path = output_dir / f"batch_{batch_index:02d}.json"
        write_structured_schema(schema_path, len(batch))
        (output_dir / f"prompt_{batch_index:02d}.txt").write_text(prompt_text, encoding="utf-8")

        predictions = None
        if batch_output_path.exists():
            try:
                payload = json.loads(batch_output_path.read_text(encoding="utf-8"))
                candidate_predictions = payload.get("predictions")
                if isinstance(candidate_predictions, list) and len(candidate_predictions) == len(batch):
                    predictions = candidate_predictions
            except json.JSONDecodeError:
                predictions = None

        if predictions is None:
            predictions = run_codex_batch(
                image_paths=[Path(record["pair_sheet_path"]) for record in demo_records]
                + [Path(record["pair_sheet_path"]) for record in batch],
                prompt_text=prompt_text,
                model=model,
                reasoning_effort=reasoning_effort,
                schema_path=schema_path,
                output_json_path=batch_output_path,
            )

        for record, prediction in zip(batch, predictions):
            record["predicted_relation"] = prediction["predicted_relation"]
            record["runner_up_relation"] = prediction["runner_up_relation"]
            record["confidence"] = float(prediction["confidence"])
            record["generation_gap"] = prediction["generation_gap"]
            record["older_face_side"] = prediction["older_face_side"]
            record["left_gender_guess"] = prediction["left_gender_guess"]
            record["right_gender_guess"] = prediction["right_gender_guess"]
            record["derived_relation"] = derive_relation_from_cues(record)

    metrics = compute_slice_metrics(working_records)
    write_json(output_dir / "metrics.json", metrics)
    write_adapted_predictions_csv(output_dir / "predictions.csv", working_records)
    return working_records


def compute_slice_metrics(records: List[dict]) -> dict:
    total = len(records)
    exact_correct = sum(1 for record in records if record["predicted_relation"] == record["relation"])
    bucket_correct = sum(
        1 for record in records if record["predicted_relation"] in GRANDPARENT_RELATIONS
    )
    declared_gap_correct = sum(
        1 for record in records if record.get("generation_gap") == "two_generation_apart"
    )
    per_relation = {}
    confusion = {
        relation: {predicted: 0 for predicted in RELATIONS}
        for relation in GRANDPARENT_RELATIONS
    }

    for relation in GRANDPARENT_RELATIONS:
        rel_records = [record for record in records if record["relation"] == relation]
        rel_total = len(rel_records)
        rel_correct = sum(1 for record in rel_records if record["predicted_relation"] == relation)
        per_relation[relation] = {
            "count": rel_total,
            "correct": rel_correct,
            "accuracy": rel_correct / rel_total if rel_total else 0.0,
        }

    for record in records:
        confusion[record["relation"]][record["predicted_relation"]] += 1

    metric_rows = []
    for relation in GRANDPARENT_RELATIONS:
        tp = confusion[relation][relation]
        fp = sum(
            confusion[other][relation]
            for other in GRANDPARENT_RELATIONS
            if other != relation
        )
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
        float(record["confidence"]) for record in records if record["predicted_relation"] == record["relation"]
    ]
    incorrect_confidences = [
        float(record["confidence"]) for record in records if record["predicted_relation"] != record["relation"]
    ]

    payload = {
        "total_pairs": total,
        "total_images": total * 2,
        "relations_in_slice": GRANDPARENT_RELATIONS,
        "overall_accuracy": exact_correct / total if total else 0.0,
        "grandparent_bucket_accuracy": bucket_correct / total if total else 0.0,
        "active_macro_precision": sum(row[0] for row in metric_rows) / len(metric_rows),
        "active_macro_recall": sum(row[1] for row in metric_rows) / len(metric_rows),
        "active_macro_f1": sum(row[2] for row in metric_rows) / len(metric_rows),
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

    if any("generation_gap" in record for record in records):
        payload["declared_two_generation_rate"] = declared_gap_correct / total if total else 0.0
        payload["derived_grandparent_rate"] = (
            sum(1 for record in records if record.get("derived_relation") in GRANDPARENT_RELATIONS) / total
            if total
            else 0.0
        )

    return payload


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


def write_zero_shot_predictions_csv(path: Path, records: List[dict]) -> None:
    fieldnames = [
        "sample_id",
        "relation",
        "predicted_relation",
        "confidence",
        "correct",
        "predicted_group",
        "p1_rel",
        "p2_rel",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "relation": record["relation"],
                    "predicted_relation": record["predicted_relation"],
                    "confidence": record["confidence"],
                    "correct": int(record["predicted_relation"] == record["relation"]),
                    "predicted_group": RELATION_GROUP[record["predicted_relation"]],
                    "p1_rel": record["p1_rel"],
                    "p2_rel": record["p2_rel"],
                }
            )


def write_adapted_predictions_csv(path: Path, records: List[dict]) -> None:
    fieldnames = [
        "sample_id",
        "relation",
        "predicted_relation",
        "runner_up_relation",
        "confidence",
        "generation_gap",
        "older_face_side",
        "left_gender_guess",
        "right_gender_guess",
        "derived_relation",
        "correct",
        "predicted_group",
        "p1_rel",
        "p2_rel",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "relation": record["relation"],
                    "predicted_relation": record["predicted_relation"],
                    "runner_up_relation": record["runner_up_relation"],
                    "confidence": record["confidence"],
                    "generation_gap": record["generation_gap"],
                    "older_face_side": record["older_face_side"],
                    "left_gender_guess": record["left_gender_guess"],
                    "right_gender_guess": record["right_gender_guess"],
                    "derived_relation": record["derived_relation"],
                    "correct": int(record["predicted_relation"] == record["relation"]),
                    "predicted_group": RELATION_GROUP[record["predicted_relation"]],
                    "p1_rel": record["p1_rel"],
                    "p2_rel": record["p2_rel"],
                }
            )


def write_pair_level_comparison(
    path: Path,
    zero_shot_records: List[dict],
    adapted_records: List[dict],
) -> None:
    zero_by_id = {record["sample_id"]: record for record in zero_shot_records}
    adapted_by_id = {record["sample_id"]: record for record in adapted_records}

    fieldnames = [
        "sample_id",
        "relation",
        "zero_shot_predicted_relation",
        "zero_shot_confidence",
        "zero_shot_group",
        "zero_shot_correct",
        "adapted_predicted_relation",
        "adapted_runner_up_relation",
        "adapted_confidence",
        "adapted_generation_gap",
        "adapted_older_face_side",
        "adapted_left_gender_guess",
        "adapted_right_gender_guess",
        "adapted_derived_relation",
        "adapted_group",
        "adapted_correct",
        "changed_prediction",
    ]

    ordered_ids = [record["sample_id"] for record in sorted(zero_shot_records, key=lambda row: row["slice_order"])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in ordered_ids:
            zero_record = zero_by_id[sample_id]
            adapted_record = adapted_by_id[sample_id]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "relation": zero_record["relation"],
                    "zero_shot_predicted_relation": zero_record["predicted_relation"],
                    "zero_shot_confidence": zero_record["confidence"],
                    "zero_shot_group": RELATION_GROUP[zero_record["predicted_relation"]],
                    "zero_shot_correct": int(
                        zero_record["predicted_relation"] == zero_record["relation"]
                    ),
                    "adapted_predicted_relation": adapted_record["predicted_relation"],
                    "adapted_runner_up_relation": adapted_record["runner_up_relation"],
                    "adapted_confidence": adapted_record["confidence"],
                    "adapted_generation_gap": adapted_record["generation_gap"],
                    "adapted_older_face_side": adapted_record["older_face_side"],
                    "adapted_left_gender_guess": adapted_record["left_gender_guess"],
                    "adapted_right_gender_guess": adapted_record["right_gender_guess"],
                    "adapted_derived_relation": adapted_record["derived_relation"],
                    "adapted_group": RELATION_GROUP[adapted_record["predicted_relation"]],
                    "adapted_correct": int(
                        adapted_record["predicted_relation"] == adapted_record["relation"]
                    ),
                    "changed_prediction": int(
                        zero_record["predicted_relation"] != adapted_record["predicted_relation"]
                    ),
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_records = load_records_from_manifest(args.root, args.baseline_manifest)
    selected_records = select_grandparent_slice(
        baseline_records,
        pairs_per_relation=args.pairs_per_relation,
        seed=args.seed,
    )

    demo_manifest_path = args.adaptation_dir / "demo_manifest_seven_shot.json"
    demo_records = load_records_from_manifest(args.root, demo_manifest_path)

    selection_setup = {
        "seed": args.seed,
        "pairs_per_relation": args.pairs_per_relation,
        "baseline_manifest": str(args.baseline_manifest),
        "adaptation_dir": str(args.adaptation_dir),
        "relations": GRANDPARENT_RELATIONS,
        "sample_ids_by_relation": {
            relation: [
                record["sample_id"] for record in selected_records if record["relation"] == relation
            ]
            for relation in GRANDPARENT_RELATIONS
        },
    }
    write_json(args.output_dir / "selection_setup.json", selection_setup)
    save_manifest(args.output_dir / "manifest.json", selected_records)

    temp_dir = Path(tempfile.mkdtemp(prefix="codex_vlm_fiw_grandparent_slice_"))
    try:
        pair_root = temp_dir / "pair_sheets"
        create_pair_sheets(selected_records, pair_root / "query")
        create_pair_sheets(demo_records, pair_root / "demos")

        zero_shot_records = run_zero_shot(
            records=selected_records,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            batch_size=args.batch_size,
            output_dir=args.output_dir / "zero_shot",
        )
        adapted_records = run_adapted(
            records=selected_records,
            demo_records=demo_records,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            batch_size=args.batch_size,
            output_dir=args.output_dir / "adapted_seven_shot_v1",
        )

        zero_metrics = json.loads((args.output_dir / "zero_shot" / "metrics.json").read_text())
        adapted_metrics = json.loads(
            (args.output_dir / "adapted_seven_shot_v1" / "metrics.json").read_text()
        )

        comparison = {
            "focused_slice": {
                "total_pairs": len(selected_records),
                "pairs_per_relation": args.pairs_per_relation,
                "relations": GRANDPARENT_RELATIONS,
            },
            "zero_shot": {
                "overall_accuracy": zero_metrics["overall_accuracy"],
                "grandparent_bucket_accuracy": zero_metrics["grandparent_bucket_accuracy"],
                "active_macro_precision": zero_metrics["active_macro_precision"],
                "active_macro_recall": zero_metrics["active_macro_recall"],
                "active_macro_f1": zero_metrics["active_macro_f1"],
            },
            "adapted_seven_shot_v1": {
                "overall_accuracy": adapted_metrics["overall_accuracy"],
                "grandparent_bucket_accuracy": adapted_metrics["grandparent_bucket_accuracy"],
                "active_macro_precision": adapted_metrics["active_macro_precision"],
                "active_macro_recall": adapted_metrics["active_macro_recall"],
                "active_macro_f1": adapted_metrics["active_macro_f1"],
                "declared_two_generation_rate": adapted_metrics.get("declared_two_generation_rate"),
                "derived_grandparent_rate": adapted_metrics.get("derived_grandparent_rate"),
            },
        }
        write_json(args.output_dir / "comparison_summary.json", comparison)
        write_pair_level_comparison(
            args.output_dir / "pair_level_comparison.csv",
            zero_shot_records=zero_shot_records,
            adapted_records=adapted_records,
        )

        print("\nGrandparent-only focused comparison complete")
        print(f"Output directory: {args.output_dir}")
        print(
            "Accuracy | zero-shot: "
            f"{zero_metrics['overall_accuracy']:.4f} | adapted: {adapted_metrics['overall_accuracy']:.4f}"
        )
        print(
            "Grandparent bucket rate | zero-shot: "
            f"{zero_metrics['grandparent_bucket_accuracy']:.4f} | "
            f"adapted: {adapted_metrics['grandparent_bucket_accuracy']:.4f}"
        )
    finally:
        if args.keep_temp:
            kept_dir = args.output_dir / "pair_sheets"
            if kept_dir.exists():
                shutil.rmtree(kept_dir)
            shutil.copytree(pair_root, kept_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
