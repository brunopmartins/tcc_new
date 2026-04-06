#!/usr/bin/env python3
"""
Run a zero-shot Codex VLM kinship-relation experiment on FIW test pairs.

The script samples valid positive FIW pairs, builds temporary side-by-side
pair sheets, calls `codex exec` in batches, and writes reproducible outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageOps, ImageDraw


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
        default=None,
        help="Directory for manifests and metrics.",
    )
    parser.add_argument(
        "--total-images",
        type=int,
        default=150,
        help="Total number of individual images. Must be even.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of pair sheets per Codex call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260406,
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
        help="Keep generated pair sheets after the run.",
    )
    return parser.parse_args()


def load_valid_positive_pairs(root: Path) -> Dict[str, List[dict]]:
    csv_path = root / "datasets" / "FIW" / "track-I" / "test-pairs.csv"
    fiw_root = root / "datasets" / "FIW" / "FIDs"
    by_relation: Dict[str, List[dict]] = defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["labels"] != "1":
                continue
            relation = row["ptype"]
            if relation not in RELATIONS:
                continue

            p1_abs = fiw_root / row["p1"]
            p2_abs = fiw_root / row["p2"]
            if not p1_abs.exists() or not p2_abs.exists():
                continue

            record = {
                "relation": relation,
                "p1_rel": row["p1"],
                "p2_rel": row["p2"],
                "p1_abs": str(p1_abs),
                "p2_abs": str(p2_abs),
            }
            by_relation[relation].append(record)

    return by_relation


def build_allocation(by_relation: Dict[str, List[dict]], total_pairs: int) -> Dict[str, int]:
    if total_pairs < len(RELATIONS):
        raise ValueError("Need at least one pair per relation.")

    base = total_pairs // len(RELATIONS)
    remainder = total_pairs % len(RELATIONS)

    allocation = {relation: base for relation in RELATIONS}
    abundance_order = sorted(RELATIONS, key=lambda rel: len(by_relation[rel]), reverse=True)
    for relation in abundance_order[:remainder]:
        allocation[relation] += 1

    # If a scarce relation cannot meet the ideal balanced target, cap it at the
    # available count and redistribute the shortfall to richer relations.
    deficit = 0
    for relation, needed in allocation.items():
        available = len(by_relation[relation])
        if available < needed:
            deficit += needed - available
            allocation[relation] = available

    if deficit:
        while deficit > 0:
            progress_made = False
            for relation in abundance_order:
                spare = len(by_relation[relation]) - allocation[relation]
                if spare <= 0:
                    continue
                allocation[relation] += 1
                deficit -= 1
                progress_made = True
                if deficit == 0:
                    break
            if not progress_made:
                break

    if deficit:
        raise ValueError(
            "Not enough valid FIW pairs to satisfy the requested sample size. "
            f"Short by {deficit} pairs after redistribution."
        )

    return allocation


def sample_pairs(
    by_relation: Dict[str, List[dict]],
    allocation: Dict[str, int],
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    sampled: List[dict] = []
    sample_index = 1

    for relation in RELATIONS:
        candidates = list(by_relation[relation])
        rng.shuffle(candidates)
        for record in candidates[: allocation[relation]]:
            sampled.append(
                {
                    **record,
                    "sample_id": f"S{sample_index:03d}",
                    "relation_name": RELATION_NAMES[relation],
                }
            )
            sample_index += 1

    rng.shuffle(sampled)
    for index, record in enumerate(sampled, start=1):
        record["batch_order"] = index
    return sampled


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
                        "predicted_relation": {
                            "type": "string",
                            "enum": RELATIONS,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["predicted_relation", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }
    path.write_text(json.dumps(schema), encoding="utf-8")


def chunked(records: List[dict], batch_size: int) -> List[List[dict]]:
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


def run_batch(
    batch_records: List[dict],
    pair_sheet_paths: List[Path],
    model: str,
    reasoning_effort: str,
    schema_path: Path,
    output_json_path: Path,
) -> List[dict]:
    prompt = (
        f"You will receive {len(batch_records)} attached images. "
        "Each attached image is one FIW kinship pair sheet containing the left and right face "
        "for a single positive pair. "
        "Classify each attached pair independently using only the visual evidence in that "
        "attached image. Do not run shell commands. Do not inspect files. "
        f"Return a JSON object with a predictions array of exactly {len(batch_records)} items, "
        "in the same order as the attachments. "
        "For each item, choose exactly one label from: "
        + ", ".join(RELATIONS)
        + "."
    )

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

    for image_path in pair_sheet_paths:
        command.extend(["--image", str(image_path)])
    command.append("-")

    result = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=300,
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


def compute_metrics(records: List[dict]) -> dict:
    total = len(records)
    correct = sum(1 for record in records if record["predicted_relation"] == record["relation"])

    per_relation = {}
    confusion: Dict[str, Dict[str, int]] = {rel: {pred: 0 for pred in RELATIONS} for rel in RELATIONS}

    for relation in RELATIONS:
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
    for relation in RELATIONS:
        tp = confusion[relation][relation]
        fp = sum(confusion[other][relation] for other in RELATIONS if other != relation)
        fn = sum(confusion[relation][other] for other in RELATIONS if other != relation)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metric_rows.append((precision, recall, f1))
        per_relation[relation].update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_precision = sum(row[0] for row in metric_rows) / len(metric_rows)
    macro_recall = sum(row[1] for row in metric_rows) / len(metric_rows)
    macro_f1 = sum(row[2] for row in metric_rows) / len(metric_rows)

    confidences = [float(record["confidence"]) for record in records]
    correct_confidences = [float(record["confidence"]) for record in records if record["predicted_relation"] == record["relation"]]
    incorrect_confidences = [float(record["confidence"]) for record in records if record["predicted_relation"] != record["relation"]]

    return {
        "task": "closed_set_relation_classification",
        "total_pairs": total,
        "total_images": total * 2,
        "overall_accuracy": correct / total if total else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
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
        "confidence",
        "correct",
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
                    "relation_name": record["relation_name"],
                    "predicted_relation": record["predicted_relation"],
                    "predicted_relation_name": RELATION_NAMES[record["predicted_relation"]],
                    "confidence": record["confidence"],
                    "correct": int(record["predicted_relation"] == record["relation"]),
                    "p1_rel": record["p1_rel"],
                    "p2_rel": record["p2_rel"],
                }
            )


def main() -> None:
    args = parse_args()
    if args.total_images % 2 != 0:
        raise ValueError("--total-images must be even because each sample is an image pair.")

    total_pairs = args.total_images // 2
    if args.output_dir is None:
        args.output_dir = args.root / "data" / f"codex_vlm_fiw_{args.total_images}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_relation = load_valid_positive_pairs(args.root)
    allocation = build_allocation(by_relation, total_pairs)
    sampled_records = sample_pairs(by_relation, allocation, args.seed)

    temp_dir = Path(tempfile.mkdtemp(prefix="codex_vlm_fiw_"))
    batch_dir = temp_dir / "pair_sheets"
    batch_dir.mkdir(parents=True, exist_ok=True)

    try:
        for record in sampled_records:
            record["pair_sheet_path"] = str(batch_dir / f"{record['sample_id']}.jpg")
            create_pair_sheet(record, Path(record["pair_sheet_path"]))

        manifest_path = args.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(sampled_records, indent=2), encoding="utf-8")

        config_payload = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "seed": args.seed,
            "total_images": args.total_images,
            "total_pairs": total_pairs,
            "batch_size": args.batch_size,
            "allocation": allocation,
            "available_valid_pairs": {relation: len(by_relation[relation]) for relation in RELATIONS},
        }
        (args.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

        batch_records_list = chunked(sampled_records, args.batch_size)
        batch_outputs = []
        for batch_index, batch_records in enumerate(batch_records_list, start=1):
            schema_path = temp_dir / f"schema_{batch_index:02d}.json"
            batch_output_path = args.output_dir / f"batch_{batch_index:02d}.json"
            write_schema(schema_path, len(batch_records))
            predictions = run_batch(
                batch_records=batch_records,
                pair_sheet_paths=[Path(record["pair_sheet_path"]) for record in batch_records],
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                schema_path=schema_path,
                output_json_path=batch_output_path,
            )
            for record, prediction in zip(batch_records, predictions):
                record["predicted_relation"] = prediction["predicted_relation"]
                record["confidence"] = float(prediction["confidence"])
            batch_outputs.append(
                {
                    "batch_index": batch_index,
                    "samples": [record["sample_id"] for record in batch_records],
                    "output_path": str(batch_output_path),
                }
            )
            print(
                f"Completed batch {batch_index}/{len(batch_records_list)} "
                f"with {len(batch_records)} pairs."
            )

        metrics = compute_metrics(sampled_records)
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (args.output_dir / "batches.json").write_text(json.dumps(batch_outputs, indent=2), encoding="utf-8")
        write_predictions_csv(args.output_dir / "predictions.csv", sampled_records)

        print("\nRun complete")
        print(f"Output directory: {args.output_dir}")
        print(f"Pairs evaluated: {metrics['total_pairs']}")
        print(f"Images evaluated: {metrics['total_images']}")
        print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
    finally:
        if args.keep_temp:
            kept_dir = args.output_dir / "pair_sheets"
            if kept_dir.exists():
                shutil.rmtree(kept_dir)
            shutil.copytree(batch_dir, kept_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
