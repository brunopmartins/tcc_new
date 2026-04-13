#!/usr/bin/env python3
"""
Run a hierarchical attribute-based VLM pipeline on the fixed grandparent slice.

Pipeline:
  1. Face profiling: estimate gender + age bucket for left/right faces separately.
  2. Pair adjudication: estimate generation gap + older face side from the pair sheet.
  3. Deterministic readout: map older/younger gender to the final grandparent relation.

Two deterministic readout policies are evaluated from the same stage outputs:
  - pair_primary: older side comes from pair adjudication
  - age_primary: older side comes from age buckets, falling back to pair adjudication
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_codex_vlm_fiw_grandparent_slice import (
    FACE_SIDE_ENUM,
    GAP_ENUM,
    GRANDPARENT_RELATIONS,
    RELATION_GROUP,
    RELATION_NAMES,
    compute_slice_metrics,
    create_pair_sheets,
    load_records_from_manifest,
    run_codex_batch,
    save_manifest,
    write_json,
)


AGE_BUCKETS = ["child_or_teen", "young_adult", "middle_aged", "elderly"]
AGE_RANK = {label: index for index, label in enumerate(AGE_BUCKETS)}
FORCED_GENDERS = ["male", "female"]
OLDER_ENUM = ["left", "right"]

GRANDPARENT_MAPPING = {
    ("male", "female"): "gfgd",
    ("male", "male"): "gfgs",
    ("female", "female"): "gmgd",
    ("female", "male"): "gmgs",
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
        "--slice-manifest",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new/data/codex_vlm_fiw_grandparent_slice_12/manifest.json"),
        help="Fixed 12-pair grandparent slice manifest.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new/data/codex_vlm_fiw_grandparent_prompt_sweep_12"),
        help="Existing prompt-sweep directory used for reference results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new/data/codex_vlm_fiw_grandparent_attribute_pipeline_12"),
        help="Directory for pipeline artifacts.",
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


def write_face_profile_schema(path: Path) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "left_gender_guess": {"type": "string", "enum": FORCED_GENDERS},
                        "right_gender_guess": {"type": "string", "enum": FORCED_GENDERS},
                        "left_gender_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "right_gender_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "left_age_bucket": {"type": "string", "enum": AGE_BUCKETS},
                        "right_age_bucket": {"type": "string", "enum": AGE_BUCKETS},
                        "left_age_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "right_age_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": [
                        "left_gender_guess",
                        "right_gender_guess",
                        "left_gender_confidence",
                        "right_gender_confidence",
                        "left_age_bucket",
                        "right_age_bucket",
                        "left_age_confidence",
                        "right_age_confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }
    write_json(path, schema)


def write_pair_adjudication_schema(path: Path) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "generation_gap": {"type": "string", "enum": GAP_ENUM},
                        "older_face_side": {"type": "string", "enum": OLDER_ENUM},
                        "pair_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": [
                        "generation_gap",
                        "older_face_side",
                        "pair_confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }
    write_json(path, schema)


def build_face_profile_prompt() -> str:
    return (
        "You will receive exactly 2 attached face images from the same kinship pair.\n"
        "Attachment 1 is the LEFT face. Attachment 2 is the RIGHT face.\n\n"
        "Analyze each face separately first. Return a JSON object with a predictions array "
        "containing exactly 1 item.\n\n"
        "For that single item, provide:\n"
        "- left_gender_guess: male or female\n"
        "- right_gender_guess: male or female\n"
        "- left_gender_confidence\n"
        "- right_gender_confidence\n"
        "- left_age_bucket: child_or_teen, young_adult, middle_aged, or elderly\n"
        "- right_age_bucket: child_or_teen, young_adult, middle_aged, or elderly\n"
        "- left_age_confidence\n"
        "- right_age_confidence\n\n"
        "Use only the visual evidence in the attached images. Do not inspect files. Return JSON only.\n"
    )


def build_pair_adjudication_prompt(face_profile: dict) -> str:
    hint_payload = {
        "left_gender_guess": face_profile["left_gender_guess"],
        "right_gender_guess": face_profile["right_gender_guess"],
        "left_age_bucket": face_profile["left_age_bucket"],
        "right_age_bucket": face_profile["right_age_bucket"],
    }
    return (
        "You will receive exactly 1 attached image containing a left-right kinship pair sheet.\n"
        "Use the image as the primary evidence. The following face-profile estimates are soft hints, not ground truth:\n"
        f"{json.dumps(hint_payload, indent=2)}\n\n"
        "Return a JSON object with a predictions array containing exactly 1 item.\n"
        "For that single item, provide:\n"
        "- generation_gap: same_generation, one_generation_apart, two_generation_apart, or uncertain\n"
        "- older_face_side: left or right\n"
        "- pair_confidence\n\n"
        "Decide the older face side by comparing the two faces directly in the pair image.\n"
        "Return JSON only.\n"
    )


def run_single_prediction(
    image_paths: List[Path],
    prompt_text: str,
    schema_writer,
    schema_path: Path,
    output_json_path: Path,
    model: str,
    reasoning_effort: str,
) -> dict:
    predictions = None
    if output_json_path.exists():
        try:
            payload = json.loads(output_json_path.read_text(encoding="utf-8"))
            candidate_predictions = payload.get("predictions")
            if isinstance(candidate_predictions, list) and len(candidate_predictions) == 1:
                predictions = candidate_predictions
        except json.JSONDecodeError:
            predictions = None

    if predictions is None:
        schema_writer(schema_path)
        predictions = run_codex_batch(
            image_paths=image_paths,
            prompt_text=prompt_text,
            model=model,
            reasoning_effort=reasoning_effort,
            schema_path=schema_path,
            output_json_path=output_json_path,
        )
    return predictions[0]


def older_side_from_age_buckets(face_profile: dict) -> str:
    left_rank = AGE_RANK[face_profile["left_age_bucket"]]
    right_rank = AGE_RANK[face_profile["right_age_bucket"]]
    if left_rank > right_rank:
        return "left"
    if right_rank > left_rank:
        return "right"
    if face_profile["left_age_confidence"] >= face_profile["right_age_confidence"]:
        return "left"
    return "right"


def map_relation_from_side(face_profile: dict, older_side: str) -> str:
    older_gender = (
        face_profile["left_gender_guess"]
        if older_side == "left"
        else face_profile["right_gender_guess"]
    )
    younger_gender = (
        face_profile["right_gender_guess"]
        if older_side == "left"
        else face_profile["left_gender_guess"]
    )
    return GRANDPARENT_MAPPING[(older_gender, younger_gender)]


def build_prediction_rows(records: List[dict]) -> tuple[List[dict], List[dict]]:
    pair_primary_rows = []
    age_primary_rows = []

    for record in records:
        face_profile = record["face_profile"]
        pair_judgment = record["pair_judgment"]

        pair_primary_side = pair_judgment["older_face_side"]
        age_primary_side = older_side_from_age_buckets(face_profile)

        pair_primary_relation = map_relation_from_side(face_profile, pair_primary_side)
        age_primary_relation = map_relation_from_side(face_profile, age_primary_side)

        shared = {
            "sample_id": record["sample_id"],
            "relation": record["relation"],
            "relation_name": record["relation_name"],
            "p1_rel": record["p1_rel"],
            "p2_rel": record["p2_rel"],
            "generation_gap": pair_judgment["generation_gap"],
            "older_face_side": pair_primary_side,
            "face_profile_side": age_primary_side,
            "pair_confidence": pair_judgment["pair_confidence"],
            "left_gender_guess": face_profile["left_gender_guess"],
            "right_gender_guess": face_profile["right_gender_guess"],
            "left_gender_confidence": face_profile["left_gender_confidence"],
            "right_gender_confidence": face_profile["right_gender_confidence"],
            "left_age_bucket": face_profile["left_age_bucket"],
            "right_age_bucket": face_profile["right_age_bucket"],
            "left_age_confidence": face_profile["left_age_confidence"],
            "right_age_confidence": face_profile["right_age_confidence"],
            "derived_relation": pair_primary_relation,
        }

        pair_primary_rows.append(
            {
                **shared,
                "predicted_relation": pair_primary_relation,
                "runner_up_relation": "",
                "confidence": pair_judgment["pair_confidence"],
            }
        )
        age_primary_rows.append(
            {
                **shared,
                "predicted_relation": age_primary_relation,
                "runner_up_relation": "",
                "confidence": (
                    face_profile["left_age_confidence"] + face_profile["right_age_confidence"]
                )
                / 2,
            }
        )

    return pair_primary_rows, age_primary_rows


def write_prediction_table(path: Path, rows: List[dict], policy_name: str) -> None:
    fieldnames = [
        "sample_id",
        "relation",
        "predicted_relation",
        "correct",
        "policy_name",
        "generation_gap",
        "older_face_side",
        "face_profile_side",
        "pair_confidence",
        "confidence",
        "left_gender_guess",
        "right_gender_guess",
        "left_gender_confidence",
        "right_gender_confidence",
        "left_age_bucket",
        "right_age_bucket",
        "left_age_confidence",
        "right_age_confidence",
        "p1_rel",
        "p2_rel",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "relation": row["relation"],
                    "predicted_relation": row["predicted_relation"],
                    "correct": int(row["predicted_relation"] == row["relation"]),
                    "policy_name": policy_name,
                    "generation_gap": row["generation_gap"],
                    "older_face_side": row["older_face_side"],
                    "face_profile_side": row["face_profile_side"],
                    "pair_confidence": row["pair_confidence"],
                    "confidence": row["confidence"],
                    "left_gender_guess": row["left_gender_guess"],
                    "right_gender_guess": row["right_gender_guess"],
                    "left_gender_confidence": row["left_gender_confidence"],
                    "right_gender_confidence": row["right_gender_confidence"],
                    "left_age_bucket": row["left_age_bucket"],
                    "right_age_bucket": row["right_age_bucket"],
                    "left_age_confidence": row["left_age_confidence"],
                    "right_age_confidence": row["right_age_confidence"],
                    "p1_rel": row["p1_rel"],
                    "p2_rel": row["p2_rel"],
                }
            )


def summarize_metrics(metrics: dict) -> dict:
    return {
        "overall_accuracy": metrics["overall_accuracy"],
        "grandparent_bucket_accuracy": metrics["grandparent_bucket_accuracy"],
        "active_macro_precision": metrics["active_macro_precision"],
        "active_macro_recall": metrics["active_macro_recall"],
        "active_macro_f1": metrics["active_macro_f1"],
        "mean_confidence": metrics["mean_confidence"],
        "declared_two_generation_rate": metrics.get("declared_two_generation_rate"),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records_from_manifest(args.root, args.slice_manifest)
    save_manifest(args.output_dir / "manifest.json", records)
    write_json(
        args.output_dir / "config.json",
        {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "slice_manifest": str(args.slice_manifest),
            "policies": ["pair_primary", "age_primary"],
        },
    )

    reference_summary = {
        "best_prompt_sweep": json.loads(
            (args.reference_dir / "comparison_summary.json").read_text(encoding="utf-8")
        ),
    }
    write_json(args.output_dir / "reference_summary.json", reference_summary)

    temp_dir = Path(tempfile.mkdtemp(prefix="codex_vlm_fiw_grandparent_attribute_pipeline_"))
    pair_root = temp_dir / "pair_sheets"
    stage1_dir = args.output_dir / "stage1_face_profile"
    stage2_dir = args.output_dir / "stage2_pair_adjudication"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    try:
        create_pair_sheets(records, pair_root / "query")

        enriched_records = []
        for record in records:
            sample_id = record["sample_id"]

            face_prompt_path = stage1_dir / f"{sample_id}_prompt.txt"
            face_schema_path = stage1_dir / f"{sample_id}_schema.json"
            face_output_path = stage1_dir / f"{sample_id}.json"
            face_prompt = build_face_profile_prompt()
            face_prompt_path.write_text(face_prompt, encoding="utf-8")
            face_profile = run_single_prediction(
                image_paths=[Path(record["p1_abs"]), Path(record["p2_abs"])],
                prompt_text=face_prompt,
                schema_writer=write_face_profile_schema,
                schema_path=face_schema_path,
                output_json_path=face_output_path,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
            )

            pair_prompt_path = stage2_dir / f"{sample_id}_prompt.txt"
            pair_schema_path = stage2_dir / f"{sample_id}_schema.json"
            pair_output_path = stage2_dir / f"{sample_id}.json"
            pair_prompt = build_pair_adjudication_prompt(face_profile)
            pair_prompt_path.write_text(pair_prompt, encoding="utf-8")
            pair_judgment = run_single_prediction(
                image_paths=[Path(record["pair_sheet_path"])],
                prompt_text=pair_prompt,
                schema_writer=write_pair_adjudication_schema,
                schema_path=pair_schema_path,
                output_json_path=pair_output_path,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
            )

            enriched_records.append(
                {
                    **record,
                    "face_profile": face_profile,
                    "pair_judgment": pair_judgment,
                }
            )

        pair_primary_rows, age_primary_rows = build_prediction_rows(enriched_records)

        pair_primary_metrics = compute_slice_metrics(pair_primary_rows)
        age_primary_metrics = compute_slice_metrics(age_primary_rows)

        pair_primary_metrics["policy_name"] = "pair_primary"
        age_primary_metrics["policy_name"] = "age_primary"

        write_json(args.output_dir / "metrics_pair_primary.json", pair_primary_metrics)
        write_json(args.output_dir / "metrics_age_primary.json", age_primary_metrics)
        write_prediction_table(
            args.output_dir / "predictions_pair_primary.csv",
            pair_primary_rows,
            "pair_primary",
        )
        write_prediction_table(
            args.output_dir / "predictions_age_primary.csv",
            age_primary_rows,
            "age_primary",
        )

        summary = {
            "pair_primary": summarize_metrics(pair_primary_metrics),
            "age_primary": summarize_metrics(age_primary_metrics),
            "pair_stage_two_generation_rate": pair_primary_metrics.get("declared_two_generation_rate"),
            "pair_stage_bucket_rate": pair_primary_metrics["grandparent_bucket_accuracy"],
        }
        write_json(args.output_dir / "comparison_summary.json", summary)

        best_policy_name = "pair_primary"
        best_policy_metrics = pair_primary_metrics
        if (
            age_primary_metrics["overall_accuracy"],
            age_primary_metrics["active_macro_f1"],
        ) > (
            pair_primary_metrics["overall_accuracy"],
            pair_primary_metrics["active_macro_f1"],
        ):
            best_policy_name = "age_primary"
            best_policy_metrics = age_primary_metrics

        print("\nGrandparent attribute pipeline complete")
        print(f"Output directory: {args.output_dir}")
        print(
            "Pair-primary | acc="
            f"{pair_primary_metrics['overall_accuracy']:.4f} | "
            f"macro_f1={pair_primary_metrics['active_macro_f1']:.4f}"
        )
        print(
            "Age-primary  | acc="
            f"{age_primary_metrics['overall_accuracy']:.4f} | "
            f"macro_f1={age_primary_metrics['active_macro_f1']:.4f}"
        )
        print(
            "Best policy: "
            f"{best_policy_name} | acc={best_policy_metrics['overall_accuracy']:.4f} | "
            f"macro_f1={best_policy_metrics['active_macro_f1']:.4f}"
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
