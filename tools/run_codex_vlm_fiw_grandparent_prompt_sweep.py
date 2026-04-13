#!/usr/bin/env python3
"""
Run an expanded prompt sweep on the fixed 12-pair grandparent FIW slice.

This diagnostic script reuses the same 3-pairs-per-relation manifest already
created for the grandparent-only slice and compares new directed prompts
against the previously saved zero-shot and seven-shot results.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_codex_vlm_fiw_grandparent_slice import (
    FACE_SIDE_ENUM,
    GAP_ENUM,
    GENDER_ENUM,
    GRANDPARENT_RELATIONS,
    RELATION_NAMES,
    RELATIONS,
    compute_slice_metrics,
    create_pair_sheets,
    derive_relation_from_cues,
    load_records_from_manifest,
    run_codex_batch,
    save_manifest,
    write_adapted_predictions_csv,
    write_json,
)


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
        help="Fixed grandparent-only slice manifest.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new/data/codex_vlm_fiw_grandparent_slice_12"),
        help="Existing focused-run directory used for reference baselines.",
    )
    parser.add_argument(
        "--adaptation-dir",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new/data/codex_vlm_fiw_domain_adapt_1500"),
        help="Domain-adaptation artifact directory with demo manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/bruno/Desktop/tcc_new/data/codex_vlm_fiw_grandparent_prompt_sweep_12"),
        help="Directory for the prompt sweep artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Number of query pairs per Codex call.",
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


def chunked(records: List[dict], batch_size: int) -> List[List[dict]]:
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


def relation_glossary(labels: Iterable[str]) -> str:
    return "\n".join(
        f"- {label} = {RELATION_NAMES[label]}" for label in labels
    )


def write_structured_schema(path: Path, batch_size: int, labels: List[str]) -> None:
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
                        "predicted_relation": {"type": "string", "enum": labels},
                        "runner_up_relation": {"type": "string", "enum": labels},
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


def build_prompt(config_name: str, demos: List[dict], batch_records: List[dict]) -> str:
    labels = RELATIONS
    oracle_prefix = ""
    demo_prefix = ""

    if config_name.startswith("oracle_4way") or config_name.startswith("fewshot_oracle_4way"):
        labels = GRANDPARENT_RELATIONS
        oracle_prefix = (
            "Important: in this diagnostic run, every query pair belongs to exactly one of these "
            "four grandparent-grandchild relations only: gfgd, gfgs, gmgd, gmgs. "
            "Do not choose parent-child or sibling labels.\n\n"
        )

    if demos:
        demo_lines = []
        for index, demo in enumerate(demos, start=1):
            demo_lines.append(
                f"- Attachment {index}: labeled example, relation = {demo['relation']} "
                f"({RELATION_NAMES[demo['relation']]})"
            )
        first_query_attachment = len(demos) + 1
        last_query_attachment = len(demos) + len(batch_records)
        demo_prefix = (
            f"You will receive {len(demos) + len(batch_records)} attached images.\n"
            "The first attachments are labeled few-shot examples. The remaining attachments are unlabeled query pairs.\n\n"
            "Labeled examples:\n"
            + "\n".join(demo_lines)
            + "\n\n"
            f"Query attachments are {first_query_attachment} through {last_query_attachment}. "
        )
    else:
        demo_prefix = (
            f"You will receive {len(batch_records)} attached images.\n"
            "Each attachment is one FIW kinship pair sheet containing a left face and a right face.\n\n"
        )

    shared_header = (
        demo_prefix
        + oracle_prefix
        + "Possible relation labels for this run:\n"
        + relation_glossary(labels)
        + "\n\n"
        + f"Return a JSON object with a predictions array containing exactly {len(batch_records)} items "
        "in the same order as the query attachments only.\n\n"
        "For each query pair, provide:\n"
        "- predicted_relation\n"
        "- runner_up_relation\n"
        "- confidence\n"
        "- generation_gap: same_generation, one_generation_apart, two_generation_apart, or uncertain\n"
        "- older_face_side: left, right, similar_age, or uncertain\n"
        "- left_gender_guess: male, female, or uncertain\n"
        "- right_gender_guess: male, female, or uncertain\n\n"
        "Use only the visual evidence in the image. Do not use shell commands. Do not inspect files. Return JSON only.\n"
    )

    if config_name == "directed_11way_guardrail_v1":
        return (
            shared_header
            + "\nDecision rules:\n"
            "1. First decide whether the pair looks same_generation, one_generation_apart, or two_generation_apart.\n"
            "2. If the apparent age separation looks closer to a grandparent-grandchild relation, avoid defaulting to parent-child labels.\n"
            "3. Use the likely older person's gender and the younger person's gender to map between gfgd, gfgs, gmgd, and gmgs.\n"
            "4. Only choose fd/fs/md/ms when the visual gap looks like one generation rather than two.\n"
        )

    if config_name == "directed_11way_guardrail_v2":
        return (
            shared_header
            + "\nUse this elimination table internally:\n"
            "- same_generation -> bb, ss, sibs\n"
            "- one_generation_apart -> fd, fs, md, ms\n"
            "- two_generation_apart -> gfgd, gfgs, gmgd, gmgs\n"
            "Treat a clearly elderly-vs-younger pair as evidence for the two_generation_apart bucket unless the faces look plausibly parent-child.\n"
            "Inside the selected bucket, map older/younger gender carefully before choosing the final label.\n"
        )

    if config_name == "oracle_4way_zero_shot_v1":
        return (
            shared_header
            + "\nDecision process:\n"
            "1. Decide which face is older.\n"
            "2. Estimate the older face gender.\n"
            "3. Estimate the younger face gender.\n"
            "4. Map directly:\n"
            "   older male + younger female -> gfgd\n"
            "   older male + younger male -> gfgs\n"
            "   older female + younger female -> gmgd\n"
            "   older female + younger male -> gmgs\n"
            "5. Use facial resemblance only to break ties when age/gender are uncertain.\n"
        )

    if config_name == "oracle_4way_zero_shot_v2":
        return (
            shared_header
            + "\nBe conservative about gender cues but strict about the label mapping:\n"
            "- If the older face is male, choose only gfgd or gfgs.\n"
            "- If the older face is female, choose only gmgd or gmgs.\n"
            "- If the younger face is female, choose only gfgd or gmgd.\n"
            "- If the younger face is male, choose only gfgs or gmgs.\n"
            "Never leave the final label inconsistent with your own older_face_side and gender guesses.\n"
        )

    if config_name == "fewshot_oracle_4way_v1":
        return (
            shared_header
            + "\nUse the labeled grandparent examples as visual anchors.\n"
            "For each query pair, first choose the closest grandparent bucket by older-face gender, then decide the grandchild gender, and only then use resemblance cues to confirm the choice.\n"
            "Keep the final label consistent with the auxiliary fields.\n"
        )

    raise ValueError(f"Unknown config: {config_name}")


def initialize_records(records: List[dict]) -> List[dict]:
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
    return working_records


def run_prompt_config(
    records: List[dict],
    demos: List[dict],
    config_name: str,
    batch_size: int,
    model: str,
    reasoning_effort: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(output_dir / "manifest.json", records)
    if demos:
        save_manifest(output_dir / "demo_manifest.json", demos)

    labels = RELATIONS
    if config_name.startswith("oracle_4way") or config_name.startswith("fewshot_oracle_4way"):
        labels = GRANDPARENT_RELATIONS

    write_json(
        output_dir / "config.json",
        {
            "config_name": config_name,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "batch_size": batch_size,
            "label_space": labels,
            "uses_demos": bool(demos),
            "demo_sample_ids": [demo["sample_id"] for demo in demos],
        },
    )

    working_records = initialize_records(records)
    batches = chunked(working_records, batch_size)

    for batch_index, batch in enumerate(batches, start=1):
        schema_path = output_dir / f"schema_{batch_index:02d}.json"
        batch_output_path = output_dir / f"batch_{batch_index:02d}.json"
        prompt_path = output_dir / f"prompt_{batch_index:02d}.txt"

        write_structured_schema(schema_path, len(batch), labels)
        prompt_text = build_prompt(config_name, demos, batch)
        prompt_path.write_text(prompt_text, encoding="utf-8")

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
                image_paths=[Path(record["pair_sheet_path"]) for record in demos]
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
    return metrics


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

    slice_records = load_records_from_manifest(args.root, args.slice_manifest)
    full_demos = load_records_from_manifest(
        args.root, args.adaptation_dir / "demo_manifest_full.json"
    )
    grandparent_demos = [
        demo for demo in full_demos if demo["relation"] in GRANDPARENT_RELATIONS
    ]

    reference_summary = {
        "zero_shot_reference": summarize_metrics(
            json.loads((args.reference_dir / "zero_shot" / "metrics.json").read_text(encoding="utf-8"))
        ),
        "seven_shot_v1_reference": summarize_metrics(
            json.loads(
                (args.reference_dir / "adapted_seven_shot_v1" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
        ),
    }
    write_json(args.output_dir / "reference_summary.json", reference_summary)
    save_manifest(args.output_dir / "manifest.json", slice_records)
    save_manifest(args.output_dir / "grandparent_demo_manifest.json", grandparent_demos)

    temp_dir = Path(tempfile.mkdtemp(prefix="codex_vlm_fiw_grandparent_prompt_sweep_"))
    pair_root = temp_dir / "pair_sheets"
    try:
        create_pair_sheets(slice_records, pair_root / "query")
        create_pair_sheets(grandparent_demos, pair_root / "demos")

        configs = [
            {"name": "directed_11way_guardrail_v1", "demos": []},
            {"name": "directed_11way_guardrail_v2", "demos": []},
            {"name": "oracle_4way_zero_shot_v1", "demos": []},
            {"name": "oracle_4way_zero_shot_v2", "demos": []},
            {"name": "fewshot_oracle_4way_v1", "demos": grandparent_demos},
        ]

        results = []
        for config in configs:
            config_metrics = run_prompt_config(
                records=slice_records,
                demos=config["demos"],
                config_name=config["name"],
                batch_size=args.batch_size,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                output_dir=args.output_dir / config["name"],
            )
            results.append(
                {
                    "config_name": config["name"],
                    "uses_demos": bool(config["demos"]),
                    **summarize_metrics(config_metrics),
                }
            )

        results.sort(
            key=lambda row: (
                row["overall_accuracy"],
                row["active_macro_f1"],
                row["grandparent_bucket_accuracy"],
            ),
            reverse=True,
        )

        write_json(args.output_dir / "prompt_sweep_summary.json", results)
        comparison = {
            "slice_manifest": str(args.slice_manifest),
            "reference_results": reference_summary,
            "new_prompt_results": results,
        }
        write_json(args.output_dir / "comparison_summary.json", comparison)

        top = results[0]
        print("\nGrandparent prompt sweep complete")
        print(f"Output directory: {args.output_dir}")
        print(
            "Best new prompt: "
            f"{top['config_name']} | acc={top['overall_accuracy']:.4f} | "
            f"bucket={top['grandparent_bucket_accuracy']:.4f} | "
            f"macro_f1={top['active_macro_f1']:.4f}"
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
