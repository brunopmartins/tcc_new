"""
Shared experiment protocol utilities for kinship classification models.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from config import DataConfig
from evaluation import (
    collect_predictions,
    compute_metrics_from_predictions,
    find_optimal_threshold,
)


PROTOCOL_VERSION = "kinship_eval_v1"


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def apply_data_root_override(
    data_config: DataConfig,
    dataset_type: str,
    data_root: Optional[str],
) -> DataConfig:
    """Apply a dataset-specific root override in-place."""
    if not data_root:
        return data_config

    if dataset_type == "kinface":
        data_config.kinface_i_root = data_root
    elif dataset_type == "fiw":
        data_config.fiw_root = data_root
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return data_config


def resolve_dataset_root(
    data_config: DataConfig,
    dataset_type: str,
) -> str:
    """Resolve the dataset root from shared config."""
    if dataset_type == "kinface":
        return data_config.kinface_i_root
    if dataset_type == "fiw":
        return data_config.fiw_root
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def update_checkpoint_metadata(checkpoint_path: Path, metadata: Dict[str, Any]) -> None:
    """Merge metadata into an existing checkpoint."""
    if not checkpoint_path.exists():
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    protocol = checkpoint.get("protocol", {})
    protocol.update(metadata)
    checkpoint["protocol"] = protocol
    torch.save(checkpoint, checkpoint_path)


def update_checkpoint_payload(checkpoint_path: Path, updates: Dict[str, Any]) -> None:
    """Merge top-level fields into an existing checkpoint."""
    if not checkpoint_path.exists():
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.update(updates)
    torch.save(checkpoint, checkpoint_path)


def load_best_checkpoint(model: torch.nn.Module, checkpoint_dir: str, device: torch.device) -> Dict[str, Any]:
    """Load the best checkpoint into a model and return the checkpoint payload."""
    checkpoint_path = Path(checkpoint_dir) / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def get_checkpoint_threshold(
    checkpoint: Dict[str, Any],
    default: float = 0.5,
) -> float:
    """Resolve the threshold stored by the shared protocol."""
    protocol = checkpoint.get("protocol", {})
    threshold = protocol.get("selected_threshold")
    if threshold is None:
        threshold = checkpoint.get("selected_threshold")
    return float(threshold) if threshold is not None else float(default)


def build_protocol_metadata(
    *,
    train_dataset: str,
    test_dataset: str,
    threshold: float,
    threshold_metric: str,
    split_seed: int,
    negative_ratio: float,
    monitor_metric: str,
    args: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a protocol metadata block stored in checkpoints and JSON sidecars."""
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "split_seed": split_seed,
        "negative_ratio": negative_ratio,
        "selected_threshold": float(threshold),
        "threshold_metric": threshold_metric,
        "threshold_source": "validation_split",
        "monitor_metric": monitor_metric,
        "seed": getattr(args, "seed", split_seed),
        "cli_args": vars(args),
    }
    if extra:
        metadata.update(extra)
    return metadata


def evaluate_with_validation_threshold(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold_metric: str = "f1",
) -> Dict[str, Any]:
    """
    Calibrate the operating threshold on validation predictions and evaluate on test.
    """
    val_bundle = collect_predictions(model, val_loader, device)
    threshold, best_threshold_metric = find_optimal_threshold(
        val_bundle["predictions"],
        val_bundle["labels"],
        metric=threshold_metric,
    )
    val_metrics = compute_metrics_from_predictions(
        predictions=val_bundle["predictions"],
        labels=val_bundle["labels"],
        threshold=threshold,
        relations=val_bundle["relations"],
        demographics=val_bundle["demographics"],
    )

    test_bundle = collect_predictions(model, test_loader, device)
    test_metrics = compute_metrics_from_predictions(
        predictions=test_bundle["predictions"],
        labels=test_bundle["labels"],
        threshold=threshold,
        relations=test_bundle["relations"],
        demographics=test_bundle["demographics"],
    )

    return {
        "threshold": threshold,
        "threshold_metric_name": threshold_metric,
        "threshold_metric_value": best_threshold_metric,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "validation_bundle": val_bundle,
        "test_bundle": test_bundle,
    }


def aggregate_numeric_metrics(results: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate a list of result dicts into mean/std summaries."""
    results = list(results)
    if not results:
        return {}

    numeric_keys = sorted(
        {
            key
            for item in results
            for key, value in item.items()
            if isinstance(value, (int, float))
        }
    )

    summary: Dict[str, float] = {}
    for key in numeric_keys:
        values = np.array([float(item[key]) for item in results if key in item], dtype=float)
        if values.size == 0:
            continue
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std())
    return summary
