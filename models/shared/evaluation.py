"""
Shared evaluation utilities for kinship classification models.

The repository-wide protocol is:
1. Collect scalar predictions from model outputs in a shared way.
2. Select thresholds on validation only.
3. Report threshold-free and thresholded metrics together.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class KinshipMetrics:
    """Comprehensive metrics calculator for kinship verification."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all accumulated predictions."""
        self.all_predictions = []
        self.all_labels = []
        self.all_relations = []
        self.all_demographics = []

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        relations: Optional[List[str]] = None,
        demographics: Optional[List[str]] = None,
    ):
        """Update metrics with a batch of predictions."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        self.all_predictions.extend(np.asarray(predictions).flatten())
        self.all_labels.extend(np.asarray(labels).flatten())

        if relations is not None:
            self.all_relations.extend(relations)
        if demographics is not None:
            self.all_demographics.extend(demographics)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics for the accumulated predictions."""
        predictions = np.asarray(self.all_predictions, dtype=float)
        labels = np.asarray(self.all_labels, dtype=int)

        if predictions.size == 0 or labels.size == 0:
            return {}

        binary_preds = (predictions > self.threshold).astype(int)
        metrics: Dict[str, Any] = {
            "accuracy": accuracy_score(labels, binary_preds),
            "balanced_accuracy": balanced_accuracy_score(labels, binary_preds),
            "precision": precision_score(labels, binary_preds, zero_division=0),
            "recall": recall_score(labels, binary_preds, zero_division=0),
            "f1": f1_score(labels, binary_preds, zero_division=0),
        }

        try:
            metrics["roc_auc"] = roc_auc_score(labels, predictions)
        except ValueError:
            metrics["roc_auc"] = 0.0

        try:
            metrics["average_precision"] = average_precision_score(labels, predictions)
        except ValueError:
            metrics["average_precision"] = 0.0

        metrics.update(self._compute_tar_at_far(predictions, labels))

        if self.all_relations:
            metrics["per_relation"] = self._compute_per_relation_metrics(
                predictions, labels, binary_preds
            )

        if self.all_demographics:
            metrics["fairness"] = self._compute_fairness_metrics(
                predictions, labels, binary_preds
            )

        return metrics

    def _compute_tar_at_far(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute TAR (true accept rate) at several FAR targets."""
        metrics = {}

        try:
            fpr, tpr, _ = roc_curve(labels, predictions)
            for far_target in [0.001, 0.01, 0.1]:
                idx = np.argmin(np.abs(fpr - far_target))
                metrics[f"tar@far={far_target}"] = float(tpr[idx])
        except Exception:
            for far_target in [0.001, 0.01, 0.1]:
                metrics[f"tar@far={far_target}"] = 0.0

        return metrics

    def _compute_per_relation_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        binary_preds: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics per relation type."""
        relations = np.asarray(self.all_relations)
        unique_relations = np.unique(relations)

        per_relation = {}
        for rel in unique_relations:
            mask = relations == rel
            if mask.sum() == 0:
                continue

            rel_preds = predictions[mask]
            rel_labels = labels[mask]
            rel_binary = binary_preds[mask]

            per_relation[rel] = {
                "accuracy": accuracy_score(rel_labels, rel_binary),
                "balanced_accuracy": balanced_accuracy_score(rel_labels, rel_binary),
                "f1": f1_score(rel_labels, rel_binary, zero_division=0),
                "count": int(mask.sum()),
            }

            try:
                per_relation[rel]["roc_auc"] = roc_auc_score(rel_labels, rel_preds)
            except ValueError:
                per_relation[rel]["roc_auc"] = 0.0

        return per_relation

    def _compute_fairness_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        binary_preds: np.ndarray,
    ) -> Dict[str, float]:
        """Compute fairness metrics across demographic groups."""
        demographics = np.asarray(self.all_demographics)
        unique_groups = np.unique(demographics)

        group_accuracies = {}
        group_tpr = {}
        group_fpr = {}

        for group in unique_groups:
            mask = demographics == group
            if mask.sum() == 0:
                continue

            group_labels = labels[mask]
            group_binary = binary_preds[mask]

            group_accuracies[group] = accuracy_score(group_labels, group_binary)

            pos_mask = group_labels == 1
            if pos_mask.sum() > 0:
                group_tpr[group] = float((group_binary[pos_mask] == 1).mean())

            neg_mask = group_labels == 0
            if neg_mask.sum() > 0:
                group_fpr[group] = float((group_binary[neg_mask] == 1).mean())

        fairness_metrics: Dict[str, Any] = {
            "group_accuracies": group_accuracies,
            "group_tpr": group_tpr,
            "group_fpr": group_fpr,
        }

        if group_accuracies:
            accs = list(group_accuracies.values())
            fairness_metrics["accuracy_gap"] = max(accs) - min(accs)
            fairness_metrics["accuracy_std"] = float(np.std(accs))

        if group_tpr:
            tprs = list(group_tpr.values())
            fairness_metrics["tpr_gap"] = max(tprs) - min(tprs)

        return fairness_metrics


def _extract_prediction_tensor(outputs: Any) -> torch.Tensor:
    """Normalize model outputs into a scalar score per sample in [0, 1]."""
    predictions: Optional[torch.Tensor] = None

    if isinstance(outputs, dict):
        if "scores" in outputs:
            predictions = outputs["scores"]
        elif "probabilities" in outputs:
            predictions = outputs["probabilities"]
        elif "logits" in outputs:
            predictions = outputs["logits"]
        elif "emb1" in outputs and "emb2" in outputs:
            predictions = (F.cosine_similarity(outputs["emb1"], outputs["emb2"], dim=1) + 1) / 2
        else:
            raise ValueError("Unsupported model output dictionary for evaluation.")
    elif isinstance(outputs, tuple):
        if (
            len(outputs) >= 2
            and isinstance(outputs[0], torch.Tensor)
            and isinstance(outputs[1], torch.Tensor)
            and outputs[0].shape == outputs[1].shape
            and outputs[0].ndim >= 2
        ):
            predictions = (F.cosine_similarity(outputs[0], outputs[1], dim=1) + 1) / 2
        else:
            predictions = outputs[0]
    else:
        predictions = outputs

    if not isinstance(predictions, torch.Tensor):
        predictions = torch.as_tensor(predictions)

    if predictions.ndim > 1:
        if predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
        elif predictions.ndim == 2 and predictions.shape[0] == 1:
            predictions = predictions.squeeze(0)
        elif predictions.ndim >= 2 and predictions.shape[1] == 1:
            predictions = predictions[:, 0]
        else:
            raise ValueError(
                "Evaluation requires one scalar prediction per sample. "
                f"Received tensor with shape {tuple(predictions.shape)}."
            )

    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    return predictions.detach()


def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Collect predictions, labels, and metadata from a dataloader."""
    model.eval()

    all_predictions: List[float] = []
    all_labels: List[int] = []
    all_relations: List[str] = []
    all_demographics: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            labels = batch["label"]
            relations = batch.get("relation", [None] * len(labels))
            demographics = batch.get("demographics", [None] * len(labels))

            outputs = model(img1, img2)
            predictions = _extract_prediction_tensor(outputs).cpu()

            all_predictions.extend(predictions.numpy().flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())

            if relations and relations[0] is not None:
                all_relations.extend(relations)
            if demographics and demographics[0] is not None:
                all_demographics.extend(demographics)

    return {
        "predictions": np.asarray(all_predictions, dtype=float),
        "labels": np.asarray(all_labels, dtype=int),
        "relations": all_relations,
        "demographics": all_demographics,
    }


def compute_metrics_from_predictions(
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    relations: Optional[List[str]] = None,
    demographics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute metrics directly from prediction arrays."""
    metrics_calculator = KinshipMetrics(threshold=threshold)
    metrics_calculator.update(
        predictions=predictions,
        labels=labels,
        relations=relations,
        demographics=demographics,
    )
    return metrics_calculator.compute()


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a model on a dataloader."""
    bundle = collect_predictions(model, dataloader, device)
    return compute_metrics_from_predictions(
        predictions=bundle["predictions"],
        labels=bundle["labels"],
        threshold=threshold,
        relations=bundle["relations"] or None,
        demographics=bundle["demographics"] or None,
    )


def find_optimal_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find an operating threshold using only the provided split."""
    best_threshold = 0.5
    best_metric = -1.0

    for threshold in np.arange(0.1, 0.95, 0.05):
        binary_preds = (predictions > threshold).astype(int)

        if metric == "f1":
            score = f1_score(labels, binary_preds, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(labels, binary_preds)
        elif metric == "balanced":
            score = balanced_accuracy_score(labels, binary_preds)
        else:
            score = accuracy_score(labels, binary_preds)

        if score > best_metric:
            best_metric = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_metric


def print_metrics(metrics: Dict[str, Any], prefix: str = ""):
    """Pretty print metrics dictionary."""
    print(f"\n{prefix}Evaluation Results:")
    print("=" * 50)

    basic_metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    for name in basic_metrics:
        if name in metrics:
            print(f"{name.replace('_', ' ').title():15s}: {metrics[name]:.4f}")

    tar_metrics = [key for key in metrics if key.startswith("tar@far")]
    if tar_metrics:
        print("\nTAR @ FAR:")
        for name in tar_metrics:
            print(f"  {name}: {metrics[name]:.4f}")

    if "per_relation" in metrics:
        print("\nPer-Relation Metrics:")
        for relation, relation_metrics in metrics["per_relation"].items():
            print(
                f"  {relation}: Acc={relation_metrics['accuracy']:.4f}, "
                f"Balanced Acc={relation_metrics['balanced_accuracy']:.4f}, "
                f"F1={relation_metrics['f1']:.4f}, "
                f"AUC={relation_metrics['roc_auc']:.4f}, "
                f"N={relation_metrics['count']}"
            )

    if "fairness" in metrics:
        print("\nFairness Metrics:")
        fairness = metrics["fairness"]
        if "accuracy_gap" in fairness:
            print(f"  Accuracy Gap: {fairness['accuracy_gap']:.4f}")
        if "tpr_gap" in fairness:
            print(f"  TPR Gap (Equalized Odds): {fairness['tpr_gap']:.4f}")

        if "group_accuracies" in fairness:
            print("  Per-Group Accuracy:")
            for group, acc in fairness["group_accuracies"].items():
                print(f"    {group}: {acc:.4f}")

    print("=" * 50)

