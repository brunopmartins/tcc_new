"""
Shared evaluation utilities for kinship classification models.
Includes accuracy, ROC-AUC, F1, per-relation metrics, and fairness metrics.
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from collections import defaultdict


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
        """
        Update metrics with new batch.
        
        Args:
            predictions: Model predictions (logits or probabilities) [B]
            labels: Ground truth labels [B]
            relations: Relation types for each sample
            demographics: Demographic groups for each sample
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        self.all_predictions.extend(predictions.flatten())
        self.all_labels.extend(labels.flatten())
        
        if relations is not None:
            self.all_relations.extend(relations)
        if demographics is not None:
            self.all_demographics.extend(demographics)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        # Convert to binary predictions
        binary_preds = (predictions > self.threshold).astype(int)
        
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(labels, binary_preds)
        metrics["precision"] = precision_score(labels, binary_preds, zero_division=0)
        metrics["recall"] = recall_score(labels, binary_preds, zero_division=0)
        metrics["f1"] = f1_score(labels, binary_preds, zero_division=0)
        
        # ROC-AUC (requires continuous predictions)
        try:
            metrics["roc_auc"] = roc_auc_score(labels, predictions)
        except ValueError:
            metrics["roc_auc"] = 0.0
        
        # TAR @ FAR thresholds
        metrics.update(self._compute_tar_at_far(predictions, labels))
        
        # Per-relation metrics
        if self.all_relations:
            metrics["per_relation"] = self._compute_per_relation_metrics(
                predictions, labels, binary_preds
            )
        
        # Fairness metrics
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
        """Compute TAR (True Accept Rate) at various FAR thresholds."""
        metrics = {}
        
        try:
            fpr, tpr, thresholds = roc_curve(labels, predictions)
            
            # TAR @ FAR = 0.1%, 1%, 10%
            for far_target in [0.001, 0.01, 0.1]:
                idx = np.argmin(np.abs(fpr - far_target))
                metrics[f"tar@far={far_target}"] = tpr[idx]
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
        relations = np.array(self.all_relations)
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
        demographics = np.array(self.all_demographics)
        unique_groups = np.unique(demographics)
        
        # Per-group accuracy
        group_accuracies = {}
        group_tpr = {}  # True positive rate
        group_fpr = {}  # False positive rate
        
        for group in unique_groups:
            mask = demographics == group
            if mask.sum() == 0:
                continue
            
            group_labels = labels[mask]
            group_binary = binary_preds[mask]
            
            group_accuracies[group] = accuracy_score(group_labels, group_binary)
            
            # TPR = TP / (TP + FN)
            pos_mask = group_labels == 1
            if pos_mask.sum() > 0:
                group_tpr[group] = (group_binary[pos_mask] == 1).mean()
            
            # FPR = FP / (FP + TN)
            neg_mask = group_labels == 0
            if neg_mask.sum() > 0:
                group_fpr[group] = (group_binary[neg_mask] == 1).mean()
        
        fairness_metrics = {
            "group_accuracies": group_accuracies,
            "group_tpr": group_tpr,
            "group_fpr": group_fpr,
        }
        
        # Aggregate fairness metrics
        if group_accuracies:
            accs = list(group_accuracies.values())
            fairness_metrics["accuracy_gap"] = max(accs) - min(accs)
            fairness_metrics["accuracy_std"] = np.std(accs)
        
        if group_tpr:
            tprs = list(group_tpr.values())
            fairness_metrics["tpr_gap"] = max(tprs) - min(tprs)  # Equalized odds
        
        return fairness_metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate model on a dataloader.
    
    Args:
        model: PyTorch model
        dataloader: Test/validation dataloader
        device: Device to run evaluation on
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics_calculator = KinshipMetrics(threshold=threshold)
    
    with torch.no_grad():
        for batch in dataloader:
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            labels = batch["label"]
            relations = batch.get("relation", [None] * len(labels))
            
            # Forward pass
            outputs = model(img1, img2)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                predictions = outputs[0]  # First element is usually the logits
            else:
                predictions = outputs
            
            # Apply sigmoid if needed
            if predictions.min() < 0 or predictions.max() > 1:
                predictions = torch.sigmoid(predictions)
            
            metrics_calculator.update(
                predictions=predictions,
                labels=labels,
                relations=relations if relations[0] is not None else None,
            )
    
    return metrics_calculator.compute()


def find_optimal_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        metric: Metric to optimize ("f1", "accuracy", "balanced")
    
    Returns:
        (optimal_threshold, best_metric_value)
    """
    best_threshold = 0.5
    best_metric = 0.0
    
    for threshold in np.arange(0.1, 0.95, 0.05):
        binary_preds = (predictions > threshold).astype(int)
        
        if metric == "f1":
            score = f1_score(labels, binary_preds, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(labels, binary_preds)
        elif metric == "balanced":
            # Balanced accuracy
            from sklearn.metrics import balanced_accuracy_score
            score = balanced_accuracy_score(labels, binary_preds)
        else:
            score = accuracy_score(labels, binary_preds)
        
        if score > best_metric:
            best_metric = score
            best_threshold = threshold
    
    return best_threshold, best_metric


def print_metrics(metrics: Dict, prefix: str = ""):
    """Pretty print metrics dictionary."""
    print(f"\n{prefix}Evaluation Results:")
    print("=" * 50)
    
    # Basic metrics
    basic_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for m in basic_metrics:
        if m in metrics:
            print(f"{m.capitalize():15s}: {metrics[m]:.4f}")
    
    # TAR @ FAR
    tar_metrics = [k for k in metrics if k.startswith("tar@far")]
    if tar_metrics:
        print("\nTAR @ FAR:")
        for m in tar_metrics:
            print(f"  {m}: {metrics[m]:.4f}")
    
    # Per-relation metrics
    if "per_relation" in metrics:
        print("\nPer-Relation Metrics:")
        for rel, rel_metrics in metrics["per_relation"].items():
            print(f"  {rel}: Acc={rel_metrics['accuracy']:.4f}, "
                  f"F1={rel_metrics['f1']:.4f}, "
                  f"AUC={rel_metrics['roc_auc']:.4f}, "
                  f"N={rel_metrics['count']}")
    
    # Fairness metrics
    if "fairness" in metrics:
        print("\nFairness Metrics:")
        fm = metrics["fairness"]
        if "accuracy_gap" in fm:
            print(f"  Accuracy Gap: {fm['accuracy_gap']:.4f}")
        if "tpr_gap" in fm:
            print(f"  TPR Gap (Equalized Odds): {fm['tpr_gap']:.4f}")
        
        if "group_accuracies" in fm:
            print("  Per-Group Accuracy:")
            for group, acc in fm["group_accuracies"].items():
                print(f"    {group}: {acc:.4f}")
    
    print("=" * 50)
