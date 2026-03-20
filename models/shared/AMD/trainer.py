"""
AMD ROCm-optimized training utilities for kinship classification models.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from rocm_utils import (
    check_rocm_availability,
    clear_rocm_cache,
    get_rocm_device,
    optimize_for_rocm,
    setup_rocm_environment,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TrainConfig
from evaluation import evaluate_model


class ROCmTrainer:
    """ROCm-optimized trainer with shared validation/selection protocol."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        config: TrainConfig,
        device: Optional[torch.device] = None,
        rocm_device_id: int = 0,
        monitor_metric: Optional[str] = None,
    ):
        setup_rocm_environment()

        is_available, status = check_rocm_availability()
        if is_available:
            print(f"ROCm Status: {status}")
        else:
            print(f"Warning: ROCm not available - {status}")
            print("Falling back to CPU training")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        self.monitor_metric = monitor_metric or config.monitor_metric

        self.device = device or get_rocm_device(rocm_device_id)
        print(f"Training device: {self.device}")

        self.model.to(self.device)
        self.model = optimize_for_rocm(self.model)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8,
        )

        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(config.num_epochs - config.warmup_epochs, 1),
                eta_min=config.min_lr,
            )
        elif config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
            )
        else:
            self.scheduler = None

        self.use_amp = config.use_amp and self.device.type == "cuda"
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.best_metric = float("-inf")
        self.best_epoch = -1
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_accuracy": [],
            "val_auc": [],
            "monitored_metric": self.monitor_metric,
        }

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> float:
        """Train for one epoch with ROCm optimizations."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        clear_rocm_cache()

        pbar = tqdm(self.train_loader, desc="Training (ROCm)")
        for batch in pbar:
            img1 = batch["img1"].to(self.device, non_blocking=True)
            img2 = batch["img2"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = self.model(img1, img2)
                    loss = self._compute_loss(outputs, labels)

                self.scaler.scale(loss).backward()

                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(img1, img2)
                loss = self._compute_loss(outputs, labels)
                loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    def _compute_loss(self, outputs, labels):
        """Compute loss based on output format."""
        if isinstance(outputs, tuple):
            if len(outputs) >= 2:
                emb1, emb2 = outputs[0], outputs[1]
                return self.loss_fn(emb1, emb2, labels)
        return self.loss_fn(outputs.squeeze(), labels)

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        return evaluate_model(self.model, self.val_loader, self.device)

    def on_epoch_start(self, epoch: int) -> None:
        """Hook for subclasses that need to change state before an epoch."""
        return None

    def train(self) -> Dict:
        """Full training loop with ROCm optimizations."""
        print(f"\n{'=' * 60}")
        print("Starting ROCm-optimized training")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"Monitoring validation metric: {self.monitor_metric}")
        print(f"{'=' * 60}\n")

        final_metrics: Dict[str, float] = {}
        for epoch in range(self.config.num_epochs):
            self.on_epoch_start(epoch + 1)
            start_time = time.time()

            if epoch < self.config.warmup_epochs:
                warmup_lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            train_loss = self.train_epoch()
            val_metrics = self.validate()
            final_metrics = val_metrics

            if self.scheduler and epoch >= self.config.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.monitor_metric, 0.0))
                else:
                    self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))
            self.history["val_auc"].append(val_metrics.get("roc_auc", 0.0))

            elapsed = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]["lr"]
            current_metric = val_metrics.get(self.monitor_metric, float("-inf"))
            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_metrics.get('accuracy', 0.0):.4f} | "
                f"Val AUC: {val_metrics.get('roc_auc', 0.0):.4f} | "
                f"{self.monitor_metric}: {current_metric:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            if current_metric > self.best_metric + self.config.min_delta:
                self.best_metric = current_metric
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.save_checkpoint("best.pt", val_metrics, epoch + 1)
                print(f"  -> New best model! {self.monitor_metric}: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt", val_metrics, epoch + 1)

            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                clear_rocm_cache()

        self.save_checkpoint("final.pt", final_metrics, epoch + 1)

        print(f"\n{'=' * 60}")
        print("Training complete!")
        print(f"Best validation {self.monitor_metric}: {self.best_metric:.4f}")
        print(f"{'=' * 60}")
        return self.history

    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None, epoch: Optional[int] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": self.history,
            "config": self.config,
            "epoch": epoch,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "monitor_metric": self.monitor_metric,
            "platform": "AMD_ROCm",
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]

        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.best_epoch = checkpoint.get("best_epoch", self.best_epoch)
        self.monitor_metric = checkpoint.get("monitor_metric", self.monitor_metric)
        return checkpoint.get("metrics", {})


Trainer = ROCmTrainer


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    config: TrainConfig,
    device: Optional[torch.device] = None,
) -> Dict:
    """Convenience function to train a model on ROCm."""
    trainer = ROCmTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        device=device,
    )
    return trainer.train()
