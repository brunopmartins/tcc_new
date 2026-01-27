"""
AMD ROCm-optimized training utilities for kinship classification models.

This trainer is specifically optimized for AMD GPUs using the ROCm platform.
Key optimizations include:
- ROCm-specific AMP (Automatic Mixed Precision) handling
- MIOpen convolution optimization
- HIP memory management
- Gradient scaling tuned for ROCm
"""
import os
import time
from pathlib import Path
from typing import Dict, Optional, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

# ROCm-specific imports
from rocm_utils import (
    setup_rocm_environment,
    check_rocm_availability,
    get_rocm_device,
    optimize_for_rocm,
    clear_rocm_cache,
    ROCmAMPContext,
    print_rocm_info,
)

# Add parent directory for shared config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TrainConfig
from evaluation import evaluate_model, print_metrics


class ROCmTrainer:
    """
    ROCm-optimized trainer for kinship classification models.

    This trainer includes AMD-specific optimizations:
    - Proper HIP device initialization
    - ROCm-compatible AMP (Automatic Mixed Precision)
    - MIOpen benchmark mode for faster convolutions
    - Memory-efficient gradient accumulation
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        config: TrainConfig,
        device: torch.device = None,
        rocm_device_id: int = 0,
    ):
        # Setup ROCm environment first
        setup_rocm_environment()

        # Check ROCm availability
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

        # Get ROCm device
        self.device = device or get_rocm_device(rocm_device_id)
        print(f"Training device: {self.device}")

        # Move model to device and apply ROCm optimizations
        self.model.to(self.device)
        self.model = optimize_for_rocm(self.model)

        # Optimizer with ROCm-friendly settings
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            # ROCm works well with default eps but can be tuned
            eps=1e-8,
        )

        # Scheduler
        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs - config.warmup_epochs,
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

        # ROCm-optimized AMP
        # Note: ROCm AMP works but may have different numerical behavior
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.amp_context = None
        if self.use_amp:
            print("Enabling ROCm AMP (Automatic Mixed Precision)")
            self.amp_context = ROCmAMPContext(enabled=True)

        # For non-context manager AMP usage
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Tracking
        self.best_metric = 0.0
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_auc": [],
        }

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> float:
        """Train for one epoch with ROCm optimizations."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Clear cache at start of epoch for better memory management
        clear_rocm_cache()

        pbar = tqdm(self.train_loader, desc="Training (ROCm)")
        for batch in pbar:
            img1 = batch["img1"].to(self.device, non_blocking=True)
            img2 = batch["img2"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)  # More memory efficient

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

        return total_loss / num_batches

    def _compute_loss(self, outputs, labels):
        """Compute loss based on output format."""
        if isinstance(outputs, tuple):
            if len(outputs) >= 2:
                emb1, emb2 = outputs[0], outputs[1]
                return self.loss_fn(emb1, emb2, labels)
        else:
            return self.loss_fn(outputs.squeeze(), labels)

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        metrics = evaluate_model(
            self.model,
            self.val_loader,
            self.device,
        )
        return metrics

    def train(self) -> Dict:
        """Full training loop with ROCm optimizations."""
        print(f"\n{'='*60}")
        print("Starting ROCm-optimized training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"{'='*60}\n")

        for epoch in range(self.config.num_epochs):
            start_time = time.time()

            # Warmup
            if epoch < self.config.warmup_epochs:
                warmup_lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler and epoch >= self.config.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["accuracy"])
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_auc"].append(val_metrics.get("roc_auc", 0))

            # Print progress
            elapsed = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val AUC: {val_metrics.get('roc_auc', 0):.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpointing
            if val_metrics["accuracy"] > self.best_metric:
                self.best_metric = val_metrics["accuracy"]
                self.patience_counter = 0
                self.save_checkpoint("best.pt", val_metrics)
                print(f"  -> New best model! Accuracy: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt", val_metrics)

            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            # Periodic cache clearing for long training runs
            if (epoch + 1) % 10 == 0:
                clear_rocm_cache()

        # Final checkpoint
        self.save_checkpoint("final.pt", val_metrics)

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best validation accuracy: {self.best_metric:.4f}")
        print(f"{'='*60}")

        return self.history

    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": self.history,
            "config": self.config,
            "platform": "AMD_ROCm",
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

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

        return checkpoint.get("metrics", {})


# Alias for compatibility with existing code
Trainer = ROCmTrainer


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    config: TrainConfig,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Convenience function to train a model on ROCm.

    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        loss_fn: Loss function
        config: Training configuration
        device: Device to train on

    Returns:
        Training history
    """
    trainer = ROCmTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        device=device,
    )

    return trainer.train()
