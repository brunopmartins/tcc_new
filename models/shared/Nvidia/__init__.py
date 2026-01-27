"""
NVIDIA CUDA shared utilities for kinship classification models.

This package provides NVIDIA-specific implementations of training utilities.
"""
from .trainer import Trainer, train_model

__all__ = [
    "Trainer",
    "train_model",
]
