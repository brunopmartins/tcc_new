"""
AMD ROCm-optimized shared utilities for kinship classification models.

This package provides AMD-specific implementations of training utilities,
device management, and optimizations for running on AMD GPUs with ROCm.
"""
from .rocm_utils import (
    setup_rocm_environment,
    check_rocm_availability,
    get_rocm_device,
    optimize_for_rocm,
    rocm_memory_stats,
    clear_rocm_cache,
    ROCmAMPContext,
    print_rocm_info,
)

from .trainer import ROCmTrainer, train_model

__all__ = [
    # ROCm utilities
    "setup_rocm_environment",
    "check_rocm_availability",
    "get_rocm_device",
    "optimize_for_rocm",
    "rocm_memory_stats",
    "clear_rocm_cache",
    "ROCmAMPContext",
    "print_rocm_info",
    # Training
    "ROCmTrainer",
    "train_model",
]
