"""
Compatibility wrapper for the default shared trainer.

By default, the generic shared trainer maps to the NVIDIA/CPU implementation.
AMD ROCm entrypoints prepend `shared/AMD` to `sys.path` and continue to use the
ROCm-specific trainer module directly.
"""
from Nvidia.trainer import Trainer, train_model

__all__ = ["Trainer", "train_model"]

