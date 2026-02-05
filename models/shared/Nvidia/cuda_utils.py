"""
NVIDIA CUDA utilities for kinship classification models.

This module provides CUDA-specific utilities, device setup, and optimizations
for running PyTorch models on NVIDIA GPUs.
"""
import os
import sys
import warnings
from typing import Optional, Tuple


def setup_cuda_environment(
    visible_devices: Optional[str] = None,
    memory_fraction: Optional[float] = None,
    allow_growth: bool = True,
) -> None:
    """
    Configure CUDA environment variables for optimal performance.

    Args:
        visible_devices: Comma-separated GPU indices (e.g., "0,1")
        memory_fraction: Fraction of GPU memory to allocate (0.0-1.0)
        allow_growth: Allow memory growth instead of pre-allocating

    Example:
        setup_cuda_environment(visible_devices="0", memory_fraction=0.9)
    """
    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    # cuDNN optimizations
    os.environ["CUDNN_BENCHMARK"] = "1"


def check_cuda_availability() -> Tuple[bool, str]:
    """
    Check if CUDA is available and properly configured.

    Returns:
        Tuple of (is_available, status_message)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA not available. Check PyTorch installation and drivers."

        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()

        if device_count == 0:
            return False, f"CUDA {cuda_version} detected but no GPUs found."

        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        return True, f"CUDA {cuda_version} with {device_count} GPU(s): {device_names}"

    except ImportError:
        return False, "PyTorch not installed."
    except Exception as e:
        return False, f"Error checking CUDA: {str(e)}"


def get_cuda_device(device_id: int = 0) -> "torch.device":
    """
    Get a CUDA device with proper error handling.

    Args:
        device_id: GPU device index

    Returns:
        torch.device for the specified CUDA GPU
    """
    import torch

    is_available, message = check_cuda_availability()

    if not is_available:
        warnings.warn(f"CUDA not available: {message}. Falling back to CPU.")
        return torch.device("cpu")

    if device_id >= torch.cuda.device_count():
        warnings.warn(f"Device {device_id} not available. Using device 0.")
        device_id = 0

    return torch.device(f"cuda:{device_id}")


def optimize_for_cuda(model: "torch.nn.Module") -> "torch.nn.Module":
    """
    Apply CUDA-specific optimizations to a model.

    Args:
        model: PyTorch model to optimize

    Returns:
        Optimized model
    """
    import torch

    # Enable TF32 for Ampere GPUs (faster matrix multiplication)
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    return model


def cuda_memory_stats() -> dict:
    """
    Get CUDA GPU memory statistics.

    Returns:
        Dictionary with memory statistics
    """
    import torch

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    stats = {}
    for i in range(torch.cuda.device_count()):
        stats[f"gpu_{i}"] = {
            "allocated_mb": torch.cuda.memory_allocated(i) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(i) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(i) / 1024**2,
        }

    return stats


def clear_cuda_cache() -> None:
    """Clear CUDA GPU memory cache."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_cuda_info() -> None:
    """Print detailed CUDA system information."""
    import torch

    print("=" * 60)
    print("NVIDIA CUDA System Information")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    if hasattr(torch.backends, 'cudnn'):
        print(f"cuDNN version: {torch.backends.cudnn.version()}")

    if torch.cuda.is_available():
        print(f"\nGPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multi-Processor Count: {props.multi_processor_count}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("\nNo GPU available")

    print("=" * 60)


if __name__ == "__main__":
    print_cuda_info()
    is_available, message = check_cuda_availability()
    print(f"\nCUDA Available: {is_available}")
    print(f"Status: {message}")
