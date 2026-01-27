"""
AMD ROCm utilities for kinship classification models.

This module provides ROCm-specific utilities, device setup, and optimizations
for running PyTorch models on AMD GPUs.

Key differences from NVIDIA CUDA:
1. Uses HIP (Heterogeneous-compute Interface for Portability) backend
2. Different environment variables for device control
3. ROCm-specific memory management and optimizations
4. AMP (Automatic Mixed Precision) works via torch.cuda.amp but may behave differently
"""
import os
import sys
import warnings
from typing import Optional, Tuple


def setup_rocm_environment(
    visible_devices: Optional[str] = None,
    memory_fraction: Optional[float] = None,
    gfx_version: Optional[str] = None,
    disable_flash_attention: bool = False,
) -> None:
    """
    Configure ROCm environment variables for optimal performance.

    Args:
        visible_devices: Comma-separated GPU indices (e.g., "0,1")
        memory_fraction: Fraction of GPU memory to allocate (0.0-1.0)
        gfx_version: Override GFX version for compatibility (e.g., "10.3.0")
        disable_flash_attention: Disable flash attention if causing issues

    Example:
        setup_rocm_environment(visible_devices="0", memory_fraction=0.9)
    """
    # HIP visible devices (equivalent to CUDA_VISIBLE_DEVICES)
    if visible_devices is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = visible_devices
        os.environ["ROCR_VISIBLE_DEVICES"] = visible_devices

    # GFX version override for compatibility with newer GPUs
    if gfx_version is not None:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = gfx_version

    # Memory allocation settings
    if memory_fraction is not None:
        # PyTorch ROCm uses similar fraction-based allocation
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = f"max_split_size_mb:512"

    # Disable flash attention if it causes issues on certain ROCm versions
    if disable_flash_attention:
        os.environ["PYTORCH_ROCM_FLASH_ATTN_ENABLED"] = "0"

    # Enable ROCm-specific optimizations
    os.environ["MIOPEN_FIND_MODE"] = "FAST"  # Faster convolution algorithm selection
    os.environ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"] = "0"  # Stability

    # HSA settings for better memory management
    os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"


def check_rocm_availability() -> Tuple[bool, str]:
    """
    Check if ROCm/HIP is available and properly configured.

    Returns:
        Tuple of (is_available, status_message)
    """
    try:
        import torch

        # Check if PyTorch was built with ROCm support
        if not torch.cuda.is_available():
            return False, "CUDA/ROCm not available. Check PyTorch installation."

        # Check for HIP backend (ROCm)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            hip_version = torch.version.hip
            device_count = torch.cuda.device_count()

            if device_count == 0:
                return False, f"ROCm/HIP {hip_version} detected but no GPUs found."

            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            return True, f"ROCm/HIP {hip_version} with {device_count} GPU(s): {device_names}"

        # Fallback: might be CUDA
        return False, "PyTorch appears to be built with CUDA, not ROCm."

    except ImportError:
        return False, "PyTorch not installed."
    except Exception as e:
        return False, f"Error checking ROCm: {str(e)}"


def get_rocm_device(device_id: int = 0) -> "torch.device":
    """
    Get a ROCm device with proper error handling.

    Args:
        device_id: GPU device index

    Returns:
        torch.device for the specified ROCm GPU
    """
    import torch

    is_available, message = check_rocm_availability()

    if not is_available:
        warnings.warn(f"ROCm not available: {message}. Falling back to CPU.")
        return torch.device("cpu")

    if device_id >= torch.cuda.device_count():
        warnings.warn(f"Device {device_id} not available. Using device 0.")
        device_id = 0

    return torch.device(f"cuda:{device_id}")


def optimize_for_rocm(model: "torch.nn.Module") -> "torch.nn.Module":
    """
    Apply ROCm-specific optimizations to a model.

    Args:
        model: PyTorch model to optimize

    Returns:
        Optimized model
    """
    import torch

    # Enable memory efficient attention if available (ROCm 5.4+)
    if hasattr(torch.backends, 'cuda'):
        # Enable TF32 equivalent on ROCm (if supported)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except:
            pass

    # Enable cudnn benchmark for consistent input sizes
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    return model


def rocm_memory_stats() -> dict:
    """
    Get ROCm GPU memory statistics.

    Returns:
        Dictionary with memory statistics
    """
    import torch

    if not torch.cuda.is_available():
        return {"error": "ROCm not available"}

    stats = {}
    for i in range(torch.cuda.device_count()):
        stats[f"gpu_{i}"] = {
            "allocated_mb": torch.cuda.memory_allocated(i) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(i) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(i) / 1024**2,
        }

    return stats


def clear_rocm_cache() -> None:
    """Clear ROCm GPU memory cache."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class ROCmAMPContext:
    """
    Context manager for Automatic Mixed Precision on ROCm.

    ROCm AMP works similarly to CUDA AMP but may have different behavior
    for certain operations. This wrapper provides consistent handling.

    Usage:
        with ROCmAMPContext(enabled=True) as amp:
            with amp.autocast():
                output = model(input)
            amp.scale_loss(loss).backward()
            amp.step(optimizer)
    """

    def __init__(self, enabled: bool = True, dtype=None):
        """
        Initialize AMP context.

        Args:
            enabled: Whether to enable mixed precision
            dtype: Data type for autocast (default: float16)
        """
        import torch

        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype or torch.float16
        self.scaler = None

    def __enter__(self):
        if self.enabled:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        return self

    def __exit__(self, *args):
        pass

    def autocast(self):
        """Get autocast context manager."""
        from torch.cuda.amp import autocast
        return autocast(enabled=self.enabled, dtype=self.dtype)

    def scale_loss(self, loss):
        """Scale loss for gradient scaling."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer):
        """Optimizer step with gradient unscaling."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_(self, optimizer):
        """Unscale gradients for gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)


def print_rocm_info() -> None:
    """Print detailed ROCm system information."""
    import torch

    print("=" * 60)
    print("AMD ROCm System Information")
    print("=" * 60)

    # PyTorch info
    print(f"PyTorch version: {torch.__version__}")

    # ROCm/HIP info
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f"HIP version: {torch.version.hip}")
    else:
        print("HIP version: Not available (may be CUDA build)")

    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multi-Processor Count: {props.multi_processor_count}")
    else:
        print("\nNo GPU available")

    # Environment variables
    print("\nRelevant Environment Variables:")
    env_vars = [
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "MIOPEN_FIND_MODE",
        "PYTORCH_HIP_ALLOC_CONF",
    ]
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    # Test ROCm setup
    print_rocm_info()
    is_available, message = check_rocm_availability()
    print(f"\nROCm Available: {is_available}")
    print(f"Status: {message}")
