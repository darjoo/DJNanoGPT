"""Utilities for collecting system information."""

import platform
from typing import Any

import psutil
import torch


def get_system_info(device: str) -> dict[str, Any]:
    """Collect system information for logging.

    Args:
        device: Device being used ('cuda' or 'cpu')

    Returns:
        Dictionary containing system information
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": device,
        "cpu_count": psutil.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

    if device == "cuda" and torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

    return info


def get_memory_usage(device: str) -> dict[str, float]:
    """Get current memory usage statistics.

    Args:
        device: Device being used ('cuda' or 'cpu')

    Returns:
        Dictionary containing memory usage metrics
    """
    memory_stats = {}

    # CPU memory
    memory_stats["cpu_memory_used_gb"] = round(psutil.virtual_memory().used / (1024**3), 2)
    memory_stats["cpu_memory_percent"] = psutil.virtual_memory().percent

    # GPU memory
    if device == "cuda" and torch.cuda.is_available():
        memory_stats["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
        memory_stats["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 2)
        memory_stats["gpu_memory_percent"] = round(
            (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100, 2
        )

    return memory_stats
