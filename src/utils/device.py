"""
src/utils/device.py - Universal Compute Device Management (CUDA / Apple Silicon MPS / CPU)
========================================================================================

Provides cross-platform device auto-detection and hardware management:
  - NVIDIA GPUs via CUDA (`torch.device("cuda")`, XGBoost `device="cuda"`)
  - Apple Silicon Mac GPUs via Metal Performance Shaders (`torch.device("mps")`)
  - Universal multi-threaded CPU fallback

Priority:
  1. Explicit argument (`device="cuda"|"mps"|"cpu"|"auto"`)
  2. Environment variable `MG_ECFM_DEVICE`
  3. Automatic hardware inspection
"""
from __future__ import annotations

import os
import platform
from typing import Any, Dict, Optional

import torch


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Return the optimal compute device.

    Auto-detects:
      1. CUDA (NVIDIA GPU on Windows / Linux)
      2. MPS (Apple Silicon Metal on macOS M1/M2/M3/M4)
      3. CPU fallback
    """
    target = requested or os.environ.get("MG_ECFM_DEVICE") or "auto"
    target = str(target).strip().lower()

    if target in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[device] CUDA requested but unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")

    if target in ("mps", "apple", "metal"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[device] Apple Silicon MPS requested but unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")

    if target == "cpu":
        return torch.device("cpu")

    # Auto-detection
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_xgb_device(device: Optional[torch.device] = None) -> str:
    """Return XGBoost compute device ('cuda' for NVIDIA GPU, otherwise 'cpu')."""
    dev = device or resolve_device()
    if dev.type == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_hardware_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Diagnostic information about the detected hardware."""
    dev = device or resolve_device()
    info: Dict[str, Any] = {
        "device": str(dev),
        "device_type": dev.type,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if dev.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = getattr(torch.version, "cuda", "unknown")
    elif dev.type == "mps":
        info["gpu_name"] = "Apple Silicon GPU (Metal Performance Shaders)"
        info["mps_available"] = True
    else:
        info["cpu_threads"] = torch.get_num_threads()
    return info

