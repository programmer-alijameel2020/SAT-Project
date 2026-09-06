"""
Compute-device detection and selection.

torch is treated as an OPTIONAL dependency here. The geometry, tracker,
and placeholder detector modules in this prototype are all OpenCV/NumPy/
SciPy operations working on small per-frame, per-track data -- at that
scale, GPU transfer overhead exceeds any compute benefit, so those
modules deliberately stay on CPU regardless of what this reports.

Where this actually matters: once the placeholder SimpleMotionDetector is
replaced with a real neural detector (YOLOv8/CenterNet-style, per Phase 3
of the research roadmap), that component should automatically run on GPU
if one is available. This module is the single place that decision gets
made -- any future neural component should call get_device() rather than
reimplementing its own detection logic.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

_cached: Optional["DeviceInfo"] = None


@dataclass
class DeviceInfo:
    backend: str                       # "torch" | "none"
    device: str                        # "cuda:0" | "cpu"
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    opencv_cuda_available: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _check_opencv_cuda() -> bool:
    """pip's opencv-python-headless ships without CUDA modules compiled
    in, so this will almost always report False unless OpenCV was built
    from source with CUDA support. Checked defensively (the attribute
    doesn't exist at all on a non-CUDA build)."""
    try:
        import cv2
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except (ImportError, AttributeError, cv2.error):
        return False


def detect_device(prefer_gpu: bool = True, force_refresh: bool = False) -> DeviceInfo:
    """Detects whether a CUDA-capable GPU is usable via torch, if torch is
    installed. Cached after the first call within a process -- device
    availability doesn't change mid-run."""
    global _cached
    if _cached is not None and not force_refresh:
        return _cached

    opencv_cuda = _check_opencv_cuda()

    try:
        import torch
    except ImportError:
        _cached = DeviceInfo(
            backend="none", device="cpu", gpu_available=False,
            opencv_cuda_available=opencv_cuda,
            note=("torch is not installed. This only matters once a real neural "
                  "detector replaces the current placeholder -- install torch "
                  "(see README) to enable GPU acceleration for that component. "
                  "The geometry/tracker/association code is CPU-only regardless "
                  "of GPU presence (OpenCV/NumPy operations too small to benefit "
                  "from GPU transfer)."),
        )
        return _cached

    if prefer_gpu and torch.cuda.is_available():
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        _cached = DeviceInfo(
            backend="torch", device=f"cuda:{idx}", gpu_available=True,
            gpu_name=props.name,
            gpu_memory_gb=round(props.total_memory / (1024 ** 3), 2),
            opencv_cuda_available=opencv_cuda,
            note=("GPU detected and will be used automatically by any torch-based "
                  "neural component (e.g. a real detector swapped in for "
                  "SimpleMotionDetector). Current geometry/tracker code remains "
                  "CPU-bound OpenCV/NumPy regardless -- that's expected, not a "
                  "missed optimization."),
        )
    else:
        _cached = DeviceInfo(
            backend="torch", device="cpu", gpu_available=False,
            opencv_cuda_available=opencv_cuda,
            note="torch installed but no CUDA-capable GPU detected -- running on CPU.",
        )
    return _cached


def get_device_string() -> str:
    return detect_device().device