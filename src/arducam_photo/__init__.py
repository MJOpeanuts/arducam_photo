"""Lightweight one-shot photo capture for the Arducam B0494C. Import has no side effects."""

from .config import AcquisitionInfo, CaptureConfig, CaptureResult, SettingReport
from .engine import capture
from .errors import (
    CameraOpenError, CameraReadError, CameraSetupError, CaptureBusyError,
    CaptureCancelled, CaptureConfigError, CaptureError, IspError, RawBufferError, SaveError,
)
from .storage import capture_and_save, save_png

__all__ = [
    "AcquisitionInfo", "CaptureConfig", "CaptureResult", "SettingReport", "capture",
    "capture_and_save", "save_png", "CaptureError", "CaptureConfigError", "CaptureBusyError",
    "CaptureCancelled", "CameraOpenError", "CameraSetupError", "CameraReadError",
    "RawBufferError", "IspError", "SaveError",
]
__version__ = "0.1.0"
