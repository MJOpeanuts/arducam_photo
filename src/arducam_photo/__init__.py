"""Lightweight one-shot photo capture for the Arducam B0494C. Import has no side effects."""

from .config import (
    AcquisitionInfo, AcquisitionResult, CaptureConfig, CaptureResult, SettingReport,
    CaptureMode, MODES, PATHS, COLOR_720P, COLOR_4K, COLOR_12MP, NATIVE_108MP,
)
from .engine import acquire, capture
from .processing import ProcessingRecipe, process_image
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
    "AcquisitionResult", "acquire", "CaptureMode", "MODES", "PATHS", "ProcessingRecipe", "process_image",
    "COLOR_720P", "COLOR_4K", "COLOR_12MP", "NATIVE_108MP",
]
__version__ = "0.1.0"
