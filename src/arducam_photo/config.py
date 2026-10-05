"""Capture configuration and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .errors import CaptureConfigError

NATIVE_108MP = "native_108mp"
COLOR_720P = "color_720p"
PATHS = (NATIVE_108MP, COLOR_720P)
APIS = ("msmf", "dshow", "any")


@dataclass(frozen=True)
class CaptureConfig:
    """Everything one capture needs. Immutable; copied again at capture start.

    camera_index: OpenCV index. Not a stable hardware identity.
    api: "msmf" (reference), "dshow" or "any". No automatic fallback.
    path: "native_108mp" (USB 3 required) or "color_720p". Chosen explicitly.
    focus: CAP_PROP_FOCUS value (reference demo range 0..1023), or None to leave untouched.
    ccm_path: manufacturer tuning JSON (arducam_108mp.json). Required for
        native_108mp when apply_ccm is True; never read for color_720p.
    apply_ccm: native_108mp only. False skips color correction (recorded in result).
    stabilization_reads: valid frames read before the kept one (reference: 5, not universal).
    max_failed_reads / max_invalid_buffers: bounds for the stabilization loop.
    """

    camera_index: int = 0
    api: str = "msmf"
    path: str = NATIVE_108MP
    focus: Optional[int] = None
    ccm_path: Optional[str] = None
    apply_ccm: bool = True
    stabilization_reads: int = 5
    max_failed_reads: int = 5
    max_invalid_buffers: int = 5

    def validate(self) -> None:
        if self.path not in PATHS:
            raise CaptureConfigError(f"unknown path {self.path!r}; expected one of {PATHS}")
        if self.api not in APIS:
            raise CaptureConfigError(f"unknown api {self.api!r}; expected one of {APIS}")
        if not isinstance(self.camera_index, int) or self.camera_index < 0:
            raise CaptureConfigError("camera_index must be an integer >= 0")
        if self.focus is not None and (not isinstance(self.focus, int) or self.focus < 0):
            raise CaptureConfigError("focus must be None or an integer >= 0")
        for name in ("stabilization_reads", "max_failed_reads", "max_invalid_buffers"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 0:
                raise CaptureConfigError(f"{name} must be an integer >= 0")
        if self.path == NATIVE_108MP and self.apply_ccm and not self.ccm_path:
            raise CaptureConfigError(
                "native_108mp with color correction needs ccm_path (arducam_108mp.json); "
                "pass it explicitly or set apply_ccm=False"
            )


@dataclass(frozen=True)
class SettingReport:
    """One setting. readback is what the driver reports, NOT proof of physical effect."""

    requested: Any
    accepted: Optional[bool]  # return value of cap.set(); None if not attempted
    readback: Any = None


@dataclass(frozen=True)
class AcquisitionInfo:
    config: CaptureConfig  # frozen copy used for this capture
    api: str
    camera_index: int
    path: str
    transport_size: tuple  # (width, height) requested from the driver
    settings: dict = field(default_factory=dict)  # name -> SettingReport
    frames_read: int = 0
    failed_reads: int = 0
    invalid_buffers: int = 0
    ccm_requested: bool = False
    ccm_applied: bool = False
    ccm_path: Optional[str] = None
    duration_s: float = 0.0
    # USB speed is never measured here; only inferred from the mode/buffer received.
    usb_speed_measured: bool = False


@dataclass(frozen=True)
class CaptureResult:
    image: Any  # numpy uint8 array (height, width, 3), BGR
    width: int
    height: int
    info: AcquisitionInfo
