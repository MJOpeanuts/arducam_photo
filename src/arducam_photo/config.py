"""Capture configuration and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional

from .errors import CaptureConfigError

NATIVE_108MP = "native_108mp"
COLOR_720P = "color_720p"
COLOR_4K = "color_4k"
COLOR_12MP = "color_12mp"


@dataclass(frozen=True)
class CaptureMode:
    name: str
    description: str
    width: int
    height: int
    fps: int
    raw: bool = False

    @property
    def transport_size(self) -> tuple:
        return self.width, self.height


MODES = {
    COLOR_720P: CaptureMode(COLOR_720P, "720p colour (1280 × 720)", 1280, 720, 10),
    COLOR_4K: CaptureMode(COLOR_4K, "4K colour (3840 × 2160)", 3840, 2160, 10),
    COLOR_12MP: CaptureMode(COLOR_12MP, "12 MP colour (4000 × 3000)", 4000, 3000, 7),
    NATIVE_108MP: CaptureMode(
        NATIVE_108MP, "Native RAW (6000 × 9000 transport; 12000 × 9000 Bayer)", 6000, 9000, 1, True
    ),
}
PATHS = tuple(MODES)
APIS = ("msmf", "dshow", "any")


@dataclass(frozen=True)
class CaptureConfig:
    """Everything one capture needs. Immutable; copied again at capture start.

    camera_index: OpenCV index. Not a stable hardware identity.
    api: "msmf" (reference), "dshow" or "any". No automatic fallback.
    path: one of MODES; native_108mp requires USB 3. Chosen explicitly.
    focus: CAP_PROP_FOCUS value (reference demo range 0..1023), or None to leave untouched.
    ccm_path: manufacturer tuning JSON (arducam_108mp.json). Required for
        native_108mp rendering when apply_ccm is True; never read by acquisition.
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
    fps: Optional[float] = None

    def validate(self) -> None:
        if self.path not in PATHS:
            raise CaptureConfigError(f"unknown path {self.path!r}; expected one of {PATHS}")
        if self.api not in APIS:
            raise CaptureConfigError(f"unknown api {self.api!r}; expected one of {APIS}")
        if type(self.camera_index) is not int or self.camera_index < 0:
            raise CaptureConfigError("camera_index must be an integer >= 0")
        if self.focus is not None and (type(self.focus) is not int or self.focus < 0):
            raise CaptureConfigError("focus must be None or an integer >= 0")
        for name in ("stabilization_reads", "max_failed_reads", "max_invalid_buffers"):
            v = getattr(self, name)
            if type(v) is not int or v < 0:
                raise CaptureConfigError(f"{name} must be an integer >= 0")
        if type(self.apply_ccm) is not bool:
            raise CaptureConfigError("apply_ccm must be boolean")
        if self.fps is not None and (
            isinstance(self.fps, bool) or not isinstance(self.fps, (int, float))
            or not math.isfinite(self.fps) or self.fps <= 0
        ):
            raise CaptureConfigError("fps must be None or a finite number > 0")


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
    received_shape: tuple = ()
    reconstructed_shape: tuple = ()
    dtype: str = "uint8"
    timings: dict = field(default_factory=dict)
    capture_id: str = ""
    captured_at: str = ""


@dataclass(frozen=True)
class AcquisitionResult:
    original: Any  # owned uint8 Bayer HxW or BGR HxWx3, never rendered
    info: AcquisitionInfo


@dataclass(frozen=True)
class CaptureResult:
    image: Any  # numpy uint8 array (height, width, 3), BGR
    width: int
    height: int
    info: AcquisitionInfo
