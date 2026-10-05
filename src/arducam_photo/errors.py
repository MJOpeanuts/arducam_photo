"""Distinct, explicit error types for each stage of a capture."""


class CaptureError(Exception):
    """Base class for all errors raised by arducam_photo."""


class CaptureConfigError(CaptureError):
    """Invalid configuration (including missing/invalid CCM tuning file)."""


class CaptureBusyError(CaptureError):
    """A capture is already running in this process."""


class CaptureCancelled(CaptureError):
    """Capture cancelled between two native calls."""


class CameraOpenError(CaptureError):
    """The camera index could not be opened with the requested API."""


class CameraSetupError(CaptureError):
    """The driver rejected or could not apply the requested path/settings."""


class CameraReadError(CaptureError):
    """Too many failed reads, or no usable frame within the bounds."""


class RawBufferError(CaptureError):
    """The received buffer has the wrong type, size or shape."""


class IspError(CaptureError):
    """Black level, demosaicing or color correction failed."""


class SaveError(CaptureError):
    """PNG encoding, writing or verification failed."""
