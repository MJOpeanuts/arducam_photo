"""Backend-transparent OpenCV camera controls shared by preview and capture."""

from __future__ import annotations

from .config import SettingReport


_ORDER = (
    ("auto_exposure", "CAP_PROP_AUTO_EXPOSURE"),
    ("auto_wb", "CAP_PROP_AUTO_WB"),
    ("focus", "CAP_PROP_FOCUS"),
    ("exposure", "CAP_PROP_EXPOSURE"),
    ("gain", "CAP_PROP_GAIN"),
    ("wb_temperature", "CAP_PROP_WB_TEMPERATURE"),
    ("brightness", "CAP_PROP_BRIGHTNESS"),
    ("contrast", "CAP_PROP_CONTRAST"),
    ("saturation", "CAP_PROP_SATURATION"),
)


def apply_camera_settings(cap, values, cv2_module):
    """Attempt only populated controls; never infer ranges, units, or mode values."""
    reports = {}
    for name, constant in _ORDER:
        requested = getattr(values, name, None) if not isinstance(values, dict) else values.get(name)
        if requested is None:
            continue
        prop = getattr(cv2_module, constant, None)
        if prop is None:
            reports[name] = SettingReport(
                requested, None, attempted=False,
                error=f"{constant} is unavailable in this OpenCV build",
            )
            continue
        accepted, readback, error, readback_error = None, None, None, None
        try:
            accepted = bool(cap.set(prop, requested))
        except Exception as exc:
            error = str(exc)
        try:
            readback = cap.get(prop)
        except Exception as exc:
            readback_error = str(exc)
        reports[name] = SettingReport(
            requested, accepted, readback, attempted=True, error=error,
            readback_error=readback_error,
        )
    return reports


def read_camera_settings(cap, reports, cv2_module):
    """Read the current values immediately after stabilization, preserving set results."""
    updated = dict(reports)
    for name, constant in _ORDER:
        previous = updated.get(name)
        if previous is None:
            continue
        prop = getattr(cv2_module, constant, None)
        if prop is None:
            continue
        try:
            readback = cap.get(prop)
            updated[name] = SettingReport(
                previous.requested, previous.accepted, readback,
                attempted=previous.attempted, error=previous.error,
                readback_error=None, mode=previous.mode,
            )
        except Exception as exc:
            updated[name] = SettingReport(
                previous.requested, previous.accepted, previous.readback,
                attempted=previous.attempted, error=previous.error,
                readback_error=str(exc), mode=previous.mode,
            )
    return updated
