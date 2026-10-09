"""Preview adapter: a camera session for framing only (never used for the photo itself).

Blocking native calls (open, set, read) cannot be interrupted or timed out from Python;
``close`` is only effective between calls. Owned and used by a single worker thread.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from .. import engine
from ..camera_settings import apply_camera_settings
from ..config import PATHS, SettingReport
from ..engine import _import_cv2, _open_video_capture
from ..errors import CameraOpenError, CameraSetupError, CaptureBusyError

log = logging.getLogger("arducam_photo")

PREVIEW_W, PREVIEW_H = 1280, 720
PREVIEW_FPS = {path: 10 for path in PATHS}  # framing only; never promises full photo FoV


def _claim_camera():
    if not engine._busy.acquire(blocking=False):
        raise CaptureBusyError("another camera acquisition or preview is already running")
    if engine._release_failure is not None:
        engine._busy.release()
        raise CameraSetupError(f"previous camera release failed; restart required: {engine._release_failure}")


def _release_camera(cap):
    if cap is not None:
        try:
            cap.release()
        except Exception as error:
            engine._release_failure = f"camera release failed: {error}"
            log.exception("release failed")


class PreviewSession:
    def __init__(self, camera_index: int, api: str, path: str, opener: Optional[Callable] = None):
        self.camera_index, self.api, self.path = camera_index, api, path
        self._opener = opener or _open_video_capture
        self._cap = None
        self._owns_camera = False
        self.settings: dict = {}
        self.failed_reads = 0

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> dict:
        _claim_camera()
        self._owns_camera = True
        cap = None
        try:
            cap = self._opener(self.camera_index, self.api)
            if cap is None or not cap.isOpened():
                raise CameraOpenError(
                    f"impossible d'ouvrir la caméra (index {self.camera_index}, {self.api}); "
                    "fermez les autres programmes qui l'utilisent et vérifiez l'USB")
            cv2 = _import_cv2()
            self._cap = cap
            self._set("width", cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_W)
            self._set("height", cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_H)
            self._set("fps", cv2.CAP_PROP_FPS, PREVIEW_FPS[self.path])  # a mismatch is not an error
        except BaseException:
            self._cap = None
            try:
                _release_camera(cap)
            finally:
                self._owns_camera = False
                engine._busy.release()
            raise
        return self.settings

    def _set(self, name, prop, value) -> SettingReport:
        try:
            ok = bool(self._cap.set(prop, value))
            rb = self._cap.get(prop)
        except Exception as e:
            raise CameraSetupError(f"réglage {name} impossible: {e}") from e
        rep = self.settings[name] = SettingReport(value, ok, rb)
        return rep

    def set_focus(self, value: int) -> SettingReport:
        """Returns requested value, cap.set result and driver read-back (distinct)."""
        if self._cap is None:
            raise CameraSetupError("preview non ouvert")
        return self._set("focus", _import_cv2().CAP_PROP_FOCUS, value)

    def apply_settings(self, values) -> dict:
        if self._cap is None:
            raise CameraSetupError("preview non ouvert")
        reports = apply_camera_settings(self._cap, values, _import_cv2())
        self.settings.update(reports)
        return reports

    def read(self) -> Optional[np.ndarray]:
        """One BGR uint8 HxWx3 frame, or None on a failed/invalid read (counted)."""
        if self._cap is None:
            return None
        try:
            ok, frame = self._cap.read()
        except Exception:
            ok, frame = False, None
        if (not ok or not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3):
            self.failed_reads += 1
            return None
        return frame

    def close(self) -> None:
        cap, self._cap = self._cap, None
        try:
            _release_camera(cap)
        finally:
            if self._owns_camera:
                self._owns_camera = False
                engine._busy.release()


def probe_cameras(max_index: int = 5, api: str = "msmf", opener: Optional[Callable] = None) -> list:
    """Indices that open. Opens then releases each index: only on explicit user request."""
    opener = opener or _open_video_capture
    found = []
    _claim_camera()
    try:
        for i in range(max_index + 1):
            cap = None
            try:
                cap = opener(i, api)
                if cap is not None and cap.isOpened():
                    found.append(i)
            except Exception:
                log.exception("probe %d failed", i)
            finally:
                _release_camera(cap)
            if engine._release_failure is not None:
                raise CameraSetupError(engine._release_failure)
    finally:
        engine._busy.release()
    return found
