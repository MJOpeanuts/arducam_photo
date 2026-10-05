"""Preview adapter: a camera session for framing only (never used for the photo itself).

Blocking native calls (open, set, read) cannot be interrupted or timed out from Python;
``close`` is only effective between calls. Owned and used by a single worker thread.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from ..config import NATIVE_108MP, SettingReport
from ..engine import _import_cv2, _open_video_capture
from ..errors import CameraOpenError, CameraSetupError

log = logging.getLogger("arducam_photo")

PREVIEW_W, PREVIEW_H = 1280, 720
PREVIEW_FPS = {NATIVE_108MP: 30, "color_720p": 10}  # starting points, "if available"


class PreviewSession:
    def __init__(self, camera_index: int, api: str, path: str, opener: Optional[Callable] = None):
        self.camera_index, self.api, self.path = camera_index, api, path
        self._opener = opener or _open_video_capture
        self._cap = None
        self.settings: dict = {}
        self.failed_reads = 0

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> dict:
        cap = self._opener(self.camera_index, self.api)
        try:
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
                if cap is not None:
                    cap.release()
            except Exception:
                log.exception("release failed")
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
        if cap is not None:
            try:
                cap.release()
            except Exception:
                log.exception("release failed")


def probe_cameras(max_index: int = 5, api: str = "msmf", opener: Optional[Callable] = None) -> list:
    """Indices that open. Opens then releases each index: only on explicit user request."""
    opener = opener or _open_video_capture
    found = []
    for i in range(max_index + 1):
        cap = None
        try:
            cap = opener(i, api)
            if cap is not None and cap.isOpened():
                found.append(i)
        except Exception:
            log.exception("probe %d failed", i)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
    return found
