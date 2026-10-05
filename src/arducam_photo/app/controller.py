"""Single camera worker: owns and serialises every camera operation.

States (guarded by a lock): idle -> opening -> preview -> closing -> idle, and idle -> capturing -> idle.
Preview and capture can never hold the camera together: capture requires ``idle`` and preview
opening requires ``idle``. Events go to ``listener(event, **payload)`` from the worker thread;
a UI must marshal them (Qt queued signal). Frame notifications are bounded: at most one is
outstanding until the consumer calls ``take_frame``; only the latest frame is kept.

Limits: OpenCV open/set/read/release are native, not interruptible, and no timeout is promised.
``shutdown`` waits for the worker (optionally bounded by ``timeout``) but cannot abort a blocked call.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np

from .. import engine
from ..config import CaptureConfig
from ..ccm import load_ccm_file
from ..config import NATIVE_108MP
from ..errors import CaptureBusyError, CaptureError
from ..storage import save_png
from .paths import ensure_dir, unique_photo_path
from .preview import PreviewSession, probe_cameras

log = logging.getLogger("arducam_photo")

IDLE, OPENING, PREVIEW, CLOSING, CAPTURING = "idle", "opening", "preview", "closing", "capturing"
THUMB_MAX = 1280


class ModeError(CaptureError):
    """Operation not allowed in the current state (e.g. second capture, preview during capture)."""


@dataclasses.dataclass(frozen=True)
class CaptureJob:
    config: CaptureConfig  # frozen copy of the profile-derived configuration
    photo_dir: str


def _thumbnail(image: np.ndarray) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    s = min(1.0, THUMB_MAX / max(h, w))
    if s >= 1.0:
        return image.copy()
    return cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


class CameraController:
    def __init__(self, listener: Optional[Callable] = None, *, opener: Optional[Callable] = None,
                 capture_fn: Optional[Callable] = None, save_fn: Optional[Callable] = None,
                 clock: Callable = time.monotonic):
        self._listener = listener or (lambda event, **kw: None)
        self._opener = opener
        self._capture_fn = capture_fn or engine.capture
        self._save_fn = save_fn or save_png
        self._clock = clock
        self._q: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self._state = IDLE
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[PreviewSession] = None  # worker thread only
        self._latest: Optional[np.ndarray] = None
        self._frame_pending = False
        self._acq_times: deque = deque(maxlen=64)
        self._disp_times: deque = deque(maxlen=64)
        self._cancel = threading.Event()

    # ---- public, thread-safe ----
    @property
    def state(self) -> str:
        return self._state

    @property
    def camera_in_use(self) -> bool:
        return self._state != IDLE

    @property
    def can_change_mode(self) -> bool:
        return self._state != CAPTURING and self._state not in (OPENING, CLOSING)

    def open_preview(self, camera_index: int, api: str, path: str) -> None:
        self._transition(IDLE, OPENING, "le preview ne peut pas démarrer (état %s)")
        self._submit(("open", camera_index, api, path))

    def close_preview(self) -> None:
        with self._lock:
            if self._state != PREVIEW:
                raise ModeError(f"pas de preview actif (état {self._state})")
            self._state = CLOSING
        self._submit(("close",))

    def set_focus(self, value: int) -> None:
        if self._state not in (OPENING, PREVIEW):
            raise ModeError("réglage du focus uniquement pendant le preview")
        self._submit(("focus", int(value)))

    def capture(self, job: CaptureJob) -> None:
        """Reserve the camera synchronously (double trigger refused) and run on the worker."""
        with self._lock:
            if self._state == CAPTURING:
                raise CaptureBusyError("une capture est déjà en cours")
            if self._state != IDLE:
                raise ModeError(f"capture impossible pendant l'état {self._state}; quittez le preview d'abord")
            self._state = CAPTURING
        self._submit(("capture", job))

    def verify_ccm(self, path: Optional[str]) -> None:
        """Validate the colour-correction file on the worker (no camera access, any state)."""
        self._submit(("ccm", path))

    def probe(self, max_index: int = 5, api: str = "msmf") -> None:
        self._transition(IDLE, IDLE, "détection impossible (état %s)")
        self._submit(("probe", max_index, api))

    def take_frame(self) -> Optional[np.ndarray]:
        """Latest frame (owned by the caller, never mutated by the worker); re-arms notification."""
        with self._lock:
            frame, self._latest, self._frame_pending = self._latest, None, False
            if frame is not None:
                self._disp_times.append(self._clock())
        return frame

    def fps(self) -> tuple:
        """(acquired fps, displayed fps) measured over the recent window, not CAP_PROP_FPS."""
        with self._lock:
            return self._rate(self._acq_times), self._rate(self._disp_times)

    def clear_frame(self) -> None:
        with self._lock:
            self._latest, self._frame_pending = None, False
            self._acq_times.clear()
            self._disp_times.clear()

    def shutdown(self, timeout: Optional[float] = None) -> bool:
        """Release the camera and stop the worker. Returns False if it did not stop in time."""
        t = self._thread
        if t is None:
            return True
        self._cancel.set()
        self._q.put(("stop",))
        t.join(timeout)
        return not t.is_alive()

    # ---- internals ----
    def _rate(self, times) -> float:
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    def _transition(self, expect: str, new: str, msg: str) -> None:
        with self._lock:
            if self._state != expect:
                raise ModeError(msg % self._state)
            self._state = new

    def _set_state(self, new: str) -> None:
        with self._lock:
            self._state = new

    def _submit(self, cmd) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="camera-worker", daemon=True)
                self._thread.start()
        self._q.put(cmd)

    def _emit(self, event: str, **kw) -> None:
        try:
            self._listener(event, **kw)
        except Exception:
            log.exception("listener failed for %s", event)

    def _run(self) -> None:
        try:
            while True:
                try:
                    cmd = self._q.get_nowait() if self._session else self._q.get()
                except queue.Empty:
                    self._pump_frame()
                    continue
                if cmd[0] == "stop":
                    break
                self._handle(cmd)
        finally:
            if self._session is not None:
                self._session.close()
                self._session = None

    def _handle(self, cmd) -> None:
        kind = cmd[0]
        if kind == "open":
            self._do_open(*cmd[1:])
        elif kind == "close":
            self._do_close()
        elif kind == "focus":
            self._do_focus(cmd[1])
        elif kind == "capture":
            self._do_capture(cmd[1])
        elif kind == "ccm":
            try:
                load_ccm_file(cmd[1])
                self._emit("ccm_checked", path=cmd[1], ok=True, error=None)
            except Exception as e:
                self._emit("ccm_checked", path=cmd[1], ok=False, error=e)
        elif kind == "probe":
            self._emit("cameras", indices=probe_cameras(cmd[1], cmd[2], self._opener))

    def _do_open(self, index, api, path) -> None:
        session = PreviewSession(index, api, path, self._opener)
        try:
            settings = session.open()
        except Exception as e:
            self._set_state(IDLE)
            self._emit("preview_failed", error=e)
            return
        self._session = session
        self.clear_frame()
        self._set_state(PREVIEW)
        self._emit("preview_opened", settings=settings)

    def _do_close(self) -> None:
        if self._session is not None:
            self._session.close()  # camera released before the event is sent
            self._session = None
        self.clear_frame()
        self._set_state(IDLE)
        self._emit("preview_closed")

    def _do_focus(self, value) -> None:
        if self._session is None:
            return
        try:
            rep = self._session.set_focus(value)
        except Exception as e:
            self._emit("focus_failed", requested=value, error=e)
            return
        self._emit("focus_applied", requested=value, accepted=rep.accepted, readback=rep.readback)

    def _pump_frame(self) -> None:
        frame = self._session.read()
        if frame is None:
            if self._session.failed_reads and self._session.failed_reads % 50 == 0:
                self._emit("preview_warning", failed_reads=self._session.failed_reads)
            time.sleep(0.005)  # avoid a hot loop on a failing camera
            return
        with self._lock:
            self._acq_times.append(self._clock())
            self._latest = frame  # previous unread frame is dropped: no backlog
            notify = not self._frame_pending
            self._frame_pending = True
        if notify:
            self._emit("frame")

    def _do_capture(self, job: CaptureJob) -> None:
        try:
            self._emit("capture_step", step="Préparation et vérification du profil")
            cfg = job.config
            if cfg.path == NATIVE_108MP and cfg.apply_ccm:
                load_ccm_file(cfg.ccm_path)  # fail before the camera is touched
            self._emit("capture_step", step="Acquisition et traitement (moteur, appel bloquant)")
            result = self._capture_fn(dataclasses.replace(job.config), cancel_event=self._cancel)
            try:
                self._emit("capture_step", step="Enregistrement du PNG")
                ensure_dir(job.photo_dir)
                path = self._save_fn(result.image, unique_photo_path(job.photo_dir))
                thumb = _thumbnail(result.image)
                info, size = result.info, (result.width, result.height)
            finally:
                del result  # free ~324 MB of pixels promptly
            self._emit("capture_done", path=path, thumbnail=thumb, info=info, size=size)
        except Exception as e:
            self._emit("capture_failed", error=e, recoverable=isinstance(e, (CaptureError, OSError)))
        finally:
            self._set_state(IDLE)  # the engine has released the camera in every case
            self._emit("capture_idle")
