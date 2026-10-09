"""One-shot capture. Blocking; run it outside a UI thread.

Blocking native calls (not interruptible, no timeout promised): VideoCapture
open, cap.set and cap.read. Cancellation is checked only between them.
COM: OpenCV's MSMF backend initialises COM itself in the calling thread; this
module does not touch COM, so open/read/release all happen in the caller thread.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4
from typing import Callable, Optional

import numpy as np

from .ccm import load_ccm_file
from .camera_settings import apply_camera_settings, read_camera_settings
from .config import AcquisitionInfo, AcquisitionResult, CaptureConfig, CaptureResult, SettingReport, NATIVE_108MP, MODES
from .errors import (
    CameraOpenError, CameraReadError, CameraSetupError, CaptureBusyError,
    CaptureCancelled, RawBufferError,
)
from .isp import process_raw

log = logging.getLogger("arducam_photo")
log.addHandler(logging.NullHandler())

TRANSPORT_W, TRANSPORT_H = 6000, 9000
RAW_ROWS, RAW_COLS = 9000, 12000
RAW_BYTES = RAW_ROWS * RAW_COLS  # 108_000_000
P720_W, P720_H, P720_FPS = 1280, 720, 10

_busy = threading.Lock()
_release_failure: Optional[str] = None


def _open_video_capture(index: int, api: str):
    """Default opener; cv2 is imported lazily so importing the package is inert."""
    import cv2

    api_id = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}[api]
    return cv2.VideoCapture(index, api_id)


def _import_cv2():
    import cv2

    return cv2


def _check_cancel(ev: Optional[threading.Event]) -> None:
    if ev is not None and ev.is_set():
        raise CaptureCancelled("capture cancelled")


def validate_raw_buffer(frame) -> np.ndarray:
    """Validate uint8 packed data, without claiming a hardware-verified layout.

    Accepted layouts are flat (N,) or (1,N), reconstructed Bayer (H,2W)
    with an optional singleton channel, or packed transport (H,W,2).
    H and 2W come from RAW_ROWS/RAW_COLS, and N must equal RAW_BYTES.
    """
    if not isinstance(frame, np.ndarray):
        raise RawBufferError(f"RAW buffer is {type(frame).__name__}, expected numpy.ndarray")
    if frame.dtype != np.uint8:
        raise RawBufferError(f"RAW buffer dtype is {frame.dtype}, expected uint8")
    if frame.size != RAW_BYTES or frame.nbytes != RAW_BYTES:
        raise RawBufferError(
            f"RAW buffer has {frame.size} elements / {frame.nbytes} bytes (shape {frame.shape}); "
            f"expected {RAW_BYTES}. The camera is probably not in the 6000x9000 USB 3 mode."
        )
    plausible = {
        (RAW_BYTES,), (1, RAW_BYTES),
        (RAW_ROWS, RAW_COLS), (RAW_ROWS, RAW_COLS, 1),
        (RAW_ROWS, RAW_COLS // 2, 2),
    }
    if frame.shape not in plausible:
        raise RawBufferError(f"implausible RAW transport shape {frame.shape}; expected packed Bayer/YUY2")
    return frame


def validate_720p_frame(frame) -> np.ndarray:
    return validate_color_frame(frame, P720_W, P720_H)


def validate_color_frame(frame, width, height) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise RawBufferError(f"frame is {type(frame).__name__}, expected numpy.ndarray")
    if (frame.dtype != np.uint8 or frame.shape != (height, width, 3)
            or frame.nbytes != height * width * 3):
        raise RawBufferError(
            f"colour frame is dtype {frame.dtype} shape {frame.shape}; expected uint8 ({height}, {width}, 3) BGR"
        )
    return frame


def capture(
    config: CaptureConfig,
    *,
    cancel_event: Optional[threading.Event] = None,
    _opener: Optional[Callable] = None,
) -> CaptureResult:
    """Compatibility facade: acquire and release, then render the owned original."""
    acquired = acquire(config, cancel_event=cancel_event, _opener=_opener)
    if acquired.info.release_error:
        error = CameraSetupError(acquired.info.release_error)
        error.acquisition_result = acquired
        raise error
    _check_cancel(cancel_event)
    cfg = acquired.info.config
    native = cfg.path == NATIVE_108MP
    started = time.monotonic()
    ccms = load_ccm_file(cfg.ccm_path) if native and cfg.apply_ccm else None
    image = process_raw(acquired.original, ccms) if native else acquired.original.copy()
    elapsed = time.monotonic() - started
    info = dataclasses.replace(
        acquired.info, ccm_requested=native and cfg.apply_ccm,
        ccm_applied=ccms is not None, ccm_path=cfg.ccm_path if native and cfg.apply_ccm else None,
        duration_s=acquired.info.duration_s + elapsed,
        timings={**acquired.info.timings, "processing_s": elapsed},
    )
    h, w = image.shape[:2]
    return CaptureResult(image=image, width=w, height=h, info=info)


def acquire(
    config: CaptureConfig,
    cancel_event: Optional[threading.Event] = None,
    _opener: Optional[Callable] = None,
) -> AcquisitionResult:
    """Acquire without ISP, copy the original, then attempt release before returning.

    A completed original survives release failure: inspect info.release_error,
    archive first, and report the failure. Further acquisitions are refused until
    process restart because the driver's ownership state is unknown.
    """
    cfg = dataclasses.replace(config)  # frozen copy for this acquisition
    cfg.validate()
    if not _busy.acquire(blocking=False):
        raise CaptureBusyError("another capture is already running")
    try:
        if _release_failure is not None:
            raise CameraSetupError(
                f"previous camera release failed; restart before another acquisition: {_release_failure}"
            )
        return _run(cfg, cancel_event, _opener or _open_video_capture)
    finally:
        _busy.release()


def _run(cfg, cancel, opener) -> AcquisitionResult:
    global _release_failure
    t0 = time.monotonic()
    native = cfg.path == NATIVE_108MP
    mode = MODES[cfg.path]
    capture_id = str(uuid4())
    captured_at = datetime.now(timezone.utc).isoformat()
    _check_cancel(cancel)

    log.info("open api=%s index=%d path=%s", cfg.api, cfg.camera_index, cfg.path)
    cap = opener(cfg.camera_index, cfg.api)
    result = None
    release_error = None
    try:
        if cap is None or not cap.isOpened():
            raise CameraOpenError(
                f"cannot open camera index {cfg.camera_index} with api {cfg.api}; "
                "check the index, that no other program holds the camera, and the USB link"
            )
        cv2 = _import_cv2()
        settings = {}

        def setp(name, prop, value, required=False):
            try:
                ok = bool(cap.set(prop, value))
                rb = cap.get(prop)
            except Exception as e:
                raise CameraSetupError(f"setting {name} failed: {e}") from e
            settings[name] = SettingReport(value, ok, rb)
            log.info("set %s=%s accepted=%s readback=%s", name, value, ok, rb)
            if required and not ok:
                raise CameraSetupError(f"driver rejected required setting {name}={value}")
            _check_cancel(cancel)

        if native:
            transport = (TRANSPORT_W, TRANSPORT_H)
            setp("width", cv2.CAP_PROP_FRAME_WIDTH, TRANSPORT_W, True)
            setp("height", cv2.CAP_PROP_FRAME_HEIGHT, TRANSPORT_H, True)
            setp("convert_rgb", cv2.CAP_PROP_CONVERT_RGB, 0, True)
        else:
            transport = mode.transport_size
            setp("width", cv2.CAP_PROP_FRAME_WIDTH, mode.width, True)
            setp("height", cv2.CAP_PROP_FRAME_HEIGHT, mode.height, True)
        setp("fps", cv2.CAP_PROP_FPS, mode.fps if cfg.fps is None else cfg.fps)
        settings.update(apply_camera_settings(cap, cfg, cv2))

        setup_s = time.monotonic() - t0
        read_started = time.monotonic()
        validate = validate_raw_buffer if native else lambda f: validate_color_frame(f, mode.width, mode.height)
        frame, frames_read, failed, invalid = _stabilize(cap, cfg, validate, cancel)

        _check_cancel(cancel)
        readback_at = datetime.now(timezone.utc).isoformat()
        settings = read_camera_settings(cap, settings, cv2)
        read_s = time.monotonic() - read_started
        copy_started = time.monotonic()
        received_shape = tuple(frame.shape)
        original = frame.reshape(RAW_ROWS, RAW_COLS).copy() if native else frame.copy()
        del frame
        h, w = original.shape[:2]
        info = AcquisitionInfo(
            config=cfg, api=cfg.api, camera_index=cfg.camera_index, path=cfg.path,
            transport_size=transport, settings=settings, frames_read=frames_read,
            failed_reads=failed, invalid_buffers=invalid,
            duration_s=time.monotonic() - t0,
            received_shape=received_shape, reconstructed_shape=tuple(original.shape),
            dtype=str(original.dtype), capture_id=capture_id, captured_at=captured_at,
            settings_readback_at=readback_at,
            timings={"setup_s": setup_s, "read_s": read_s, "copy_s": time.monotonic() - copy_started},
        )
        log.info("done %dx%d reads=%d failed=%d invalid=%d ccm=%s %.2fs",
                 w, h, frames_read, failed, invalid, info.ccm_applied, info.duration_s)
        result = AcquisitionResult(original=original, info=info)
    finally:
        release_started = time.monotonic()
        try:
            if cap is not None:
                cap.release()
        except Exception as error:
            log.exception("release failed")
            release_error = f"camera release failed: {error}"
            _release_failure = release_error
            if result is None:
                raise CameraSetupError(release_error) from error
        release_s = time.monotonic() - release_started
    elapsed = time.monotonic() - t0
    return dataclasses.replace(result, info=dataclasses.replace(
        result.info, duration_s=elapsed,
        timings={**result.info.timings, "release_s": release_s, "acquisition_s": elapsed},
        camera_released=release_error is None, release_error=release_error,
    ))


def _stabilize(cap, cfg, validate, cancel):
    """Read until stabilization_reads + 1 valid frames; the last valid frame is kept.

    Failed reads and invalid buffers are counted separately and bounded.
    Transient bad frames are tolerated up to the bounds.
    """
    needed = cfg.stabilization_reads + 1
    good = failed = invalid = 0
    frame = None
    while good < needed:
        _check_cancel(cancel)
        try:
            ok, data = cap.read()
        except Exception as e:
            raise CameraReadError(f"read raised: {e}") from e
        if not ok:
            failed += 1
            if failed > cfg.max_failed_reads:
                raise CameraReadError(f"{failed} failed reads (limit {cfg.max_failed_reads})")
            continue
        try:
            frame = validate(data)
        except RawBufferError as e:
            invalid += 1
            log.warning("invalid buffer %d/%d: %s", invalid, cfg.max_invalid_buffers, e)
            if invalid > cfg.max_invalid_buffers:
                raise RawBufferError(f"{invalid} invalid buffers (limit {cfg.max_invalid_buffers}); last: {e}") from e
            continue
        good += 1
    return frame, good + failed + invalid, failed, invalid
