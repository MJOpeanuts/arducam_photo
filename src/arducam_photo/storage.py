"""Optional PNG helper. The capture engine does not depend on it."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from .config import CaptureConfig, CaptureResult
from .engine import capture
from .errors import SaveError


def _verify(path: str, image: np.ndarray) -> None:
    import cv2

    try:
        with open(path, "rb") as f:
            back = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    except OSError as e:
        raise SaveError(f"cannot re-read {path}: {e}") from e
    if back is None or back.shape != image.shape or back.dtype != image.dtype or not np.array_equal(back, image):
        raise SaveError(f"verification failed: {path} differs from the captured image")


def save_png(image: np.ndarray, path, overwrite: bool = False) -> str:
    """Write a BGR uint8 image as PNG atomically (temp file in the same directory),
    verify by decoding it back, and never replace an existing file unless overwrite=True."""
    import cv2

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise SaveError("image must be a uint8 HxWx3 BGR array")
    path = os.path.abspath(os.fspath(path))
    if not path.lower().endswith(".png"):
        raise SaveError("output path must end with .png")
    if os.path.exists(path) and not overwrite:
        raise SaveError(f"{path} exists; pass overwrite=True to replace it")
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise SaveError("PNG encoding failed")
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        raise SaveError(f"output directory does not exist: {directory}")
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".png", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(buf.tobytes())
            f.flush()
            os.fsync(f.fileno())
        del buf
        _verify(tmp, image)
        if overwrite:
            os.replace(tmp, path)
        else:
            try:
                os.link(tmp, path)  # atomic and fails if the target appeared meanwhile
            except FileExistsError as e:
                raise SaveError(f"{path} exists; pass overwrite=True to replace it") from e
            except OSError:
                # No hard links on this filesystem: best-effort check, small race window remains.
                if os.path.exists(path):
                    raise SaveError(f"{path} exists; pass overwrite=True to replace it")
                os.replace(tmp, path)
        _verify(path, image)
    except SaveError:
        raise
    except OSError as e:
        raise SaveError(f"cannot write {path}: {e}") from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def capture_and_save(config: CaptureConfig, path, overwrite: bool = False) -> CaptureResult:
    """Capture then save a verified PNG. Used by both examples."""
    result = capture(config)
    save_png(result.image, path, overwrite=overwrite)
    return result
