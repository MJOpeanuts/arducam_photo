"""Minimal RAW processing for the native 108 MP path (black level, demosaic, CCM)."""

from __future__ import annotations

import cv2
import math
import numpy as np

from .ccm import apply_ccm_inplace
from .errors import IspError

BLACK_LEVEL = 16


def process_raw(raw_2d: np.ndarray, ccms=None, *, black_level=BLACK_LEVEL, temperature=4000) -> np.ndarray:
    """raw_2d: uint8 Bayer (rows, cols). Returns BGR uint8 (rows, cols, 3).

    Follows the reference sequence: subtract 16 (saturating), demosaic with
    COLOR_BayerGR2RGB, then optional CCM at 4000 K. The reference treats the
    demosaic output as BGR (OpenCV's Bayer constant naming is inverted), so
    the array is returned as-is and used as BGR throughout.
    """
    if (not isinstance(raw_2d, np.ndarray) or raw_2d.dtype != np.uint8
            or raw_2d.ndim != 2 or min(raw_2d.shape) < 3):
        raise IspError("RAW processing expects a uint8 Bayer HxW image at least 3x3")
    if type(black_level) is not int or not 0 <= black_level <= 255:
        raise IspError("black_level must be an integer in 0..255")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise IspError("temperature must be finite and > 0")
    try:
        black = cv2.subtract(raw_2d, black_level)
        img = cv2.cvtColor(black, cv2.COLOR_BayerGR2RGB)
        if ccms is not None:
            apply_ccm_inplace(img, ccms, temperature)
    except IspError:
        raise
    except Exception as e:  # cv2.error, MemoryError, ...
        raise IspError(f"RAW processing failed: {e}") from e
    return img
