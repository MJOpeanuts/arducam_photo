"""Minimal RAW processing for the native 108 MP path (black level, demosaic, CCM)."""

from __future__ import annotations

import cv2
import numpy as np

from .ccm import apply_ccm_inplace
from .errors import IspError

BLACK_LEVEL = 16


def process_raw(raw_2d: np.ndarray, ccms=None) -> np.ndarray:
    """raw_2d: uint8 Bayer (rows, cols). Returns BGR uint8 (rows, cols, 3).

    Follows the reference sequence: subtract 16 (saturating), demosaic with
    COLOR_BayerGR2RGB, then optional CCM at 4000 K. The reference treats the
    demosaic output as BGR (OpenCV's Bayer constant naming is inverted), so
    the array is returned as-is and used as BGR throughout.
    """
    try:
        black = cv2.subtract(raw_2d, BLACK_LEVEL)
        img = cv2.cvtColor(black, cv2.COLOR_BayerGR2RGB)
        if ccms is not None:
            apply_ccm_inplace(img, ccms)
    except IspError:
        raise
    except Exception as e:  # cv2.error, MemoryError, ...
        raise IspError(f"RAW processing failed: {e}") from e
    return img
