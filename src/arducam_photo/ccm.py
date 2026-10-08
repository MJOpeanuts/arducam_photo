"""Manufacturer tuning file (arducam_108mp.json) loading and color correction.

Re-implemented from the documented behaviour of the reference demo (see README);
no manufacturer code or data is copied.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from .errors import CaptureConfigError, IspError

REFERENCE_CT = 4000
_ROWS_PER_CHUNK = 256  # bounds temporary float memory (~256*12000*3*4 B per chunk)

_RGB2Y = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.500], [0.500, -0.419, -0.081]])
_Y2RGB = np.array([[1.0, 0.0, 1.402], [1.0, -0.345, -0.714], [1.0, 1.771, 0.0]])


def load_ccm_file(path) -> list:
    """Load and validate the tuning file; return list of {"ct": float, "ccm": 3x3 ndarray}."""
    if not path:
        raise CaptureConfigError("no CCM tuning file path given")
    p = os.fspath(path)
    if not os.path.isfile(p):
        raise CaptureConfigError(f"CCM tuning file not found: {p}")
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise CaptureConfigError(f"CCM tuning file unreadable or not valid JSON: {p}: {e}") from e
    ccms = data.get("ccms") if isinstance(data, dict) else None
    return validate_ccm_entries(ccms, source=f"CCM tuning file {p}")


def validate_ccm_entries(ccms, *, source="CCM snapshot") -> list:
    """Copy and normalize validated JSON-compatible tuning entries."""
    if not isinstance(ccms, list) or not ccms:
        raise CaptureConfigError(f"{source}: missing non-empty 'ccms' list")
    out = []
    last_ct = -math.inf
    for i, e in enumerate(ccms):
        if not isinstance(e, dict) or "ct" not in e or "ccm" not in e:
            raise CaptureConfigError(f"{source}: entry {i} needs 'ct' and 'ccm'")
        ct, m = e["ct"], e["ccm"]
        if isinstance(ct, bool) or not isinstance(ct, (int, float)) or not math.isfinite(ct) or ct <= 0:
            raise CaptureConfigError(f"{source}: entry {i} has invalid 'ct'")
        if ct <= last_ct:
            raise CaptureConfigError(f"{source}: 'ct' values must be strictly increasing")
        last_ct = ct
        try:
            arr = np.array(m, dtype=np.float64)
        except (TypeError, ValueError) as ex:
            raise CaptureConfigError(f"{source}: entry {i} 'ccm' not numeric") from ex
        if arr.size != 9 or not np.all(np.isfinite(arr)):
            raise CaptureConfigError(f"{source}: entry {i} 'ccm' must hold 9 finite numbers")
        out.append({"ct": float(ct), "ccm": arr.reshape(3, 3)})
    return out


def select_ccm(ccms: list, ct: float = REFERENCE_CT) -> np.ndarray:
    """Interpolate the 3x3 RGB CCM at color temperature ct (reference: 4000 K)."""
    if (isinstance(ct, bool) or not isinstance(ct, (int, float))
            or not math.isfinite(ct) or ct <= 0):
        raise IspError("CCM temperature must be finite and > 0")
    if not ccms:
        raise IspError("CCM list must not be empty")
    if ct <= ccms[0]["ct"]:
        return ccms[0]["ccm"].copy()
    if ct >= ccms[-1]["ct"]:
        return ccms[-1]["ccm"].copy()
    idx = next(i for i, e in enumerate(ccms) if e["ct"] >= ct)
    lam = (ct - ccms[idx - 1]["ct"]) / (ccms[idx]["ct"] - ccms[idx - 1]["ct"])
    return lam * ccms[idx]["ccm"] + (1.0 - lam) * ccms[idx - 1]["ccm"]


def bgr_matrix(ccm_rgb: np.ndarray, saturation: float = 1.0) -> np.ndarray:
    """Matrix M such that out_bgr_row = in_bgr_row @ M (reference: saturation, rot180, transpose)."""
    s = np.diag([1.0, saturation, saturation])
    m = _Y2RGB @ s @ _RGB2Y @ ccm_rgb
    return np.rot90(m, 2).T


def apply_ccm_inplace(img: np.ndarray, ccms: list, ct: float = REFERENCE_CT) -> np.ndarray:
    """Apply color correction to a BGR uint8 image in place, in row chunks."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise IspError("color correction expects a uint8 HxWx3 image")
    m = bgr_matrix(select_ccm(ccms, ct)).astype(np.float32)
    for y in range(0, img.shape[0], _ROWS_PER_CHUNK):
        chunk = img[y : y + _ROWS_PER_CHUNK]
        flat = chunk.reshape(-1, 3).astype(np.float32)
        res = np.clip(flat @ m, 0, 255)
        chunk[...] = res.astype(np.uint8).reshape(chunk.shape)
    return img
