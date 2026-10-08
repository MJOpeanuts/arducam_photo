"""Deterministic, non-destructive rendering of an acquired original."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .ccm import load_ccm_file, validate_ccm_entries
from .errors import IspError
from .isp import process_raw


@dataclass(frozen=True)
class ProcessingRecipe:
    name: str = "reference"
    black_level: int = 16
    apply_ccm: bool = True
    temperature: float = 4000
    grayscale: bool = False
    clahe_clip: float = 0
    clahe_grid: int = 8
    threshold: str = "none"
    adaptive_block: int = 11
    adaptive_c: float = 2
    max_dimension: int | None = None
    version: int = 1

    def validate(self) -> None:
        if self.name not in ("minimal", "black_level", "reference"):
            raise IspError("unknown processing recipe name")
        if type(self.version) is not int or self.version != 1:
            raise IspError("unsupported recipe version")
        for name in ("apply_ccm", "grayscale"):
            if type(getattr(self, name)) is not bool:
                raise IspError(f"{name} must be boolean")
        for name, lower, upper in (
            ("black_level", 0, 255), ("clahe_grid", 1, 64), ("adaptive_block", 3, 255),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < lower or (upper is not None and value > upper):
                raise IspError(f"invalid {name}")
        if self.adaptive_block % 2 != 1:
            raise IspError("adaptive_block must be odd")
        for name in ("temperature", "clahe_clip", "adaptive_c"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise IspError(f"{name} must be finite")
        if self.temperature <= 0 or self.clahe_clip < 0:
            raise IspError("temperature must be > 0 and clahe_clip >= 0")
        if self.threshold not in ("none", "otsu", "adaptive"):
            raise IspError("threshold must be none, otsu or adaptive")
        if self.max_dimension is not None and (
            type(self.max_dimension) is not int or self.max_dimension < 1
        ):
            raise IspError("max_dimension must be None or an integer >= 1")


def process_image(original, recipe, *, raw=False, ccm_path=None, ccm_entries=None) -> np.ndarray:
    """Render uint8 Bayer or BGR without changing or aliasing the original.

    Minimal RAW only demosaics; black_level subtracts the requested offset;
    reference additionally applies explicitly requested CCM. Colour starts as
    an unchanged copy. Thresholding implies grayscale, CLAHE uses luminance
    for colour, and reduction preserves aspect ratio without upscaling.
    Provided CCM entries take precedence over a path, so a recorded snapshot
    renders independently of subsequent tuning-file changes.
    """
    if not isinstance(recipe, ProcessingRecipe):
        raise IspError("recipe must be a ProcessingRecipe")
    recipe.validate()
    if type(raw) is not bool:
        raise IspError("raw must be boolean")
    if not isinstance(original, np.ndarray) or original.dtype != np.uint8:
        raise IspError("original must be a uint8 numpy array")
    if raw:
        if original.ndim != 2 or min(original.shape) < 3:
            raise IspError("RAW original must be a Bayer image at least 3x3")
    elif original.ndim != 3 or original.shape[2] != 3 or min(original.shape[:2]) < 1:
        raise IspError("colour original must be a non-empty BGR HxWx3 image")
    try:
        if raw:
            ccms = None
            if recipe.name == "reference" and recipe.apply_ccm:
                ccms = (validate_ccm_entries(ccm_entries) if ccm_entries is not None
                        else load_ccm_file(ccm_path))
            image = process_raw(
                original, ccms, black_level=0 if recipe.name == "minimal" else recipe.black_level,
                temperature=recipe.temperature,
            )
        else:
            image = original.copy()
        if recipe.max_dimension is not None:
            height, width = image.shape[:2]
            scale = min(1.0, recipe.max_dimension / max(height, width))
            if scale < 1:
                image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                                   interpolation=cv2.INTER_AREA)
        if recipe.grayscale or recipe.threshold != "none":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if recipe.clahe_clip > 0:
            clahe = cv2.createCLAHE(recipe.clahe_clip, (recipe.clahe_grid, recipe.clahe_grid))
            if image.ndim == 2:
                image = clahe.apply(image)
            else:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                lab[..., 0] = clahe.apply(lab[..., 0])
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        if recipe.threshold == "otsu":
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif recipe.threshold == "adaptive":
            image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, recipe.adaptive_block, recipe.adaptive_c)
        return image
    except cv2.error as error:
        raise IspError(f"image processing failed: {error}") from error
