"""Verified image exports and capture-to-archive compatibility helpers."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, replace

import numpy as np

from .config import CaptureConfig, CaptureResult
from .errors import CameraSetupError, SaveError


@dataclass(frozen=True)
class OutputOptions:
    format: str = "png"
    jpeg_quality: int = 95

    def validate(self):
        if self.format.lower() not in ("png", "jpeg", "jpg"):
            raise SaveError("output format must be png or jpeg")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise SaveError("JPEG quality must be an integer from 1 to 100")


def _validate_image(image):
    if (
        not isinstance(image, np.ndarray) or image.dtype != np.uint8
        or image.ndim not in (2, 3)
        or (image.ndim == 3 and image.shape[2] != 3)
        or not image.size
    ):
        raise SaveError("image must be a nonempty uint8 grayscale or HxWx3 BGR array")


def _verify(path: str, image: np.ndarray, *, lossless=True) -> None:
    import cv2

    try:
        with open(path, "rb") as f:
            back = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    except (OSError, cv2.error) as e:
        raise SaveError(f"cannot re-read {path}: {e}") from e
    if (
        back is None or back.shape != image.shape or back.dtype != image.dtype
        or (lossless and not np.array_equal(back, image))
    ):
        raise SaveError(f"verification failed: {path} differs from the exported image")


def _protect_original(path):
    # An exported image must never replace an archive's immutable source.
    from .archive import _read_manifest, _confined_path

    manifest = os.path.join(os.path.dirname(path), "acquisition.json")
    if os.path.isfile(manifest):
        metadata = _read_manifest(manifest)
        original = _confined_path(manifest, metadata["files"]["original"]["path"])
        if os.path.realpath(path) == os.path.realpath(original):
            raise SaveError("an archived original cannot be overwritten")
        if os.path.exists(path) and os.path.samefile(path, original):
            raise SaveError("an archived original cannot be overwritten")


def _discard_publication(path, identity):
    try:
        current = os.lstat(path)
        if (current.st_dev, current.st_ino) == identity:
            os.unlink(path)
    except OSError:
        pass


def save_image(image, path, overwrite=False, options=None, *, _timings=None) -> str:
    """Verify an export before publishing it; no-overwrite publication is atomic."""
    import cv2

    _validate_image(image)
    path = os.path.abspath(os.fspath(path))
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        raise SaveError("output path must end with .png, .jpg or .jpeg")
    options = options or OutputOptions("png" if ext == ".png" else "jpeg")
    options.validate()
    if (ext == ".png") != (options.format.lower() == "png"):
        raise SaveError("output format and filename extension disagree")
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        raise SaveError(f"output directory does not exist: {directory}")
    _protect_original(path)
    if os.path.lexists(path) and not overwrite:
        raise SaveError(f"{path} exists; pass overwrite=True to replace it")
    scratch = os.path.join(directory, f".write-{uuid.uuid4().hex}{ext}")
    owned = False
    published = False
    published_identity = None
    timings = _timings if _timings is not None else {}
    try:
        started = time.monotonic()
        params = [] if ext == ".png" else [cv2.IMWRITE_JPEG_QUALITY, options.jpeg_quality]
        ok, buf = cv2.imencode(ext, image, params)
        if not ok:
            raise SaveError("image encoding failed")
        timings["encode_s"] = time.monotonic() - started
        started = time.monotonic()
        with open(scratch, "xb") as f:
            owned = True
            f.write(buf.tobytes())
            f.flush()
            os.fsync(f.fileno())
        timings["write_s"] = time.monotonic() - started
        started = time.monotonic()
        _verify(scratch, image, lossless=ext == ".png")
        timings["pre_verify_s"] = time.monotonic() - started
        stat = os.stat(scratch)
        published_identity = (stat.st_dev, stat.st_ino)
        started = time.monotonic()
        if overwrite:
            _protect_original(path)
            os.replace(scratch, path)
        else:
            # Fail closed on filesystems without hard links; never fall back to replace.
            os.link(scratch, path)
        published = True
        timings["publish_s"] = time.monotonic() - started
        started = time.monotonic()
        _verify(path, image, lossless=ext == ".png")
        timings["post_verify_s"] = time.monotonic() - started
        timings["verify_s"] = timings["pre_verify_s"] + timings["post_verify_s"]
    except SaveError:
        if published:
            _discard_publication(path, published_identity)
        raise
    except (OSError, cv2.error) as e:
        if published:
            _discard_publication(path, published_identity)
        raise SaveError(f"cannot write {path}: {e}") from e
    finally:
        if owned and os.path.exists(scratch):
            os.unlink(scratch)
    return path


def save_png(image, path, overwrite: bool = False) -> str:
    """Save a byte-exact uint8 grayscale or BGR PNG."""
    return save_image(image, path, overwrite, OutputOptions())


def capture_and_save(config: CaptureConfig, path, overwrite: bool = False) -> CaptureResult:
    """Archive the acquisition before rendering and exporting a compatible result."""
    from . import engine
    from .archive import archive_acquisition, record_error, save_output, _update_manifest, snapshot_ccm
    from .config import NATIVE_108MP
    from .processing import ProcessingRecipe, process_image

    total_started = time.monotonic()
    acquisition = engine.acquire(config)
    manifest = archive_acquisition(acquisition, os.path.dirname(os.path.abspath(os.fspath(path))))
    release_error = getattr(acquisition.info, "release_error", None)
    if release_error:
        error = CameraSetupError(f"camera release failed: {release_error}")
        error.manifest_path = manifest
        error.archive_path = manifest
        try:
            record_error(manifest, "release", error)
        except SaveError:
            pass
        raise error
    cfg = acquisition.info.config
    native = cfg.path == NATIVE_108MP
    recipe = ProcessingRecipe(apply_ccm=cfg.apply_ccm)
    started = time.monotonic()
    try:
        ccm = snapshot_ccm(cfg.ccm_path, recipe.temperature) if native and recipe.apply_ccm else None
        processing_options = {"ccm_entries": ccm["entries"]} if ccm is not None else {}
        image = process_image(acquisition.original, recipe, raw=native, ccm_path=cfg.ccm_path,
                              **processing_options)
        elapsed = time.monotonic() - started
        info = replace(
            acquisition.info, ccm_requested=native and cfg.apply_ccm,
            ccm_applied=native and cfg.apply_ccm,
            ccm_path=cfg.ccm_path if native and cfg.apply_ccm else None,
            duration_s=acquisition.info.duration_s + elapsed,
            timings={**acquisition.info.timings, "processing_s": elapsed},
        )
        result = CaptureResult(image=image, width=image.shape[1], height=image.shape[0], info=info)
    except Exception as e:
        e.manifest_path = manifest
        e.archive_path = manifest
        try:
            record_error(manifest, "processing", e)
        except SaveError:
            pass
        raise
    try:
        requested_path = os.path.abspath(os.fspath(path))
        extension = os.path.splitext(requested_path)[1].lower()
        options = OutputOptions("png" if extension == ".png" else "jpeg")
        canonical = save_output(result.image, manifest, recipe, options, ccm_provenance=ccm,
                                timings={"processing_s": info.timings["processing_s"]})
        export_timings = {}
        save_image(result.image, requested_path, overwrite=overwrite, options=options,
                   _timings=export_timings)
        elapsed = time.monotonic() - total_started
        def link_export(data):
            for output in data["outputs"]:
                if output["path"] == os.path.basename(canonical):
                    output["export_path"] = requested_path
                    output["timings"].update({"export": export_timings, "processing_s": info.timings["processing_s"],
                                              "total_s": elapsed})
            data["timings"].update({"processing_s": info.timings["processing_s"], "total_s": elapsed})
        _update_manifest(manifest, link_export)
    except Exception as e:
        e.manifest_path = manifest
        e.archive_path = manifest
        try:
            record_error(manifest, "output", e)
        except SaveError:
            pass
        raise
    return result
