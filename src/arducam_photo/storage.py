"""Verified image exports and capture-to-archive compatibility helpers."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, replace

import numpy as np

from .config import CaptureConfig, CaptureResult
from .errors import CameraSetupError, SaveError


def begin_capture_bundle(acquisition, directory) -> str:
    """Publish verified RAW (when available) and a strict per-capture JSON manifest."""
    from .archive import _jsonable, _versions
    from .config import NATIVE_108MP

    directory = os.path.abspath(os.fspath(directory))
    os.makedirs(directory, exist_ok=True)
    info = acquisition.info
    try:
        capture_id = str(uuid.UUID(info.capture_id)) if info.capture_id else str(uuid.uuid4())
    except (ValueError, TypeError, AttributeError) as exc:
        raise SaveError(f"invalid capture id: {exc}") from exc
    marker = os.path.join(directory, f".{capture_id}.incomplete")
    manifest_path = os.path.join(directory, f"{capture_id}.json")
    raw_path = os.path.join(directory, f"{capture_id}.raw")
    png_path = os.path.join(directory, f"{capture_id}.png")
    if any(os.path.lexists(path) for path in (manifest_path, raw_path, png_path)):
        raise SaveError("capture id collides with existing artifacts; no files overwritten")
    try:
        with open(marker, "xb") as f:
            f.write(b"capture publication incomplete\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError as exc:
        raise SaveError(f"capture id already used or cannot create completion marker: {exc}") from exc
    original = acquisition.original
    if not isinstance(original, np.ndarray) or original.dtype != np.uint8 or not original.size:
        raise SaveError("capture original must be a nonempty uint8 array")
    native = info.path == NATIVE_108MP
    raw_record = None
    if native:
        expected_shape = tuple(info.reconstructed_shape or original.shape)
        expected_bytes = int(np.prod(expected_shape))
        if (original.dtype != np.uint8 or not original.flags.c_contiguous
                or tuple(original.shape) != expected_shape or original.nbytes != expected_bytes):
            raise SaveError("native RAW shape, dtype or storage order does not match acquisition metadata")
        raw_tmp = os.path.join(directory, f".{capture_id}.{uuid.uuid4().hex}.raw.tmp")
        digest = hashlib.sha256()
        try:
            with open(raw_tmp, "xb") as f:
                view = memoryview(original).cast("B")
                for offset in range(0, len(view), 1024 * 1024):
                    chunk = view[offset:offset + 1024 * 1024]
                    f.write(chunk)
                    digest.update(chunk)
                f.flush()
                os.fsync(f.fileno())
            if os.path.getsize(raw_tmp) != original.nbytes or _sha256_file(raw_tmp) != digest.hexdigest():
                raise SaveError("RAW size or SHA-256 verification failed")
            os.link(raw_tmp, raw_path)
            os.unlink(raw_tmp)
        except Exception as exc:
            raise SaveError(f"cannot preserve native RAW: {exc}") from exc
        raw_record = {
            "status": "preserved", "file": os.path.basename(raw_path),
            "dimensions": list(original.shape), "dtype": "uint8", "size_bytes": original.nbytes,
            "organization": "headerless C-order Bayer bytes, row-major",
            "bayer_pattern": "RGGB software convention; physical sensor pattern unvalidated",
            "bayer_pattern_status": "derived from existing pipeline convention; hardware not validated",
            "sha256": digest.hexdigest(),
        }
    else:
        raw_record = {"status": "unavailable", "file": None, "bayer_pattern": None,
                      "bayer_pattern_status": "not applicable",
                      "reason": f"{info.path} is a color capture; no native Bayer RAW is available"}

    config = _jsonable(info.config)
    versions = _versions()
    data = {
        "schema_version": 1, "capture_id": capture_id, "captured_at": info.captured_at,
        "program": {"version": versions.get("program"), "commit": versions.get("commit")},
        "acquisition": {
            "path": info.path, "camera": {
                "identity": None, "identity_status": "device identity unavailable through OpenCV",
                "index": info.camera_index, "backend": info.api,
            },
            "transport_dimensions": list(info.transport_size),
            "image_dimensions": list(original.shape[:2][::-1]),
            "dtype": info.dtype, "requested_parameters": config,
            "parameter_reports": _jsonable(info.settings),
            "readback_at": info.settings_readback_at,
            "readback_note": info.settings_readback_note,
            "stabilization": {
                "frames_read": info.frames_read, "failed_reads": info.failed_reads,
                "invalid_buffers": info.invalid_buffers,
            },
            "raw": raw_record,
        },
        "processing": {"status": "pending", "black_level": None, "demosaic": None,
                       "color_correction": None, "temperature": None,
                       "matrix": None, "tuning_file": None},
        "png": {"status": "pending", "file": os.path.basename(png_path)},
    }
    _write_capture_json(manifest_path, data, new=True)
    return manifest_path


def finish_capture_bundle(manifest_path, image, processing, ccm=None) -> str:
    from .archive import _jsonable

    directory = os.path.dirname(os.path.abspath(manifest_path))
    capture_id = os.path.splitext(os.path.basename(manifest_path))[0]
    png_path = os.path.join(directory, f"{capture_id}.png")
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("capture_id") != capture_id:
        raise SaveError("capture manifest identity mismatch")
    save_image(image, png_path, options=OutputOptions())
    data["processing"] = {
        "status": "complete", "parameters": _jsonable(processing),
        "black_level": processing.black_level if data["acquisition"]["path"] == "native_108mp" else None,
        "demosaic": "COLOR_BayerGR2RGB" if data["acquisition"]["path"] == "native_108mp" else None,
        "color_correction": bool(ccm and ccm.get("requested")),
        "temperature": ccm.get("temperature") if ccm else None,
        "matrix": ccm.get("effective_bgr_matrix") if ccm else None,
        "tuning_file": ({k: ccm.get(k) for k in ("path", "sha256")} if ccm else None),
    }
    data["png"] = {"status": "preserved", "file": os.path.basename(png_path),
                   "size_bytes": os.path.getsize(png_path), "sha256": _sha256_file(png_path)}
    _write_capture_json(manifest_path, data)
    marker = os.path.join(directory, f".{capture_id}.incomplete")
    os.unlink(marker)
    return png_path


def fail_capture_bundle(manifest_path, stage, error):
    if not manifest_path or not os.path.isfile(manifest_path):
        return
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    was_rendering = stage in ("render", "output")
    data["processing"] = {
        "status": "failed" if was_rendering else "not_run",
        "raw_preserved": data.get("acquisition", {}).get("raw", {}).get("status") == "preserved",
    }
    data["failure"] = {"stage": stage, "error": str(error)}
    data["png"] = {"status": "failed", "file": os.path.basename(manifest_path)[:-5] + ".png"}
    _write_capture_json(manifest_path, data)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_capture_json(path, data, *, new=False):
    from .archive import _jsonable

    payload = json.dumps(_jsonable(data), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    tmp = os.path.join(os.path.dirname(path), f".{uuid.uuid4().hex}.json.tmp")
    with open(tmp, "xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    published = False
    try:
        if new:
            os.link(tmp, path)
        else:
            os.replace(tmp, path)
        published = True
        with open(path, encoding="utf-8") as f:
            verified = json.load(f)
        if verified.get("capture_id") != data.get("capture_id"):
            raise SaveError("capture JSON identity verification failed")
        raw = verified.get("acquisition", {}).get("raw", {})
        if raw.get("status") == "preserved":
            raw_path = os.path.join(os.path.dirname(path), raw["file"])
            if os.path.getsize(raw_path) != raw["size_bytes"] or _sha256_file(raw_path) != raw["sha256"]:
                raise SaveError("capture JSON does not match RAW size/SHA-256")
    finally:
        if published and os.path.exists(tmp):
            os.unlink(tmp)


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
    directory = os.path.dirname(os.path.abspath(os.fspath(path)))
    capture_manifest = (begin_capture_bundle(acquisition, directory)
                        if hasattr(acquisition, "info") and hasattr(acquisition, "original") else None)
    try:
        manifest = archive_acquisition(acquisition, directory)
    except Exception as e:
        if capture_manifest:
            fail_capture_bundle(capture_manifest, "archive", e)
        raise
    if capture_manifest is None:
        capture_manifest = begin_capture_bundle(acquisition, directory)
    release_error = getattr(acquisition.info, "release_error", None)
    if release_error:
        error = CameraSetupError(f"camera release failed: {release_error}")
        error.manifest_path = manifest
        error.archive_path = manifest
        try:
            record_error(manifest, "release", error)
        except SaveError:
            pass
        fail_capture_bundle(capture_manifest, "release", error)
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
        fail_capture_bundle(capture_manifest, "processing", e)
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
        finish_capture_bundle(capture_manifest, result.image, recipe, ccm)
    except Exception as e:
        e.manifest_path = manifest
        e.archive_path = manifest
        try:
            record_error(manifest, "output", e)
        except SaveError:
            pass
        fail_capture_bundle(capture_manifest, "output", e)
        raise
    return result
