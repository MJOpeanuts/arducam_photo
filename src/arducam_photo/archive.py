"""Immutable acquisition originals, validated offline loading, and export provenance."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import struct
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import COLOR_720P, NATIVE_108MP, PATHS
from .errors import SaveError
from .storage import OutputOptions, save_image

MAX_BYTES = 512 * 1024 * 1024
MAX_PIXELS = 120_000_000
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_metadata_lock = threading.RLock()


@dataclasses.dataclass(frozen=True)
class LoadedAcquisition:
    original: np.ndarray
    metadata: dict
    manifest_path: str | None
    raw: bool


def _now():
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise SaveError(f"metadata value is not serializable: {type(value).__name__}")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(shape, raw):
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != (2 if raw else 3)
        or any(type(x) is not int or x <= 0 for x in shape)
        or (not raw and shape[2] != 3)
        or math.prod(shape[:2]) > MAX_PIXELS
        or math.prod(shape) > MAX_BYTES
    ):
        raise SaveError("invalid or oversized original shape")
    return tuple(shape)


def _confined_path(manifest, relative):
    if not isinstance(relative, str) or not relative or os.path.isabs(relative):
        raise SaveError("archive file path must be relative")
    root = os.path.realpath(os.path.dirname(manifest))
    target = os.path.realpath(os.path.join(root, relative))
    if target == root or os.path.commonpath((root, target)) != root:
        raise SaveError("archive file escapes its acquisition directory")
    return target


def _read_manifest(path):
    try:
        if os.path.getsize(path) > MAX_MANIFEST_BYTES:
            raise SaveError("manifest is too large")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise SaveError("unsupported acquisition manifest schema")
        uuid.UUID(data["id"])
        mode = data["path"]
        if mode not in PATHS:
            raise SaveError("invalid acquisition mode")
        raw = mode == NATIVE_108MP
        record = data["files"]["original"]
        shape = _shape(record["shape"], raw)
        if record["dtype"] != "uint8" or data["dtype"] != "uint8":
            raise SaveError("original dtype must be uint8")
        if tuple(data["reconstructed_shape"]) != shape:
            raise SaveError("manifest original shapes disagree")
        info = data["info"]
        if not isinstance(info, dict) or info.get("path") != mode:
            raise SaveError("manifest and acquisition modes disagree")
        if info.get("dtype", "uint8") != "uint8":
            raise SaveError("acquisition dtype must be uint8")
        if tuple(info.get("reconstructed_shape", shape)) != shape:
            raise SaveError("acquisition and original shapes disagree")
        config = info.get("config", {})
        if not isinstance(config, dict) or config.get("path", mode) != mode:
            raise SaveError("config and acquisition modes disagree")
        received = data["received_shape"]
        if (not isinstance(received, list) or not 1 <= len(received) <= 3
                or any(type(x) is not int or x <= 0 for x in received)
                or math.prod(received) != math.prod(shape)):
            raise SaveError("received and reconstructed byte counts disagree")
        suffix = ".npy" if raw else ".png"
        if not record["path"].lower().endswith(suffix):
            raise SaveError("original format does not match acquisition mode")
        _confined_path(path, record["path"])
        if type(record["size"]) is not int or not 0 < record["size"] <= MAX_BYTES:
            raise SaveError("invalid original file size")
        sha = record["sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise SaveError("invalid original checksum")
        if not isinstance(data["outputs"], list) or not isinstance(data["errors"], list):
            raise SaveError("invalid manifest history")
        return data
    except SaveError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        raise SaveError(f"invalid acquisition manifest: {e}") from e


def _atomic_json(path, data, *, new=False):
    scratch = os.path.join(os.path.dirname(path), f".metadata-{uuid.uuid4().hex}")
    owned = False
    try:
        payload = json.dumps(_jsonable(data), ensure_ascii=False, indent=2, allow_nan=False).encode()
        if len(payload) > MAX_MANIFEST_BYTES:
            raise SaveError("manifest is too large")
        with open(scratch, "xb") as f:
            owned = True
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        if new:
            os.link(scratch, path)
        else:
            os.replace(scratch, path)
    finally:
        if owned and os.path.exists(scratch):
            os.unlink(scratch)


def _validated_npy(path, expected=None):
    """Read only the bounded header and exact file size before numpy maps any data."""
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f, max_header_size=4096)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f, max_header_size=4096)
            else:
                raise SaveError("unsupported NPY header version")
            shape = _shape(shape, True)
            if dtype != np.dtype("uint8") or fortran:
                raise SaveError("original NPY must be C-order uint8, never pickle")
            if expected is not None and shape != tuple(expected):
                raise SaveError("NPY header and manifest shape disagree")
            if os.fstat(f.fileno()).st_size != f.tell() + math.prod(shape):
                raise SaveError("NPY file has truncated or trailing data")
        return np.load(path, mmap_mode="r", allow_pickle=False, max_header_size=4096)
    except SaveError:
        raise
    except (OSError, ValueError, EOFError, TypeError) as e:
        raise SaveError(f"invalid NPY original: {e}") from e


def _image_header(path):
    """Check dimensions/type before letting OpenCV allocate the decoded image."""
    with open(path, "rb") as f:
        start = f.read(32)
        if start.startswith(b"\x89PNG\r\n\x1a\n"):
            if len(start) < 29 or start[12:16] != b"IHDR" or struct.unpack(">I", start[8:12])[0] != 13:
                raise SaveError("invalid PNG header")
            width, height = struct.unpack(">II", start[16:24])
            if start[24] != 8 or start[25] not in (2, 6):
                raise SaveError("source PNG must be 8-bit color")
            if start[26:28] != b"\0\0":
                raise SaveError("unsupported PNG compression/filter")
            channels = 4 if start[25] == 6 else 3
        elif start[:2] == b"\xff\xd8":
            f.seek(2)
            while True:
                if f.read(1) != b"\xff":
                    raise SaveError("invalid JPEG header")
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if not marker or marker in (b"\xda", b"\xd9"):
                    raise SaveError("JPEG has no dimensions")
                length_bytes = f.read(2)
                if len(length_bytes) != 2:
                    raise SaveError("truncated JPEG header")
                length = struct.unpack(">H", length_bytes)[0]
                if length < 2:
                    raise SaveError("invalid JPEG segment")
                if marker[0] in (0xC0, 0xC1, 0xC2):
                    frame = f.read(6)
                    if len(frame) != 6 or frame[0] != 8 or frame[5] != 3:
                        raise SaveError("source JPEG must be 8-bit color")
                    height, width = struct.unpack(">HH", frame[1:5])
                    channels = 3
                    break
                f.seek(length - 2, 1)
                if f.tell() > MAX_BYTES:
                    raise SaveError("oversized JPEG header")
        else:
            raise SaveError("source must be a PNG or JPEG color image")
    _shape([height, width, 3], False)
    if height * width * channels > MAX_BYTES:
        raise SaveError("oversized color image")
    return height, width, channels


def _load_color(path, expected=None, *, archived=False):
    import cv2

    if not 0 < os.path.getsize(path) <= MAX_BYTES:
        raise SaveError("invalid or oversized color file")
    h, w, channels = _image_header(path)
    if archived and channels != 3:
        raise SaveError("archived original must have three BGR channels")
    if expected is not None and (h, w, 3) != tuple(expected):
        raise SaveError("color header and manifest shape disagree")
    with open(path, "rb") as f:
        back = cv2.imdecode(
            np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if back is None or back.dtype != np.uint8 or back.shape != (h, w, 3):
        raise SaveError("color image decode failed")
    return back


def _versions():
    import cv2

    try:
        program = importlib.metadata.version("arducam-photo")
    except importlib.metadata.PackageNotFoundError:
        program = "0.1.0"
    result = {"program": program, "numpy": np.__version__, "opencv": cv2.__version__, "python": platform.python_version()}
    # Optional Git identity without invoking a shell or requiring git installation.
    root = Path(__file__).resolve().parents[2]
    try:
        head = (root / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            head = (root / ".git" / head[5:]).read_text().strip()
        if len(head) == 40 and all(c in "0123456789abcdef" for c in head):
            result["commit"] = head
    except OSError:
        pass
    return result


def archive_acquisition(acquisition, directory) -> str:
    """Persist and verify the untouched original, then publish a schema-1 manifest."""
    started = time.monotonic()
    directory = os.path.abspath(os.fspath(directory))
    original, info = acquisition.original, acquisition.info
    info_data = _jsonable(info)
    if not isinstance(info_data, dict):
        raise SaveError("acquisition info must be a dataclass or mapping")
    mode = info_data["path"]
    if mode not in PATHS:
        raise SaveError("unsupported acquisition mode")
    raw = mode == NATIVE_108MP
    if not isinstance(original, np.ndarray) or original.dtype != np.uint8:
        raise SaveError("original must be uint8")
    _shape(original.shape, raw)
    os.makedirs(directory, exist_ok=True)
    identifier = str(uuid.uuid4())
    folder = os.path.join(directory, identifier)
    # mkdir is exclusive. Never delete a colliding acquisition.
    try:
        os.mkdir(folder)
    except OSError as e:
        raise SaveError(f"cannot create acquisition directory: {e}") from e
    manifest = os.path.join(folder, "acquisition.json")
    try:
        filename = "original.npy" if raw else "original.png"
        original_path = os.path.join(folder, filename)
        if raw:
            with open(original_path, "xb") as f:
                np.save(f, np.ascontiguousarray(original), allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            back = _validated_npy(original_path, original.shape)
            for row in range(0, original.shape[0], 128):
                if not np.array_equal(back[row:row + 128], original[row:row + 128]):
                    raise SaveError("NPY verification failed")
            del back
        else:
            save_image(original, original_path)
        record = {
            "path": filename, "shape": list(original.shape), "dtype": "uint8",
            "size": os.path.getsize(original_path), "sha256": _sha256(original_path),
            "format": "npy" if raw else "png",
        }
        data = {
            "schema_version": 1, "id": identifier, "created_at": _now(),
            "captured_at": info_data.get("captured_at") or _now(),
            "versions": _versions(), "path": mode, "dtype": "uint8",
            "received_shape": info_data.get("received_shape", list(original.shape)),
            "reconstructed_shape": list(original.shape), "info": info_data,
            "camera": {"index": info_data.get("camera_index"), "api": info_data.get("api"),
                       "settings": info_data.get("settings", {})},
            "provenance": {"bayer": "reference GR convention" if raw else None,
                           "demosaic": "COLOR_BayerGR2RGB" if raw else None, "opencv": "BGR",
                           "validation": "reference convention; hardware unvalidated"},
            "counts": {k: info_data.get(k, 0) for k in ("frames_read", "failed_reads", "invalid_buffers")},
            "timings": dict(info_data.get("timings") or {}),
            "files": {"original": record}, "errors": [], "outputs": [],
        }
        data["timings"]["archive_s"] = time.monotonic() - started
        _atomic_json(manifest, data, new=True)
        _read_manifest(manifest)
        return manifest
    except Exception as e:
        shutil.rmtree(folder)
        if isinstance(e, SaveError):
            raise
        raise SaveError(f"cannot archive acquisition: {e}") from e


def load_acquisition(path) -> LoadedAcquisition:
    """Load a validated archive or a clearly identified external color source."""
    path = os.path.abspath(os.fspath(path))
    try:
        manifest = path if path.lower().endswith(".json") else os.path.join(os.path.dirname(path), "acquisition.json")
        if os.path.isfile(manifest):
            data = _read_manifest(manifest)
            record = data["files"]["original"]
            original_path = _confined_path(manifest, record["path"])
            if path != manifest and os.path.realpath(path) != original_path:
                raise SaveError("file is not this manifest's original")
            if os.path.getsize(original_path) != record["size"] or _sha256(original_path) != record["sha256"]:
                raise SaveError("original size or SHA256 does not match manifest")
            raw = data["path"] == NATIVE_108MP
            original = (_validated_npy(original_path, record["shape"]) if raw
                        else _load_color(original_path, record["shape"], archived=True))
            return LoadedAcquisition(original, data, manifest, raw)
        if path.lower().endswith((".npy", ".json")):
            raise SaveError("original NPY requires its acquisition manifest")
        original = _load_color(path)
        data = {
            "schema_version": 1, "id": str(uuid.uuid4()), "source": "external_color_image",
            "source_path": path, "created_at": _now(), "path": COLOR_720P,
            "dtype": "uint8", "reconstructed_shape": list(original.shape),
            "provenance": {"opencv": "BGR", "validation": "external source; not a camera acquisition"},
            "files": {"source": {"path": path, "size": os.path.getsize(path), "sha256": _sha256(path)}},
            "outputs": [], "errors": [],
        }
        return LoadedAcquisition(original, data, None, False)
    except SaveError:
        raise
    except (OSError, ValueError, TypeError) as e:
        raise SaveError(f"cannot load acquisition: {e}") from e


def _update_manifest(manifest, change):
    """Serialize thread/process writers; publish complete metadata with atomic replace."""
    manifest = os.path.abspath(os.fspath(manifest))
    lock_path = manifest + ".lock"
    with _metadata_lock:
        deadline = time.monotonic() + 5
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise SaveError("acquisition metadata is locked by another writer")
                time.sleep(0.02)
        try:
            data = _read_manifest(manifest)
            change(data)
            _atomic_json(manifest, data)
        finally:
            os.unlink(lock_path)


def record_error(manifest_path, stage, error):
    """Append a processing/render/export failure without changing the original."""
    try:
        _update_manifest(manifest_path, lambda data: data["errors"].append({
            "stage": str(stage), "type": type(error).__name__,
            "message": str(error), "at": _now(),
        }))
    except OSError as e:
        raise SaveError(f"cannot record acquisition error: {e}") from e


def save_output(image, manifest_path, recipe, options=None) -> str:
    """Write a unique derived image and atomically link full provenance to its archive."""
    options = options or OutputOptions()
    options.validate()
    manifest_path = os.path.abspath(os.fspath(manifest_path))
    _read_manifest(manifest_path)
    recipe_data = _jsonable(recipe)
    if not isinstance(recipe_data, dict):
        raise SaveError("recipe must contain full parameters as a dataclass or mapping")
    from .processing import ProcessingRecipe
    try:
        normalized = ProcessingRecipe(**recipe_data)
        normalized.validate()
        recipe_data = _jsonable(normalized)
    except (TypeError, ValueError) as e:
        raise SaveError(f"invalid output recipe: {e}") from e
    identifier = str(uuid.uuid4())
    ext = "png" if options.format.lower() == "png" else "jpg"
    filename = f"output-{identifier}.{ext}"
    path = os.path.join(os.path.dirname(manifest_path), filename)
    started = time.monotonic()
    published = False
    try:
        save_image(image, path, options=options)
        published = True
        record = {
            "id": identifier, "acquisition_id": _read_manifest(manifest_path)["id"],
            "created_at": _now(), "path": filename, "shape": list(image.shape),
            "dtype": str(image.dtype), "size": os.path.getsize(path), "sha256": _sha256(path),
            "recipe": recipe_data, "options": _jsonable(options),
            "versions": _versions(), "timings": {"save_s": time.monotonic() - started},
        }
        def append(data):
            data["outputs"].append(record)
            data["files"][identifier] = {k: record[k] for k in ("path", "shape", "dtype", "size", "sha256")}
        _update_manifest(manifest_path, append)
        return path
    except Exception as e:
        if published:
            os.unlink(path)
        try:
            record_error(manifest_path, "output", e)
        except (SaveError, OSError):
            pass
        if isinstance(e, SaveError):
            raise
        raise SaveError(f"cannot save acquisition output: {e}") from e
