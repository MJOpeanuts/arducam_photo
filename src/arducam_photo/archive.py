"""Immutable acquisition originals, validated offline loading, and export provenance."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import io
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

from .config import NATIVE_108MP, PATHS, MODES
from .errors import CaptureConfigError, IspError, SaveError
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


def _mode_shape(mode):
    descriptor = MODES[mode]
    return ((descriptor.height, descriptor.width * 2) if descriptor.raw
            else (descriptor.height, descriptor.width, 3))


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
        if os.path.basename(os.fspath(path)) != "acquisition.json":
            raise SaveError("archive manifest must be named acquisition.json")
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
        if shape != _mode_shape(mode):
            raise SaveError("original geometry does not match acquisition mode")
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
        if record["path"] != "original" + suffix:
            raise SaveError("archive original must be a directly adjacent original.npy or original.png")
        _confined_path(path, record["path"])
        if type(record["size"]) is not int or not 0 < record["size"] <= MAX_BYTES:
            raise SaveError("invalid original file size")
        sha = record["sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise SaveError("invalid original checksum")
        if not isinstance(data["outputs"], list) or not isinstance(data["errors"], list):
            raise SaveError("invalid manifest history")
        for key, item in data["files"].items():
            if key == "original":
                continue
            _confined_path(path, item["path"])
            _shape(item["shape"], raw=len(item["shape"]) == 2)
            if (item["dtype"] != "uint8" or type(item["size"]) is not int
                    or not 0 < item["size"] <= MAX_BYTES
                    or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in item["sha256"])):
                raise SaveError("invalid output file record")
        for output in data["outputs"]:
            uuid.UUID(output["id"])
            if output["acquisition_id"] != data["id"]:
                raise SaveError("output acquisition identity mismatch")
            file_record = data["files"][output["id"]]
            if any(output[k] != file_record[k] for k in ("path", "shape", "dtype", "size", "sha256")):
                raise SaveError("output history and file record disagree")
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
            if start[24] != 8 or start[25] not in (0, 2, 4, 6):
                raise SaveError("source PNG must be 8-bit grayscale or color")
            if start[26:28] != b"\0\0":
                raise SaveError("unsupported PNG compression/filter")
            channels = {0: 1, 2: 3, 4: 2, 6: 4}[start[25]]
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
                    if len(frame) != 6 or frame[0] != 8 or frame[5] not in (1, 3):
                        raise SaveError("source JPEG must be 8-bit grayscale or color")
                    height, width = struct.unpack(">HH", frame[1:5])
                    channels = frame[5]
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


def snapshot_ccm(path, temperature=4000):
    """Snapshot validated tuning entries and effective reference matrices before rendering."""
    from .ccm import load_ccm_file, select_ccm, bgr_matrix

    if not path:
        raise CaptureConfigError("no CCM tuning file path given")
    path = os.path.abspath(os.fspath(path))
    try:
        if not 0 < os.path.getsize(path) <= MAX_MANIFEST_BYTES:
            raise SaveError("CCM tuning file is empty or too large")
        checksum = _sha256(path)
        entries = load_ccm_file(path)
        matrix = select_ccm(entries, temperature)
        if checksum != _sha256(path):
            raise SaveError("CCM tuning file changed while snapshotting")
    except OSError as e:
        raise CaptureConfigError(f"cannot snapshot CCM tuning file: {e}") from e
    return {
        "requested": True, "version": 1, "path": path, "sha256": checksum,
        "temperature": temperature,
        "entries": [{"ct": entry["ct"], "ccm": entry["ccm"].tolist()} for entry in entries],
        "effective_rgb_matrix": matrix.tolist(),
        "effective_bgr_matrix": bgr_matrix(matrix).astype(np.float32).tolist(),
        "convention": "reference saturation=1, RGB CCM, rot180/transpose BGR float32",
    }


def _ccm_provenance(info_data, recipe=None):
    config = info_data.get("config") or {}
    path = config.get("ccm_path")
    requested = info_data.get("path") == NATIVE_108MP and bool(config.get("apply_ccm"))
    if recipe is not None:
        requested = (info_data.get("path") == NATIVE_108MP
                     and recipe["name"] == "reference" and recipe["apply_ccm"])
    result = {"requested": requested, "path": path, "sha256": None}
    if requested and path:
        try:
            if 0 < os.path.getsize(path) <= MAX_MANIFEST_BYTES:
                result["sha256"] = _sha256(path)
            result = snapshot_ccm(path, recipe["temperature"] if recipe is not None else 4000)
        except (OSError, CaptureConfigError, SaveError, IspError) as e:
            result["error"] = str(e)
    return result


def _source_parent(path, checksum, size):
    manifest = os.path.join(os.path.dirname(path), "acquisition.json")
    if not os.path.isfile(manifest) or not os.path.basename(path).startswith("output-"):
        return None
    data = _read_manifest(manifest)
    for output in data["outputs"]:
        if _confined_path(manifest, output["path"]) == os.path.realpath(path):
            return {
                "acquisition_id": data["id"], "output_id": output["id"],
                "manifest_path": manifest,
                "sha256_verified": output["sha256"] == checksum and output["size"] == size,
            }
    return None


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
    if original.shape != _mode_shape(mode):
        raise SaveError("original geometry does not match acquisition mode")
    os.makedirs(directory, exist_ok=True)
    try:
        identifier = str(uuid.UUID(info_data["capture_id"])) if info_data.get("capture_id") else str(uuid.uuid4())
    except (ValueError, TypeError, AttributeError) as e:
        raise SaveError("invalid acquisition capture_id") from e
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
        original_timings = {}
        if raw:
            phase = time.monotonic()
            contiguous = np.ascontiguousarray(original)
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header, np.lib.format.header_data_from_array_1_0(contiguous))
            original_timings["encode_s"] = time.monotonic() - phase
            phase = time.monotonic()
            with open(original_path, "xb") as f:
                f.write(header.getvalue())
                f.write(memoryview(contiguous).cast("B"))
                f.flush()
                os.fsync(f.fileno())
            original_timings["write_s"] = time.monotonic() - phase
            phase = time.monotonic()
            back = _validated_npy(original_path, original.shape)
            for row in range(0, original.shape[0], 128):
                if not np.array_equal(back[row:row + 128], original[row:row + 128]):
                    raise SaveError("NPY verification failed")
            del back
            original_timings["verify_s"] = time.monotonic() - phase
        else:
            save_image(original, original_path, _timings=original_timings)
        phase = time.monotonic()
        checksum = _sha256(original_path)
        original_timings["hash_s"] = time.monotonic() - phase
        record = {
            "path": filename, "shape": list(original.shape), "dtype": "uint8",
            "size": os.path.getsize(original_path), "sha256": checksum,
            "format": "npy" if raw else "png",
            "timings": original_timings,
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
                           "validation": "reference convention; hardware unvalidated",
                           "ccm": _ccm_provenance(info_data)},
            "counts": {k: info_data.get(k, 0) for k in ("frames_read", "failed_reads", "invalid_buffers")},
            "timings": dict(info_data.get("timings") or {}),
            "files": {"original": record}, "errors": [], "outputs": [],
        }
        data["timings"]["archive_s"] = time.monotonic() - started
        data["timings"]["original"] = original_timings
        acquisition_duration = info_data.get("duration_s") or data["timings"].get("acquisition_s", 0)
        data["timings"]["total_until_archive_s"] = acquisition_duration + data["timings"]["archive_s"]
        _atomic_json(manifest, data, new=True)
        _read_manifest(manifest)
        completed_archive_s = time.monotonic() - started
        def final_timings(metadata):
            metadata["timings"]["archive_s"] = completed_archive_s
            metadata["timings"]["total_until_archive_s"] = acquisition_duration + completed_archive_s
        try:
            _update_manifest(manifest, final_timings)
        except (SaveError, OSError):
            # The complete archive is already published; optional timing refinement
            # must never delete a verified original if a later write cannot complete.
            pass
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
        if os.path.isfile(manifest) and (
            path == manifest or os.path.basename(path).lower() in ("original.npy", "original.png")
        ):
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
        source_hash = _sha256(path)
        source_size = os.path.getsize(path)
        data = {
            "schema_version": 1,
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "sha256:" + source_hash)),
            "source": "external_color_image",
            "source_path": path, "created_at": _now(), "path": "external_image",
            "dtype": "uint8", "reconstructed_shape": list(original.shape),
            "provenance": {"opencv": "BGR", "validation": "external source; not a camera acquisition"},
            "files": {"source": {"path": path, "size": source_size, "sha256": source_hash}},
            "outputs": [], "errors": [],
        }
        parent = _source_parent(path, source_hash, source_size)
        if parent is not None:
            data["parent"] = parent
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


def update_timings(manifest_path, timings, output_path=None):
    """Atomically merge phase/total timings, optionally into one linked output too."""
    manifest_path = os.path.abspath(os.fspath(manifest_path))
    values = _jsonable(timings)
    if not isinstance(values, dict):
        raise SaveError("timings must be a mapping")
    target = None
    if output_path is not None:
        target = os.path.realpath(os.path.join(os.path.dirname(manifest_path), os.fspath(output_path)))
    def update(data):
        if target is not None:
            output = next((item for item in data["outputs"]
                           if _confined_path(manifest_path, item["path"]) == target), None)
            if output is None:
                raise SaveError("timing output is not linked to this acquisition")
            output["timings"].update(values)
            data["timings"]["last_output"] = dict(output["timings"])
        data["timings"].update(values)
    try:
        _update_manifest(manifest_path, update)
    except OSError as e:
        raise SaveError(f"cannot update acquisition timings: {e}") from e


def _normalize_recipe(recipe):
    recipe_data = _jsonable(recipe)
    if not isinstance(recipe_data, dict):
        raise SaveError("recipe must contain full parameters as a dataclass or mapping")
    from .processing import ProcessingRecipe
    try:
        normalized = ProcessingRecipe(**recipe_data)
        normalized.validate()
        recipe_data = _jsonable(normalized)
    except (TypeError, ValueError, IspError) as e:
        raise SaveError(f"invalid output recipe: {e}") from e
    return recipe_data


def _external_source(source):
    """Build lightweight source provenance without decoding/copying its pixels."""
    expected = source if isinstance(source, dict) else None
    if expected is not None:
        if expected.get("source") != "external_color_image":
            raise SaveError("output source metadata must identify an external image")
        path = expected["source_path"]
    else:
        path = os.path.abspath(os.fspath(source))
    path = os.path.abspath(os.fspath(path))
    size = os.path.getsize(path)
    if not 0 < size <= MAX_BYTES:
        raise SaveError("invalid source file size")
    h, w, channels = _image_header(path)
    checksum = _sha256(path)
    identifier = str(uuid.uuid5(uuid.NAMESPACE_URL, "sha256:" + checksum))
    if expected is not None:
        prior = expected["files"]["source"]
        if (checksum != prior["sha256"] or size != prior["size"]
                or identifier != expected["id"]
                or [h, w, 3] != expected["reconstructed_shape"]):
            raise SaveError("external source changed since processing")
    result = {
        "id": identifier, "kind": "external_image", "path": path,
        "size": size, "sha256": checksum, "shape": [h, w, 3], "dtype": "uint8",
        "encoded_channels": channels,
    }
    parent = expected.get("parent") if expected is not None else _source_parent(path, checksum, size)
    if parent is not None:
        result["parent"] = parent
    return result


def _save_external_output(image, source, recipe, options, directory, ccm, caller_timings):
    published = sidecar_published = False
    path = sidecar = None
    try:
        source_record = _external_source(source)
        directory = os.path.abspath(os.fspath(directory or os.path.dirname(source_record["path"])))
        identifier = str(uuid.uuid4())
        extension = "png" if options.format.lower() == "png" else "jpg"
        path = os.path.join(directory, f"output-{identifier}.{extension}")
        sidecar = os.path.splitext(path)[0] + ".json"
        started = time.monotonic()
        timings = dict(caller_timings)
        save_image(image, path, options=options, _timings=timings)
        published = True
        phase = time.monotonic()
        checksum = _sha256(path)
        timings["hash_s"] = time.monotonic() - phase
        timings["save_s"] = time.monotonic() - started
        output = {
            "id": identifier, "source_id": source_record["id"], "created_at": _now(),
            "path": os.path.basename(path), "shape": list(image.shape), "dtype": str(image.dtype),
            "size": os.path.getsize(path), "sha256": checksum,
            "recipe": recipe, "options": _jsonable(options), "timings": timings,
            "ccm": ccm or {"requested": False, "applied": False},
        }
        _atomic_json(sidecar, {
            "schema_version": 1, "kind": "external_source_output",
            "source": source_record, "output": output, "versions": _versions(),
            "errors": [],
        }, new=True)
        sidecar_published = True
        return path
    except Exception as e:
        if published:
            os.unlink(path)
        if sidecar_published:
            os.unlink(sidecar)
        if isinstance(e, SaveError):
            raise
        raise SaveError(f"cannot export external source: {e}") from e


def save_output(image, manifest_path, recipe, options=None, *, directory=None,
                ccm_provenance=None, timings=None) -> str:
    """Export with archive links, or an external-source sidecar without copying its original.

    The source argument accepts a manifest path, external image path, LoadedAcquisition,
    or the small metadata mapping returned when loading an external image.
    """
    options = options or OutputOptions()
    options.validate()
    recipe_data = _normalize_recipe(recipe)
    ccm_provenance = _jsonable(ccm_provenance)
    caller_timings = _jsonable(timings or {})
    if not isinstance(caller_timings, dict):
        raise SaveError("output timings must be a mapping")
    if isinstance(manifest_path, LoadedAcquisition):
        manifest_path = manifest_path.manifest_path or manifest_path.metadata
    if isinstance(manifest_path, dict):
        return _save_external_output(image, manifest_path, recipe_data, options, directory,
                                     ccm_provenance, caller_timings)
    manifest_path = os.path.abspath(os.fspath(manifest_path))
    if not manifest_path.lower().endswith(".json"):
        if manifest_path.lower().endswith(".npy"):
            raise SaveError("RAW outputs require an acquisition manifest")
        return _save_external_output(image, manifest_path, recipe_data, options, directory,
                                     ccm_provenance, caller_timings)
    source_metadata = _read_manifest(manifest_path)
    ccm_required = (source_metadata["path"] == NATIVE_108MP
                    and recipe_data["name"] == "reference" and recipe_data["apply_ccm"])
    if ccm_required and ccm_provenance is None:
        error = SaveError("RAW reference output requires the actual pre-render CCM snapshot")
        record_error(manifest_path, "output", error)
        raise error
    if ccm_required:
        from .ccm import select_ccm, validate_ccm_entries
        try:
            if (ccm_provenance["temperature"] != recipe_data["temperature"]
                    or ccm_provenance["version"] != 1):
                raise SaveError("CCM snapshot and output recipe disagree")
            effective = select_ccm(validate_ccm_entries(ccm_provenance["entries"]), recipe_data["temperature"])
            if not np.array_equal(effective, np.asarray(ccm_provenance["effective_rgb_matrix"])):
                raise SaveError("CCM snapshot effective matrix disagrees with its tuning entries")
        except (KeyError, TypeError, ValueError, CaptureConfigError, IspError) as e:
            raise SaveError(f"invalid render CCM snapshot: {e}") from e
    identifier = str(uuid.uuid4())
    ext = "png" if options.format.lower() == "png" else "jpg"
    filename = f"output-{identifier}.{ext}"
    path = os.path.join(os.path.dirname(manifest_path), filename)
    started = time.monotonic()
    output_timings = dict(caller_timings)
    published = False
    try:
        save_image(image, path, options=options, _timings=output_timings)
        published = True
        phase = time.monotonic()
        checksum = _sha256(path)
        output_timings["hash_s"] = time.monotonic() - phase
        record = {
            "id": identifier, "acquisition_id": _read_manifest(manifest_path)["id"],
            "created_at": _now(), "path": filename, "shape": list(image.shape),
            "dtype": str(image.dtype), "size": os.path.getsize(path), "sha256": checksum,
            "recipe": recipe_data, "options": _jsonable(options),
            "versions": _versions(), "timings": {
                **output_timings, "save_s": time.monotonic() - started},
            "ccm": ccm_provenance or {"requested": False, "applied": False},
        }
        def append(data):
            data["outputs"].append(record)
            data["files"][identifier] = {k: record[k] for k in ("path", "shape", "dtype", "size", "sha256")}
            data["timings"]["last_output"] = record["timings"]
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
