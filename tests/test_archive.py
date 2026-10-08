import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arducam_photo import archive
from arducam_photo.config import COLOR_720P, NATIVE_108MP
from arducam_photo.errors import SaveError
from arducam_photo.storage import OutputOptions, save_png


def acquisition(raw=False):
    image = np.arange(120 if raw else 360, dtype=np.uint8).reshape(
        (10, 12) if raw else (10, 12, 3))
    info = {
        "path": NATIVE_108MP if raw else COLOR_720P, "api": "any", "camera_index": 0,
        "config": {"path": NATIVE_108MP if raw else COLOR_720P},
        "received_shape": [10, 6, 2] if raw else [10, 12, 3],
        "reconstructed_shape": list(image.shape), "dtype": "uint8",
        "settings": {"width": {"requested": 12, "accepted": True, "readback": 12}},
        "frames_read": 6, "timings": {"acquire_s": 0.1},
    }
    return SimpleNamespace(original=image, info=info)


def write_metadata(manifest, data):
    Path(manifest).write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("raw", [False, True])
def test_archive_roundtrip_and_provenance(tmp_path, raw):
    a = acquisition(raw)
    before = a.original.copy()
    manifest = archive.archive_acquisition(a, tmp_path)
    loaded = archive.load_acquisition(manifest)
    assert Path(manifest).is_absolute()
    assert loaded.raw is raw
    assert loaded.manifest_path == manifest
    assert np.array_equal(loaded.original, before)
    assert np.array_equal(a.original, before)
    metadata = loaded.metadata
    original = Path(manifest).parent / metadata["files"]["original"]["path"]
    assert np.array_equal(archive.load_acquisition(original).original, before)
    assert metadata["schema_version"] == 1
    assert metadata["created_at"].endswith("+00:00")
    assert metadata["counts"]["frames_read"] == 6
    assert metadata["camera"]["settings"]["width"]["accepted"] is True
    assert metadata["provenance"]["validation"] == "reference convention; hardware unvalidated"
    assert metadata["files"]["original"]["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert set(metadata["versions"]) >= {"program", "numpy", "opencv"}
    assert metadata["timings"]["archive_s"] >= 0
    if raw:
        assert loaded.original.flags.writeable is False
    assert len(list(Path(manifest).parent.iterdir())) == 2


def test_unique_archives_and_outputs(tmp_path):
    a = acquisition()
    m1 = archive.archive_acquisition(a, tmp_path)
    m2 = archive.archive_acquisition(a, tmp_path)
    assert m1 != m2
    recipe = {"name": "minimal", "black_level": 0, "grayscale": False}
    p1 = archive.save_output(a.original, m1, recipe)
    p2 = archive.save_output(a.original, m1, recipe, OutputOptions("jpeg", 93))
    assert p1 != p2
    loaded = archive.load_acquisition(m1)
    outputs = loaded.metadata["outputs"]
    assert len(outputs) == 2
    assert all(outputs[0]["recipe"][k] == v for k, v in recipe.items())
    assert outputs[0]["recipe"]["version"] == 1
    assert "temperature" in outputs[0]["recipe"] and "max_dimension" in outputs[0]["recipe"]
    assert outputs[1]["options"]["jpeg_quality"] == 93
    assert outputs[0]["acquisition_id"] == loaded.metadata["id"]
    assert np.array_equal(loaded.original, a.original)


def test_original_cannot_be_replaced_even_with_overwrite(tmp_path):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    original = Path(m).parent / "original.png"
    before = original.read_bytes()
    with pytest.raises(SaveError, match="original"):
        save_png(np.zeros((10, 12, 3), np.uint8), original, overwrite=True)
    assert original.read_bytes() == before


def test_processing_errors_keep_original_and_thread_updates(tmp_path):
    m = archive.archive_acquisition(acquisition(True), tmp_path)
    before = (Path(m).parent / "original.npy").read_bytes()
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda i: archive.record_error(m, "processing", ValueError(str(i))), range(12)))
    loaded = archive.load_acquisition(m)
    assert len(loaded.metadata["errors"]) == 12
    assert (Path(m).parent / "original.npy").read_bytes() == before
    assert not Path(m + ".lock").exists()


@pytest.mark.parametrize("field,value", [
    ("dtype", "object"), ("schema_version", 2), ("schema_version", True),
    ("path", "unknown"), ("reconstructed_shape", [120000000, 120000000]),
    ("received_shape", [1, 1]), ("id", "not-a-uuid"),
])
def test_reject_malformed_manifest_before_loading(tmp_path, field, value, monkeypatch):
    m = archive.archive_acquisition(acquisition(True), tmp_path)
    data = json.loads(Path(m).read_text())
    data[field] = value
    write_metadata(m, data)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("must reject before numpy loads"))
    with pytest.raises(SaveError):
        archive.load_acquisition(m)


@pytest.mark.parametrize("relative", ["../outside.npy", "/etc/passwd"])
def test_path_escape_rejected(tmp_path, relative):
    m = archive.archive_acquisition(acquisition(True), tmp_path)
    data = json.loads(Path(m).read_text())
    data["files"]["original"]["path"] = relative
    write_metadata(m, data)
    with pytest.raises(SaveError):
        archive.load_acquisition(m)


def test_symlink_escape_rejected(tmp_path):
    m = archive.archive_acquisition(acquisition(True), tmp_path / "archives")
    outside = tmp_path / "outside.npy"
    outside.write_bytes((Path(m).parent / "original.npy").read_bytes())
    original = Path(m).parent / "original.npy"
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(SaveError, match="escapes"):
        archive.load_acquisition(m)


def test_corruption_rejected(tmp_path):
    m = archive.archive_acquisition(acquisition(True), tmp_path)
    original = Path(m).parent / "original.npy"
    payload = bytearray(original.read_bytes())
    payload[-1] ^= 255
    original.write_bytes(payload)
    with pytest.raises(SaveError, match="SHA256"):
        archive.load_acquisition(m)


@pytest.mark.parametrize("bad", ["object", "truncated", "trailing", "oversized"])
def test_npy_header_checked_before_numpy_load(tmp_path, bad, monkeypatch):
    path = tmp_path / "bad.npy"
    if bad == "object":
        np.save(path, np.array([{"unsafe": True}], dtype=object))
    elif bad == "oversized":
        with path.open("wb") as f:
            np.lib.format.write_array_header_1_0(f, {
                "shape": (1_000_000_000, 1_000_000_000), "fortran_order": False, "descr": "|u1",
            })
    else:
        np.save(path, acquisition(True).original, allow_pickle=False)
        data = path.read_bytes()
        path.write_bytes(data[:-1] if bad == "truncated" else data + b"x")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("must reject before numpy loads"))
    with pytest.raises(SaveError):
        archive._validated_npy(path)


@pytest.mark.parametrize("ext", ["png", "jpg"])
def test_external_source_is_identified(tmp_path, ext):
    from arducam_photo.storage import save_image
    path = tmp_path / f"source.{ext}"
    save_image(acquisition().original, path)
    loaded = archive.load_acquisition(path)
    assert loaded.raw is False and loaded.manifest_path is None
    assert loaded.metadata["source"] == "external_color_image"
    assert loaded.metadata["source_path"] == str(path)
    assert loaded.original.shape == (10, 12, 3)


def test_unmanifested_npy_rejected(tmp_path):
    path = tmp_path / "original.npy"
    np.save(path, acquisition(True).original)
    with pytest.raises(SaveError, match="manifest"):
        archive.load_acquisition(path)


def test_oversized_png_rejected_before_decode(tmp_path, monkeypatch):
    import cv2
    import struct
    path = tmp_path / "huge.png"
    save_png(acquisition().original, path)
    data = bytearray(path.read_bytes())
    data[16:24] = struct.pack(">II", 100000000, 100000000)
    path.write_bytes(data)
    monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: pytest.fail("must reject before allocation"))
    with pytest.raises(SaveError):
        archive.load_acquisition(path)


def test_archive_collision_does_not_delete_old_folder(tmp_path, monkeypatch):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    identifier = Path(m).parent.name
    before = Path(m).read_bytes()
    import uuid
    monkeypatch.setattr(archive.uuid, "uuid4", lambda: uuid.UUID(identifier))
    with pytest.raises(SaveError):
        archive.archive_acquisition(acquisition(), tmp_path)
    assert Path(m).read_bytes() == before


def test_archive_disk_failure_cleans_only_new_folder(tmp_path, monkeypatch):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(archive, "_atomic_json", fail)
    with pytest.raises(SaveError, match="disk full"):
        archive.archive_acquisition(acquisition(True), tmp_path)
    assert list(tmp_path.iterdir()) == [Path(m).parent]
    assert archive.load_acquisition(m).original.shape == (10, 12, 3)


def test_output_manifest_failure_removes_unlinked_output(tmp_path, monkeypatch):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    before = (Path(m).parent / "original.png").read_bytes()
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(archive, "_atomic_json", fail)
    with pytest.raises(SaveError):
        archive.save_output(acquisition().original, m, {"name": "minimal"})
    assert len(list(Path(m).parent.iterdir())) == 2
    assert (Path(m).parent / "original.png").read_bytes() == before
