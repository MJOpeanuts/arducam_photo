import hashlib
import json
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arducam_photo import archive
from arducam_photo.config import COLOR_720P, NATIVE_108MP
from arducam_photo.errors import SaveError
from arducam_photo.storage import OutputOptions, save_png


@pytest.fixture(autouse=True)
def tiny_archive_geometry(monkeypatch):
    for name, descriptor in archive.MODES.items():
        monkeypatch.setitem(archive.MODES, name, replace(
            descriptor, height=10, width=6 if descriptor.raw else 12))


def acquisition(raw=False):
    image = np.arange(120 if raw else 360, dtype=np.uint8).reshape(
        (10, 12) if raw else (10, 12, 3))
    info = {
        "path": NATIVE_108MP if raw else COLOR_720P, "api": "any", "camera_index": 0,
        "config": {"path": NATIVE_108MP if raw else COLOR_720P},
        "received_shape": [10, 6, 2] if raw else [10, 12, 3],
        "reconstructed_shape": list(image.shape), "dtype": "uint8",
        "settings": {"width": {"requested": 12, "accepted": True, "readback": 12}},
        "frames_read": 6, "duration_s": 0.1, "timings": {"acquire_s": 0.1},
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
    assert metadata["timings"]["total_until_archive_s"] == 0.1 + metadata["timings"]["archive_s"]
    assert set(metadata["timings"]["original"]) >= {"encode_s", "write_s", "verify_s", "hash_s"}
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


@pytest.mark.parametrize("mode", ["color_4k", "color_12mp"])
def test_additional_color_modes_archive(tmp_path, mode):
    a = acquisition()
    a.info["path"] = mode
    a.info["config"]["path"] = mode
    m = archive.archive_acquisition(a, tmp_path)
    assert archive.load_acquisition(m).metadata["path"] == mode


def test_output_path_escape_in_metadata_is_rejected(tmp_path):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    archive.save_output(acquisition().original, m, {"name": "minimal"})
    data = json.loads(Path(m).read_text())
    identifier = data["outputs"][0]["id"]
    data["outputs"][0]["path"] = "../escape.png"
    data["files"][identifier]["path"] = "../escape.png"
    write_metadata(m, data)
    with pytest.raises(SaveError, match="escapes"):
        archive.load_acquisition(m)


def test_failed_output_is_recorded_without_changing_original(tmp_path):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    with pytest.raises(SaveError):
        archive.save_output(np.ones((3, 3), np.float64), m, {"name": "minimal"})
    loaded = archive.load_acquisition(m)
    assert len(loaded.metadata["errors"]) == 1
    assert loaded.metadata["errors"][0]["stage"] == "output"
    assert loaded.metadata["outputs"] == []


def test_capture_id_preserved_and_ccm_content_provenance(tmp_path):
    import uuid
    a = acquisition(True)
    a.info["capture_id"] = str(uuid.uuid4())
    tuning = tmp_path / "tuning.json"
    tuning.write_text('{"ccms": [{"ct": 4000, "ccm": [[1,0,0],[0,1,0],[0,0,1]]}]}')
    a.info["config"].update({"apply_ccm": True, "ccm_path": str(tuning)})
    m = archive.archive_acquisition(a, tmp_path / "archives")
    loaded = archive.load_acquisition(m)
    assert loaded.metadata["id"] == a.info["capture_id"] == Path(m).parent.name
    assert loaded.metadata["provenance"]["ccm"]["sha256"] == hashlib.sha256(tuning.read_bytes()).hexdigest()
    assert loaded.metadata["provenance"]["ccm"]["effective_rgb_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


@pytest.mark.parametrize("ext", ["png", "jpg"])
def test_grayscale_source_decodes_as_bgr(tmp_path, ext):
    from arducam_photo.storage import save_image
    path = tmp_path / f"gray.{ext}"
    image = acquisition(True).original
    save_image(image, path)
    loaded = archive.load_acquisition(path)
    assert loaded.original.shape == (10, 12, 3)
    assert np.array_equal(loaded.original[:, :, 0], loaded.original[:, :, 1])
    assert loaded.manifest_path is None


def test_geometry_must_match_mode(tmp_path):
    a = acquisition()
    a.original = np.zeros((11, 12, 3), dtype=np.uint8)
    with pytest.raises(SaveError, match="geometry"):
        archive.archive_acquisition(a, tmp_path)


@pytest.mark.parametrize("source_kind", ["path", "loaded", "metadata"])
def test_external_exports_sidecar_without_copying_source(tmp_path, source_kind, monkeypatch):
    from arducam_photo.processing import ProcessingRecipe
    source = tmp_path / "input.png"
    save_png(acquisition().original, source)
    before = source.read_bytes()
    loaded = archive.load_acquisition(source)
    argument = {"path": source, "loaded": loaded, "metadata": loaded.metadata}[source_kind]
    def no_decode(*args, **kwargs):
        pytest.fail("export must not reload or copy external original")
    monkeypatch.setattr(archive, "_load_color", no_decode)
    out = archive.save_output(loaded.original[:, :, 0], argument,
                              ProcessingRecipe(grayscale=True, apply_ccm=False))
    sidecar = Path(out).with_suffix(".json")
    data = json.loads(sidecar.read_text())
    assert source.read_bytes() == before
    assert data["kind"] == "external_source_output"
    assert data["source"]["id"] == loaded.metadata["id"]
    assert data["source"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert data["output"]["recipe"]["grayscale"] is True
    assert data["output"]["source_id"] == data["source"]["id"]
    assert data["output"]["shape"] == [10, 12]
    assert not list(tmp_path.glob("*/acquisition.json"))
    assert len(list(tmp_path.iterdir())) == 3


def test_external_source_changes_are_rejected_at_export(tmp_path):
    source = tmp_path / "input.png"
    save_png(acquisition().original, source)
    loaded = archive.load_acquisition(source)
    save_png(np.zeros((10, 12, 3), np.uint8), source, overwrite=True)
    with pytest.raises(SaveError, match="changed"):
        archive.save_output(loaded.original, loaded.metadata, {"name": "minimal"})
    assert list(tmp_path.iterdir()) == [source]


def test_saved_grayscale_archive_output_can_reload_as_external_source(tmp_path):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    out = archive.save_output(acquisition(True).original, m,
                              {"name": "minimal", "grayscale": True})
    loaded = archive.load_acquisition(out)
    assert loaded.original.shape == (10, 12, 3)
    assert loaded.manifest_path is None
    assert loaded.metadata["source"] == "external_color_image"
    assert loaded.metadata["path"] == "external_image"
    assert loaded.metadata["parent"]["acquisition_id"] == Path(m).parent.name
    assert loaded.metadata["parent"]["sha256_verified"] is True
    second = archive.load_acquisition(out)
    assert second.metadata["id"] == loaded.metadata["id"]
    exported = archive.save_output(loaded.original, loaded.metadata, {"name": "minimal"})
    data = json.loads(Path(exported).with_suffix(".json").read_text())
    assert data["source"]["parent"] == loaded.metadata["parent"]


def test_external_sidecar_failure_cleans_only_new_output(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    save_png(acquisition().original, source)
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(archive, "_atomic_json", fail)
    with pytest.raises(SaveError, match="disk full"):
        archive.save_output(acquisition().original, source, {"name": "minimal"})
    assert list(tmp_path.iterdir()) == [source]


def test_effective_ccm_snapshot_is_preserved_after_file_replacement(tmp_path):
    from arducam_photo.processing import ProcessingRecipe
    tuning = tmp_path / "tuning.json"
    tuning.write_text(json.dumps({"ccms": [
        {"ct": 3000, "ccm": np.eye(3).tolist()},
        {"ct": 5000, "ccm": (2 * np.eye(3)).tolist()},
    ]}))
    snapshot = archive.snapshot_ccm(tuning, 4000)
    assert snapshot["effective_rgb_matrix"] == (1.5 * np.eye(3)).tolist()
    a = acquisition(True)
    a.info["config"].update({"apply_ccm": True, "ccm_path": str(tuning)})
    m = archive.archive_acquisition(a, tmp_path / "archives")
    tuning.write_text("invalid replacement")
    archive.save_output(acquisition().original, m, ProcessingRecipe(),
                        ccm_provenance=snapshot)
    stored = archive.load_acquisition(m).metadata["outputs"][0]["ccm"]
    assert stored == snapshot
    assert len(stored["entries"]) == 2
    assert len(stored["effective_bgr_matrix"]) == 3


def test_invalid_tuning_does_not_destroy_archived_original(tmp_path):
    a = acquisition(True)
    tuning = tmp_path / "invalid.json"
    tuning.write_text("not JSON")
    a.info["config"].update({"apply_ccm": True, "ccm_path": str(tuning)})
    m = archive.archive_acquisition(a, tmp_path / "archives")
    loaded = archive.load_acquisition(m)
    assert np.array_equal(loaded.original, a.original)
    assert loaded.metadata["provenance"]["ccm"]["error"]
    assert loaded.metadata["provenance"]["ccm"]["sha256"]


def test_library_native_capture_uses_exact_snapshot_without_rereading_tuning(tmp_path, monkeypatch):
    from arducam_photo import engine, processing
    from arducam_photo.config import AcquisitionInfo, AcquisitionResult, CaptureConfig
    from arducam_photo.storage import capture_and_save
    a = acquisition(True)
    tuning = tmp_path / "tuning.json"
    tuning.write_text(json.dumps({"ccms": [{"ct": 4000, "ccm": np.eye(3).tolist()}]}))
    expected_hash = hashlib.sha256(tuning.read_bytes()).hexdigest()
    cfg = CaptureConfig(path=NATIVE_108MP, ccm_path=str(tuning))
    info = AcquisitionInfo(
        config=cfg, api="any", camera_index=0, path=NATIVE_108MP, transport_size=(6, 10),
        received_shape=(10, 6, 2), reconstructed_shape=(10, 12), dtype="uint8",
    )
    monkeypatch.setattr(engine, "acquire", lambda config: AcquisitionResult(a.original, info))
    renderer = processing.process_image
    def replace_file_after_snapshot(*args, **kwargs):
        assert kwargs["ccm_entries"]
        tuning.write_text("changed during rendering")
        return renderer(*args, **kwargs)
    monkeypatch.setattr(processing, "process_image", replace_file_after_snapshot)
    result = capture_and_save(cfg, tmp_path / "rendered.png")
    assert result.image.shape == (10, 12, 3)
    manifest, = tmp_path.glob("*/acquisition.json")
    loaded = archive.load_acquisition(manifest)
    assert loaded.metadata["outputs"][0]["ccm"]["sha256"] == expected_hash
    assert loaded.metadata["outputs"][0]["ccm"]["effective_rgb_matrix"] == np.eye(3).tolist()


def test_nonstandard_manifest_name_is_rejected(tmp_path):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    custom = Path(m).with_name("custom.json")
    Path(m).rename(custom)
    with pytest.raises(SaveError, match="named acquisition.json"):
        archive.load_acquisition(custom)


def test_nested_original_layout_is_rejected(tmp_path):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    root = Path(m).parent
    nested = root / "nested"
    nested.mkdir()
    (root / "original.png").rename(nested / "original.png")
    data = json.loads(Path(m).read_text())
    data["files"]["original"]["path"] = "nested/original.png"
    write_metadata(m, data)
    with pytest.raises(SaveError, match="directly adjacent"):
        archive.load_acquisition(m)


def test_library_release_failure_archives_original_without_rendering(tmp_path, monkeypatch):
    from arducam_photo import engine, processing
    from arducam_photo.config import AcquisitionInfo, AcquisitionResult, CaptureConfig
    from arducam_photo.errors import CameraSetupError
    from arducam_photo.storage import capture_and_save
    a = acquisition(True)
    cfg = CaptureConfig(path=NATIVE_108MP, apply_ccm=False)
    info = AcquisitionInfo(
        config=cfg, api="any", camera_index=0, path=NATIVE_108MP, transport_size=(6, 10),
        received_shape=(10, 6, 2), reconstructed_shape=(10, 12), dtype="uint8",
        camera_released=False, release_error="native release failed",
    )
    monkeypatch.setattr(engine, "acquire", lambda config: AcquisitionResult(a.original, info))
    def no_render(*args, **kwargs):
        pytest.fail("release failure must not render or report success")
    monkeypatch.setattr(processing, "process_image", no_render)
    with pytest.raises(CameraSetupError, match="release failed") as raised:
        capture_and_save(cfg, tmp_path / "result.png")
    loaded = archive.load_acquisition(raised.value.manifest_path)
    assert raised.value.archive_path == raised.value.manifest_path
    assert np.array_equal(loaded.original, a.original)
    assert loaded.metadata["info"]["camera_released"] is False
    assert loaded.metadata["errors"][-1]["stage"] == "release"
    assert loaded.metadata["outputs"] == []
    assert not (tmp_path / "result.png").exists()


def test_caller_render_timings_are_linked_with_measured_save_timings(tmp_path):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    archive.save_output(acquisition().original, m, {"name": "minimal"},
                        timings={"render_s": 0.125})
    loaded = archive.load_acquisition(m)
    timings = loaded.metadata["outputs"][0]["timings"]
    assert timings["render_s"] == 0.125
    assert set(timings) >= {"encode_s", "write_s", "pre_verify_s", "post_verify_s", "hash_s", "save_s"}
    assert loaded.metadata["timings"]["last_output"] == timings


def test_offline_output_records_actual_ccm_not_original_capture_config(tmp_path):
    from arducam_photo.processing import ProcessingRecipe
    old = tmp_path / "capture-tuning.json"
    actual = tmp_path / "offline-tuning.json"
    old.write_text(json.dumps({"ccms": [{"ct": 4000, "ccm": np.eye(3).tolist()}]}))
    actual.write_text(json.dumps({"ccms": [{"ct": 4000, "ccm": (2 * np.eye(3)).tolist()}]}))
    a = acquisition(True)
    a.info["config"].update({"apply_ccm": True, "ccm_path": str(old)})
    m = archive.archive_acquisition(a, tmp_path / "archives")
    snapshot = archive.snapshot_ccm(actual)
    archive.save_output(acquisition().original, m, ProcessingRecipe(),
                        ccm_provenance=snapshot)
    loaded = archive.load_acquisition(m)
    assert loaded.metadata["provenance"]["ccm"]["path"] == str(old)
    output = loaded.metadata["outputs"][0]
    assert output["ccm"]["path"] == str(actual)
    assert output["ccm"]["effective_rgb_matrix"] == (2 * np.eye(3)).tolist()
    assert output["ccm"]["sha256"] != loaded.metadata["provenance"]["ccm"]["sha256"]


def test_raw_reference_output_without_actual_snapshot_is_rejected(tmp_path):
    from arducam_photo.processing import ProcessingRecipe
    m = archive.archive_acquisition(acquisition(True), tmp_path)
    with pytest.raises(SaveError, match="actual pre-render CCM snapshot"):
        archive.save_output(acquisition().original, m, ProcessingRecipe())
    loaded = archive.load_acquisition(m)
    assert loaded.metadata["outputs"] == []
    assert loaded.metadata["errors"][-1]["stage"] == "output"
    assert np.array_equal(loaded.original, acquisition(True).original)


def test_public_timing_updates_preserve_original_and_output_phases(tmp_path):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    out = archive.save_output(acquisition().original, m, {"name": "minimal"})
    archive.update_timings(m, {"render_s": 0.25, "total_until_output_s": 0.75}, output_path=out)
    archive.update_timings(m, {"failed_render_s": 0.1})
    loaded = archive.load_acquisition(m)
    assert loaded.metadata["timings"]["render_s"] == 0.25
    assert loaded.metadata["timings"]["failed_render_s"] == 0.1
    output = loaded.metadata["outputs"][0]
    assert output["timings"]["total_until_output_s"] == 0.75
    assert "encode_s" in output["timings"]
    assert loaded.metadata["timings"]["last_output"] == output["timings"]
    assert np.array_equal(loaded.original, acquisition().original)


def test_public_timing_update_rejects_unlinked_output_atomically(tmp_path):
    m = archive.archive_acquisition(acquisition(), tmp_path)
    before = Path(m).read_bytes()
    with pytest.raises(SaveError, match="not linked"):
        archive.update_timings(m, {"render_s": 1}, output_path=tmp_path / "unknown.png")
    assert Path(m).read_bytes() == before


def test_final_archive_timing_failure_preserves_published_original(tmp_path, monkeypatch):
    writer = archive._atomic_json
    def fail_only_refinement(path, data, *, new=False):
        if not new:
            raise OSError("disk full after successful publication")
        return writer(path, data, new=new)
    monkeypatch.setattr(archive, "_atomic_json", fail_only_refinement)
    a = acquisition(True)
    manifest = archive.archive_acquisition(a, tmp_path)
    loaded = archive.load_acquisition(manifest)
    assert np.array_equal(loaded.original, a.original)
    assert loaded.metadata["timings"]["total_until_archive_s"] >= 0.1
    assert {p.name for p in Path(manifest).parent.iterdir()} == {"acquisition.json", "original.npy"}
