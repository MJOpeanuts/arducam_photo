import numpy as np
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import arducam_photo as ap
from arducam_photo.config import AcquisitionInfo, AcquisitionResult, SettingReport
from conftest import FakeCap, opener_for


def img():
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)


def test_png_roundtrip_exact_and_no_temp_left(tmp_path):
    import cv2
    a = img()
    out = tmp_path / "a.png"
    ap.save_png(a, out)
    assert np.array_equal(cv2.imread(str(out), cv2.IMREAD_UNCHANGED), a)
    assert [p.name for p in tmp_path.iterdir()] == ["a.png"]


def test_existing_file_kept_without_overwrite(tmp_path):
    out = tmp_path / "a.png"
    out.write_bytes(b"keep")
    with pytest.raises(ap.SaveError):
        ap.save_png(img(), out)
    assert out.read_bytes() == b"keep"
    ap.save_png(img(), out, overwrite=True)
    assert out.read_bytes() != b"keep"


def test_bad_inputs(tmp_path):
    with pytest.raises(ap.SaveError):
        ap.save_png(np.zeros((3, 3), np.float32), tmp_path / "a.png")
    with pytest.raises(ap.SaveError):
        ap.save_png(img(), tmp_path / "nodir" / "a.png")


def test_capture_and_save_writes_720p_png(tmp_path, monkeypatch):
    from arducam_photo import storage
    from arducam_photo import engine
    frame = np.zeros((720, 1280, 3), np.uint8)
    acquire = engine.acquire
    monkeypatch.setattr(engine, "acquire", lambda cfg: acquire(
        cfg, _opener=opener_for(FakeCap([(True, frame)]))))
    r = ap.capture_and_save(ap.CaptureConfig(path="color_720p"), tmp_path / "x.png")
    assert (r.width, r.height) == (1280, 720)
    assert (tmp_path / "x.png").exists()
    assert len(list(tmp_path.glob("*/acquisition.json"))) == 1
    import json
    manifest, = tmp_path.glob("*/acquisition.json")
    metadata = json.loads(manifest.read_text())
    assert len(metadata["outputs"]) == 1
    assert metadata["outputs"][0]["acquisition_id"] == metadata["id"]
    assert metadata["outputs"][0]["export_path"] == str(tmp_path / "x.png")
    assert metadata["outputs"][0]["recipe"]["version"] == 1
    assert metadata["outputs"][0]["timings"]["total_s"] > 0
    assert "ccm" in metadata["outputs"][0]
    bundle, = tmp_path.glob("*.json")
    capture_id = json.loads(bundle.read_text())["capture_id"]
    assert bundle.name == f"{capture_id}.json"
    assert (tmp_path / f"{capture_id}.png").exists()
    assert not (tmp_path / f"{capture_id}.raw").exists()
    assert not (tmp_path / f".{capture_id}.incomplete").exists()


def test_grayscale_png_roundtrip(tmp_path):
    import cv2
    a = np.arange(120, dtype=np.uint8).reshape(10, 12)
    out = tmp_path / "gray.png"
    ap.save_png(a, out)
    assert np.array_equal(cv2.imread(str(out), cv2.IMREAD_UNCHANGED), a)


def test_jpeg_preserves_dimensions_not_exact_values(tmp_path):
    import cv2
    from arducam_photo.storage import save_image, OutputOptions
    a = img()
    out = tmp_path / "a.jpg"
    save_image(a, out, options=OutputOptions("jpeg", 95))
    b = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert b.shape == a.shape and b.dtype == a.dtype
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("options", [("webp", 95), ("jpeg", 0), ("jpeg", 101), ("jpeg", True)])
def test_bad_output_options(tmp_path, options):
    from arducam_photo.storage import save_image, OutputOptions
    with pytest.raises(ap.SaveError):
        save_image(img(), tmp_path / "a.jpg", options=OutputOptions(*options))
    assert not list(tmp_path.iterdir())


def test_no_unsafe_replace_fallback(tmp_path, monkeypatch):
    from arducam_photo import storage
    def no_link(*args):
        raise OSError("hard links unavailable")
    monkeypatch.setattr(storage.os, "link", no_link)
    with pytest.raises(ap.SaveError):
        storage.save_png(img(), tmp_path / "a.png")
    assert not list(tmp_path.iterdir())


def test_failed_verification_leaves_existing_image(tmp_path, monkeypatch):
    from arducam_photo import storage
    out = tmp_path / "a.png"
    out.write_bytes(b"keep")
    def fail(*args, **kwargs):
        raise ap.SaveError("bad decode")
    monkeypatch.setattr(storage, "_verify", fail)
    with pytest.raises(ap.SaveError):
        storage.save_png(img(), out, overwrite=True)
    assert out.read_bytes() == b"keep"
    assert [p.name for p in tmp_path.iterdir()] == ["a.png"]


@pytest.mark.parametrize("stage", ["processing", "output"])
def test_capture_failure_preserves_original_and_records_error(tmp_path, monkeypatch, stage):
    from arducam_photo import engine, processing, storage, archive
    frame = np.arange(720 * 1280 * 3, dtype=np.uint8).reshape(720, 1280, 3)
    acquire = engine.acquire
    monkeypatch.setattr(engine, "acquire", lambda cfg: acquire(
        cfg, _opener=opener_for(FakeCap([(True, frame)]))))
    def fail(*args, **kwargs):
        raise ValueError("test failure")
    monkeypatch.setattr(processing if stage == "processing" else storage,
                        "process_image" if stage == "processing" else "save_image", fail)
    with pytest.raises(ValueError, match="test failure") as raised:
        ap.capture_and_save(ap.CaptureConfig(path="color_720p", stabilization_reads=0), tmp_path / "x.png")
    manifest, = tmp_path.glob("*/acquisition.json")
    assert raised.value.manifest_path == str(manifest)
    assert raised.value.archive_path == str(manifest)
    loaded = archive.load_acquisition(manifest)
    assert np.array_equal(loaded.original, frame)
    assert loaded.metadata["errors"][-1]["stage"] == stage
    assert not (tmp_path / "x.png").exists()


def test_pre_and_post_publication_verification(tmp_path, monkeypatch):
    from arducam_photo import storage
    calls = []
    verify = storage._verify
    def checked(path, image, **kwargs):
        calls.append(path)
        verify(path, image, **kwargs)
    monkeypatch.setattr(storage, "_verify", checked)
    timings = {}
    out = tmp_path / "a.png"
    storage.save_image(img(), out, _timings=timings)
    assert len(calls) == 2
    assert calls[-1] == str(out)
    assert timings["verify_s"] == timings["pre_verify_s"] + timings["post_verify_s"]


def test_archive_failure_has_no_recoverable_archive_attribute(tmp_path, monkeypatch):
    from arducam_photo import archive, engine
    monkeypatch.setattr(engine, "acquire", lambda cfg: object())
    def fail(*args, **kwargs):
        raise ap.SaveError("disk full")
    monkeypatch.setattr(archive, "archive_acquisition", fail)
    with pytest.raises(ap.SaveError, match="disk full") as raised:
        ap.capture_and_save(ap.CaptureConfig(path="color_720p"), tmp_path / "out.png")
    assert not hasattr(raised.value, "archive_path")
    assert not hasattr(raised.value, "manifest_path")


def _bundle_acquisition(path="native_108mp"):
    capture_id = str(uuid4())
    raw = np.arange(24, dtype=np.uint8).reshape(4, 6)
    frame = raw if path == "native_108mp" else np.zeros((4, 6, 3), np.uint8)
    cfg = ap.CaptureConfig(path=path, apply_ccm=False, stabilization_reads=0)
    info = AcquisitionInfo(
        config=cfg, api="msmf", camera_index=2, path=path, transport_size=(3, 4),
        settings={"focus": SettingReport(0, False, float("nan"))},
        frames_read=6, failed_reads=1, invalid_buffers=2, received_shape=(24,),
        reconstructed_shape=tuple(frame.shape), capture_id=capture_id,
        captured_at=datetime.now(timezone.utc).isoformat(),
        settings_readback_at=datetime.now(timezone.utc).isoformat(),
    )
    return AcquisitionResult(frame, info)


def test_capture_bundle_raw_json_png_share_id_and_verify_integrity(tmp_path, monkeypatch):
    import hashlib
    import json
    from arducam_photo import storage

    acquisition = _bundle_acquisition()
    manifest = storage.begin_capture_bundle(acquisition, tmp_path)
    capture_id = acquisition.info.capture_id
    raw_path = tmp_path / f"{capture_id}.raw"
    json_path = tmp_path / f"{capture_id}.json"
    marker = tmp_path / f".{capture_id}.incomplete"
    assert raw_path.read_bytes() == acquisition.original.tobytes()
    data = json.loads(json_path.read_text())
    assert data["capture_id"] == capture_id
    assert data["acquisition"]["raw"]["size_bytes"] == acquisition.original.nbytes
    assert data["acquisition"]["raw"]["sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert data["acquisition"]["parameter_reports"]["focus"]["readback"] is None
    assert b"NaN" not in json_path.read_bytes()
    assert marker.exists()

    def save_fake(image, path, **kwargs):
        open(path, "xb").write(b"png data")
        return str(path)

    monkeypatch.setattr(storage, "save_image", save_fake)
    from arducam_photo.processing import ProcessingRecipe
    recipe = ProcessingRecipe(apply_ccm=False)
    png = storage.finish_capture_bundle(manifest, np.zeros((4, 6, 3), np.uint8), recipe)
    assert png == str(tmp_path / f"{capture_id}.png")
    assert (tmp_path / f"{capture_id}.png").read_bytes() == b"png data"
    assert not marker.exists()
    assert json.loads(json_path.read_text())["png"]["status"] == "preserved"


def test_capture_bundle_failure_keeps_raw_and_detectable_marker(tmp_path):
    import json
    from arducam_photo.storage import begin_capture_bundle, fail_capture_bundle

    acquisition = _bundle_acquisition()
    manifest = begin_capture_bundle(acquisition, tmp_path)
    fail_capture_bundle(manifest, "render", RuntimeError("render failed"))
    capture_id = acquisition.info.capture_id
    assert (tmp_path / f"{capture_id}.raw").read_bytes() == acquisition.original.tobytes()
    data = json.loads((tmp_path / f"{capture_id}.json").read_text())
    assert data["processing"]["status"] == "failed"
    assert data["processing"]["raw_preserved"] is True
    assert (tmp_path / f".{capture_id}.incomplete").exists()


def test_720p_capture_bundle_has_no_raw(tmp_path):
    import json
    from arducam_photo.storage import begin_capture_bundle

    acquisition = _bundle_acquisition("color_720p")
    manifest = begin_capture_bundle(acquisition, tmp_path)
    capture_id = acquisition.info.capture_id
    assert not (tmp_path / f"{capture_id}.raw").exists()
    data = json.loads(open(manifest, encoding="utf8").read())
    assert data["acquisition"]["raw"]["status"] == "unavailable"
    assert data["acquisition"]["raw"]["file"] is None


def test_capture_bundle_never_overwrites_existing_artifacts(tmp_path):
    from arducam_photo.storage import begin_capture_bundle

    acquisition = _bundle_acquisition("color_720p")
    (tmp_path / f"{acquisition.info.capture_id}.json").write_text("keep")
    with pytest.raises(ap.SaveError):
        begin_capture_bundle(acquisition, tmp_path)
    assert (tmp_path / f"{acquisition.info.capture_id}.json").read_text() == "keep"
    assert not (tmp_path / f".{acquisition.info.capture_id}.incomplete").exists()
