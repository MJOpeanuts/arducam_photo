import numpy as np
import pytest

import arducam_photo as ap
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
