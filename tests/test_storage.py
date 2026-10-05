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
        ap.save_png(np.zeros((3, 3), np.uint8), tmp_path / "a.png")
    with pytest.raises(ap.SaveError):
        ap.save_png(img(), tmp_path / "nodir" / "a.png")


def test_capture_and_save_writes_720p_png(tmp_path, monkeypatch):
    from arducam_photo import storage
    frame = np.zeros((720, 1280, 3), np.uint8)
    monkeypatch.setattr(storage, "capture", lambda cfg: ap.capture(cfg, _opener=opener_for(FakeCap([(True, frame)]))))
    r = ap.capture_and_save(ap.CaptureConfig(path="color_720p"), tmp_path / "x.png")
    assert (r.width, r.height) == (1280, 720)
    assert (tmp_path / "x.png").exists()
