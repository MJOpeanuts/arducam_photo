import json
import subprocess
import sys
import threading

import numpy as np
import pytest

import arducam_photo as ap
from arducam_photo import engine
from arducam_photo.isp import process_raw
from conftest import FakeCap, good_raw, opener_for

BAD = (True, np.zeros(10, np.uint8))


def native_cfg(ccm, **kw):
    return ap.CaptureConfig(path="native_108mp", ccm_path=ccm, **kw)


def test_import_has_no_side_effects(tmp_path):
    code = (
        "import threading, os, sys; t=threading.active_count(); f=set(os.listdir('.'));"
        "import arducam_photo; assert threading.active_count()==t; assert set(os.listdir('.'))==f;"
        "import arducam_photo.engine as e; assert e._busy.locked() is False"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(tmp_path))


def test_native_transient_then_valid(small_raw, ccm_file):
    cap = FakeCap([(False, None), BAD, good_raw()])
    r = ap.capture(native_cfg(ccm_file, stabilization_reads=2), _opener=opener_for(cap))
    assert r.image.shape == (8, 12, 3) and r.image.dtype == np.uint8
    assert r.info.failed_reads == 1 and r.info.invalid_buffers == 1 and r.info.frames_read == 5
    assert r.info.ccm_applied and r.info.usb_speed_measured is False
    assert cap.released == 1
    assert (6000, 9000) == r.info.transport_size


def test_convert_rgb_disabled_and_setting_order(small_raw, ccm_file):
    import cv2
    cap = FakeCap([good_raw()])
    ap.capture(native_cfg(ccm_file, focus=300, stabilization_reads=0), _opener=opener_for(cap))
    props = [p for p, _ in cap.set_calls]
    assert (cv2.CAP_PROP_CONVERT_RGB, 0) in cap.set_calls
    assert props.index(cv2.CAP_PROP_FRAME_WIDTH) < props.index(cv2.CAP_PROP_FOCUS)


def test_wrong_size_buffer_never_reshaped(small_raw, ccm_file):
    cap = FakeCap([BAD])
    with pytest.raises(ap.RawBufferError):
        ap.capture(native_cfg(ccm_file, max_invalid_buffers=2), _opener=opener_for(cap))
    assert cap.read_calls == 3 and cap.released == 1


@pytest.mark.parametrize("bad", [
    (True, np.zeros(96, np.uint16)), (True, "x"), (True, None), (True, np.zeros(95, np.uint8)),
])
def test_buffer_type_and_dtype_rejected(small_raw, bad):
    with pytest.raises(ap.RawBufferError):
        engine.validate_raw_buffer(bad[1])


def test_read_failure_bounded(small_raw, ccm_file):
    cap = FakeCap([(False, None)])
    with pytest.raises(ap.CameraReadError):
        ap.capture(native_cfg(ccm_file, max_failed_reads=3), _opener=opener_for(cap))
    assert cap.read_calls == 4 and cap.released == 1


def test_open_failure(ccm_file):
    cap = FakeCap([good_raw()], opened=False)
    with pytest.raises(ap.CameraOpenError):
        ap.capture(native_cfg(ccm_file), _opener=opener_for(cap))
    assert cap.released == 1 and cap.read_calls == 0


def test_no_silent_fallback_to_720p(small_raw, ccm_file):
    frame720 = (True, np.zeros((720, 1280, 3), np.uint8))
    cap = FakeCap([frame720])
    with pytest.raises(ap.RawBufferError):
        ap.capture(native_cfg(ccm_file, max_invalid_buffers=1), _opener=opener_for(cap))


def test_released_on_isp_error(small_raw, ccm_file, monkeypatch):
    monkeypatch.setattr(engine, "process_raw", lambda *a: (_ for _ in ()).throw(ap.IspError("boom")))
    cap = FakeCap([good_raw()])
    with pytest.raises(ap.IspError):
        ap.capture(native_cfg(ccm_file, stabilization_reads=0), _opener=opener_for(cap))
    assert cap.released == 1


def test_cancel_between_native_calls(small_raw, ccm_file):
    ev = threading.Event()
    ev.set()
    cap = FakeCap([good_raw()])
    with pytest.raises(ap.CaptureCancelled):
        ap.capture(native_cfg(ccm_file), cancel_event=ev, _opener=opener_for(cap))


def test_concurrent_capture_refused(small_raw, ccm_file):
    inside, go = threading.Event(), threading.Event()

    class Slow(FakeCap):
        def read(self):
            inside.set()
            go.wait(5)
            return good_raw()

    out = {}
    t = threading.Thread(target=lambda: out.update(r=ap.capture(
        native_cfg(ccm_file, stabilization_reads=0), _opener=opener_for(Slow([])))))
    t.start()
    assert inside.wait(5)
    with pytest.raises(ap.CaptureBusyError):
        ap.capture(native_cfg(ccm_file), _opener=opener_for(FakeCap([good_raw()])))
    go.set()
    t.join()
    assert "r" in out
    # lock released: next capture works
    ap.capture(native_cfg(ccm_file, stabilization_reads=0), _opener=opener_for(FakeCap([good_raw()])))


def test_no_camera_kept_between_calls(small_raw, ccm_file):
    caps = []
    for _ in range(3):
        c = FakeCap([good_raw()])
        caps.append(c)
        ap.capture(native_cfg(ccm_file, stabilization_reads=1), _opener=opener_for(c))
    assert [c.released for c in caps] == [1, 1, 1]
    assert all(c.read_calls == 2 for c in caps)  # bounded reads, no continuous stream


def test_config_frozen_at_start(small_raw, ccm_file):
    cfg = native_cfg(ccm_file, focus=100, stabilization_reads=0)
    r = ap.capture(cfg, _opener=opener_for(FakeCap([good_raw()])))
    with pytest.raises(Exception):
        cfg.focus = 5  # frozen
    assert r.info.config.focus == 100 and r.info.settings["focus"].requested == 100


@pytest.mark.parametrize("make", ["missing", "badjson", "nocct", "badmatrix", "unordered"])
def test_ccm_missing_or_invalid(tmp_path, small_raw, make):
    p = tmp_path / "t.json"
    if make == "missing":
        p = tmp_path / "nope.json"
    elif make == "badjson":
        p.write_text("{")
    elif make == "nocct":
        p.write_text(json.dumps({"ccms": [{"ccm": [1] * 9}]}))
    elif make == "badmatrix":
        p.write_text(json.dumps({"ccms": [{"ct": 4000, "ccm": [1] * 8}]}))
    else:
        i = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        p.write_text(json.dumps({"ccms": [{"ct": 5000, "ccm": i}, {"ct": 3000, "ccm": i}]}))
    cap = FakeCap([good_raw()])
    op = opener_for(cap)
    with pytest.raises(ap.CaptureConfigError):
        ap.capture(native_cfg(str(p)), _opener=op)
    assert op.calls == []  # camera never opened


def test_native_requires_ccm_path_or_opt_out(small_raw):
    with pytest.raises(ap.CaptureConfigError):
        ap.capture(ap.CaptureConfig(path="native_108mp"), _opener=opener_for(FakeCap([])))
    r = ap.capture(ap.CaptureConfig(path="native_108mp", apply_ccm=False, stabilization_reads=0),
                   _opener=opener_for(FakeCap([good_raw()])))
    assert r.info.ccm_applied is False and r.info.ccm_requested is False


def test_720p_no_raw_no_ccm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not be called")
    monkeypatch.setattr(engine, "process_raw", boom)
    monkeypatch.setattr(engine, "load_ccm_file", boom)
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    cap = FakeCap([(True, frame)])
    r = ap.capture(ap.CaptureConfig(path="color_720p", ccm_path="/does/not/exist.json"),
                   _opener=opener_for(cap))
    assert (r.width, r.height) == (1280, 720) and np.array_equal(r.image, frame)
    assert r.image is not frame and r.info.ccm_applied is False and cap.released == 1


def test_720p_wrong_format_rejected():
    cap = FakeCap([(True, np.zeros((720, 1280), np.uint8))])
    with pytest.raises(ap.RawBufferError):
        ap.capture(ap.CaptureConfig(path="color_720p", max_invalid_buffers=1), _opener=opener_for(cap))
    assert cap.released == 1


def test_isp_channel_order_and_black_level():
    raw = np.zeros((8, 8), np.uint8)
    raw[0::2, 1::2] = 216  # sites that OpenCV GR demosaic maps to channel index 2
    assert process_raw(raw)[4, 4].tolist() == [0, 0, 200]
    raw = np.zeros((8, 8), np.uint8)
    raw[1::2, 0::2] = 216
    assert process_raw(raw)[4, 4].tolist() == [200, 0, 0]
    assert process_raw(np.full((8, 8), 10, np.uint8)).max() == 0  # below black level


def test_ccm_applied_on_channels(tmp_path):
    from arducam_photo.ccm import load_ccm_file, apply_ccm_inplace
    p = tmp_path / "t.json"
    sw = [0, 0, 1, 0, 1, 0, 1, 0, 0]  # swaps R and B in RGB space
    p.write_text(json.dumps({"ccms": [{"ct": 3000, "ccm": sw}]}))
    img = np.zeros((600, 4, 3), np.uint8)  # > one chunk
    img[..., 0] = 200  # blue
    apply_ccm_inplace(img, load_ccm_file(str(p)))
    assert img[0, 0, 2] >= 190 and img[0, 0, 0] <= 10 and img[-1, -1, 2] >= 190
