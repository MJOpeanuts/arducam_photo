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
    assert op.calls == [(0, "msmf")]  # rendering validates CCM after acquisition
    assert cap.released == 1


def test_native_requires_ccm_path_or_opt_out(small_raw):
    cap = FakeCap([good_raw()])
    with pytest.raises(ap.CaptureConfigError):
        ap.capture(ap.CaptureConfig(path="native_108mp", stabilization_reads=0), _opener=opener_for(cap))
    assert cap.released == 1 and cap.read_calls == 1
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


def test_acquire_independent_of_ccm_owns_original(small_raw, monkeypatch):
    def forbidden(*args):
        raise AssertionError("acquisition must not render or load CCM")
    monkeypatch.setattr(engine, "process_raw", forbidden)
    monkeypatch.setattr(engine, "load_ccm_file", forbidden)
    source = good_raw()[1]

    class InvalidatingCap(FakeCap):
        def release(self):
            source[:] = 0
            super().release()

    cap = InvalidatingCap([(True, source)])
    result = ap.acquire(ap.CaptureConfig(ccm_path="missing.json", stabilization_reads=0),
                        _opener=opener_for(cap))
    assert cap.released == 1
    assert result.original.shape == (8, 12) and np.all(result.original == 100)
    assert not np.shares_memory(result.original, source)
    assert result.info.received_shape == (96,)
    assert result.info.reconstructed_shape == (8, 12)
    assert result.info.dtype == "uint8"
    assert result.info.capture_id and result.info.captured_at.endswith("+00:00")
    assert result.info.timings["acquisition_s"] >= result.info.timings["read_s"] >= 0
    assert result.info.timings["release_s"] >= 0
    assert result.info.camera_released and result.info.release_error is None
    assert not result.info.ccm_applied


def test_release_before_processing(small_raw, monkeypatch):
    cap = FakeCap([good_raw()])

    def process(raw, ccms):
        assert cap.released == 1
        assert not engine._busy.locked()
        return np.zeros((*raw.shape, 3), np.uint8)

    monkeypatch.setattr(engine, "process_raw", process)
    result = ap.capture(ap.CaptureConfig(apply_ccm=False, stabilization_reads=0),
                        _opener=opener_for(cap))
    assert result.image.shape == (8, 12, 3)


def test_release_before_ccm_load(small_raw, monkeypatch):
    cap = FakeCap([good_raw()])

    def load(path):
        assert cap.released == 1
        raise ap.CaptureConfigError("invalid tuning at render time")

    monkeypatch.setattr(engine, "load_ccm_file", load)
    with pytest.raises(ap.CaptureConfigError):
        ap.capture(ap.CaptureConfig(stabilization_reads=0), _opener=opener_for(cap))


def test_acquire_cancel_after_read_releases(small_raw):
    event = threading.Event()

    class CancellingCap(FakeCap):
        def read(self):
            event.set()
            return good_raw()

    cap = CancellingCap([])
    with pytest.raises(ap.CaptureCancelled):
        ap.acquire(ap.CaptureConfig(stabilization_reads=0), event, opener_for(cap))
    assert cap.released == 1 and not engine._busy.locked()


def test_acquire_setup_rejection_releases():
    class RejectingCap(FakeCap):
        def set(self, prop, value):
            return False

    cap = RejectingCap([])
    with pytest.raises(ap.CameraSetupError):
        ap.acquire(ap.CaptureConfig(), _opener=opener_for(cap))
    assert cap.released == 1 and not engine._busy.locked()


def test_release_has_separate_timing(small_raw):
    import time

    class SlowRelease(FakeCap):
        def release(self):
            time.sleep(0.01)
            super().release()

    result = ap.acquire(ap.CaptureConfig(stabilization_reads=0),
                        _opener=opener_for(SlowRelease([good_raw()])))
    timings = result.info.timings
    assert timings["release_s"] >= 0.01
    assert timings["acquisition_s"] >= timings["release_s"] + timings["read_s"] + timings["copy_s"]


def test_release_failure_preserves_original_and_blocks_next_capture(small_raw, monkeypatch):
    monkeypatch.setattr(engine, "_release_failure", None)
    source = good_raw()[1]

    class BrokenRelease(FakeCap):
        def release(self):
            source[:] = 0
            raise RuntimeError("driver cannot release")

    acquired = ap.acquire(ap.CaptureConfig(stabilization_reads=0),
                          _opener=opener_for(BrokenRelease([(True, source)])))
    assert np.all(acquired.original == 100) and acquired.original.shape == (8, 12)
    assert acquired.info.camera_released is False
    assert "driver cannot release" in acquired.info.release_error
    assert acquired.info.timings["release_s"] >= 0
    opener = opener_for(FakeCap([good_raw()]))
    with pytest.raises(ap.CameraSetupError, match="restart"):
        ap.acquire(ap.CaptureConfig(), _opener=opener)
    assert opener.calls == [] and not engine._busy.locked()


def test_capture_release_error_carries_acquisition(small_raw, monkeypatch):
    monkeypatch.setattr(engine, "_release_failure", None)

    class BrokenRelease(FakeCap):
        def release(self):
            raise RuntimeError("release failed")

    with pytest.raises(ap.CameraSetupError) as caught:
        ap.capture(ap.CaptureConfig(stabilization_reads=0),
                   _opener=opener_for(BrokenRelease([good_raw()])))
    assert caught.value.acquisition_result.original.shape == (8, 12)
    assert caught.value.acquisition_result.info.release_error


def test_native_mode_geometry():
    assert (engine.TRANSPORT_W, engine.TRANSPORT_H) == (6000, 9000)
    assert (engine.RAW_ROWS, engine.RAW_COLS, engine.RAW_BYTES) == (9000, 12000, 108_000_000)
    assert ap.MODES["native_108mp"].raw


@pytest.mark.parametrize("path,width,height,fps", [
    ("color_720p", 1280, 720, 10),
    ("color_4k", 3840, 2160, 10),
    ("color_12mp", 4000, 3000, 7),
])
def test_colour_modes_exact_shape_and_fps(path, width, height, fps):
    import cv2
    frame = np.zeros((height, width, 3), np.uint8)
    frame[..., 0] = 37
    cap = FakeCap([(True, frame)])
    result = ap.acquire(ap.CaptureConfig(path=path, stabilization_reads=0), _opener=opener_for(cap))
    assert result.original.shape == frame.shape and result.original.dtype == np.uint8
    assert result.info.transport_size == (width, height)
    assert (cv2.CAP_PROP_FPS, fps) in cap.set_calls
    assert (cv2.CAP_PROP_CONVERT_RGB, 0) not in cap.set_calls
    assert np.array_equal(result.original, frame) and not np.shares_memory(result.original, frame)
    assert cap.released == 1


@pytest.mark.parametrize("fps", [None, 2.5])
def test_native_fps_default_or_explicit(small_raw, fps):
    import cv2
    cap = FakeCap([good_raw()])
    ap.acquire(ap.CaptureConfig(fps=fps, stabilization_reads=0), _opener=opener_for(cap))
    assert (cv2.CAP_PROP_FPS, 1 if fps is None else fps) in cap.set_calls
    assert all(value != 60 for prop, value in cap.set_calls if prop == cv2.CAP_PROP_FPS)


@pytest.mark.parametrize("shape", [(96,), (1, 96), (8, 12), (8, 12, 1), (8, 6, 2)])
def test_raw_plausible_shapes(small_raw, shape):
    frame = np.zeros(shape, np.uint8)
    assert engine.validate_raw_buffer(frame) is frame


@pytest.mark.parametrize("shape", [(4, 24), (2, 4, 12), (8, 4, 3), (6, 8, 2)])
def test_raw_same_byte_count_nonsense_rejected(small_raw, shape):
    with pytest.raises(ap.RawBufferError):
        engine.validate_raw_buffer(np.zeros(shape, np.uint8))


@pytest.mark.parametrize("fps", [0, -1, True, float("inf"), float("nan"), "10"])
def test_invalid_fps_before_camera(fps):
    opener = opener_for(FakeCap([]))
    with pytest.raises(ap.CaptureConfigError):
        ap.acquire(ap.CaptureConfig(fps=fps), _opener=opener)
    assert opener.calls == []


@pytest.mark.parametrize("dtype,shape", [
    (np.uint16, (2160, 3840, 3)), (np.uint8, (1080, 1920, 3)),
    (np.uint8, (2160, 3840, 4)), (np.uint8, (2160, 3840)),
])
def test_4k_rejects_wrong_received_format(dtype, shape):
    cap = FakeCap([(True, np.zeros(shape, dtype))])
    with pytest.raises(ap.RawBufferError):
        ap.acquire(ap.CaptureConfig(path="color_4k", max_invalid_buffers=0),
                   _opener=opener_for(cap))
    assert cap.released == 1
