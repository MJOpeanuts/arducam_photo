import dataclasses
import json
import os
import threading
import time

import numpy as np
import pytest

from arducam_photo import CaptureBusyError, CaptureConfigError, CaptureResult
from arducam_photo.config import AcquisitionInfo
from arducam_photo.app.controller import CameraController, CaptureJob, ModeError
from arducam_photo.app.model import (CAPTURE_MODE, DISCARD, PREVIEW_MODE, SAVE, START, STAY, NeedsDecision,
                                     ProfileMissing, SessionModel)
from arducam_photo.app.paths import unique_photo_path
from arducam_photo.app.profile import (JsonProfileStore, JsonSettingsStore, ShootingProfile, StoreError)
from conftest import ccm_file  # noqa: F401  (fixture)


class VideoCap:
    """Fake camera for preview: 720p BGR frames; counts reads and releases."""

    registry = []

    def __init__(self):
        self.props, self.reads, self.released, self.opened = {}, 0, 0, True
        self.focus_sets = []
        VideoCap.registry.append(self)

    def isOpened(self):
        return self.opened

    def set(self, p, v):
        import cv2
        if p == cv2.CAP_PROP_FOCUS:
            self.focus_sets.append(v)
        self.props[p] = v
        return True

    def get(self, p):
        return self.props.get(p, 0.0) - 1 if p == 28 else self.props.get(p, 0.0)  # focus readback differs

    def read(self):
        self.reads += 1
        time.sleep(0.002)
        return True, np.zeros((720, 1280, 3), np.uint8)

    def release(self):
        self.released += 1


class Env:
    def __init__(self, tmp_path, ccm=None):
        VideoCap.registry = []
        self.events, self.cv = [], threading.Condition()
        self.captures, self.gate = [], threading.Event()
        self.gate.set()
        self.states = []
        self.profiles = JsonProfileStore(str(tmp_path / "profiles.json"))
        self.settings = JsonSettingsStore(str(tmp_path / "settings.json"))
        self.tmp = tmp_path
        self.fail_capture = None
        self.model = SessionModel(self.profiles, self.settings, self._ui, opener=lambda i, a: VideoCap(),
                                  capture_fn=self._capture, save_fn=self._save)
        if ccm:
            self.model.set_ccm_path(ccm)

    def _ui(self, event, **kw):
        with self.cv:
            self.events.append((event, kw))
            self.cv.notify_all()

    def _capture(self, cfg, cancel_event=None):
        self.states.append(self.model.controller.state)
        self.gate.wait(5)
        self.captures.append(cfg)
        cam = VideoCap()  # engine opens/releases its own camera
        cam.release()
        if self.fail_capture:
            raise self.fail_capture
        info = AcquisitionInfo(config=cfg, api=cfg.api, camera_index=cfg.camera_index, path=cfg.path,
                               transport_size=(1, 1), ccm_applied=cfg.apply_ccm)
        return CaptureResult(np.zeros((90, 120, 3), np.uint8), 120, 90, info)

    def _save(self, image, path):
        open(path, "wb").write(b"png")
        return path

    def wait(self, event, timeout=5, nth=1):
        end = time.time() + timeout
        with self.cv:
            while sum(1 for e, _ in self.events if e == event) < nth:
                if not self.cv.wait(max(0.0, end - time.time())):
                    raise AssertionError(f"timeout waiting {event}; got {[e for e, _ in self.events]}")

    def count(self, event):
        return sum(1 for e, _ in self.events if e == event)

    def preview(self):
        self.model.enter_preview()
        self.wait("mode_changed")
        assert self.model.mode == PREVIEW_MODE

    def set_focus_and_save(self, v=300):
        self.model.set_focus(v)
        self.wait("focus_applied")
        self.model.save_profile()

    def close(self):
        assert self.model.shutdown(5)


@pytest.fixture
def env(tmp_path, ccm_file):  # noqa: F811
    e = Env(tmp_path, ccm_file)
    e.wait("ccm_checked")
    yield e
    e.close()


def to_capture(env, decision=None):
    assert env.model.request_capture_mode(decision)
    env.wait("mode_changed", nth=2)
    assert env.model.mode == CAPTURE_MODE


def test_start_does_not_open_camera(env):
    time.sleep(0.05)
    assert VideoCap.registry == [] and env.model.mode == START


def test_preview_to_capture_releases_camera_and_never_reads_while_waiting(env):
    env.preview()
    env.set_focus_and_save()
    to_capture(env)
    cam = VideoCap.registry[0]
    assert cam.released == 1 and env.model.controller.state == "idle"
    assert env.count("video_cleared") == 1
    reads = cam.reads
    time.sleep(0.15)
    assert cam.reads == reads and len(VideoCap.registry) == 1  # no read, no reopen while waiting


def test_capture_uses_saved_focus_and_frozen_profile_no_auto_preview(env):
    env.preview(); env.set_focus_and_save(321)
    to_capture(env)
    env.gate.clear()
    env.model.trigger()
    env.wait("capture_step")
    # profile changes on disk/model while capturing must not alter the running capture
    env.model.saved = dataclasses.replace(env.model.saved, focus_requested=5)
    env.gate.set()
    env.wait("capture_idle")
    cfg = env.captures[0]
    assert cfg.focus == 321 and cfg.apply_ccm and cfg.ccm_path == env.model.settings.ccm_path
    assert env.states == ["capturing"]
    assert env.model.mode == CAPTURE_MODE and env.model.controller.state == "idle"
    assert len(VideoCap.registry) == 2  # preview camera + engine's own; no preview restart
    assert env.model.last_photo and os.path.exists(env.model.last_photo)


def test_double_trigger_and_mode_changes_refused_during_capture(env):
    env.preview(); env.set_focus_and_save()
    to_capture(env)
    env.gate.clear()
    env.model.trigger()
    with pytest.raises(CaptureBusyError):
        env.model.trigger()
    with pytest.raises(ModeError):
        env.model.enter_preview()
    with pytest.raises(ModeError):
        env.model.controller.open_preview(0, "msmf", "color_720p")
    env.gate.set()
    env.wait("capture_idle")
    assert len(env.captures) == 1


def test_capture_error_releases_and_stays_in_capture(env):
    from arducam_photo import CameraReadError
    env.preview(); env.set_focus_and_save()
    to_capture(env)
    env.fail_capture = CameraReadError("boom")
    env.model.trigger()
    env.wait("capture_failed"); env.wait("capture_idle")
    assert env.model.mode == CAPTURE_MODE and env.model.controller.state == "idle"
    ev = [kw for e, kw in env.events if e == "capture_failed"][0]
    assert ev["recoverable"] is True
    env.fail_capture = None
    env.model.trigger()  # a new attempt is possible
    env.wait("capture_done")


def test_unexpected_error_also_returns_to_idle(env):
    env.preview(); env.set_focus_and_save()
    to_capture(env)
    env.fail_capture = RuntimeError("bug")
    env.model.trigger()
    env.wait("capture_idle")
    assert env.model.controller.state == "idle"


def test_missing_ccm_not_silently_disabled(tmp_path):
    e = Env(tmp_path)  # no ccm configured
    try:
        e.preview(); e.set_focus_and_save()
        to_capture(e)
        with pytest.raises(CaptureConfigError):
            e.model.trigger()
        assert e.captures == []
    finally:
        e.close()


def test_invalid_ccm_reported_before_camera(tmp_path):
    bad = tmp_path / "bad.json"; bad.write_text("{}")
    e = Env(tmp_path, str(bad))
    try:
        e.wait("ccm_checked")
        assert [kw for ev, kw in e.events if ev == "ccm_checked"][0]["ok"] is False
        e.preview(); e.set_focus_and_save()
        to_capture(e)
        e.model.trigger()  # worker validates ccm before touching the camera
        e.wait("capture_failed"); e.wait("capture_idle")
        assert e.captures == []
    finally:
        e.close()


def test_no_valid_profile_requires_preview(env):
    assert env.model.request_capture_mode()
    assert env.model.mode == CAPTURE_MODE
    with pytest.raises(ProfileMissing):
        env.model.trigger()


def test_unsaved_changes_decisions(env):
    env.preview()
    env.model.set_focus(100); env.wait("focus_applied")
    assert env.model.has_unsaved()
    with pytest.raises(NeedsDecision):
        env.model.request_capture_mode()
    assert env.model.request_capture_mode(STAY) is False
    assert env.model.mode == PREVIEW_MODE and env.model.controller.state == "preview"
    to_capture(env, DISCARD)
    assert env.profiles.load("index:0") is None
    # save path
    env.model.enter_preview(); env.wait("mode_changed", nth=3)
    env.model.set_focus(200); env.wait("focus_applied", nth=2)
    to_capture(env_wait(env, 4), SAVE)
    assert env.profiles.load("index:0").focus_requested == 200


def env_wait(env, n):
    return type("W", (), {"model": env.model, "wait": lambda self, ev, nth=1: env.wait(ev, nth=n),
                          })()


def test_saved_focus_reapplied_on_preview_start_and_profile_reload(env, tmp_path):
    env.preview(); env.set_focus_and_save(444)
    to_capture(env)
    env.model.enter_preview()
    env.wait("mode_changed", nth=3)
    env.wait("focus_applied", nth=2)
    assert VideoCap.registry[-1].focus_sets == [444]
    # relaunch: a new model reads the persisted profile
    e2 = Env(tmp_path)
    try:
        e2.model.request_capture_mode()
        p = e2.model.saved
        assert p.focus_requested == 444 and p.focus_readback == 443 and p.is_valid
    finally:
        e2.close()


def test_focus_requested_and_readback_distinct(env):
    env.preview()
    env.model.set_focus(300); env.wait("focus_applied")
    d = env.model.draft
    assert d.focus_requested == 300 and d.focus_readback == 299 and d.focus_verified is None
    with pytest.raises(CaptureConfigError):
        env.model.set_focus(5000)


def test_profile_without_focus_cannot_be_saved(env):
    env.preview()
    with pytest.raises(CaptureConfigError):
        env.model.save_profile()


def test_preview_only_keeps_latest_frame_and_bounded_notifications(env):
    env.preview()
    time.sleep(0.3)
    assert env.count("frame") == 1  # no consumer: a single outstanding notification
    f = env.model.controller.take_frame()
    assert f is not None and f.shape == (720, 1280, 3)
    env.wait("frame", nth=2)
    acq, disp = env.model.controller.fps()
    assert acq > 0 and disp >= 0


def test_preview_open_failure(tmp_path):
    cam_events = []
    c = CameraController(lambda e, **kw: cam_events.append(e), opener=lambda i, a: None)
    c.open_preview(0, "msmf", "native_108mp")
    for _ in range(100):
        if "preview_failed" in cam_events:
            break
        time.sleep(0.02)
    assert "preview_failed" in cam_events and c.state == "idle"
    c.shutdown(5)


def test_shutdown_releases_camera(env):
    env.preview()
    assert env.model.shutdown(5)
    assert VideoCap.registry[0].released == 1


def test_profile_store_is_injectable_and_corrupt_file_kept(tmp_path):
    p = tmp_path / "profiles.json"
    p.write_text("{not json")
    st = JsonProfileStore(str(p))
    with pytest.raises(StoreError):
        st.load("index:0")
    st.save(ShootingProfile(camera_key="index:0", focus_requested=1))
    assert any(n.startswith("profiles.json.corrupt-") for n in os.listdir(tmp_path))  # not deleted
    assert st.load("index:0").focus_requested == 1


def test_unique_photo_path_never_overwrites(tmp_path):
    a = unique_photo_path(str(tmp_path), now=1_700_000_000)
    open(a, "wb").close()
    b = unique_photo_path(str(tmp_path), now=1_700_000_000)
    assert a != b and not os.path.exists(b)


def test_engine_independent_of_qt():
    import subprocess, sys
    code = ("import sys, arducam_photo, arducam_photo.app, arducam_photo.app.model;"
            "assert not any(m.startswith('PySide6') for m in sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True)
