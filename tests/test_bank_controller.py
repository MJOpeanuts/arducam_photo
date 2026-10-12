"""Camera-worker exclusivity and immutable capture-bank provenance."""

import dataclasses
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from arducam_photo import archive, engine
from arducam_photo.app.controller import (
    ADJUSTING, PREVIEW, RESTORING, CameraController, CaptureJob, ModeError,
)
from arducam_photo.app.preview import PreviewSession
from arducam_photo.config import COLOR_720P, MODES, NATIVE_108MP, CaptureConfig


class Events:
    def __init__(self):
        self.items = []
        self.condition = threading.Condition()

    def __call__(self, event, **payload):
        with self.condition:
            self.items.append((event, payload))
            self.condition.notify_all()

    def wait(self, event, count=1):
        deadline = time.monotonic() + 5
        with self.condition:
            while sum(name == event for name, _ in self.items) < count:
                assert self.condition.wait(max(0, deadline - time.monotonic())), self.items
            return [payload for name, payload in self.items if name == event][-1]


class Camera:
    def __init__(self, frame=None):
        self.frame = np.zeros((8, 12, 3), np.uint8) if frame is None else frame
        self.props = {cv2.CAP_PROP_FOCUS: 31, cv2.CAP_PROP_GAIN: 2}
        self.calls = []
        self.released = 0

    def record(self, operation, *args):
        self.calls.append((threading.get_ident(), operation, args))

    def isOpened(self):
        self.record("isOpened")
        return True

    def set(self, prop, value):
        self.record("set", prop, value)
        if prop == cv2.CAP_PROP_CONTRAST:
            return False
        self.props[prop] = value
        return True

    def get(self, prop):
        self.record("get", prop)
        if prop == cv2.CAP_PROP_BRIGHTNESS:
            return float("nan")
        if prop == cv2.CAP_PROP_SATURATION:
            raise RuntimeError("unreadable")
        return self.props.get(prop, 0)

    def read(self):
        self.record("read")
        time.sleep(0.001)
        return True, self.frame

    def release(self):
        self.record("release")
        self.released += 1


@pytest.fixture
def preview_worker():
    events, camera = Events(), Camera()
    worker = CameraController(events, opener=lambda *_: camera)
    worker.open_preview(0, "any", COLOR_720P)
    events.wait("preview_opened")
    yield worker, events, camera
    assert worker.shutdown(5)
    assert camera.released == 1


def fake_adjustment(monkeypatch, run, restore):
    monkeypatch.setitem(sys.modules, "arducam_photo.auto_adjust", SimpleNamespace(
        AutoAdjuster=lambda: SimpleNamespace(run=run), restore_settings=restore))


def assert_blocked(worker):
    for operation in (
        lambda: worker.set_focus(12),
        lambda: worker.apply_camera_settings({"gain": 5}),
        worker.close_preview,
        worker.probe,
        worker.probe_controls,
        lambda: worker.capture(CaptureJob(CaptureConfig(), ".")),
        lambda: worker.auto_adjust({}, object()),
        lambda: worker.reject_adjustment({}),
    ):
        with pytest.raises(ModeError):
            operation()


def test_adjustment_reserves_preview_and_uses_only_existing_worker(preview_worker, monkeypatch):
    worker, events, camera = preview_worker
    entered, finish = threading.Event(), threading.Event()
    result = SimpleNamespace(before={"gain": 2}, proposed={"gain": 7})
    sessions = []

    def run(session, before, options, cancelled):
        sessions.append(session)
        assert before == {"gain": 2}
        assert options == {"test": True}
        session.apply_settings({"gain": 7})
        session.read()
        entered.set()
        assert finish.wait(5)
        assert not cancelled()
        return result

    fake_adjustment(monkeypatch, run, lambda *_: pytest.fail("successful proposal restored"))
    before = {"gain": 2}
    worker.auto_adjust(before, {"test": True})
    before["gain"] = 99
    assert worker.state == ADJUSTING
    try:
        assert entered.wait(5)
        events.wait("adjustment_started")
        assert_blocked(worker)
        calls = len(camera.calls)
        time.sleep(0.03)
        assert len(camera.calls) == calls  # no independent preview pump during the sequence
    finally:
        finish.set()
    assert events.wait("adjustment_proposed")["result"] is result
    assert worker.state == PREVIEW and sessions == [worker._session]
    assert {thread for thread, _, _ in camera.calls} == {worker._thread.ident}
    worker.apply_camera_settings({"gain": 4})
    events.wait("camera_settings_applied")
    assert camera.props[cv2.CAP_PROP_GAIN] == 4


def test_cancellation_restores_and_does_not_cancel_worker(preview_worker, monkeypatch):
    worker, events, camera = preview_worker
    entered = threading.Event()

    def run(session, before, options, cancelled):
        session.apply_settings({"gain": 99})
        entered.set()
        deadline = time.monotonic() + 5
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert cancelled()
        session.apply_settings({"gain": 2})
        error = RuntimeError("adjustment cancelled")
        error.restoration_warnings = ("unavailable exposure",)
        raise error

    fake_adjustment(monkeypatch, run, lambda *_: pytest.fail("service already restored"))
    worker.auto_adjust({"gain": 2}, {})
    assert entered.wait(5)
    worker.cancel_adjustment()
    failure = events.wait("adjustment_failed")
    assert str(failure["error"]) == "adjustment cancelled"
    assert failure["restoration_warnings"] == ["unavailable exposure"]
    assert worker.state == PREVIEW and camera.props[cv2.CAP_PROP_GAIN] == 2
    assert not worker._cancel.is_set()
    worker.set_focus(45)
    events.wait("focus_applied")


def test_rejection_restores_asynchronously_and_blocks_commands(preview_worker, monkeypatch):
    worker, events, camera = preview_worker
    entered, finish = threading.Event(), threading.Event()

    def restore(session, before):
        entered.set()
        assert finish.wait(5)
        session.apply_settings(before)
        return ["automatic mode unavailable"]

    fake_adjustment(monkeypatch, lambda *_: None, restore)
    before = {"gain": 2}
    worker.reject_adjustment(before)
    before["gain"] = 88
    assert worker.state == RESTORING
    try:
        assert entered.wait(5)
        events.wait("adjustment_restoration_started")
        assert_blocked(worker)
        calls = len(camera.calls)
        time.sleep(0.03)
        assert len(camera.calls) == calls
    finally:
        finish.set()
    assert events.wait("adjustment_restored")["warnings"] == ["automatic mode unavailable"]
    assert worker.state == PREVIEW and camera.props[cv2.CAP_PROP_GAIN] == 2


def test_restoration_exception_is_reported_and_preview_resumes(preview_worker, monkeypatch):
    worker, events, _ = preview_worker

    def restore(*_):
        raise RuntimeError("driver unavailable")

    fake_adjustment(monkeypatch, lambda *_: None, restore)
    worker.reject_adjustment({"gain": 2})
    assert events.wait("adjustment_restored")["warnings"] == [
        "restoration failed: driver unavailable"]
    assert worker.state == PREVIEW


def test_service_restoration_warnings_are_not_followed_by_requested_rollback(
        preview_worker, monkeypatch):
    worker, events, camera = preview_worker

    class RestoredError(RuntimeError):
        restoration_warnings = ("driver rejected exposure restoration",)

    def run(session, before, options, cancelled):
        raise RestoredError("already restored")

    fake_adjustment(monkeypatch, run, lambda *_: pytest.fail("duplicate unsafe rollback"))
    worker.auto_adjust({"gain": 88}, {})
    failure = events.wait("adjustment_failed")
    assert failure["restoration_warnings"] == ["driver rejected exposure restoration"]
    assert camera.props[cv2.CAP_PROP_GAIN] == 2 and worker.state == PREVIEW


def test_generic_service_error_does_not_write_requested_controls(preview_worker, monkeypatch):
    worker, events, camera = preview_worker

    def run(session, before, options, cancelled):
        session.apply_settings({"gain": 7})
        session.apply_settings({"gain": 2})
        raise RuntimeError("unexpected failure")

    fake_adjustment(monkeypatch, run, lambda *_: pytest.fail("unexpected wrapper rollback"))
    worker.auto_adjust({"gain": 88, "focus": 99}, {})
    assert events.wait("adjustment_failed")["restoration_warnings"] == []
    assert camera.props[cv2.CAP_PROP_GAIN] == 2
    assert not any(operation == "set" and args[0] == cv2.CAP_PROP_FOCUS
                   for _, operation, args in camera.calls)


def test_cancellation_before_proposal_restores_even_if_service_returns(preview_worker, monkeypatch):
    worker, events, camera = preview_worker

    def run(session, before, options, cancelled):
        session.apply_settings({"gain": 7})
        worker.cancel_adjustment()
        return SimpleNamespace(original_settings={"gain": 2})

    def restore(session, before):
        assert before == {"gain": 2}
        session.apply_settings(before)
        return []

    fake_adjustment(monkeypatch, run, restore)
    worker.auto_adjust({"gain": 88}, {})
    assert str(events.wait("adjustment_failed")["error"]) == "adjustment cancelled"
    assert camera.props[cv2.CAP_PROP_GAIN] == 2
    assert not any(name == "adjustment_proposed" for name, _ in events.items)


def test_rejection_prefers_service_observed_original_to_requested_bank(
        preview_worker, monkeypatch):
    worker, events, camera = preview_worker

    def run(session, before, options, cancelled):
        session.apply_settings({"gain": 7})
        return SimpleNamespace(
            before={"gain": 88}, proposed={"gain": 7}, original_settings={"gain": 2})

    def restore(session, before):
        assert before == {"gain": 2}
        session.apply_settings(before)
        return []

    fake_adjustment(monkeypatch, run, restore)
    worker.auto_adjust({"gain": 88}, {})
    events.wait("adjustment_proposed")
    worker.reject_adjustment({"gain": 88})
    events.wait("adjustment_restored")
    assert camera.props[cv2.CAP_PROP_GAIN] == 2


def test_shutdown_signals_both_cancellations(preview_worker, monkeypatch):
    worker, events, _ = preview_worker
    entered = threading.Event()

    def run(session, before, options, cancelled):
        entered.set()
        deadline = time.monotonic() + 5
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert cancelled()
        raise RuntimeError("stopped")

    fake_adjustment(monkeypatch, run, lambda *_: [])
    worker.auto_adjust({}, {})
    assert entered.wait(5)
    assert worker.shutdown(5)
    events.wait("adjustment_failed")
    assert worker._cancel.is_set() and worker._adjust_cancel.is_set()


def test_snapshot_reads_observed_controls_not_persisted_requests():
    camera = Camera()
    session = PreviewSession(0, "any", COLOR_720P, lambda *_: camera)
    session.open()
    try:
        session.apply_settings({"focus": 100, "auto_exposure": 0.25,
                                "auto_exposure_mode": "manual"})
        camera.props[cv2.CAP_PROP_FOCUS] = 31
        snapshot = session.snapshot_settings()
        assert snapshot["focus"] == 31 and session.settings["focus"].requested == 100
        assert snapshot["brightness"] is None and snapshot["saturation"] is None
        assert snapshot["auto_exposure"] == 0.25
        assert snapshot["auto_exposure_mode"] is None
        assert snapshot["auto_wb_mode"] is None
        assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot
    finally:
        session.close()


def test_explicit_control_probe_requires_driver_set_acceptance(preview_worker):
    worker, events, camera = preview_worker
    assert not any(event == "camera_capabilities" for event, _ in events.items)
    worker.probe_controls()
    reports = events.wait("camera_capabilities")["reports"]
    assert reports["focus"].accepted is True
    assert reports["focus"].requested == reports["focus"].readback == 31
    assert reports["contrast"].readback == 0 and reports["contrast"].accepted is False
    assert reports["brightness"].accepted is None and not reports["brightness"].attempted
    assert reports["saturation"].accepted is None and not reports["saturation"].attempted
    assert {thread for thread, _, _ in camera.calls} == {worker._thread.ident}
    assert worker.state == PREVIEW


def test_control_probe_failure_still_emits_capabilities_and_unlocks(preview_worker, monkeypatch):
    worker, events, _ = preview_worker

    def fail_probe():
        raise RuntimeError("probe driver failed")

    monkeypatch.setattr(worker._session, "probe_controls", fail_probe)
    worker.probe_controls()
    payload = events.wait("camera_capabilities")
    assert payload["reports"] == {}
    assert str(payload["error"]) == "probe driver failed"
    assert worker.state == PREVIEW


@pytest.mark.parametrize("raw", [False, True])
def test_capture_freezes_bank_snapshot_in_both_manifests(tmp_path, monkeypatch, raw):
    mode = NATIVE_108MP if raw else COLOR_720P
    monkeypatch.setitem(MODES, mode, dataclasses.replace(
        MODES[mode], width=6 if raw else 12, height=8))
    if raw:
        monkeypatch.setattr(engine, "RAW_ROWS", 8)
        monkeypatch.setattr(engine, "RAW_COLS", 12)
        monkeypatch.setattr(engine, "RAW_BYTES", 96)
        monkeypatch.setattr(engine, "TRANSPORT_W", 6)
        monkeypatch.setattr(engine, "TRANSPORT_H", 8)
    original = np.full((96,) if raw else (8, 12, 3), 70, np.uint8)
    camera, events = Camera(original), Events()
    config = CaptureConfig(api="any", path=mode, focus=31, gain=2, apply_ccm=False,
                           stabilization_reads=1)
    snapshot = {"id": "bank-1", "name": "workbench", "revision": 3, "schema_version": 1,
                "configuration": {"camera": {"focus": 31, "gain": 2},
                                  "observed": {"gain": None}}}
    expected = json.loads(json.dumps(snapshot))
    job = CaptureJob(config, str(tmp_path), snapshot)
    assert isinstance(job.bank_snapshot, str)
    snapshot["name"] = "edited"
    snapshot["configuration"]["camera"]["focus"] = 99
    entered, finish = threading.Event(), threading.Event()

    def open_camera(*_):
        entered.set()
        assert finish.wait(5)
        return camera

    worker = CameraController(events, opener=open_camera)
    try:
        worker.capture(job)
        assert entered.wait(5)
        snapshot["revision"] = 4
        finish.set()
        events.wait("capture_idle")
        done = events.wait("capture_done")
        archive_data = json.loads(Path(done["manifest_path"]).read_text())
        bundle = next(tmp_path.glob("*.json"))
        capture_data = json.loads(bundle.read_text())
        assert done["info"].bank_snapshot == expected
        for metadata in (archive_data, capture_data):
            assert metadata["bank_snapshot"] == expected
            provenance = metadata["control_provenance"]
            assert provenance["requested"]["focus"] == 31
            assert provenance["reports"]["focus"]["accepted"] is True
            assert provenance["readbacks"]["focus"] == 31
            assert "brightness" in provenance["unavailable"]
            assert any("preview != RAW" in text for text in provenance["limitations"])
            assert any("fragile" in text for text in provenance["limitations"])
            assert any("identical exposure" in text for text in provenance["limitations"])
        loaded = archive.load_acquisition(done["manifest_path"])
        assert np.array_equal(loaded.original, original.reshape((8, 12) if raw else (8, 12, 3)))
        assert camera.released == 1
        if raw:
            assert bundle.with_suffix(".raw").read_bytes() == original.tobytes()
    finally:
        finish.set()
        assert worker.shutdown(5)


def test_capture_job_rejects_nonobject_or_nonfinite_snapshot():
    with pytest.raises(ValueError):
        CaptureJob(CaptureConfig(), ".", "[]")
    with pytest.raises(ValueError):
        CaptureJob(CaptureConfig(), ".", {"bad": float("nan")})
