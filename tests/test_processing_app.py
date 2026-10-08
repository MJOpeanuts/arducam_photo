"""Offline orchestration and archive-first capture regression tests."""

import dataclasses
import json
import os
import threading
import time

import cv2
import numpy as np
import pytest

from arducam_photo import engine
from arducam_photo import archive, processing
from arducam_photo.config import AcquisitionInfo, AcquisitionResult, CaptureConfig, COLOR_720P, NATIVE_108MP, MODES
from arducam_photo.app.controller import CameraController, CaptureJob, ModeError
from arducam_photo.app.model import CAPTURE_MODE, PROCESSING_MODE, SessionModel
from arducam_photo.app.profile import JsonProfileStore, JsonSettingsStore, ShootingProfile


class Events:
    def __init__(self):
        self.events = []
        self.condition = threading.Condition()

    def __call__(self, event, **kw):
        with self.condition:
            self.events.append((event, kw))
            self.condition.notify_all()

    def wait(self, event, count=1):
        deadline = time.monotonic() + 5
        with self.condition:
            while sum(e == event for e, _ in self.events) < count:
                assert self.condition.wait(max(0, deadline - time.monotonic())), self.events
        return [kw for e, kw in self.events if e == event][-1]


@pytest.fixture(autouse=True)
def small_test_modes(monkeypatch):
    monkeypatch.setitem(MODES, COLOR_720P, dataclasses.replace(MODES[COLOR_720P], width=16, height=12))
    monkeypatch.setitem(MODES, NATIVE_108MP, dataclasses.replace(MODES[NATIVE_108MP], width=8, height=12))


def acquisition(raw=False):
    cfg = CaptureConfig(path=NATIVE_108MP if raw else COLOR_720P, apply_ccm=raw,
                        ccm_path="missing-ccm.json" if raw else None)
    original = np.full((12, 16) if raw else (12, 16, 3), 70, np.uint8)
    info = AcquisitionInfo(config=cfg, api=cfg.api, camera_index=0, path=cfg.path,
                           transport_size=(16, 12), received_shape=original.shape,
                           reconstructed_shape=original.shape, camera_released=True)
    return AcquisitionResult(original, info)


@pytest.fixture
def controller():
    events = Events()
    worker = CameraController(events, opener=lambda *_: pytest.fail("offline processing opened a camera"))
    yield worker, events
    assert worker.shutdown(5)


def test_capture_archives_before_render_and_camera_already_released(controller, tmp_path, monkeypatch):
    worker, events = controller
    result = acquisition()
    order = []

    def acquire(*args, **kw):
        order.extend(("acquire", "release"))
        return result

    actual_archive, actual_process = archive.archive_acquisition, processing.process_image

    def persist(*args):
        assert order[-1] == "release"
        order.append("archive")
        return actual_archive(*args)

    def render(*args, **kw):
        assert order[-1] == "archive"
        assert events.wait("capture_archived")["path"]
        order.append("render")
        return actual_process(*args, **kw)

    monkeypatch.setattr(engine, "acquire", acquire)
    monkeypatch.setattr(archive, "archive_acquisition", persist)
    monkeypatch.setattr(processing, "process_image", render)
    worker.capture(CaptureJob(result.info.config, str(tmp_path)))
    events.wait("capture_idle")
    done = events.wait("capture_done")
    assert order == ["acquire", "release", "archive", "render"]
    assert os.path.isfile(done["path"]) and os.path.isfile(done["manifest_path"])
    assert set(done["timings"]) == {"acquisition_s", "archive_s", "render_s", "output_s",
                                    "total_until_archive_s", "total_until_output_s"}
    metadata = json.loads(open(done["manifest_path"], encoding="utf-8").read())
    assert metadata["timings"]["total_until_archive_s"] >= metadata["timings"]["archive_s"]
    assert metadata["outputs"][0]["timings"]["render_s"] == done["timings"]["render_s"]
    assert done["info"].timings["render_s"] == done["timings"]["render_s"]
    assert done["info"].duration_s >= done["timings"]["total_until_output_s"]
    assert worker.state == "idle"


def test_missing_ccm_keeps_raw_archive(controller, tmp_path, monkeypatch):
    worker, events = controller
    result = acquisition(raw=True)
    monkeypatch.setattr(engine, "acquire", lambda *a, **kw: result)
    worker.capture(CaptureJob(result.info.config, str(tmp_path)))
    events.wait("capture_idle")
    error = events.wait("capture_failed")
    assert error["stage"] == "render" and error["archive_path"]
    loaded = archive.load_acquisition(error["archive_path"])
    assert loaded.raw and np.array_equal(loaded.original, result.original)
    metadata = json.loads(open(error["archive_path"], encoding="utf-8").read())
    assert metadata["errors"] and not metadata["outputs"]
    assert metadata["timings"]["render_s"] >= 0
    assert error["timings"]["total_until_failure_s"] >= error["timings"]["render_s"]


def test_raw_capture_completion_reports_effective_ccm_and_all_phase_times(controller, tmp_path, monkeypatch, ccm_file):
    worker, events = controller
    original = acquisition(True)
    cfg = dataclasses.replace(original.info.config, ccm_path=ccm_file)
    original = dataclasses.replace(original, info=dataclasses.replace(original.info, config=cfg))
    monkeypatch.setattr(engine, "acquire", lambda *a, **kw: original)
    worker.capture(CaptureJob(cfg, str(tmp_path)))
    events.wait("capture_idle")
    info = events.wait("capture_done")["info"]
    assert info.ccm_requested and info.ccm_applied
    assert info.ccm_path == os.path.abspath(ccm_file)
    assert {"acquisition_s", "archive_s", "render_s", "output_s"} <= info.timings.keys()


def test_archive_failure_never_claims_success(controller, tmp_path, monkeypatch):
    worker, events = controller
    result = acquisition()
    monkeypatch.setattr(engine, "acquire", lambda *a, **kw: result)
    monkeypatch.setattr(archive, "archive_acquisition",
                        lambda *a: (_ for _ in ()).throw(OSError("archive full")))
    worker.capture(CaptureJob(result.info.config, str(tmp_path)))
    events.wait("capture_idle")
    assert events.wait("capture_failed")["archive_path"] is None
    assert not any(e == "capture_archived" for e, _ in events.events)


def test_release_failure_archives_original_but_never_renders_or_reopens(controller, tmp_path, monkeypatch):
    worker, events = controller
    result = acquisition()
    result = dataclasses.replace(result, info=dataclasses.replace(result.info, camera_released=False,
                                                                  release_error="release test failure"))
    monkeypatch.setattr(engine, "acquire", lambda *a, **kw: result)
    monkeypatch.setattr(processing, "process_image", lambda *a, **kw: pytest.fail("render after release failure"))
    worker.capture(CaptureJob(result.info.config, str(tmp_path)))
    events.wait("capture_idle")
    failed = events.wait("capture_failed")
    assert failed["stage"] == "release" and os.path.isfile(failed["archive_path"])
    with pytest.raises(ModeError, match="redémarrage requis"):
        worker.open_preview(0, "msmf", COLOR_720P)
    with pytest.raises(ModeError, match="redémarrage requis"):
        worker.capture(CaptureJob(result.info.config, str(tmp_path)))


def test_offline_load_process_save_on_worker_only(controller, tmp_path, monkeypatch):
    worker, events = controller
    manifest = archive.archive_acquisition(acquisition(), tmp_path)
    main = threading.get_ident()
    threads = []
    real_load, real_process, real_save = archive.load_acquisition, processing.process_image, archive.save_output

    def wrap(fn):
        def call(*a, **kw):
            threads.append(threading.get_ident())
            return fn(*a, **kw)
        return call

    monkeypatch.setattr(archive, "load_acquisition", wrap(real_load))
    monkeypatch.setattr(processing, "process_image", wrap(real_process))
    monkeypatch.setattr(archive, "save_output", wrap(real_save))
    worker.load_acquisition(manifest)
    events.wait("processing_idle")
    assert events.wait("processing_loaded")["raw"] is False
    recipe = processing.ProcessingRecipe(apply_ccm=False, grayscale=True, max_dimension=8)
    worker.process(manifest, recipe)
    events.wait("processing_idle", 2)
    assert events.wait("processing_done")["size"] == (8, 6)
    worker.save_processed(archive.OutputOptions(format="jpeg", jpeg_quality=80))
    events.wait("processing_idle", 3)
    saved = events.wait("processing_saved")["path"]
    assert os.path.isfile(saved) and saved.endswith(".jpg")
    assert len(set(threads)) == 1 and main not in threads
    assert archive.load_acquisition(manifest).original.shape == (12, 16, 3)


def test_processing_reserves_single_worker_and_blocks_camera(controller, tmp_path, monkeypatch):
    worker, events = controller
    manifest = archive.archive_acquisition(acquisition(), tmp_path)
    gate = threading.Event()
    actual = archive.load_acquisition

    def blocked(*a):
        assert gate.wait(5)
        return actual(*a)

    monkeypatch.setattr(archive, "load_acquisition", blocked)
    worker.load_acquisition(manifest)
    try:
        assert worker.state == "processing" and not worker.can_change_mode
        assert not worker.camera_in_use
        with pytest.raises(ModeError):
            worker.open_preview(0, "msmf", COLOR_720P)
        with pytest.raises(ModeError):
            worker.capture(CaptureJob(CaptureConfig(path=COLOR_720P), str(tmp_path)))
        with pytest.raises(ModeError):
            worker.load_acquisition(manifest)
    finally:
        gate.set()
    events.wait("processing_idle")
    assert worker._q.empty()


def test_load_failure_clears_retained_result(controller, tmp_path):
    worker, events = controller
    manifest = archive.archive_acquisition(acquisition(), tmp_path)
    worker.process(manifest, processing.ProcessingRecipe(apply_ccm=False))
    events.wait("processing_idle")
    assert worker._processed is not None
    worker.load_acquisition(str(tmp_path / "missing.json"))
    events.wait("processing_idle", 2)
    assert worker._processed is None
    worker.save_processed(archive.OutputOptions())
    events.wait("processing_idle", 3)
    assert "aucun résultat" in str(events.wait("processing_failed")["error"])


def test_raw_reference_is_minimal_demosaic_not_mosaic(controller, tmp_path):
    worker, events = controller
    manifest = archive.archive_acquisition(acquisition(True), tmp_path)
    worker.load_acquisition(manifest)
    events.wait("processing_idle")
    loaded = events.wait("processing_loaded")
    assert loaded["raw"] and loaded["thumbnail"].shape == (12, 16, 3)
    assert np.all(loaded["thumbnail"] == 70)  # no black subtraction in minimal reference


def test_external_image_can_be_processed_and_exported(controller, tmp_path):
    worker, events = controller
    path = str(tmp_path / "external.png")
    assert cv2.imwrite(path, acquisition().original)
    worker.load_acquisition(path)
    events.wait("processing_idle")
    worker.process(path, processing.ProcessingRecipe(apply_ccm=False))
    events.wait("processing_idle", 2)
    worker.save_processed(archive.OutputOptions())
    events.wait("processing_idle", 3)
    assert os.path.isfile(events.wait("processing_saved")["path"])
    assert cv2.imread(path).shape == (12, 16, 3)


def test_external_source_mutation_cannot_be_silently_relabelled(controller, tmp_path):
    worker, events = controller
    path = str(tmp_path / "external.png")
    assert cv2.imwrite(path, acquisition().original)
    worker.process(path, processing.ProcessingRecipe(apply_ccm=False))
    events.wait("processing_idle")
    assert cv2.imwrite(path, np.zeros((12, 16, 3), np.uint8))
    worker.save_processed(archive.OutputOptions())
    events.wait("processing_idle", 2)
    assert "source changed" in str(events.wait("processing_failed")["error"])


def test_changed_source_requires_reload_before_comparison(controller, tmp_path):
    worker, events = controller
    path = str(tmp_path / "external.png")
    assert cv2.imwrite(path, acquisition().original)
    worker.load_acquisition(path)
    events.wait("processing_idle")
    assert cv2.imwrite(path, np.zeros((12, 16, 3), np.uint8))
    worker.process(path, processing.ProcessingRecipe(apply_ccm=False))
    events.wait("processing_idle", 2)
    assert "source a changé" in str(events.wait("processing_failed")["error"])


def test_processing_exports_effective_ccm_snapshot_not_later_file(controller, tmp_path, ccm_file):
    import hashlib
    worker, events = controller
    manifest = archive.archive_acquisition(acquisition(True), tmp_path)
    expected_hash = hashlib.sha256(open(ccm_file, "rb").read()).hexdigest()
    worker.process(manifest, processing.ProcessingRecipe(), ccm_file)
    events.wait("processing_idle")
    events.wait("processing_done")
    os.unlink(ccm_file)
    worker.save_processed(archive.OutputOptions())
    events.wait("processing_idle", 2)
    assert os.path.isfile(events.wait("processing_saved")["path"])
    metadata = json.loads(open(manifest, encoding="utf-8").read())
    assert metadata["outputs"][-1]["ccm"]["sha256"] == expected_hash
    assert metadata["outputs"][-1]["ccm"]["effective_bgr_matrix"]


def test_processing_to_capture_never_opens_preview(tmp_path):
    opened = []
    model = SessionModel(JsonProfileStore(str(tmp_path / "p.json")), JsonSettingsStore(str(tmp_path / "s.json")),
                         opener=lambda *a: opened.append(a))
    try:
        model.request_processing_mode()
        assert model.mode == PROCESSING_MODE
        model.request_capture_mode()
        assert model.mode == CAPTURE_MODE and opened == []
    finally:
        assert model.shutdown(5)


def test_default_model_missing_bundled_ccm_still_acquires_and_archives(tmp_path, monkeypatch):
    events = Events()
    model = SessionModel(JsonProfileStore(str(tmp_path / "p.json")), JsonSettingsStore(str(tmp_path / "s.json")),
                         events)
    profile = ShootingProfile(camera_key=model.camera.key, focus_requested=123)
    model.profiles.save(profile)
    model.set_photo_dir(str(tmp_path / "photos"))
    model.request_capture_mode()
    from arducam_photo.errors import CaptureConfigError
    monkeypatch.setattr(model, "effective_ccm_path",
                        lambda: (_ for _ in ()).throw(CaptureConfigError("missing bundled")))
    monkeypatch.setattr(engine, "acquire", lambda *a, **kw: acquisition(True))
    try:
        model.trigger()
        events.wait("capture_idle")
        assert model.last_acquisition and not model.last_photo
        assert os.path.isfile(model.last_acquisition)
    finally:
        model.shutdown(5)


def test_shutdown_waits_processing_before_releasing_bundled_resource(tmp_path, monkeypatch):
    events = Events()
    model = SessionModel(JsonProfileStore(str(tmp_path / "p.json")), JsonSettingsStore(str(tmp_path / "s.json")),
                         events)
    closed = []
    model._resources.callback(lambda: closed.append(True))
    source = str(tmp_path / "source.png")
    assert cv2.imwrite(source, acquisition().original)
    gate = threading.Event()
    started = threading.Event()
    real = archive.load_acquisition

    def blocked(*args):
        started.set()
        assert gate.wait(5)
        return real(*args)

    monkeypatch.setattr(archive, "load_acquisition", blocked)
    model.request_processing_mode()
    model.load_processing(source)
    assert started.wait(5)
    try:
        with pytest.raises(ModeError):
            model.enter_preview()
        with pytest.raises(ModeError):
            model.request_capture_mode()
        assert not model.shutdown(0) and not closed
    finally:
        gate.set()
        assert model.shutdown(5)
    assert closed == [True] and model.controller._processed is None
