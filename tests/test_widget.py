import gc
import os
import dataclasses
from importlib import resources
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QEnterEvent, QImage, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea, QToolButton

from arducam_photo.app.widget import CameraWidget, bgr_to_qimage
from arducam_photo.app.model import SessionModel
from arducam_photo.app.profile import JsonProfileStore, JsonSettingsStore
from arducam_photo.app.profile import ShootingProfile
from arducam_photo.app.model import PREVIEW_MODE, CAPTURE_MODE
from arducam_photo.app.icons import lucide_icon


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_qimage_owns_pixels_after_source_freed(qapp):
    frame = np.zeros((20, 30, 3), np.uint8)
    frame[..., 2] = 200  # red in BGR
    img = bgr_to_qimage(frame)
    del frame
    gc.collect()
    junk = [np.full((20, 30, 3), 7, np.uint8) for _ in range(50)]  # reuse freed memory
    c = img.pixelColor(5, 5)
    assert (c.red(), c.green(), c.blue()) == (200, 0, 0) and img.width() == 30 and len(junk) == 50


def test_widget_start_has_no_camera_and_closes_cleanly(qapp, tmp_path):
    opened = []
    model = SessionModel(JsonProfileStore(str(tmp_path / "p.json")), JsonSettingsStore(str(tmp_path / "s.json")),
                         opener=lambda i, a: opened.append(i))
    w = CameraWidget(model)
    w.show()
    qapp.processEvents()
    assert opened == [] and model.mode == "start"
    assert w.can_close()
    assert w.shutdown()


@pytest.fixture
def widget(qapp, tmp_path):
    model = SessionModel(JsonProfileStore(str(tmp_path / "profiles.json")),
                         JsonSettingsStore(str(tmp_path / "settings.json")))
    model.saved = ShootingProfile(camera_key=model.camera.key, camera_index=model.camera.index,
                                  path=model.path, api=model.api, focus_requested=300,
                                  focus_readback=299)
    model.draft = model.saved
    w = CameraWidget(model)
    w.resize(1280, 720)
    w.show()
    qapp.processEvents()
    yield w
    model.mode = CAPTURE_MODE
    model.controller._state = "idle"
    assert w.shutdown()
    w.close()
    qapp.processEvents()


def show_mode(w, mode, qapp):
    w.model.mode = mode
    w.model.controller._state = "preview" if mode == PREVIEW_MODE else "idle"
    w._show_page(mode)
    qapp.processEvents()


def test_save_status_is_unique_and_focus_restored(widget, qapp):
    show_mode(widget, PREVIEW_MODE, qapp)
    assert widget.focus_spin.value() == widget.focus_slider.value() == 300
    assert widget.unsaved.text() == "Réglages enregistrés"
    widget.model.draft = dataclasses.replace(widget.model.saved, focus_requested=301)
    widget._refresh_preview_state()
    assert widget.unsaved.text() == "Modifications non enregistrées"
    widget._save()
    assert widget.unsaved.text() == "Réglages enregistrés"
    labels = [label.text() for label in widget.findChildren(QLabel) if label.isVisible()]
    assert labels.count("Réglages enregistrés") == 1
    widget.model.draft = dataclasses.replace(widget.model.saved, focus_requested=302)
    widget._refresh_preview_state()
    assert widget.unsaved.text() == "Modifications non enregistrées"
    assert widget.preview_msg.text() != "Réglages enregistrés."


def test_save_failure_keeps_dirty_status(widget, qapp, monkeypatch):
    show_mode(widget, PREVIEW_MODE, qapp)
    widget.model.draft = dataclasses.replace(widget.model.saved, focus_requested=301)

    def fail_save(profile):
        raise OSError("profile-storage-test-failure")

    monkeypatch.setattr(widget.model.profiles, "save", fail_save)
    widget._save()
    assert widget.unsaved.text() == "Modifications non enregistrées"
    assert widget.model.saved.focus_requested == 300
    assert widget.preview_msg.isVisible()
    assert "profile-storage-test-failure" not in widget.preview_msg.text()
    assert "profile-storage-test-failure" in widget.preview_details.text()


@pytest.mark.parametrize("mode,image_name", [(PREVIEW_MODE, "video"), (CAPTURE_MODE, "photo")])
def test_image_rescales_without_new_frame(widget, qapp, mode, image_name):
    show_mode(widget, mode, qapp)
    if mode == PREVIEW_MODE:
        widget.model.controller._latest = np.zeros((180, 320, 3), np.uint8)
        widget._on_event("frame", {})
    else:
        path = os.path.join(widget.model.photo_dir(), "unused-test-photo.png")
        widget._on_event("capture_done", {
            "path": path, "size": (320, 180), "thumbnail": np.zeros((180, 320, 3), np.uint8),
            "info": SimpleNamespace(ccm_applied=True, duration_s=0.25),
        })
    label = getattr(widget, image_name)
    qapp.processEvents()
    first = label.pixmap().size()
    widget.resize(1000, 620)
    qapp.processEvents()
    second = label.pixmap().size()
    assert second != first
    assert abs(second.width() / second.height() - 16 / 9) < 0.02
    assert second.width() <= label.width() and second.height() <= label.height()
    widget.resize(1440, 900)
    qapp.processEvents()
    assert label.pixmap().width() > second.width()


def test_fullscreen_escape(widget, qapp):
    show_mode(widget, PREVIEW_MODE, qapp)
    widget.model.controller._latest = np.zeros((180, 320, 3), np.uint8)
    widget._on_event("frame", {})
    button = next(b for b in widget.findChildren(QPushButton) if b.text() == "Plein écran")
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    fullscreen = next(w for w in qapp.topLevelWidgets() if w.isFullScreen())
    QTest.keyClick(fullscreen, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not any(w.isFullScreen() for w in qapp.topLevelWidgets())
    assert widget.video.isVisible()
    assert widget.model.mode == PREVIEW_MODE


@pytest.mark.parametrize("mode", [PREVIEW_MODE, CAPTURE_MODE])
@pytest.mark.parametrize("window_size", [(1280, 720), (838, 400)])
def test_diagnostic_collapses_without_reserved_space(widget, qapp, mode, window_size):
    widget.resize(*window_size)
    show_mode(widget, mode, qapp)
    button = next(b for b in widget.findChildren(QToolButton)
                  if b.isVisible() and "Diagnostic" in b.text())
    height = (widget.video if mode == PREVIEW_MODE else widget.photo).height()
    assert not button.isChecked()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert button.isChecked()
    assert (widget.width(), widget.height()) == window_size
    area = next(a for a in widget.findChildren(QScrollArea) if a.isVisible())
    assert area.height() <= 150 and area.widgetResizable()
    assert (widget.video if mode == PREVIEW_MODE else widget.photo).height() < height
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert not button.isChecked()
    assert (widget.video if mode == PREVIEW_MODE else widget.photo).height() == height


def test_capture_busy_failure_and_opening_availability(widget, qapp, tmp_path):
    show_mode(widget, CAPTURE_MODE, qapp)
    assert not widget.btn_open_last.isEnabled()
    assert widget.photo.pixmap() is None or widget.photo.pixmap().isNull()
    widget._set_capture_busy(True)
    assert not widget.btn_trigger.isEnabled() and not widget.btn_c_preview.isEnabled()
    widget._on_event("capture_failed", {"error": OSError("technical-test-details")})
    widget._on_event("capture_idle", {})
    assert widget.model.mode == CAPTURE_MODE
    assert widget.btn_trigger.isEnabled() and widget.btn_c_preview.isEnabled()
    assert widget.result_label.text()
    photo_path = tmp_path / "photo.png"
    img = QImage(320, 180, QImage.Format.Format_RGB32)
    img.fill(Qt.GlobalColor.darkGray)
    assert img.save(str(photo_path))
    widget.model.last_photo = str(photo_path)
    widget._on_event("capture_done", {
        "path": str(photo_path), "size": (320, 180), "thumbnail": np.zeros((180, 320, 3), np.uint8),
        "info": SimpleNamespace(ccm_applied=True, duration_s=0.25),
    })
    assert widget.btn_open_last.isEnabled()
    assert "technical-test-details" not in widget.result_label.text()
    assert widget.btn_open_dir.height() == widget.btn_open_last.height()


def test_bundled_svg_icons(qapp):
    for name in ("squirrel", "refresh-cw"):
        ref = resources.files("arducam_photo.resources").joinpath("icons", name + ".svg")
        assert b'viewBox="0 0 24 24"' in ref.read_bytes()
        icon = lucide_icon(name, "#FF963F")
        assert not icon.isNull()
        image = icon.pixmap(24, 24).toImage()
        assert any(image.pixelColor(x, y).alpha() for x in range(24) for y in range(24))
    assert "ISC License" in resources.files("arducam_photo.resources").joinpath("icons", "LICENSE").read_text()


@pytest.mark.parametrize("mode", [PREVIEW_MODE, CAPTURE_MODE])
@pytest.mark.parametrize("window_size", [(1280, 720), (1024, 576), (854, 480), (838, 400)])
def test_controls_fit_and_keyboard_focus(widget, qapp, mode, window_size):
    widget.resize(*window_size)
    show_mode(widget, mode, qapp)
    assert (widget.width(), widget.height()) == window_size
    buttons = ((widget.btn_save, widget.btn_p_capture) if mode == PREVIEW_MODE else
               (widget.btn_trigger, widget.btn_open_dir, widget.btn_open_last, widget.btn_c_preview))
    for button in buttons:
        position = button.mapTo(widget, button.rect().topLeft())
        assert position.x() >= 0 and position.y() >= 0
        assert position.x() + button.width() <= widget.width()
        assert position.y() + button.height() <= widget.height()
        assert button.width() >= button.minimumSizeHint().width()
        if button.isEnabled():
            button.setFocus(Qt.FocusReason.TabFocusReason)
            qapp.processEvents()
            assert button.hasFocus()
            QTest.mouseMove(button, button.rect().center())
    output = os.environ.get("ARDUCAM_SCREENSHOT_DIR")
    if output and window_size == (1280, 720):
        os.makedirs(output, exist_ok=True)
        assert widget.grab().save(os.path.join(output, f"{mode}.png"))


def test_action_button_palette_and_states(widget, qapp):
    show_mode(widget, CAPTURE_MODE, qapp)

    def background(button):
        image = button.grab().toImage()
        scale = image.devicePixelRatio()
        return image.pixelColor(round(button.width() * scale / 2), round(8 * scale)).name().upper()

    for button, normal, hover in (
        (widget.btn_trigger, "#FF963F", "#FFA65E"),
        (widget.btn_c_preview, "#D9DDE0", "#ECEFF1"),
    ):
        qapp.sendEvent(button, QEvent(QEvent.Type.Leave))
        button.clearFocus()
        qapp.processEvents()
        assert background(button) == normal
        center = QPointF(button.rect().center())
        global_center = QPointF(button.mapToGlobal(button.rect().center()))
        qapp.sendEvent(button, QEnterEvent(center, center, global_center))
        qapp.sendEvent(button, QMouseEvent(QEvent.Type.MouseMove, center, global_center,
                                          Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                                          Qt.KeyboardModifier.NoModifier))
        qapp.processEvents()
        assert background(button) == hover
        button.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()
        assert button.hasFocus()
        button.setEnabled(False)
        qapp.processEvents()
        assert background(button) == "#343739"
        button.setEnabled(True)
    show_mode(widget, PREVIEW_MODE, qapp)
    qapp.sendEvent(widget.btn_p_capture, QEvent(QEvent.Type.Leave))
    widget.btn_p_capture.clearFocus()
    qapp.processEvents()
    assert background(widget.btn_p_capture) == "#D9DDE0"


def test_widget_real_controller_transitions_and_capture(qapp, tmp_path):
    from test_app import Env, VideoCap

    env = Env(tmp_path)
    env.model.set_photo_dir(str(tmp_path / "photos"))
    w = CameraWidget(env.model)
    w.resize(1280, 720)
    w.show()

    def wait_until(predicate):
        for _ in range(500):
            qapp.processEvents()
            if predicate():
                return
            QTest.qWait(10)
        pytest.fail("Timed out waiting for controller/UI transition")

    def screenshot(name):
        output = os.environ.get("ARDUCAM_SCREENSHOT_DIR")
        if output:
            os.makedirs(output, exist_ok=True)
            assert w.grab().save(os.path.join(output, name + ".png"))

    try:
        w._go_preview()
        wait_until(lambda: env.model.mode == PREVIEW_MODE and w.video.pixmap() is not None
                   and not w.video.pixmap().isNull())
        w.focus_spin.setValue(300)
        wait_until(lambda: env.model.draft.focus_readback is not None)
        w._save()
        assert w.unsaved.text() == "Réglages enregistrés"
        screenshot("preview-flow")
        w._go_capture()
        wait_until(lambda: env.model.mode == CAPTURE_MODE and w.btn_trigger.isVisible())
        preview_reads = VideoCap.registry[0].reads
        assert VideoCap.registry[0].released == 1
        env.gate.clear()
        w._trigger()
        assert not w.btn_trigger.isEnabled() and not w.btn_c_preview.isEnabled()
        env.gate.set()
        wait_until(lambda: env.model.last_photo is not None and w.btn_trigger.isEnabled())
        assert env.model.mode == CAPTURE_MODE
        assert w.btn_open_dir.isEnabled() and w.btn_open_last.isEnabled()
        assert "Photo enregistrée" in w.result_label.text()
        assert VideoCap.registry[0].reads == preview_reads
        screenshot("capture-flow")
        env.fail_capture = OSError("simulated-save-or-camera-failure")
        w._trigger()
        wait_until(lambda: w.btn_trigger.isEnabled() and "Échec" in w.result_label.text())
        assert env.model.mode == CAPTURE_MODE
        assert VideoCap.registry[0].reads == preview_reads
        assert "simulated-save-or-camera-failure" in w.capture_details.text()
    finally:
        env.gate.set()
        assert w.shutdown()
        env.model.mode = CAPTURE_MODE
        w.close()
