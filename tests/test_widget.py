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
from arducam_photo.app.icons import application_icon, bundled_pixmap, lucide_icon
from arducam_photo.app.window import MainWindow


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


@pytest.fixture
def window(qapp, tmp_path):
    model = SessionModel(JsonProfileStore(str(tmp_path / "profiles.json")),
                         JsonSettingsStore(str(tmp_path / "settings.json")))
    model.saved = ShootingProfile(camera_key=model.camera.key, focus_requested=300)
    model.draft = model.saved
    win = MainWindow(model)
    win.resize(1280, 720)
    win.move(40, 50)
    win.show()
    win.activateWindow()
    QTest.qWait(20)
    yield win
    win.leave_fullscreen()
    model.mode = CAPTURE_MODE
    model.controller._state = "idle"
    win.close()
    qapp.processEvents()


@pytest.mark.parametrize("mode", [PREVIEW_MODE, CAPTURE_MODE])
@pytest.mark.parametrize("maximized", [False, True])
@pytest.mark.parametrize("exit_key", [Qt.Key.Key_Escape, Qt.Key.Key_F11])
def test_global_fullscreen_restores_window(window, qapp, mode, maximized, exit_key):
    widget = window.widget
    show_mode(widget, mode, qapp)
    if maximized:
        window.showMaximized()
    qapp.processEvents()
    geometry, state = window.geometry(), window.windowState()
    normal = window.normalGeometry()
    controller = widget.model.controller
    before = (controller.state, widget.model.mode, widget.model.draft, widget.model.saved)
    top_levels = set(qapp.topLevelWidgets())
    focus = widget.focus_spin if mode == PREVIEW_MODE else widget.btn_trigger
    focus.setFocus()
    QTest.keyClick(focus, Qt.Key.Key_F11)
    qapp.processEvents()
    assert window.isFullScreen()
    assert set(qapp.topLevelWidgets()) == top_levels
    assert widget.footer.isVisible()
    assert widget.rect().contains(widget.footer.geometry())
    assert all(b.text() == "Quitter le plein écran" for b in widget._fullscreen_controls)
    assert (widget.btn_save if mode == PREVIEW_MODE else widget.btn_trigger).isVisible()
    QTest.keyClick(focus, exit_key)
    qapp.processEvents()
    assert not window.isFullScreen()
    assert window.windowState() == state
    assert window.geometry() == geometry
    if maximized:
        window.showNormal()
        qapp.processEvents()
        assert window.geometry() == normal
    else:
        assert window.normalGeometry() == normal
    assert (controller.state, widget.model.mode, widget.model.draft, widget.model.saved) == before
    assert all(b.text() == "Plein écran" for b in widget._fullscreen_controls)


def test_diagnostic_fullscreen_and_persistent_preference(window, qapp):
    widget = window.widget
    show_mode(widget, PREVIEW_MODE, qapp)
    widget.preview_diagnostic_toggle.setChecked(True)
    widget._fullscreen_controls[0].click()
    assert window.isFullScreen()
    widget._startup_controls[0].setChecked(True)
    assert all(c.isChecked() for c in widget._startup_controls)
    settings = JsonSettingsStore(widget.model.settings_store.path)
    assert settings.load().start_fullscreen
    fresh = SessionModel(widget.model.profiles, settings)
    try:
        assert fresh.settings.start_fullscreen
        assert fresh.settings.camera_index == widget.model.settings.camera_index
        assert fresh.settings.photo_dir == widget.model.settings.photo_dir
    finally:
        fresh.shutdown()
    widget._startup_controls[1].setChecked(False)
    assert not settings.load().start_fullscreen


def test_quit_uses_existing_guards(window, qapp, monkeypatch):
    widget = window.widget
    show_mode(widget, PREVIEW_MODE, qapp)
    widget.model.draft = dataclasses.replace(widget.model.saved, focus_requested=301)
    monkeypatch.setattr(widget, "_ask_unsaved", lambda: "stay")
    quit_button = next(b for b in widget.preview_diagnostic.findChildren(QPushButton)
                       if b.text() == "Quitter l’application")
    quit_button.click()
    assert window.isVisible()
    monkeypatch.setattr(widget, "_info", lambda text: None)
    widget.model.controller._state = "capturing"
    quit_button.click()
    assert window.isVisible()
    widget.model.controller._state = "preview"
    monkeypatch.setattr(widget, "_ask_unsaved", lambda: "save")
    shutdown = widget.shutdown
    monkeypatch.setattr(widget, "shutdown", lambda: False)
    quit_button.click()
    assert window.isVisible()
    assert not widget.model.has_unsaved()
    monkeypatch.setattr(widget, "shutdown", shutdown)
    quit_button.click()
    assert not window.isVisible()


def test_original_app_icon_footer_and_clean_header(widget, qapp, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    icon = application_icon()
    assert not icon.isNull()
    original = bundled_pixmap("icons/nuts-app.png").toImage()
    assert original.pixelColor(0, 0).alpha() == 0
    logo = bundled_pixmap("powered by_white.png")
    for mode in (PREVIEW_MODE, CAPTURE_MODE):
        show_mode(widget, mode, qapp)
        assert widget.footer.source.toImage() == logo.toImage()
        displayed = widget.footer.pixmap()
        assert abs(displayed.width() / displayed.height() - logo.width() / logo.height()) < .1
        assert widget.footer.isVisible()
        assert widget.rect().contains(widget.footer.geometry())
        assert not (widget.preview_diagnostic if mode == PREVIEW_MODE else widget.capture_diagnostic).isVisible()
        labels = [label.text() for label in widget.findChildren(QLabel) if label.isVisible()]
        assert "Preview / Réglages" not in labels and "Profil enregistré" not in labels
        assert not widget.profile_label.isVisible()
        assert not hasattr(widget, "fullscreen_video")


@pytest.mark.parametrize("missing", ["icons/nuts-app.svg", "icons/nuts-app.png",
                                   "icons/nuts-app.ico", "powered by_white.png"])
def test_missing_original_resource_is_named(qapp, monkeypatch, tmp_path, missing):
    import arducam_photo.app.icons as icons

    root = resources.files("arducam_photo.resources")
    for name in ("icons/nuts-app.svg", "icons/nuts-app.png", "icons/nuts-app.ico",
                 "powered by_white.png"):
        if name != missing:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(root.joinpath(name).read_bytes())
    monkeypatch.setattr(icons.resources, "files", lambda package: tmp_path)
    with pytest.raises(FileNotFoundError, match=missing):
        if missing.startswith("icons/"):
            application_icon()
        else:
            bundled_pixmap(missing)


def test_startup_default_and_fullscreen_after_relaunch(qapp, monkeypatch, tmp_path):
    from PySide6 import QtWidgets
    from arducam_photo.app import main

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(QtWidgets, "QApplication", lambda argv: qapp)
    expected_fullscreen = False

    def run():
        qapp.processEvents()
        win = next(w for w in qapp.topLevelWidgets()
                   if isinstance(w, MainWindow) and w.isVisible())
        assert win.isFullScreen() == expected_fullscreen
        assert not qapp.windowIcon().isNull() and not win.windowIcon().isNull()
        assert win.widget.model.controller.state == "idle"
        win.widget._startup_controls[0].setChecked(True)
        win.close()
        return 0

    monkeypatch.setattr(qapp, "exec", run)
    assert main.main([]) == 0
    expected_fullscreen = True
    assert main.main([]) == 0


def test_windows_app_id_is_stable(monkeypatch):
    import ctypes
    from arducam_photo.app import main

    calls = []
    monkeypatch.setattr(main.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(shell32=SimpleNamespace(
        SetCurrentProcessExplicitAppUserModelID=calls.append)), raising=False)
    main.configure_windows_app_id()
    assert calls == ["MJOpeanuts.ArducamCapture"]


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
    widget.ensurePolished()
    qapp.processEvents()
    show_mode(widget, mode, qapp)
    # Allow small differences in window sizing across platforms (±2 pixels tolerance)
    actual_width, actual_height = widget.width(), widget.height()
    expected_width, expected_height = window_size
    assert abs(actual_width - expected_width) <= 2, f"Width mismatch: expected {expected_width}, got {actual_width}"
    assert abs(actual_height - expected_height) <= 2, f"Height mismatch: expected {expected_height}, got {actual_height}"
    buttons = ((widget.btn_save, widget.btn_p_capture) if mode == PREVIEW_MODE else
               (widget.btn_trigger, widget.btn_open_dir, widget.btn_open_last, widget.btn_c_preview))
    assert widget.rect().contains(widget.footer.geometry())
    footer_top = widget.footer.geometry().top()
    for button in buttons:
        position = button.mapTo(widget, button.rect().topLeft())
        assert position.x() >= 0 and position.y() >= 0
        assert position.x() + button.width() <= widget.width()
        assert position.y() + button.height() <= widget.height()
        assert position.y() + button.height() < footer_top
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
    window = MainWindow(env.model)
    w = window.widget
    window.resize(1280, 720)
    window.show()

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
        camera = VideoCap.registry[0]
        window.toggle_fullscreen()
        qapp.processEvents()
        window.leave_fullscreen()
        qapp.processEvents()
        assert VideoCap.registry == [camera]
        assert camera.released == 0
        screenshot("preview-flow")
        w._go_capture()
        wait_until(lambda: env.model.mode == CAPTURE_MODE and w.btn_trigger.isVisible())
        preview_reads = VideoCap.registry[0].reads
        assert VideoCap.registry[0].released == 1
        window.toggle_fullscreen()
        qapp.processEvents()
        assert env.model.mode == CAPTURE_MODE
        window.leave_fullscreen()
        qapp.processEvents()
        assert VideoCap.registry == [camera]
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
        window.close()
