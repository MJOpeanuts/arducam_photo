"""Standalone application: ``python -m arducam_photo.app``."""

from __future__ import annotations

import sys


def configure_windows_app_id():
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MJOpeanuts.ArducamCapture")


def main(argv=None) -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from .icons import application_icon
    from .model import SessionModel
    from .profile import JsonProfileStore, JsonSettingsStore
    from .window import MainWindow

    configure_windows_app_id()
    app = QApplication(argv if argv is not None else sys.argv)
    app.setWindowIcon(application_icon())
    model = SessionModel(JsonProfileStore(), JsonSettingsStore())
    win = MainWindow(model)
    win.resize(1280, 720)
    win.show()

    def fit_available_screen():
        available = win.screen().availableGeometry()
        frame_width = max(0, win.frameGeometry().width() - win.width())
        frame_height = max(0, win.frameGeometry().height() - win.height())
        win.resize(max(1, min(1280, available.width() - frame_width)),
                   max(1, min(720, available.height() - frame_height)))
        frame = win.frameGeometry()
        win.move(available.x() + (available.width() - frame.width()) // 2,
                 available.y() + (available.height() - frame.height()) // 2)
        if model.settings.start_fullscreen:
            win.toggle_fullscreen()

    QTimer.singleShot(0, fit_available_screen)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
