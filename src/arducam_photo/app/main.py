"""Standalone application: ``python -m arducam_photo.app``."""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMainWindow

    from .model import SessionModel
    from .profile import JsonProfileStore, JsonSettingsStore
    from .widget import CameraWidget

    app = QApplication(argv if argv is not None else sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Arducam Capture")
    model = SessionModel(JsonProfileStore(), JsonSettingsStore())
    widget = CameraWidget(model)
    win.setCentralWidget(widget)
    win.closeEvent = lambda e: (e.ignore() if not widget.can_close() else (widget.shutdown(), e.accept()))
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

    QTimer.singleShot(0, fit_available_screen)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
