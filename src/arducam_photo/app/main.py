"""Standalone application: ``python -m arducam_photo.app``."""

from __future__ import annotations

import sys


def main(argv=None) -> int:
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
    win.resize(900, 800)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
