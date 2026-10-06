"""Main-window display state and guarded application exit."""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QMainWindow

from .icons import application_icon
from .widget import CameraWidget


class MainWindow(QMainWindow):
    fullscreen_changed = Signal(bool)

    def __init__(self, model):
        super().__init__()
        self.setWindowTitle("Arducam Capture")
        self.setWindowIcon(application_icon())
        self._window_geometry = None
        self._window_state = Qt.WindowState.WindowNoState
        self._closing = False
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setInterval(1000)
        self._shutdown_timer.timeout.connect(self._poll_shutdown)
        self.widget = CameraWidget(model)
        self.setCentralWidget(self.widget)
        self.widget.fullscreen_requested.connect(self.toggle_fullscreen)
        self.widget.quit_requested.connect(self.close)
        self.fullscreen_changed.connect(self.widget.set_fullscreen_state)
        self._f11 = QShortcut(QKeySequence("F11"), self)
        self._f11.activated.connect(self.toggle_fullscreen)
        self._escape = QShortcut(QKeySequence("Escape"), self)
        self._escape.activated.connect(self.leave_fullscreen)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.leave_fullscreen()
            return
        self._window_geometry = self.normalGeometry()
        self._window_state = self.windowState()
        self.showFullScreen()
        self.fullscreen_changed.emit(True)

    def leave_fullscreen(self):
        if not self.isFullScreen():
            return
        self.showNormal()
        if self._window_geometry is not None:
            self.setGeometry(self._window_geometry)
        self.setWindowState(self._window_state)
        self.fullscreen_changed.emit(False)

    def closeEvent(self, event):
        if self._closing:
            if self._shutdown_timer.isActive():
                event.ignore()
            else:
                event.accept()
            return
        if not self.widget.can_close():
            event.ignore()
            return
        if not self.widget.shutdown():
            self._closing = True
            self.widget.setEnabled(False)
            self.statusBar().showMessage(
                "La caméra n'a pas répondu dans les 10 s. Fermeture en cours : "
                "l’application quittera automatiquement après sa libération.")
            self._shutdown_timer.start()
            event.ignore()
            return
        event.accept()

    def _poll_shutdown(self):
        if self.widget.model.shutdown(timeout=0):
            self._shutdown_timer.stop()
            self.close()
