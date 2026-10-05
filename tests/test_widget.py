import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from arducam_photo.app.widget import CameraWidget, bgr_to_qimage
from arducam_photo.app.model import SessionModel
from arducam_photo.app.profile import JsonProfileStore, JsonSettingsStore


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
