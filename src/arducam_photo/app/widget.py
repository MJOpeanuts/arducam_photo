"""Reusable PySide6 widget composing the model/controller. All camera, ISP, encoding and file
work runs on the controller's worker thread; this module only displays and forwards user input."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from ..config import COLOR_720P, NATIVE_108MP
from .controller import ModeError
from .model import (CAPTURE_MODE, DISCARD, PATH_LABELS, PREVIEW_MODE, SAVE, START, STAY,
                    NeedsDecision, ProfileMissing, SessionModel)
from .paths import open_in_system
from .profile import FOCUS_MAX, FOCUS_MIN


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """BGR uint8 HxWx3 -> QImage that owns its pixels (deep copy), so neither the numpy
    array nor the intermediate RGB buffer needs to outlive this call."""
    import cv2

    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class CameraWidget(QWidget):
    _event = Signal(str, object)  # worker thread -> UI thread (queued)

    def __init__(self, model: SessionModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model = model
        model._ui = lambda event, **kw: self._event.emit(event, kw)
        self._event.connect(self._on_event)
        self._build()
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(500)
        self._fps_timer.timeout.connect(self._update_fps)
        self._show_page(START)

    # ---- layout ----
    def _build(self) -> None:
        self.stack = QStackedWidget()
        # start page
        s = QWidget(); sl = QVBoxLayout(s)
        self.path_combo = QComboBox()
        for k in (NATIVE_108MP, COLOR_720P):
            self.path_combo.addItem(PATH_LABELS[k], k)
        self.path_combo.setCurrentIndex(self.path_combo.findData(self.model.path))
        self.path_combo.currentIndexChanged.connect(self._path_changed)
        self.cam_combo = QComboBox(); self._fill_cameras([self.model.camera.index])
        self.cam_combo.currentIndexChanged.connect(self._camera_changed)
        detect = QPushButton("Détecter les caméras (ouvre brièvement chaque index)")
        detect.clicked.connect(self._detect)
        self.usb_note = QLabel("Le parcours choisi n'est pas une mesure de la vitesse USB : "
                               "108 MP exige USB 3, 720p fonctionne en USB 2 ou 3 si le mode existe.")
        self.usb_note.setWordWrap(True)
        self.cam_note = QLabel("Identification par index OpenCV : fragile (peut changer entre deux essais).")
        self.cam_note.setWordWrap(True)
        self.ccm_label = QLabel(); self.ccm_label.setWordWrap(True)
        ccm_btn = QPushButton("Avancé : utiliser un autre arducam_108mp.json…"); ccm_btn.clicked.connect(self._pick_ccm)
        self.folder_label = QLabel(); self.folder_label.setWordWrap(True)
        folder_btn = QPushButton("Dossier des photos…"); folder_btn.clicked.connect(self._pick_folder)
        self.btn_to_preview = QPushButton("Preview / Réglages")
        self.btn_to_preview.clicked.connect(self._go_preview)
        self.btn_to_capture = QPushButton("Capture")
        self.btn_to_capture.clicked.connect(self._go_capture)
        for w in (QLabel("Parcours :"), self.path_combo, QLabel("Caméra :"), self.cam_combo, detect,
                  self.cam_note, self.usb_note, self.ccm_label, ccm_btn, self.folder_label, folder_btn):
            sl.addWidget(w)
        row = QHBoxLayout(); row.addWidget(self.btn_to_preview); row.addWidget(self.btn_to_capture)
        sl.addLayout(row); sl.addStretch()
        # preview page
        p = QWidget(); pl = QVBoxLayout(p)
        self.video = QLabel("Aucun flux"); self.video.setMinimumSize(640, 360)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps_label = QLabel("FPS acquis : – | affichés : –")
        self.focus_slider = QSlider(Qt.Orientation.Horizontal); self.focus_slider.setRange(FOCUS_MIN, FOCUS_MAX)
        self.focus_spin = QSpinBox(); self.focus_spin.setRange(FOCUS_MIN, FOCUS_MAX)
        self.focus_spin.setSpecialValueText("non défini"); self.focus_spin.setRange(FOCUS_MIN - 1, FOCUS_MAX)
        self.focus_spin.setValue(FOCUS_MIN - 1)
        self.focus_slider.valueChanged.connect(lambda v: self._focus_input(v, from_slider=True))
        self.focus_spin.valueChanged.connect(lambda v: self._focus_input(v, from_slider=False))
        self.focus_state = QLabel("Focus manuel : consigne – | valeur relue – (non vérifiable physiquement). "
                                  "Pas d'autofocus.")
        self.focus_state.setWordWrap(True)
        self.unsaved = QLabel(""); self.preview_msg = QLabel("")
        self.btn_save = QPushButton("Enregistrer les réglages"); self.btn_save.clicked.connect(self._save)
        self.btn_p_capture = QPushButton("Aller en Capture"); self.btn_p_capture.clicked.connect(self._go_capture)
        for w in (self.video, self.fps_label, self.focus_slider, self.focus_spin, self.focus_state,
                  self.unsaved, self.preview_msg, self.btn_save, self.btn_p_capture):
            pl.addWidget(w)
        # capture page
        c = QWidget(); cl = QVBoxLayout(c)
        self.profile_label = QLabel(); self.profile_label.setWordWrap(True)
        self.ccm_label2 = QLabel(); self.ccm_label2.setWordWrap(True)
        self.btn_trigger = QPushButton("DÉCLENCHER"); self.btn_trigger.setMinimumHeight(90)
        self.btn_trigger.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.btn_trigger.clicked.connect(self._trigger)
        self.step_label = QLabel(""); self.result_label = QLabel(""); self.result_label.setWordWrap(True)
        self.photo = QLabel("Aucune photo dans cette session"); self.photo.setMinimumSize(480, 320)
        self.photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_open_dir = QPushButton("Ouvrir le dossier"); self.btn_open_dir.clicked.connect(self._open_dir)
        self.btn_open_last = QPushButton("Ouvrir la dernière photo"); self.btn_open_last.clicked.connect(self._open_last)
        self.btn_c_preview = QPushButton("Preview / Réglages"); self.btn_c_preview.clicked.connect(self._go_preview)
        for w in (self.profile_label, self.ccm_label2, self.btn_trigger, self.step_label, self.result_label,
                  self.photo, self.btn_open_dir, self.btn_open_last, self.btn_c_preview):
            cl.addWidget(w)
        for page in (s, p, c):
            self.stack.addWidget(page)
        lay = QVBoxLayout(self); lay.addWidget(self.stack)

    PAGES = {START: 0, PREVIEW_MODE: 1, CAPTURE_MODE: 2}

    def _show_page(self, mode: str) -> None:
        self.stack.setCurrentIndex(self.PAGES[mode])
        self.ccm_label.setText(self.model.ccm_status())
        self.ccm_label2.setText(self.model.ccm_status())
        self.folder_label.setText(f"Dossier des photos : {self.model.photo_dir()}")
        if mode == CAPTURE_MODE:
            prof = self.model.saved
            if prof is not None and prof.is_valid:
                frag = " (identification par index : fragile)" if prof.camera_fragile else ""
                self.profile_label.setText(
                    f"Profil enregistré : {PATH_LABELS[prof.path]}, caméra {prof.camera_key}{frag}, "
                    f"focus demandé {prof.focus_requested}, relu {prof.focus_readback}")
            else:
                self.profile_label.setText("Aucun profil valide : passez par Preview / Réglages.")
            self.btn_trigger.setEnabled(bool(prof and prof.is_valid))
            self._fps_timer.stop()
        elif mode == PREVIEW_MODE:
            self._fps_timer.start()
            self._refresh_preview_state()
        else:
            self._fps_timer.stop()

    def _fill_cameras(self, indices) -> None:
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        for i in indices:
            self.cam_combo.addItem(f"Caméra index {i}", i)
        k = self.cam_combo.findData(self.model.camera.index)
        self.cam_combo.setCurrentIndex(max(k, 0))
        self.cam_combo.blockSignals(False)

    # ---- user actions ----
    def _guard(self, fn, *a):
        try:
            return fn(*a)
        except (ModeError, ProfileMissing) as e:
            self._info(str(e))
        except Exception as e:  # recoverable error shown, never crashes the UI
            QMessageBox.warning(self, "Erreur", str(e))

    def _info(self, text: str) -> None:
        QMessageBox.information(self, "Information", text)

    def _path_changed(self, _):
        self._guard(self.model.select_path, self.path_combo.currentData())
        self.ccm_label.setText(self.model.ccm_status())

    def _camera_changed(self, _):
        if self.cam_combo.currentData() is not None:
            self._guard(self.model.select_camera, self.cam_combo.currentData())

    def _detect(self):
        self._guard(self.model.controller.probe)

    def _pick_ccm(self):
        f, _ = QFileDialog.getOpenFileName(self, "arducam_108mp.json", "", "JSON (*.json)")
        if f:
            self._guard(self.model.set_ccm_path, f)

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier des photos", self.model.photo_dir())
        if d:
            self._guard(self.model.set_photo_dir, d)
            self.folder_label.setText(f"Dossier des photos : {self.model.photo_dir()}")

    def _go_preview(self):
        if self.model.mode == PREVIEW_MODE:
            return
        self._guard(self.model.enter_preview)

    def _go_capture(self):
        decision = None
        while True:
            try:
                self.model.request_capture_mode(decision)
                return
            except NeedsDecision:
                decision = self._ask_unsaved()
            except (ModeError, ProfileMissing) as e:
                self._info(str(e)); return
            except Exception as e:
                QMessageBox.warning(self, "Erreur", str(e)); return

    def _ask_unsaved(self) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("Modifications non enregistrées")
        box.setText("Des réglages ne sont pas enregistrés.")
        b_save = box.addButton("Enregistrer et continuer", QMessageBox.ButtonRole.AcceptRole)
        b_drop = box.addButton("Abandonner et continuer", QMessageBox.ButtonRole.DestructiveRole)
        b_stay = box.addButton("Rester en Preview", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(b_stay)
        box.exec()
        clicked = box.clickedButton()
        return SAVE if clicked is b_save else DISCARD if clicked is b_drop else STAY

    def _focus_input(self, value: int, from_slider: bool) -> None:
        if value < FOCUS_MIN or self.model.mode != PREVIEW_MODE:
            return
        other = self.focus_spin if from_slider else self.focus_slider
        other.blockSignals(True); other.setValue(value); other.blockSignals(False)
        self._guard(self.model.set_focus, value)
        self._refresh_preview_state()

    def _save(self):
        if self._guard(self.model.save_profile) is not None:
            self.preview_msg.setText("Réglages enregistrés.")
        self._refresh_preview_state()

    def _trigger(self):
        if self._guard(self.model.trigger) is None and self.model.controller.state == "capturing":
            self._set_capture_busy(True)
            self.result_label.setText("")

    def _set_capture_busy(self, busy: bool) -> None:
        for b in (self.btn_trigger, self.btn_c_preview):
            b.setEnabled(not busy)
        if not busy:
            prof = self.model.saved
            self.btn_trigger.setEnabled(bool(prof and prof.is_valid))

    def _open_dir(self):
        d = self.model.photo_dir()
        os.makedirs(d, exist_ok=True)
        self._guard(open_in_system, d)

    def _open_last(self):
        if self.model.last_photo and os.path.exists(self.model.last_photo):
            self._guard(open_in_system, self.model.last_photo)
        else:
            self._info("Aucune photo prise pendant cette session.")

    # ---- events (UI thread) ----
    def _on_event(self, event: str, kw: dict) -> None:
        if event == "frame":
            f = self.model.controller.take_frame()
            if f is not None and self.model.mode == PREVIEW_MODE:
                img = bgr_to_qimage(f)
                self.video.setPixmap(QPixmap.fromImage(img).scaled(
                    self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
        elif event == "video_cleared":
            self.video.clear(); self.video.setText("Aucun flux")
        elif event == "mode_changed":
            self._show_page(kw["mode"])
        elif event == "preview_failed":
            QMessageBox.warning(self, "Preview", str(kw["error"]))
        elif event in ("focus_applied", "focus_failed"):
            self._refresh_preview_state(kw)
        elif event == "cameras":
            self._fill_cameras(kw["indices"] or [self.model.camera.index])
        elif event == "ccm_checked":
            if not kw["ok"]:
                QMessageBox.warning(self, "Correction couleur", f"Fichier invalide : {kw['error']}")
            self.ccm_label.setText(self.model.ccm_status() + ("" if kw["ok"] else " — INVALIDE"))
        elif event == "capture_step":
            self.step_label.setText(f"Étape : {kw['step']}")
        elif event == "capture_done":
            w, h = kw["size"]
            info = kw["info"]
            self.photo.setPixmap(QPixmap.fromImage(bgr_to_qimage(kw["thumbnail"])).scaled(
                self.photo.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.result_label.setText(
                f"Enregistré : {kw['path']} ({w}×{h}), correction couleur appliquée : {info.ccm_applied}, "
                f"durée du moteur : {info.duration_s:.1f} s")
        elif event == "capture_failed":
            self.result_label.setText(f"Échec de la capture : {kw['error']}. Vous restez en Capture.")
        elif event == "capture_idle":
            self.step_label.setText("")
            self._set_capture_busy(False)

    def _refresh_preview_state(self, kw: Optional[dict] = None) -> None:
        d = self.model.draft
        if d is None:
            return
        rb = "–" if d.focus_readback is None else f"{d.focus_readback:g}"
        req = "–" if d.focus_requested is None else str(d.focus_requested)
        self.focus_state.setText(f"Focus manuel : consigne {req} | valeur relue {rb} "
                                 "(le relu n'est pas une preuve d'effet physique). Pas d'autofocus.")
        self.unsaved.setText("● Modifications non enregistrées" if self.model.has_unsaved() else "Réglages enregistrés")
        if kw and "error" in kw:
            self.preview_msg.setText(f"Erreur focus : {kw['error']}")

    def _update_fps(self) -> None:
        a, d = self.model.controller.fps()
        self.fps_label.setText(f"FPS acquis : {a:.1f} | affichés : {d:.1f}")

    # ---- closing ----
    def can_close(self) -> bool:
        """False if the window must stay open (capture running, or user stays in Preview)."""
        if self.model.controller.state == "capturing":
            self._info("Une capture est en cours : attendez la fin avant de fermer.")
            return False
        if self.model.has_unsaved():
            d = self._ask_unsaved()
            if d == STAY:
                return False
            if d == SAVE and self._guard(self.model.save_profile) is None:
                return False
        return True

    def shutdown(self) -> bool:
        self._fps_timer.stop()
        return self.model.shutdown(timeout=10)

    def closeEvent(self, e):
        if not self.can_close():
            e.ignore(); return
        if not self.shutdown():
            self._info("La caméra n'a pas répondu dans les 10 s ; elle peut rester occupée jusqu'à la fin de l'appel natif.")
        e.accept()
