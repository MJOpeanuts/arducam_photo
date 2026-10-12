"""Reusable PySide6 widget composing the model/controller. All camera, ISP, encoding and file
work runs on the controller's worker thread; this module only displays and forwards user input."""

from __future__ import annotations

import os
import math
from datetime import datetime
from typing import Optional

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
                               QStackedWidget, QTabBar, QToolButton, QVBoxLayout, QWidget)

from ..config import PATHS, MODES
from .controller import ModeError
from .icons import bundled_pixmap, lucide_icon
from .model import (CAPTURE_MODE, PROCESSING_MODE, DISCARD, PATH_LABELS, PREVIEW_MODE, SAVE, START, STAY,
                    NeedsDecision, ProfileMissing, ProposalPending, SessionModel)
from .paths import open_in_system
from .profile import FOCUS_MAX, FOCUS_MIN
from .processing_widget import ProcessingWidget


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """BGR uint8 HxWx3 -> QImage that owns its pixels (deep copy), so neither the numpy
    array nor the intermediate RGB buffer needs to outlive this call."""
    import cv2

    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB if frame.ndim == 2 else cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class ImageView(QLabel):
    """Keep the source image independent of the layout's available size."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.source = QPixmap()
        self.setObjectName("image")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(160, 90)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def setPixmap(self, pixmap: QPixmap) -> None:
        self.source = pixmap
        self._fit()

    def clear(self) -> None:
        self.source = QPixmap()
        super().clear()

    def _fit(self) -> None:
        if not self.source.isNull():
            super().setPixmap(self.source.scaled(
                self.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()


class CameraWidget(QWidget):
    _event = Signal(str, object)  # worker thread -> UI thread (queued)
    fullscreen_requested = Signal()
    quit_requested = Signal()

    def __init__(self, model: SessionModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model = model
        self._proposal_restoring = False
        model._ui = lambda event, **kw: self._event.emit(event, kw)
        self._event.connect(self._on_event)
        self._build()
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(500)
        self._fps_timer.timeout.connect(self._update_fps)
        self._show_page(START)
        self._refresh_banks()

    # ---- layout ----
    def _build(self) -> None:
        self._technical = {"FPS acquis / affichés": "– / –"}
        self._fullscreen_controls = []
        self._startup_controls = []
        self.setStyleSheet("""
            QWidget { background: #1B1D1F; color: #F4F3F1; font-family: "Segoe UI";
                      font-size: 13px; }
            QLabel { background: transparent; }
            QLabel#title { font-size: 23px; font-weight: 600; }
            QLabel#muted { color: #A4A7A9; }
            QLabel#image { background: #111314; border: 1px solid #343739; border-radius: 8px; }
            QPushButton, QToolButton, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #26292B; border: 1px solid #434648; border-radius: 6px;
                padding: 7px 12px; min-height: 20px; }
            QPushButton:hover, QToolButton:hover { background: #343739; }
            QPushButton:focus, QToolButton:focus, QComboBox:focus, QSpinBox:focus {
                border: 1px solid #FF963F; }
            QPushButton:disabled, QToolButton:disabled { color: #727679; background: #202224;
                border-color: #343739; }
            QPushButton#mode { background: #D9DDE0; color: #202326; border-color: #D9DDE0; }
            QPushButton#mode:hover { background: #ECEFF1; border-color: #ECEFF1; }
            QPushButton#mode:focus { border-color: #FF963F; }
            QPushButton#photo { background: #FF963F; color: #21180E;
                border-color: #FF963F; font-weight: 700; font-size: 16px; }
            QPushButton#photo:hover { background: #FFA65E; border-color: #FFA65E; }
            QPushButton#photo:focus { border-color: #F4F3F1; }
            QPushButton#photo:disabled, QPushButton#mode:disabled {
                background: #343739; color: #727679; border-color: #434648; }
            QSlider::groove:horizontal { height: 4px; background: #343739; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #FF963F; border-radius: 2px; }
            QSlider::handle:horizontal { background: #F4F3F1; width: 14px;
                margin: -5px 0; border-radius: 7px; }
            QSlider::handle:horizontal:hover { background: #FFA65E; }
            QSlider::handle:horizontal:focus { background: #FF963F; }
            QSlider::handle:horizontal:disabled { background: #727679; }
            QScrollArea { border: 1px solid #343739; border-radius: 6px; }
            QScrollArea#startSettings { border: none; }
            QToolButton#diagnostic { color: #A4A7A9; background: transparent; border: none;
                text-align: left; padding-left: 0; }
            QToolButton#diagnostic:focus { border: 1px solid #FF963F; }
        """)
        self.stack = QStackedWidget()
        # start page
        s = QWidget(); sl = QVBoxLayout(s)
        self.path_combo = QComboBox()
        for k in PATHS:
            self.path_combo.addItem(PATH_LABELS[k], k)
            self.path_combo.setItemData(self.path_combo.count() - 1, MODES[k].description, Qt.ItemDataRole.ToolTipRole)
        self.path_combo.setCurrentIndex(self.path_combo.findData(self.model.path))
        self.path_combo.currentIndexChanged.connect(self._path_changed)
        self.cam_combo = QComboBox(); self._fill_cameras([self.model.camera.index])
        self.cam_combo.currentIndexChanged.connect(self._camera_changed)
        detect = QPushButton("Détecter les caméras (ouvre brièvement chaque index)")
        self.detect_button = detect
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
        self.start_scroll = QScrollArea()
        self.start_scroll.setObjectName("startSettings")
        self.start_scroll.setAccessibleName("Configuration de la caméra")
        self.start_scroll.setWidgetResizable(True)
        self.start_scroll.setWidget(s)
        # preview page
        p = QWidget(); pl = QVBoxLayout(p)
        self.btn_p_capture = self._mode_button("Capture", self._go_capture)
        self._header(pl)
        self.video = ImageView("Aucun flux")
        self.video.setAccessibleName("Aperçu de la caméra")
        pl.addWidget(self.video, 1)
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
        self.unsaved = QLabel(""); self.unsaved.setObjectName("muted")
        self.preview_msg = QLabel(""); self.preview_msg.setWordWrap(True); self.preview_msg.hide()
        self.btn_save = QPushButton("Enregistrer"); self.btn_save.clicked.connect(self._save)
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(1, 120)
        self.fps_spin.setToolTip("Cadence photo demandée, sans garantie du pilote")
        self.fps_spin.valueChanged.connect(lambda v: self._guard(self.model.set_fps, v))
        self.focus_slider.setAccessibleName("Focus manuel")
        self.focus_spin.setAccessibleName("Valeur du focus manuel")
        self.focus_slider.setToolTip("Régler la consigne du focus manuel")
        self.focus_spin.setToolTip(f"Focus manuel · {FOCUS_MIN} à {FOCUS_MAX}")
        self.btn_save.setToolTip("Enregistrer le profil et le focus manuel")
        focus_row = QHBoxLayout()
        focus_row.addWidget(QLabel("Focus manuel"))
        focus_row.addWidget(self.focus_slider, 1); focus_row.addWidget(self.focus_spin)
        focus_row.addWidget(self.btn_save); focus_row.addWidget(self.unsaved)
        focus_row.addWidget(self.btn_p_capture)
        pl.addLayout(focus_row); pl.addWidget(self.preview_msg)
        bank_row = QHBoxLayout()
        self.bank_combo = QComboBox()
        self.bank_combo.setAccessibleName("Bank cible de chargement ou sauvegarde")
        self.btn_load_bank = QPushButton("Charger la bank")
        self.btn_load_bank.clicked.connect(self._load_bank)
        self.btn_create_bank = QPushButton("Créer une bank…")
        self.btn_create_bank.clicked.connect(self._create_bank)
        for control in (QLabel("Bank cible"), self.bank_combo, self.btn_load_bank, self.btn_create_bank):
            bank_row.addWidget(control)
            control.setVisible(self.model.banks is not None)
        pl.addLayout(bank_row)
        adjustment_row = QHBoxLayout()
        self.btn_probe_controls = QPushButton("Vérifier commandes")
        self.btn_probe_controls.clicked.connect(self._probe_controls)
        self.btn_adjust = QPushButton("Auto-ajustement…")
        self.btn_adjust.clicked.connect(self._auto_adjust)
        self.btn_cancel_adjustment = QPushButton("Annuler l’ajustement")
        self.btn_cancel_adjustment.clicked.connect(lambda: self._guard(self.model.controller.cancel_adjustment))
        self.btn_accept_proposal = QPushButton("Valider la proposition")
        self.btn_accept_proposal.clicked.connect(self._accept_proposal)
        self.btn_reject_proposal = QPushButton("Rejeter la proposition")
        self.btn_reject_proposal.clicked.connect(self._reject_proposal)
        for button in (self.btn_probe_controls, self.btn_adjust, self.btn_cancel_adjustment,
                       self.btn_accept_proposal, self.btn_reject_proposal):
            adjustment_row.addWidget(button)
        pl.addLayout(adjustment_row)
        self.adjustment_summary = QLabel()
        self.adjustment_summary.setWordWrap(True)
        self.adjustment_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.adjustment_summary.hide()
        pl.addWidget(self.adjustment_summary)
        self.camera_settings_toggle = QToolButton()
        self.camera_settings_toggle.setText("Autres réglages caméra")
        self.camera_settings_toggle.setCheckable(True)
        self.camera_settings_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.camera_settings_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.camera_settings_panel = QWidget()
        settings_form = QFormLayout(self.camera_settings_panel)
        self.camera_settings_inputs = {}
        self.camera_mode_inputs = {}
        for name, label in (("auto_exposure_mode", "Mode d’exposition"),
                            ("auto_wb_mode", "Mode balance des blancs")):
            selector = QComboBox()
            selector.addItem("Non déterminé", None)
            selector.addItem("Manuel", "manual")
            selector.addItem("Automatique", "automatic")
            selector.currentIndexChanged.connect(
                lambda _, n=name, w=selector: self._camera_mode_input(n, w)
            )
            self.camera_mode_inputs[name] = selector
            settings_form.addRow(label, selector)
        setting_labels = {
            "auto_exposure": "Mode auto exposition (valeur backend)",
            "exposure": "Exposition (valeur backend)",
            "gain": "Gain (valeur backend)",
            "auto_wb": "Balance des blancs auto (valeur backend)",
            "wb_temperature": "Température balance des blancs (valeur backend)",
            "brightness": "Luminosité (valeur backend)",
            "contrast": "Contraste (valeur backend)",
            "saturation": "Saturation (valeur backend)",
        }
        for name, label in setting_labels.items():
            editor = QLineEdit()
            editor.setPlaceholderText("inchangé")
            editor.setAccessibleName(label)
            editor.editingFinished.connect(lambda n=name, w=editor: self._camera_control_input(n, w))
            self.camera_settings_inputs[name] = editor
            settings_form.addRow(label, editor)
        self.camera_settings_note = QLabel(
            "Valeurs transmises telles quelles au backend OpenCV; unités, plages et modes de bascule "
            "dépendent du pilote. Une valeur vide signifie « ne pas modifier »."
        )
        self.camera_settings_note.setWordWrap(True)
        settings_form.addRow(self.camera_settings_note)
        self.camera_settings_panel.hide()
        self.camera_settings_toggle.toggled.connect(self._toggle_camera_settings)
        pl.addWidget(self.camera_settings_toggle)
        pl.addWidget(self.camera_settings_panel)
        self.preview_details = QLabel(); self.preview_details.setWordWrap(True)
        self._diagnostic(pl, "preview", (self.fps_label, self.focus_state, self.preview_details))
        fps_row = QHBoxLayout(); fps_row.addWidget(QLabel("Cadence photo demandée")); fps_row.addWidget(self.fps_spin)
        fps_row.addStretch()
        self.preview_diagnostic.widget().layout().addLayout(fps_row)
        # capture page
        c = QWidget(); cl = QVBoxLayout(c)
        self.btn_c_preview = self._mode_button("Preview", self._go_preview)
        self._header(cl)
        self.capture_mode_combo = QComboBox()
        self.capture_mode_combo.setAccessibleName("Mode photo")
        for path in PATHS:
            self.capture_mode_combo.addItem(PATH_LABELS[path], path)
            self.capture_mode_combo.setItemData(self.capture_mode_combo.count() - 1, MODES[path].description,
                                               Qt.ItemDataRole.ToolTipRole)
        self.capture_mode_combo.setCurrentIndex(self.capture_mode_combo.findData(self.model.path))
        self.capture_mode_combo.currentIndexChanged.connect(self._capture_path_changed)
        mode_row = QHBoxLayout(); mode_row.addWidget(QLabel("Mode photo")); mode_row.addWidget(self.capture_mode_combo)
        mode_row.addStretch(); cl.addLayout(mode_row)
        self.profile_label = QLabel(); self.profile_label.setWordWrap(True)
        self.profile_label.setObjectName("muted")
        self.ccm_label2 = QLabel(); self.ccm_label2.setWordWrap(True)
        self.btn_trigger = QPushButton("PHOTO"); self.btn_trigger.setObjectName("photo")
        self.btn_trigger.setIcon(lucide_icon("squirrel", "#21180E", 24))
        self.btn_trigger.setMinimumSize(160, 48)
        self.btn_trigger.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.btn_trigger.setAccessibleName("Prendre une photo")
        self.btn_trigger.setToolTip("Capturer avec le profil enregistré")
        self.btn_trigger.clicked.connect(self._trigger)
        self.step_label = QLabel(""); self.result_label = QLabel(""); self.result_label.setWordWrap(True)
        self.step_label.setWordWrap(True)
        cl.addWidget(QLabel("Dernière prise"))
        self.photo = ImageView("Aucune photo dans cette session")
        self.photo.setAccessibleName("Dernière photo capturée")
        cl.addWidget(self.photo, 1)
        self.btn_open_dir = QPushButton("Ouvrir le dossier"); self.btn_open_dir.clicked.connect(self._open_dir)
        self.btn_open_last = QPushButton("Ouvrir la dernière photo"); self.btn_open_last.clicked.connect(self._open_last)
        self.btn_process_last = QPushButton("Traiter la dernière acquisition")
        self.btn_process_last.clicked.connect(self._process_last)
        mode_row.addWidget(self.btn_process_last)
        for button in (self.btn_open_dir, self.btn_open_last):
            button.setFixedHeight(38)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.setToolTip(button.text())
            button.setAccessibleName(button.text())
        action_row = QHBoxLayout()
        action_row.addWidget(self.btn_trigger); action_row.addWidget(self.step_label, 1)
        action_row.addStretch(1)
        action_row.addWidget(self.btn_open_dir); action_row.addWidget(self.btn_open_last)
        action_row.addWidget(self.btn_c_preview)
        cl.addLayout(action_row); cl.addWidget(self.result_label)
        self.result_label.hide(); self.step_label.hide()
        self.capture_details = QLabel(); self.capture_details.setWordWrap(True)
        self._diagnostic(cl, "capture", (self.profile_label, self.ccm_label2, self.capture_details))
        self.processing = ProcessingWidget(self.model, self._guard)
        for page in (self.start_scroll, p, c, self.processing):
            self.stack.addWidget(page)
        lay = QVBoxLayout(self); lay.setContentsMargins(16, 12, 16, 12)
        self.tabs = QTabBar(); self.tabs.setExpanding(False)
        for label in ("Preview", "Capture", "Traitement"):
            self.tabs.addTab(label)
        self.tabs.currentChanged.connect(self._navigate)
        self.tabs.tabBarClicked.connect(
            lambda index: self._navigate(index) if self.model.mode == START and not self.model.pending else None)
        lay.addWidget(self.tabs); lay.addWidget(self.stack, 1)
        self.footer = ImageView()
        self.footer.setObjectName("footer")
        self.footer.setAccessibleName("powered by")
        self.footer.setMinimumSize(0, 0)
        self.footer.setFixedHeight(28)
        self.footer.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.footer.setPixmap(bundled_pixmap("powered by_white.png"))
        lay.addWidget(self.footer)

    def _mode_button(self, text, callback):
        button = QPushButton(text)
        button.setObjectName("mode")
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button.setIcon(lucide_icon("refresh-cw", "#202326", 18))
        button.setAccessibleName(f"Passer en {text}")
        button.setToolTip(f"Passer en {text}")
        button.clicked.connect(callback)
        return button

    def _header(self, layout):
        row = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(lucide_icon("squirrel", "#FF963F", 30).pixmap(QSize(30, 30)))
        icon.setAccessibleName("Arducam")
        row.addWidget(icon); row.addStretch()
        layout.addLayout(row)

    def _diagnostic(self, layout, name, widgets):
        toggle = QToolButton()
        toggle.setObjectName("diagnostic"); toggle.setText("Diagnostic")
        toggle.setCheckable(True); toggle.setArrowType(Qt.ArrowType.RightArrow)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        toggle.setAccessibleName("Afficher ou masquer le diagnostic")
        toggle.setToolTip("Informations techniques et détails des erreurs")
        area = QScrollArea(); area.setWidgetResizable(True); area.setMaximumHeight(150)
        area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        content = QWidget(); inner = QVBoxLayout(content)
        for widget in widgets:
            widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            inner.addWidget(widget)
        fullscreen = QPushButton("Plein écran")
        fullscreen.setToolTip("Toute la fenêtre · F11 pour basculer · Échap pour revenir")
        fullscreen.clicked.connect(self.fullscreen_requested.emit)
        startup = QCheckBox("Démarrer en plein écran")
        startup.setChecked(self.model.settings.start_fullscreen)
        startup.toggled.connect(self._set_start_fullscreen)
        quit_button = QPushButton("Quitter l’application")
        quit_button.clicked.connect(self.quit_requested.emit)
        self._fullscreen_controls.append(fullscreen)
        self._startup_controls.append(startup)
        inner.addWidget(fullscreen)
        inner.addWidget(startup)
        inner.addWidget(quit_button)
        inner.addStretch(); area.setWidget(content); area.hide()
        toggle.toggled.connect(area.setVisible)
        toggle.toggled.connect(lambda checked: toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow))
        layout.addWidget(toggle); layout.addWidget(area)
        setattr(self, f"{name}_diagnostic_toggle", toggle)
        setattr(self, f"{name}_diagnostic", area)

    def set_fullscreen_state(self, enabled):
        for button in self._fullscreen_controls:
            button.setText("Quitter le plein écran" if enabled else "Plein écran")

    def _set_start_fullscreen(self, enabled):
        self._guard(self.model.set_start_fullscreen, enabled)
        for checkbox in self._startup_controls:
            checkbox.blockSignals(True)
            checkbox.setChecked(self.model.settings.start_fullscreen)
            checkbox.blockSignals(False)

    def _refresh_diagnostic(self):
        if not self.model.capabilities:
            for key in list(self._technical):
                if key.startswith("Commandes vérifiées caméra"):
                    self._technical.pop(key)
        prof = self.model.draft if self.model.mode == PREVIEW_MODE else self.model.saved
        req = getattr(prof, "focus_requested", None)
        rb = getattr(prof, "focus_readback", None)
        lines = [
            f"Backend : {getattr(prof, 'api', self.model.api)} · "
            f"Caméra index : {getattr(prof, 'camera_index', self.model.camera.index)}",
            "Identification par index fragile : peut changer entre deux essais.",
            f"Parcours : {PATH_LABELS.get(getattr(prof, 'path', self.model.path), self.model.path)}",
            f"Focus demandé : {req if req is not None else '–'} · "
            f"relu : {rb if rb is not None else '–'} · plage : {FOCUS_MIN}–{FOCUS_MAX}",
            "Le relu ne prouve pas un effet physique. Pas d’autofocus.",
            f"Dossier : {self.model.photo_dir()}",
            self.model.ccm_status(),
        ]
        lines.extend(f"{key} : {value}" for key, value in self._technical.items())
        if self.model.banks is not None:
            lines.append("Commandes inconnues ou indisponibles désactivées : utilisez « Vérifier commandes ». "
                         "Acceptation et relu fini ne garantissent pas la qualité d’image.")
        text = "\n".join(lines)
        self.preview_details.setText(text)
        self.capture_details.setText(text)

    def _error(self, summary, error):
        self._technical["Erreur technique"] = str(error)
        self._refresh_diagnostic()
        label = self.preview_msg if self.model.mode == PREVIEW_MODE else self.result_label
        label.setText(summary + " Voir Diagnostic.")
        label.show()
        if self.model.mode == START:
            box = QMessageBox(QMessageBox.Icon.Warning, "Erreur", summary, parent=self)
            box.setDetailedText(str(error)); box.exec()

    def _refresh_open_buttons(self):
        directory = self._directory_to_open()
        self.btn_open_dir.setEnabled(os.path.isdir(directory))
        self.btn_open_dir.setToolTip(f"Ouvrir le dossier : {directory}")
        self.btn_open_last.setEnabled(bool(self.model.last_photo and os.path.isfile(self.model.last_photo)))
        self.btn_process_last.setEnabled(bool(self.model.last_acquisition)
                                         and self.model.controller.can_change_mode
                                         and self.model.proposal is None)

    PAGES = {START: 0, PREVIEW_MODE: 1, CAPTURE_MODE: 2, PROCESSING_MODE: 3}

    def _show_page(self, mode: str) -> None:
        self.stack.setCurrentIndex(self.PAGES[mode])
        self._set_capture_busy(not self.model.controller.can_change_mode)
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex({PREVIEW_MODE: 0, CAPTURE_MODE: 1, PROCESSING_MODE: 2}.get(mode, -1))
        self.tabs.blockSignals(False)
        self.capture_mode_combo.blockSignals(True)
        self.capture_mode_combo.setCurrentIndex(self.capture_mode_combo.findData(self.model.path))
        self.capture_mode_combo.blockSignals(False)
        self.processing.refresh()
        ccm_status = self.model.ccm_status()
        self.ccm_label.setText(ccm_status)
        self.ccm_label2.setText(ccm_status)
        self.folder_label.setText(f"Dossier des photos : {self.model.photo_dir()}")
        self._refresh_diagnostic()
        self._refresh_open_buttons()
        if mode == CAPTURE_MODE:
            prof = self.model.saved
            if prof is not None and prof.is_valid:
                self.profile_label.setText(
                    f"Caméra {prof.camera_index} · {PATH_LABELS[prof.path]} · "
                    f"Focus {prof.focus_requested if prof.focus_requested is not None else '–'}")
            else:
                self.profile_label.setText("Aucun profil valide : passez par Preview / Réglages.")
            if prof and prof.path != self.model.path:
                self.result_label.setText("Ce mode nécessite un nouveau profil enregistré explicitement en Preview.")
                self.result_label.show()
            elif self.result_label.text().startswith("Ce mode nécessite"):
                self.result_label.clear()
                self.result_label.hide()
            self._set_capture_busy(self.model.controller.state == "capturing")
            self._fps_timer.stop()
        elif mode == PREVIEW_MODE:
            self.preview_msg.clear(); self.preview_msg.hide()
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
        except ProposalPending:
            self._resolve_proposal_prompt()
        except (ModeError, ProfileMissing) as e:
            self._error("Action indisponible.", e)
        except Exception as e:  # recoverable error shown, never crashes the UI
            self._error("Impossible de terminer cette action.", e)

    def _info(self, text: str) -> None:
        QMessageBox.information(self, "Information", text)

    def _path_changed(self, _):
        self._guard(self.model.select_path, self.path_combo.currentData())
        self.ccm_label.setText(self.model.ccm_status())

    def _capture_path_changed(self, _):
        self._guard(self.model.select_path, self.capture_mode_combo.currentData())
        self._show_page(self.model.mode)

    def _navigate(self, index):
        {0: self._go_preview, 1: self._go_capture, 2: self._go_processing}.get(index, lambda: None)()
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex({PREVIEW_MODE: 0, CAPTURE_MODE: 1, PROCESSING_MODE: 2}.get(self.model.mode, -1))
        self.tabs.blockSignals(False)

    def _go_processing(self):
        decision = None
        while True:
            try:
                self.model.request_processing_mode(decision)
                if self.model.pending:
                    self._set_capture_busy(True)
                return
            except NeedsDecision:
                decision = self._ask_unsaved()
            except ProposalPending:
                if not self._resolve_proposal_prompt():
                    return
            except Exception as e:
                self._error("Impossible de passer en Traitement.", e)
                return

    def _process_last(self):
        self._go_processing()
        if self.model.mode == PROCESSING_MODE and self.model.last_acquisition:
            self.processing._load(self.model.last_acquisition)

    def _camera_changed(self, _):
        if self.cam_combo.currentData() is not None:
            self._guard(self.model.select_camera, self.cam_combo.currentData())

    def _detect(self):
        self._guard(self.model.controller.probe)
        self._set_capture_busy(not self.model.controller.can_change_mode)

    def _pick_ccm(self):
        f, _ = QFileDialog.getOpenFileName(self, "arducam_108mp.json", "", "JSON (*.json)")
        if f:
            self._guard(self.model.set_ccm_path, f)

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier des photos", self.model.photo_dir())
        if d:
            self._guard(self.model.set_photo_dir, d)
            self.folder_label.setText(f"Dossier des photos : {self.model.photo_dir()}")
            self._refresh_open_buttons()
            self._refresh_diagnostic()

    def _go_preview(self):
        if self.model.mode == PREVIEW_MODE:
            return
        self._guard(self.model.enter_preview)
        if self.model.pending:
            self._set_capture_busy(True)

    def _go_capture(self):
        decision = None
        while True:
            try:
                self.model.request_capture_mode(decision)
                if self.model.pending:
                    self._set_capture_busy(True)
                return
            except NeedsDecision:
                decision = self._ask_unsaved()
            except ProposalPending:
                if not self._resolve_proposal_prompt():
                    return
            except (ModeError, ProfileMissing) as e:
                self._error("Enregistrez un profil valide avant de passer en Capture.", e); return
            except Exception as e:
                self._error("Impossible de passer en Capture.", e); return

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
        if clicked is b_save and self.model.banks is not None:
            return SAVE if self._save_bank_target() else STAY
        return SAVE if clicked is b_save else DISCARD if clicked is b_drop else STAY

    def _refresh_banks(self, selected=None):
        if self.model.banks is None:
            return
        selected = selected or self.bank_combo.currentData()
        try:
            banks = self.model.list_banks()
        except Exception as error:
            self._error("Impossible de lire la liste des banks.", error)
            return
        self._bank_entries = {bank.id: bank for bank in banks}
        self.bank_combo.blockSignals(True)
        self.bank_combo.clear()
        self.bank_combo.addItem("Choisir une bank cible", None)
        for bank in banks:
            self.bank_combo.addItem(f"{bank.name} · révision {bank.revision}", bank.id)
        self.bank_combo.setCurrentIndex(max(0, self.bank_combo.findData(selected)))
        self.bank_combo.blockSignals(False)

    def _create_bank(self):
        name, accepted = QInputDialog.getText(self, "Créer une bank", "Nom de la nouvelle bank :")
        if not accepted:
            return False
        bank = self._guard(self.model.create_bank, name)
        if bank is None:
            return False
        self._refresh_banks(bank.id)
        self._refresh_preview_state()
        return True

    def _save_bank_target(self):
        bank_id = self.bank_combo.currentData()
        if bank_id is None:
            return self._create_bank()
        try:
            bank = self.model.banks.get(bank_id)
            answer = QMessageBox.question(
                self, "Remplacer la bank",
                f"Remplacer « {bank.name} », révision {bank.revision}, par les réglages courants ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return False
            saved = self.model.save_bank(bank_id, confirm_replace=True)
        except Exception as error:
            self._error("Impossible d’enregistrer la bank.", error)
            return False
        self._refresh_banks(saved.id)
        self._refresh_preview_state()
        return True

    def _load_bank(self):
        bank_id = self.bank_combo.currentData()
        if bank_id is None:
            self._info("Choisissez une bank à charger.")
            return
        decision = None
        while True:
            try:
                if self.model.load_bank(bank_id, decision):
                    self._refresh_preview_state()
                return
            except NeedsDecision:
                decision = self._ask_unsaved()
            except ProposalPending:
                if not self._resolve_proposal_prompt():
                    return
            except Exception as error:
                self._error("Impossible de charger cette bank.", error)
                return

    def _probe_controls(self):
        self._guard(self.model.controller.probe_controls)
        self._set_capture_busy(not self.model.controller.can_change_mode)

    def _adjustment_options(self):
        from ..auto_adjust import AdjustmentOptions

        dialog = QDialog(self)
        dialog.setWindowTitle("Auto-ajustement borné")
        form = QFormLayout(dialog)
        note = QLabel("Région de cadrage du preview (pas le champ photo garanti) : "
                      "x, y, largeur et hauteur normalisés entre 0 et 1. "
                      "Exposition et gain : bornes choisies par l’opérateur dans les unités du backend, "
                      "sans convention MSMF/DShow présumée. Focus manuel fixé, sans autofocus.")
        note.setWordWrap(True)
        form.addRow(note)
        roi = []
        for name, value in (("x", 0), ("y", 0), ("largeur", 1), ("hauteur", 1)):
            spin = QDoubleSpinBox(dialog)
            spin.setRange(0, 1)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setValue(value)
            spin.setObjectName("roi_" + name)
            form.addRow("Région · " + name, spin)
            roi.append(spin)
        bounds = {}
        steps = {}
        for name in ("exposure", "gain"):
            pair = []
            for bound in ("minimum", "maximum"):
                field = QLineEdit(dialog)
                field.setObjectName(name + "_" + bound)
                field.setPlaceholderText("Valeur backend obligatoire")
                form.addRow(("Exposition" if name == "exposure" else "Gain") + " · " + bound, field)
                pair.append(field)
            bounds[name] = pair
            field = QLineEdit(dialog)
            field.setObjectName(name + "_step")
            field.setPlaceholderText("Pas positif dans les unités du backend")
            form.addRow(("Exposition" if name == "exposure" else "Gain") + " · pas", field)
            steps[name + "_step"] = field
        controls = {}
        for name, label, low, high, value in (
            ("max_iterations", "Itérations maximum", 3, 1000, 20),
            ("time_budget", "Budget temps (secondes)", 1, 3600, 15),
            ("settle_frames", "Frames de stabilisation", 0, 1000, 2),
        ):
            spin = QSpinBox(dialog)
            spin.setRange(low, high)
            spin.setValue(value)
            spin.setObjectName(name)
            form.addRow(label, spin)
            controls[name] = spin
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        region = tuple(spin.value() for spin in roi)
        if region[2] <= 0 or region[3] <= 0 or region[0] + region[2] > 1 or region[1] + region[3] > 1:
            raise ValueError("La région doit être non vide et entièrement dans le preview.")
        ranges = {name + "_bounds": tuple(float(field.text()) for field in fields)
                  for name, fields in bounds.items()}
        for pair in ranges.values():
            if not all(math.isfinite(value) for value in pair) or pair[0] >= pair[1]:
                raise ValueError("Les bornes backend doivent être finies et croissantes.")
        options = AdjustmentOptions(roi=region, **ranges,
                                    **{name: float(field.text()) for name, field in steps.items()},
                                    **{name: spin.value() for name, spin in controls.items()})
        options.validate()
        return options

    def _auto_adjust(self):
        draft = self.model.draft
        if (draft is None or draft.focus_requested is None or
                draft.auto_exposure_mode != "manual" or draft.auto_exposure is None):
            self._info("Fixez le focus et choisissez explicitement le mode d’exposition manuel "
                       "avec sa valeur backend avant l’auto-ajustement.")
            return
        try:
            options = self._adjustment_options()
            if options is None:
                return
            self.model.start_auto_adjustment(options)
            self.adjustment_summary.setText("Auto-ajustement en cours. Aucun réglage n’est sauvegardé.")
            self.adjustment_summary.show()
            self._refresh_preview_state()
        except Exception as error:
            self._error("Impossible de lancer l’auto-ajustement.", error)

    def _accept_proposal(self):
        self._guard(self.model.validate_proposal)
        self._refresh_preview_state()

    def _reject_proposal(self):
        self._guard(self.model.reject_proposal)
        self._proposal_restoring = (self.model.proposal is not None
                                   and not self.model.controller.can_change_mode)
        self._refresh_preview_state()

    def _resolve_proposal_prompt(self):
        box = QMessageBox(self)
        box.setWindowTitle("Proposition en attente")
        box.setText("Validez ou rejetez la proposition avant de continuer. Valider ne sauvegarde pas.")
        accept = box.addButton("Valider", QMessageBox.ButtonRole.AcceptRole)
        reject = box.addButton("Rejeter", QMessageBox.ButtonRole.DestructiveRole)
        stay = box.addButton("Rester", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(stay)
        box.exec()
        if box.clickedButton() is accept:
            self._accept_proposal()
            return self.model.proposal is None
        if box.clickedButton() is reject:
            self._reject_proposal()
            self._info("Attendez la restauration des réglages, puis réessayez cette action.")
        return False

    def _focus_input(self, value: int, from_slider: bool) -> None:
        if self.model.mode != PREVIEW_MODE:
            return
        if value >= FOCUS_MIN:
            self._guard(self.model.set_focus, value)
        self._refresh_preview_state()

    def _toggle_camera_settings(self, expanded):
        self.camera_settings_panel.setVisible(expanded)
        self.camera_settings_toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def _camera_control_input(self, name, editor):
        text = editor.text().strip()
        try:
            value = None if not text else float(text)
        except ValueError:
            self._error("Valeur caméra invalide.", ValueError("saisir un nombre fini ou laisser vide"))
            return
        self._guard(self.model.set_camera_control, name, value)
        self._refresh_preview_state()

    def _camera_mode_input(self, name, selector):
        mode = selector.currentData()
        self._guard(self.model.set_camera_mode, name, mode)
        self._refresh_preview_state()

    def _sync_camera_settings(self, profile):
        for name, editor in self.camera_settings_inputs.items():
            value = getattr(profile, name, None)
            text = "" if value is None else str(value)
            if editor.text() != text:
                editor.blockSignals(True)
                editor.setText(text)
                editor.blockSignals(False)
        for name, selector in self.camera_mode_inputs.items():
            mode = getattr(profile, name, None)
            index = selector.findData(mode)
            selector.blockSignals(True)
            selector.setCurrentIndex(max(0, index))
            selector.blockSignals(False)
        self.camera_settings_inputs["exposure"].setEnabled(profile.auto_exposure_mode != "automatic")
        self.camera_settings_inputs["gain"].setEnabled(profile.auto_exposure_mode != "automatic")
        self.camera_settings_inputs["wb_temperature"].setEnabled(profile.auto_wb_mode != "automatic")

    def _sync_focus_controls(self, requested: Optional[int]) -> None:
        known = requested is not None
        self.focus_slider.blockSignals(True)
        self.focus_spin.blockSignals(True)
        self.focus_spin.setRange(FOCUS_MIN if known else FOCUS_MIN - 1, FOCUS_MAX)
        self.focus_spin.setSpecialValueText("" if known else "non défini")
        self.focus_spin.setValue(requested if known else FOCUS_MIN - 1)
        self.focus_slider.setValue(requested if known else FOCUS_MIN)
        self.focus_spin.blockSignals(False)
        self.focus_slider.blockSignals(False)

    def _save(self):
        saved = (self._save_bank_target() if self.model.banks is not None
                 else self._guard(self.model.save_profile) is not None)
        if saved:
            self.preview_msg.clear(); self.preview_msg.hide()
        self._refresh_preview_state()

    def _trigger(self):
        if self._guard(self.model.trigger) is None and self.model.controller.state == "capturing":
            self._set_capture_busy(True)
            self.result_label.setText("")
            self.result_label.hide()

    def _set_capture_busy(self, busy: bool) -> None:
        busy = busy or self.model.controller.state in ("adjusting", "restoring")
        pending_proposal = self.model.proposal is not None
        for b in (self.btn_trigger, self.btn_c_preview, self.btn_p_capture,
                  self.btn_to_capture, self.btn_to_preview, self.capture_mode_combo, self.tabs,
                  self.btn_process_last, self.path_combo, self.cam_combo, self.detect_button):
            b.setEnabled(not busy and not pending_proposal)
        self._refresh_editability(busy)
        if not busy and not pending_proposal:
            prof = self.model.saved
            self.btn_trigger.setEnabled(bool(prof and prof.is_valid and prof.path == self.model.path)
                                        and not self.model.controller.camera_fault)
            self._refresh_open_buttons()

    def _control_available(self, name):
        if self.model.banks is None:
            return True
        report = self.model.capabilities.get(name)
        readback = getattr(report, "readback", None)
        try:
            return bool(report and report.accepted is True and not report.error
                        and not report.readback_error and readback is not None and math.isfinite(readback))
        except (TypeError, ValueError):
            return False

    def _refresh_editability(self, busy=False):
        state = self.model.controller.state
        busy = busy or state in ("adjusting", "restoring")
        proposal = self.model.proposal is not None
        editable = (self.model.mode == PREVIEW_MODE and not self.model.pending
                    and self.model.controller.can_change_mode and not busy and not proposal)
        for control in (self.btn_save, self.fps_spin, self.bank_combo,
                        self.btn_create_bank, self.btn_load_bank):
            control.setEnabled(editable)
        for control in (self.focus_slider, self.focus_spin):
            control.setEnabled(editable and self._control_available("focus"))
        draft = self.model.draft
        for name, control in self.camera_settings_inputs.items():
            auto = (name in ("exposure", "gain") and
                    getattr(draft, "auto_exposure_mode", None) == "automatic") or (
                        name == "wb_temperature" and getattr(draft, "auto_wb_mode", None) == "automatic")
            control.setEnabled(editable and not auto and self._control_available(name))
            if self.model.banks is not None and name not in self.model.capabilities:
                control.setToolTip("Commande inconnue : utilisez « Vérifier commandes » avant de modifier.")
        for name, selector in self.camera_mode_inputs.items():
            selector.setEnabled(editable and self._control_available(name.removesuffix("_mode")))
            if self.model.banks is not None and name.removesuffix("_mode") not in self.model.capabilities:
                selector.setToolTip("Commande inconnue : utilisez « Vérifier commandes ».")
        if self.model.banks is not None and "focus" not in self.model.capabilities:
            for control in (self.focus_slider, self.focus_spin):
                control.setToolTip("Focus indisponible tant que « Vérifier commandes » n’a pas confirmé "
                                   "l’acceptation et un relu fini. Aucun autofocus.")
        self.btn_probe_controls.setEnabled(editable)
        self.btn_probe_controls.setVisible(self.model.banks is not None)
        self.btn_adjust.setVisible(self.model.banks is not None)
        self.btn_adjust.setEnabled(editable and all(self._control_available(name)
                                                   for name in ("focus", "auto_exposure", "exposure", "gain")))
        self.btn_cancel_adjustment.setEnabled(state == "adjusting" and not self._proposal_restoring)
        self.btn_cancel_adjustment.setVisible(state in ("adjusting", "restoring"))
        self.btn_accept_proposal.setEnabled(proposal and not busy)
        self.btn_reject_proposal.setEnabled(proposal and not busy)
        self.btn_accept_proposal.setVisible(proposal)
        self.btn_reject_proposal.setVisible(proposal)

    def _open_dir(self):
        d = self._directory_to_open()
        if os.path.isdir(d):
            self._guard(open_in_system, d)
        self._refresh_open_buttons()

    def _directory_to_open(self):
        if self.model.last_acquisition:
            directory = os.path.dirname(self.model.last_acquisition)
            if os.path.isdir(directory):
                return directory
        return self.model.photo_dir()

    def _open_last(self):
        if self.model.last_photo and os.path.isfile(self.model.last_photo):
            self._guard(open_in_system, self.model.last_photo)
        else:
            self._info("Aucune photo prise pendant cette session.")
        self._refresh_open_buttons()

    # ---- events (UI thread) ----
    def _on_event(self, event: str, kw: dict) -> None:
        if event == "bank_changed":
            self._refresh_banks(kw["bank"].id)
            self._refresh_preview_state()
        elif event == "camera_capabilities":
            self._record_camera_reports(kw["reports"], "Commandes vérifiées")
            for name, report in kw["reports"].items():
                text = (f"Demandé : {report.requested} · accepté : {report.accepted} · "
                        f"relu : {report.readback} · "
                        f"{'disponible' if self._control_available(name) else 'indisponible'}. "
                        "Le relu ne garantit ni effet physique ni qualité d’image.")
                if name in self.camera_settings_inputs:
                    self.camera_settings_inputs[name].setToolTip(text)
                if name + "_mode" in self.camera_mode_inputs:
                    self.camera_mode_inputs[name + "_mode"].setToolTip(text)
                if name == "focus":
                    self.focus_slider.setToolTip(text)
                    self.focus_spin.setToolTip(text)
            self._refresh_preview_state()
            if kw.get("error"):
                self._error("Impossible de vérifier les commandes caméra.", kw["error"])
        elif event in ("adjustment_started", "adjustment_proposed", "adjustment_restored",
                       "adjustment_failed", "proposal_resolved"):
            if event in ("adjustment_restored", "adjustment_failed", "proposal_resolved"):
                self._proposal_restoring = False
            if event == "adjustment_proposed":
                result = kw["result"]
                converged = getattr(result, "converged", False)
                reason = {
                    "converged": "critères atteints",
                    "max_iterations": "limite d’itérations",
                    "time_budget": "budget temps atteint",
                    "criteria_not_met": "critères non atteints",
                }.get(getattr(result, "stop_reason", ""), "arrêt à vérifier")
                self.adjustment_summary.setText(
                    f"{'Critères atteints' if converged else 'Candidat seulement, critères non atteints'} "
                    f"· {reason}\nAvant : {result.before}\nProposé : {result.proposed}\n"
                    f"Mesures : {getattr(result, 'metrics', '–')}\n"
                    f"Itérations : {getattr(result, 'iterations', '–')} · "
                    f"durée : {getattr(result, 'elapsed', '–')} s\n"
                    "Valider modifie seulement les réglages courants, sans sauvegarder.")
            elif event == "adjustment_failed":
                self._error("Auto-ajustement interrompu.", kw.get("error", "échec"))
            elif event in ("adjustment_restored", "proposal_resolved"):
                self.adjustment_summary.setText(
                    "Réglages restaurés." if event == "adjustment_restored"
                    else "Proposition validée dans le brouillon, non sauvegardée.")
            warnings = kw.get("warnings", kw.get("restoration_warnings", []))
            if warnings:
                self.adjustment_summary.setText(self.adjustment_summary.text() +
                                                "\nRestauration à vérifier : " + str(warnings))
                self._technical["Restauration à vérifier"] = str(warnings)
            self.adjustment_summary.show()
            self._refresh_preview_state()
        elif event == "frame":
            f = self.model.controller.take_frame()
            if f is not None and self.model.mode == PREVIEW_MODE:
                img = bgr_to_qimage(f)
                pixmap = QPixmap.fromImage(img)
                self.video.setPixmap(pixmap)
                dimensions = f"{img.width()} × {img.height()}"
                if self._technical.get("Dimensions du preview") != dimensions:
                    self._technical["Dimensions du preview"] = dimensions
                    self._refresh_diagnostic()
        elif event == "video_cleared":
            self.video.clear(); self.video.setText("Aucun flux")
        elif event == "mode_changed":
            self._show_page(kw["mode"])
        elif event == "preview_failed":
            self._set_capture_busy(False)
            self._error("Impossible d’ouvrir la caméra.", kw["error"])
        elif event == "preview_warning":
            self._technical["Lectures échouées"] = kw["failed_reads"]
            self._refresh_diagnostic()
        elif event in ("focus_applied", "focus_failed"):
            if "accepted" in kw:
                self._technical["Consigne acceptée par le pilote"] = "oui" if kw["accepted"] else "non"
            self._refresh_preview_state(kw)
        elif event == "camera_settings_applied":
            for name, report in kw["reports"].items():
                self._technical[f"Caméra {name}"] = (
                    f"demandé {report.requested}; essayé {report.attempted}; "
                    f"accepté {report.accepted}; relu {report.readback}; "
                    f"erreur {report.error or report.readback_error or 'aucune'}; "
                    f"mode {report.mode or 'inconnu'}"
                )
            self._refresh_preview_state()
        elif event == "camera_settings_failed":
            self._error("Impossible d’appliquer les réglages caméra.", kw["error"])
        elif event == "cameras":
            self._fill_cameras(kw["indices"] or [self.model.camera.index])
            self._set_capture_busy(False)
        elif event in ("cameras_failed", "preview_release_failed"):
            self._set_capture_busy(False)
            self._error("Caméra indisponible · voir Diagnostic.", kw["error"])
        elif event == "ccm_checked":
            if not kw["ok"]:
                self._error("Fichier de correction couleur invalide.", kw["error"])
            self.ccm_label.setText(self.model.ccm_status() + ("" if kw["ok"] else " — INVALIDE"))
            self._refresh_diagnostic()
        elif event == "capture_step":
            self._set_capture_busy(True)
            step = kw["step"]
            self._technical["Étape du contrôleur"] = step
            self.step_label.setText({
                "Préparation et vérification du profil": "Préparation du profil",
                "Acquisition et traitement (moteur, appel bloquant)": "Acquisition et traitement",
                "Enregistrement du PNG": "Enregistrement PNG",
            }.get(step, step))
            self.step_label.show()
            self._refresh_diagnostic()
        elif event == "capture_done":
            w, h = kw["size"]
            info = kw["info"]
            self.photo.setPixmap(QPixmap.fromImage(bgr_to_qimage(kw["thumbnail"])))
            try:
                timestamp = datetime.fromtimestamp(os.path.getmtime(kw["path"])).strftime("%d/%m/%Y à %H:%M:%S")
            except OSError:
                timestamp = None
            self.result_label.setText(
                f"Photo enregistrée · {timestamp}" if timestamp else "Photo enregistrée")
            self.result_label.show()
            self._technical.update({
                "Dernier fichier": kw["path"], "Dimensions": f"{w} × {h}",
                "Correction couleur appliquée": str(info.ccm_applied),
                "Durée du moteur": f"{info.duration_s:.1f} s",
                "Chronométrage détaillé": str(kw.get("timings", {})),
            })
            self._record_camera_reports(getattr(info, "settings", {}), "Capture")
            self._technical.pop("Erreur technique", None)
            self._refresh_diagnostic()
            self._refresh_open_buttons()
        elif event == "capture_failed":
            self._technical["Chronométrage détaillé"] = str(kw.get("timings", {}))
            archive = kw.get("archive_path")
            capture_manifest = kw.get("capture_manifest_path")
            if kw.get("raw_preserved"):
                status = "traitement échoué" if kw.get("stage") in ("render", "output") else "capture incomplète"
                message = f"RAW conservé, {status} · manifeste {capture_manifest}"
            elif capture_manifest:
                message = f"Capture incomplète · manifeste conservé {capture_manifest}"
            else:
                message = ("Capture incomplète · original archivé disponible en Traitement."
                           if archive else "Échec de la capture · aucun nouvel original archivé.")
            self._error(message, kw["error"])
        elif event == "capture_archived":
            self._technical["Archive de l’original"] = kw["path"]
            self._technical["Chronométrage détaillé"] = str(kw.get("timings", {}))
            self._record_camera_reports(kw["info"].settings, "Capture")
            self._refresh_open_buttons()
        elif event.startswith("processing_"):
            self.processing.on_event(event, kw)
            if event == "processing_busy":
                self.processing.busy = True
                self.processing.refresh()
                self._set_capture_busy(True)
            elif event == "processing_idle":
                self._set_capture_busy(False)
                self._technical["Durée traitement"] = f"{kw['duration_s']:.2f} s"
        elif event == "capture_idle":
            self.step_label.setText("")
            self.step_label.hide()
            self._set_capture_busy(False)

    def _refresh_preview_state(self, kw: Optional[dict] = None) -> None:
        d = self.model.draft
        if d is None:
            return
        self._sync_camera_settings(d)
        self._set_capture_busy(not self.model.controller.can_change_mode)
        self._sync_focus_controls(d.focus_requested)
        self.fps_spin.blockSignals(True)
        self.fps_spin.setValue(round(d.fps or MODES[d.path].fps))
        self.fps_spin.blockSignals(False)
        self._technical["Aperçu de cadrage"] = "720p à cadence modeste; champ photo complet non garanti"
        rb = "–" if d.focus_readback is None else f"{d.focus_readback:g}"
        req = "–" if d.focus_requested is None else str(d.focus_requested)
        self.focus_state.setText(f"Focus manuel : consigne {req} | valeur relue {rb} "
                                 "(le relu n'est pas une preuve d'effet physique). Pas d'autofocus.")
        self.unsaved.setText("Modifications non enregistrées" if self.model.has_unsaved() else "Réglages enregistrés")
        self._refresh_diagnostic()
        if kw and "error" in kw:
            self._error("Impossible d’appliquer le focus.", kw["error"])

    def _record_camera_reports(self, reports, prefix):
        for name, report in reports.items():
            self._technical[f"{prefix} caméra {name}"] = (
                f"demandé {report.requested}; essayé {report.attempted}; "
                f"accepté {report.accepted}; relu {report.readback}; "
                f"erreur {report.error or report.readback_error or 'aucune'}; mode {report.mode or 'inconnu'}"
            )

    def _update_fps(self) -> None:
        a, d = self.model.controller.fps()
        self.fps_label.setText(f"FPS acquis : {a:.1f} | affichés : {d:.1f}")
        self._technical["FPS acquis / affichés"] = f"{a:.1f} / {d:.1f}"
        self._refresh_diagnostic()

    # ---- closing ----
    def can_close(self) -> bool:
        """False if the window must stay open (capture running, or user stays in Preview)."""
        if self.model.controller.state in ("capturing", "processing", "adjusting", "restoring"):
            self._info("Une opération est en cours : attendez la fin avant de fermer.")
            return False
        if self.model.proposal is not None and not self._resolve_proposal_prompt():
            return False
        if self.model.has_unsaved():
            d = self._ask_unsaved()
            if d == STAY:
                return False
            if (d == SAVE and self.model.has_unsaved() and
                    (not self._save_bank_target() if self.model.banks is not None
                     else self._guard(self.model.save_profile) is None)):
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
