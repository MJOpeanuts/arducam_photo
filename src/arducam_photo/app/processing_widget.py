"""Camera-free, bounded comparison UI; full-resolution pixels stay on the worker."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QToolButton,
                               QFileDialog, QGraphicsView, QGraphicsScene, QFormLayout)

from ..processing import ProcessingRecipe
from ..archive import OutputOptions


class ComparisonView(QGraphicsView):
    changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(160, 120)
        self._syncing = False
        self.horizontalScrollBar().valueChanged.connect(self._changed)
        self.verticalScrollBar().valueChanged.connect(self._changed)

    def set_image(self, image):
        from .widget import bgr_to_qimage

        self.scene().clear()
        self.scene().addPixmap(QPixmap.fromImage(bgr_to_qimage(image)))
        self.scene().setSceneRect(self.scene().itemsBoundingRect())
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._changed()

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        scale = self.transform().m11() * factor
        if 0.02 <= scale <= 32:
            self.scale(factor, factor)
            self._changed()
        event.accept()

    def _changed(self, *_):
        if not self._syncing:
            center = self.mapToScene(self.viewport().rect().center())
            self.changed.emit((self.transform(), center))

    def synchronize(self, state):
        self._syncing = True
        self.setTransform(state[0])
        self.centerOn(state[1])
        self._syncing = False


class ProcessingWidget(QWidget):
    def __init__(self, model, guard):
        super().__init__()
        self.model, self.guard = model, guard
        self.has_result = False
        self.busy = False
        self.raw = False
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.open_button = QPushButton("Ouvrir une acquisition…")
        self.last_button = QPushButton("Dernière acquisition")
        self.open_button.clicked.connect(self._open)
        self.last_button.clicked.connect(self._last)
        row.addWidget(self.open_button); row.addWidget(self.last_button); row.addStretch()
        layout.addLayout(row)
        self.source_label = QLabel("Source · miniature bornée à 1280 px")
        self.result_label = QLabel("Résultat · miniature bornée à 1280 px")
        names = QHBoxLayout()
        names.addWidget(self.source_label, 1); names.addWidget(self.result_label, 1)
        layout.addLayout(names)
        self.source, self.result = ComparisonView(), ComparisonView()
        self.source.changed.connect(self.result.synchronize)
        self.result.changed.connect(self.source.synchronize)
        images = QHBoxLayout(); images.addWidget(self.source, 1); images.addWidget(self.result, 1)
        layout.addLayout(images, 1)
        controls = QHBoxLayout()
        self.recipe_combo = QComboBox()
        for label, value in (("RAW minimal", "minimal"), ("RAW niveau noir", "black_level"),
                             ("Référence / couleur", "reference")):
            self.recipe_combo.addItem(label, value)
        self.recipe_combo.setCurrentIndex(2)
        self.colour = QCheckBox("Correction couleur RAW"); self.colour.setChecked(False)
        self.recipe_combo.currentIndexChanged.connect(self.refresh)
        self.gray = QCheckBox("Gris")
        self.clahe = QCheckBox("CLAHE")
        self.threshold = QComboBox()
        for label, value in (("Sans seuil", "none"), ("Otsu", "otsu"), ("Adaptatif", "adaptive")):
            self.threshold.addItem(label, value)
        self.reduction = QComboBox()
        for label, value in (("Taille originale", None), ("Max 4096 px", 4096),
                             ("Max 2048 px", 2048), ("Max 1280 px", 1280)):
            self.reduction.addItem(label, value)
        for item in (self.recipe_combo, self.colour, self.gray, self.clahe, self.threshold, self.reduction):
            controls.addWidget(item)
        layout.addLayout(controls)
        toggle = QToolButton(); toggle.setText("Paramètres avancés"); toggle.setCheckable(True)
        advanced = QWidget(); form = QFormLayout(advanced)
        self.black = QSpinBox(); self.black.setRange(0, 254); self.black.setValue(16)
        self.temperature = QSpinBox(); self.temperature.setRange(1000, 15000); self.temperature.setValue(4000)
        self.clip = QDoubleSpinBox(); self.clip.setRange(0.1, 40); self.clip.setValue(2)
        self.grid = QSpinBox(); self.grid.setRange(1, 64); self.grid.setValue(8)
        self.block = QSpinBox(); self.block.setRange(3, 255); self.block.setSingleStep(2); self.block.setValue(11)
        self.adaptive_c = QDoubleSpinBox(); self.adaptive_c.setRange(-255, 255); self.adaptive_c.setValue(2)
        for label, control in (("RAW · niveau noir (référence minimale)", self.black),
                               ("Température couleur", self.temperature), ("CLAHE clip", self.clip),
                               ("CLAHE grille", self.grid), ("Bloc adaptatif impair", self.block),
                               ("Constante adaptative", self.adaptive_c)):
            form.addRow(label, control)
        advanced.hide(); toggle.toggled.connect(advanced.setVisible)
        layout.addWidget(toggle); layout.addWidget(advanced)
        actions = QHBoxLayout()
        self.process_button = QPushButton("Traiter")
        self.save_button = QPushButton("Enregistrer le résultat")
        self.format = QComboBox(); self.format.addItems(["PNG", "JPEG"])
        self.quality = QSpinBox(); self.quality.setRange(1, 100); self.quality.setValue(95)
        self.quality.setToolTip("Qualité JPEG")
        self.process_button.clicked.connect(self._process)
        self.save_button.clicked.connect(self._save)
        for item in (self.process_button, self.format, QLabel("Qualité JPEG"), self.quality, self.save_button):
            actions.addWidget(item)
        layout.addLayout(actions)
        self.status = QLabel("Ouvrez un acquisition.json, PNG ou JPEG. Zoom molette · glisser pour déplacer.")
        self.status.setWordWrap(True); layout.addWidget(self.status)
        self.refresh()

    def refresh(self):
        for button in (self.open_button, self.last_button, self.process_button, self.save_button):
            button.setEnabled(not self.busy)
        self.last_button.setEnabled(not self.busy and bool(self.model.last_acquisition))
        self.process_button.setEnabled(not self.busy and bool(self.model.processing_source))
        self.save_button.setEnabled(not self.busy and self.has_result)
        self.colour.setEnabled(self.raw and self.recipe_combo.currentData() == "reference")

    def _load(self, path):
        self.guard(self.model.load_processing, path)
        self.busy = self.model.controller.state == "processing"
        if self.busy:
            self.has_result = False
        self.refresh()

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Acquisition ou image", "",
                                             "Acquisitions et images (*.json *.png *.jpg *.jpeg)")
        if path:
            self._load(path)

    def _last(self):
        if self.model.last_acquisition:
            self._load(self.model.last_acquisition)

    def recipe(self):
        return ProcessingRecipe(name=self.recipe_combo.currentData(), black_level=self.black.value(),
                                apply_ccm=self.raw and self.recipe_combo.currentData() == "reference"
                                and self.colour.isChecked(), temperature=self.temperature.value(),
                                grayscale=self.gray.isChecked(),
                                clahe_clip=self.clip.value() if self.clahe.isChecked() else 0,
                                clahe_grid=self.grid.value(), threshold=self.threshold.currentData(),
                                adaptive_block=self.block.value(), adaptive_c=self.adaptive_c.value(),
                                max_dimension=self.reduction.currentData())

    def _process(self):
        self.guard(self.model.process, self.recipe())
        self.busy = self.model.controller.state == "processing"
        if self.busy:
            self.has_result = False
        self.refresh()

    def _save(self):
        self.guard(self.model.save_processed, OutputOptions(format=self.format.currentText().lower(),
                                                           jpeg_quality=self.quality.value()))
        self.busy = self.model.controller.state == "processing"; self.refresh()

    def on_event(self, event, kw):
        if event == "processing_loaded":
            self.raw = kw["raw"]
            self.source.set_image(kw["thumbnail"])
            self.result.scene().clear()
            self.has_result = False
            self.source_label.setText("Source · RAW dématricé minimal, pas mosaïque · miniature ≤1280 px"
                                      if kw["raw"] else "Source couleur · miniature ≤1280 px")
            self.status.setText(f"{kw['path']} · {kw['size'][0]} × {kw['size'][1]}")
        elif event == "processing_done":
            self.result.set_image(kw["thumbnail"])
            self.source._changed()
            self.has_result = True
            self.result_label.setText(f"Résultat · {kw['size'][0]} × {kw['size'][1]} · miniature ≤1280 px")
        elif event == "processing_saved":
            self.status.setText(f"Résultat enregistré : {kw['path']}")
        elif event == "processing_failed":
            self.status.setText(f"Échec du traitement : {kw['error']}")
        elif event == "processing_idle":
            self.busy = False
        self.refresh()
