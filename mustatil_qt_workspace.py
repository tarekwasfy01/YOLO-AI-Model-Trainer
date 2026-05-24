#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mustatil Qt Workspace

Qt/PySide6 frontend for the existing Mustatil backend.
The heavy algorithms are reused from mustatil_legacy_backend.py through a small
compatibility layer so the previous Detection, SAM2, Trainer, FormLearner and
GeoPackage export logic remains available while the UI becomes a QGIS-like
workspace with menu bar, project file and docks.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from PySide6.QtCore import Qt, QTimer, Signal, QObject, QRectF
    from PySide6.QtGui import QAction, QPixmap, QImage, QPainter, QPen, QColor, QBrush, QKeySequence
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
        QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
        QTabWidget, QTextEdit, QDockWidget, QTreeWidget, QTreeWidgetItem, QListWidget,
        QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
        QGraphicsTextItem, QStatusBar, QSpinBox, QDoubleSpinBox,
        QCheckBox, QComboBox, QSplitter, QGroupBox, QProgressBar, QInputDialog,
        QScrollArea, QSlider
    )
except Exception as exc:  # pragma: no cover - used when dependencies are missing
    raise SystemExit(
        "PySide6 is missing. Run INSTALL_QT_DEPENDENCIES.bat or install manually:\n"
        "python -m pip install PySide6 Pillow numpy\n\n" + str(exc)
    )

try:
    from PIL import Image
except Exception as exc:
    raise SystemExit("Pillow is missing: python -m pip install pillow\n" + str(exc))

import mustatil_legacy_backend as backend

APP_NAME = "Mustatil Qt Workspace"
PROJECT_EXT = ".mustatil"
IMG_EXT = getattr(backend, "IMG_EXT", {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
CLASSES = getattr(backend, "CLASSES", ["mustatil", "false_positive"])


class QtSignals(QObject):
    log = Signal(str, str)
    error = Signal(str, str)
    info = Signal(str, str)
    redraw = Signal()
    sam_redraw = Signal()
    form_redraw = Signal()


class Var:
    """Tiny Tk variable adapter used by legacy backend methods."""
    def __init__(self, value: Any = None):
        self._value = value
        self._widgets: List[Any] = []

    def get(self):
        return self._value

    def set(self, value: Any):
        self._value = value
        for w in list(self._widgets):
            try:
                if isinstance(w, QLineEdit):
                    if w.text() != str(value):
                        w.setText(str(value))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(value))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(value))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(value))
                elif isinstance(w, QComboBox):
                    ix = w.findText(str(value))
                    if ix >= 0:
                        w.setCurrentIndex(ix)
                    else:
                        w.setEditText(str(value))
                elif hasattr(w, "_mustatil_set_from_var"):
                    w._mustatil_set_from_var(value)
            except Exception:
                pass

    def bind_widget(self, widget):
        self._widgets.append(widget)
        self.set(self._value)
        return widget

    def __str__(self):
        return str(self._value)


@dataclass
class MustatilProject:
    project_name: str = "Untitled Mustatil Project"
    project_file: str = ""
    project_root: str = ""
    image_folder: str = ""
    label_folder: str = ""
    crop_folder: str = ""
    export_folder: str = ""
    weights_folder: str = ""
    sam_folder: str = ""
    runs_folder: str = ""
    train_output_folder: str = ""
    yolo_models: List[str] = field(default_factory=lambda: ["", "", ""])
    image: str = ""
    output: str = ""
    sam_model: str = "sam2_b.pt"
    train_model: str = "yolov8n.pt"
    formlearner_model: str = ""
    tile_size: int = 1024
    overlap: int = 384
    shifted_tiles: bool = True
    yolo_confidence: float = 0.05
    preview_confidence: float = 0.10
    formscore_preview: float = 0.0
    detection_min_score: float = 0.0
    detection_min_consensus: int = 1
    formscore_threshold: float = 0.50
    crop_size: int = 1024
    crop_padding: int = 128
    epochs: int = 80
    imgsz: int = 640
    batch: int = 2
    device: str = "cpu"
    classes: List[str] = field(default_factory=lambda: list(CLASSES))
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    modified: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    version: int = 1

    @staticmethod
    def from_file(path: Path) -> "MustatilProject":
        d = json.loads(path.read_text(encoding="utf-8"))
        # Also accept older project.json files.
        if "folders" in d and "project_root" not in d:
            root = path.parent
            d = {
                "project_name": root.name,
                "project_file": str(path),
                "project_root": str(root),
                "image_folder": str(root / "images"),
                "label_folder": str(root / "labels"),
                "crop_folder": str(root / "crops"),
                "export_folder": str(root / "exports"),
                "weights_folder": str(root / "weights"),
                "sam_folder": str(root / "sam2"),
                "runs_folder": str(root / "runs"),
                "classes": d.get("classes", CLASSES),
            }
        known = {f.name for f in MustatilProject.__dataclass_fields__.values()}
        clean = {k: v for k, v in d.items() if k in known}
        p = MustatilProject(**clean)
        p.project_file = str(path)
        return p

    def save(self, path: Path):
        self.modified = time.strftime("%Y-%m-%d %H:%M:%S")
        self.project_file = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


class ImageCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.setBackgroundBrush(QBrush(QColor("#f4f4f4")))

    def clear(self):
        self.scene().clear()
        self.pixmap_item = None

    def set_pil_image(self, pil_img: Image.Image):
        if pil_img is None:
            return
        im = pil_img.convert("RGBA")
        data = im.tobytes("raw", "RGBA")
        qimg = QImage(data, im.width, im.height, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg.copy())
        self.scene().clear()
        self.pixmap_item = self.scene().addPixmap(pix)
        self.scene().setSceneRect(0, 0, pix.width(), pix.height())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def add_box(self, x1, y1, x2, y2, color="lime", label=""):
        pen = QPen(QColor(color))
        pen.setWidth(2)
        self.scene().addRect(float(x1), float(y1), float(x2 - x1), float(y2 - y1), pen)
        if label:
            t = self.scene().addText(str(label))
            t.setDefaultTextColor(QColor(color))
            t.setPos(float(x1), max(0, float(y1) - 18))

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)


class DetectionOverviewCanvas(ImageCanvas):
    """Simple detection preview canvas.

    This intentionally does not reload raster pyramid/overview levels while zooming.
    It keeps one normal overview image in memory and only scales the view.
    """
    pass


class AnnotatorCanvas(ImageCanvas):
    """Canvas that lets the user draw annotation rectangles with the mouse."""
    box_drawn = Signal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.NoDrag)
        self._drawing = False
        self._start = None
        self._temp_rect = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap_item is not None:
            self._drawing = True
            self._start = self.mapToScene(event.position().toPoint())
            pen = QPen(QColor("#0057d8"))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            self._temp_rect = self.scene().addRect(QRectF(self._start, self._start), pen)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing and self._temp_rect is not None and self._start is not None:
            cur = self.mapToScene(event.position().toPoint())
            self._temp_rect.setRect(QRectF(self._start, cur).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drawing and event.button() == Qt.LeftButton:
            cur = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._start, cur).normalized()
            if self._temp_rect is not None:
                self.scene().removeItem(self._temp_rect)
            self._drawing = False
            self._start = None
            self._temp_rect = None
            if rect.width() >= 4 and rect.height() >= 4:
                self.box_drawn.emit(rect.left(), rect.top(), rect.right(), rect.bottom())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MustatilQtWorkspace(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1380, 860)
        self.signals = QtSignals()
        self.signals.log.connect(self._append_log)
        self.signals.error.connect(lambda title, txt: QMessageBox.critical(self, title, txt))
        self.signals.info.connect(lambda title, txt: QMessageBox.information(self, title, txt))
        self.signals.redraw.connect(self.redraw)
        self.signals.sam_redraw.connect(self.draw_sam_preview)
        self.signals.form_redraw.connect(self.draw_fl_preview)

        self.project_state = MustatilProject()
        self._autosave_enabled = True
        self._current_project_file: Optional[Path] = None

        self._init_legacy_state()
        self._bind_legacy_backend_methods()
        self._build_ui()
        self._build_menus()
        self._start_autosave_timer()
        self.log(backend.deps())

    # ------------------------------------------------------------------
    # Legacy-compatible state
    # ------------------------------------------------------------------
    def _init_legacy_state(self):
        self.models = [Var(""), Var(""), Var("")]
        self.image = Var("")
        self.output = Var("")
        self.project = Var("")
        self.project_create_dir = Var("")
        self.conf = Var(0.05)
        self.showconf = Var(0.10)
        self.minscore = Var(0.0)
        self.filter_score = Var(0.0)
        self.filter_consensus = Var(1)
        self.tile = Var(1024)
        self.overlap = Var(384)
        self.shift = Var(True)
        self.cropdir = Var("")
        self.cropsize = Var(1024)
        self.pad = Var(128)
        self.startcls = Var(0)
        self.sammodel = Var("sam2_b.pt")
        self.sam_source_dir = Var("")
        self.sam_out = Var("")
        self.sam_padding = Var(96)
        self.sam_max_crop = Var(1024)
        self.sam_use_ann_boxes = Var(True)
        self.sam_skip_existing = Var(True)
        self.trainmodel = Var("yolov8n.pt")
        self.train_output_dir = Var("")
        self.epochs = Var(80)
        self.imgsz = Var(640)
        self.batch = Var(2)
        self.device = Var("cpu")
        self.low_ram_mode = Var(False)
        self.auto_resume = Var(True)
        self.train_chunk_enabled = Var(True)
        self.train_chunk_size = Var(1024)
        self.train_chunk_overlap = Var(128)
        self.train_chunk_min_visible = Var(0.35)
        self.train_keep_negative_chunks = Var(True)
        self.form_project = Var("")
        self.form_model_path = Var("")
        self.form_epochs = Var(1200)
        self.fl_model_path = Var("")
        self.fl_threshold = Var(0.50)
        self.fl_output = Var("")
        self.ann_cls = Var(0)

        self.dets = []
        self.fl_kept = []
        self.fl_scored = []
        self.sam_images: List[Path] = []
        self.sam_selected_img: Optional[Path] = None
        self.sam_current_polys = []
        self.sam_polys = []
        self.sam_prompt_boxes = []
        self.preview = None
        self.origW = self.origH = 1
        self.det_preview_max = Var(2400)
        self.det_preview_zoom = 1.0
        self.det_preview_source = ""
        self.last_img = None
        self.last_geo = ("pixel", None, None, None)
        self.fl_preview_img = None
        self.sam_preview_img = None
        self.ann_imgs = []
        self.ann_i = 0
        self.ann_boxes = []
        self.ann_manifest = {}
        self.ann_selected = -1

    def _bind_legacy_backend_methods(self):
        import types
        # Pure/backend methods from the old Tk class. UI-specific methods are reimplemented in Qt below.
        names = [
            "score", "detect", "export_crops",
            "_ensure_project_folders", "_image_files_recursive", "_label_candidates_for_image",
            "_preferred_label_path_for_image", "_find_existing_label_for_image",
            "sync_reviewed_annotations_for_training", "yolo_clip_box_for_tile",
            "save_yolo_boxes_tile", "prepare_yolo_dataset",
            "_ensure_geo_from_current_image",
            "_feature_crs_name", "_feature_crs_wkt", "_px_to_map", "_dets_to_features",
            "_write_features_auto", "_safe_name", "export_geo_by_model",
            "export_formlearner_current", "export_geo", "train_formlearner", "detect_with_formlearner",
            "update_crop_manifest_status", "current_label_path", "saveann", "nextann",
        ]
        for name in names:
            if hasattr(backend.GUI, name):
                setattr(self, name, types.MethodType(getattr(backend.GUI, name), self))

    # ------------------------------------------------------------------
    # UI build helpers
    # ------------------------------------------------------------------
    def _var_line(self, var: Var, browse=None, file_filter="All files (*)"):
        w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(); var.bind_widget(edit)
        edit.textChanged.connect(var.set)
        lay.addWidget(edit, 1)
        if browse:
            btn = QPushButton("…")
            btn.clicked.connect(lambda: browse(var, file_filter))
            lay.addWidget(btn)
        return w

    def _spin(self, var: Var, minv=0, maxv=999999, step=1):
        s = QSpinBox(); s.setRange(minv, maxv); s.setSingleStep(step); var.bind_widget(s); s.valueChanged.connect(var.set); return s

    def _dspin(self, var: Var, minv=0.0, maxv=1.0, step=0.01, decimals=3):
        s = QDoubleSpinBox(); s.setRange(minv, maxv); s.setDecimals(decimals); s.setSingleStep(step); var.bind_widget(s); s.valueChanged.connect(var.set); return s

    def _filter_slider_float(self, var: Var, minv=0.0, maxv=1.0, step=0.01, decimals=2):
        box = QWidget(); lay = QHBoxLayout(box); lay.setContentsMargins(0, 0, 0, 0)
        scale = int(round((maxv - minv) / step))
        slider = QSlider(Qt.Horizontal); slider.setRange(0, scale)
        label = QLabel()
        def set_from_var(value=None):
            try:
                val = float(var.get())
            except Exception:
                val = minv
            val = max(minv, min(maxv, val))
            pos = int(round((val - minv) / step))
            if slider.value() != pos:
                slider.setValue(pos)
            label.setText(f"{val:.{decimals}f}")
        def changed(pos):
            val = minv + pos * step
            var.set(round(val, decimals + 1))
            label.setText(f"{float(var.get()):.{decimals}f}")
            self.redraw()
            try:
                self.draw_fl_preview()
            except Exception:
                pass
        slider.valueChanged.connect(changed)
        class _FloatSliderBinder:
            def _mustatil_set_from_var(self_inner, value):
                try:
                    val = max(minv, min(maxv, float(value)))
                    pos = int(round((val - minv) / step))
                    if slider.value() != pos:
                        old = slider.blockSignals(True)
                        slider.setValue(pos)
                        slider.blockSignals(old)
                    label.setText(f"{val:.{decimals}f}")
                except Exception:
                    pass
        var._widgets.append(_FloatSliderBinder())
        set_from_var()
        lay.addWidget(slider, 1); lay.addWidget(label)
        return box

    def _filter_slider_int(self, var: Var, minv=1, maxv=3):
        box = QWidget(); lay = QHBoxLayout(box); lay.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Horizontal); slider.setRange(minv, maxv)
        label = QLabel()
        def changed(pos):
            var.set(int(pos))
            label.setText(str(int(var.get())))
            self.redraw()
        slider.valueChanged.connect(changed)
        class _IntSliderBinder:
            def _mustatil_set_from_var(self_inner, value):
                try:
                    val = max(minv, min(maxv, int(value)))
                    if slider.value() != val:
                        old = slider.blockSignals(True)
                        slider.setValue(val)
                        slider.blockSignals(old)
                    label.setText(str(val))
                except Exception:
                    pass
        var._widgets.append(_IntSliderBinder())
        try:
            slider.setValue(int(var.get()))
        except Exception:
            slider.setValue(minv)
        label.setText(str(slider.value()))
        lay.addWidget(slider, 1); lay.addWidget(label)
        return box

    def _check(self, text, var: Var):
        c = QCheckBox(text); var.bind_widget(c); c.toggled.connect(var.set); return c

    def _scroll(self, child: QWidget):
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(child); return sc

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_detection_tab()
        self._build_annotator_tab()
        self._build_trainer_tab()
        self._build_sam_tab()
        self._build_formtrainer_tab()
        # FormLearner Detection is integrated into the Detection tab.
        self._build_docks()
        self.setStatusBar(QStatusBar(self))

    def _build_detection_tab(self):
        page = QWidget(); root = QSplitter(Qt.Horizontal); layout = QHBoxLayout(page); layout.addWidget(root)
        controls = QWidget(); form = QVBoxLayout(controls)

        gb = QGroupBox("YOLO models")
        g = QGridLayout(gb)
        for i, v in enumerate(self.models):
            g.addWidget(QLabel(f"Model {i+1}"), i, 0)
            g.addWidget(self._var_line(v, self.browse_file, "YOLO models (*.pt *.onnx);;All files (*)"), i, 1)
        form.addWidget(gb)

        imgbox = QGroupBox("Image, tiling and filters")
        f = QFormLayout(imgbox)
        f.addRow("Large image", self._var_line(self.image, self.browse_file, "Images (*.tif *.tiff *.jpg *.jpeg *.png *.bmp *.webp);;All files (*)"))
        f.addRow("YOLO compute confidence", self._dspin(self.conf, 0.001, 1, 0.01))
        f.addRow("Display confidence", self._dspin(self.showconf, 0.001, 1, 0.01))
        f.addRow("Preview FormScore", self._dspin(self.minscore, 0, 1, 0.01))
        f.addRow("Tile size", self._spin(self.tile, 64, 8192, 64))
        f.addRow("Overlap", self._spin(self.overlap, 0, 4096, 16))
        f.addRow("", self._check("shifted tiles", self.shift))
        form.addWidget(imgbox)

        row = QHBoxLayout()
        b = QPushButton("Load overview preview"); b.clicked.connect(self.loadprev); row.addWidget(b)
        b = QPushButton("Start tiled detection"); b.clicked.connect(lambda: self.run_task("Detection", self.detect)); row.addWidget(b)
        form.addLayout(row)
        overview_hint = QLabel("Detection preview uses a normal single overview image. Mouse wheel zoom only scales the displayed preview; it does not reload pyramids/overviews.")
        overview_hint.setWordWrap(True)
        form.addWidget(overview_hint)

        postbox = QGroupBox("Post-detection filter sliders")
        pf = QFormLayout(postbox)
        pf.addRow("Confidence filter", self._filter_slider_float(self.showconf, 0.0, 1.0, 0.01, 2))
        pf.addRow("FormScore filter", self._filter_slider_float(self.minscore, 0.0, 1.0, 0.01, 2))
        pf.addRow("Ensemble score filter", self._filter_slider_float(self.filter_score, 0.0, 4.0, 0.05, 2))
        pf.addRow("Minimum consensus", self._filter_slider_int(self.filter_consensus, 1, 3))
        hint = QLabel("These sliders only filter already computed detections/exported visible results. They do not start YOLO again.")
        hint.setWordWrap(True)
        pf.addRow("", hint)
        form.addWidget(postbox)

        cropbox = QGroupBox("Crops and exports")
        cf = QFormLayout(cropbox)
        cf.addRow("Crop folder", self._var_line(self.cropdir, self.browse_dir))
        cf.addRow("Crop size", self._spin(self.cropsize, 64, 8192, 64))
        cf.addRow("Padding", self._spin(self.pad, 0, 4096, 16))
        cf.addRow("Output", self._var_line(self.output, self.save_file, "GeoPackage (*.gpkg);;GeoJSON (*.geojson);;All files (*)"))
        form.addWidget(cropbox)
        r2 = QHBoxLayout()
        for text, func in [
            ("Export visible crops", self.export_crops),
            ("Export visible GeoJSON/GPKG", self.export_geo),
            ("Export by model", self.export_geo_by_model),
        ]:
            btn = QPushButton(text); btn.clicked.connect(lambda _=False, fn=func, label=text: self.run_task(label, fn)); r2.addWidget(btn)
        form.addLayout(r2)

        flbox = QGroupBox("FormLearner Detection")
        fl = QFormLayout(flbox)
        fl.addRow("FormLearner model", self._var_line(self.fl_model_path, self.browse_file, "JSON (*.json);;All files (*)"))
        fl.addRow("Minimum FormScore", self._dspin(self.fl_threshold, 0, 1, 0.01))
        fl.addRow("Output", self._var_line(self.fl_output, self.save_file, "GeoPackage (*.gpkg);;GeoJSON (*.geojson);;All files (*)"))
        form.addWidget(flbox)
        r3 = QHBoxLayout()
        b = QPushButton("Start FormLearner Detection")
        b.clicked.connect(lambda: self.run_task("FormLearner Detection", self.detect_with_formlearner))
        r3.addWidget(b)
        b = QPushButton("Export current FormScore filter")
        b.clicked.connect(lambda: self.run_task("Export FormLearner", self.export_formlearner_current))
        r3.addWidget(b)
        form.addLayout(r3)

        self.fl_log = QTextEdit(); self.fl_log.setReadOnly(True); self.fl_log.setMaximumHeight(150)
        form.addWidget(QLabel("FormLearner Log"))
        form.addWidget(self.fl_log)
        form.addStretch(1)

        self.image_view = DetectionOverviewCanvas()
        root.addWidget(self._scroll(controls)); root.addWidget(self.image_view); root.setSizes([430, 900])
        self.tabs.addTab(page, "Detection")

    def _build_annotator_tab(self):
        page = QWidget(); spl = QSplitter(Qt.Horizontal); lay = QHBoxLayout(page); lay.addWidget(spl)
        left = QWidget(); l = QVBoxLayout(left)
        project_box = QGroupBox("Project / image folder")
        pf = QFormLayout(project_box)
        pf.addRow("Project root", self._var_line(self.project, self.browse_dir))
        btnrow = QHBoxLayout()
        for text, func in [("Create folder", self.create_new_project_folder_dialog), ("Load project images", self.loadproj), ("Save annotation", self.saveann)]:
            b = QPushButton(text); b.clicked.connect(func); btnrow.addWidget(b)
        pf.addRow(btnrow)
        l.addWidget(project_box)

        self.ann_list_qt = QListWidget()
        self.ann_list_qt.currentRowChanged.connect(self.on_ann_image_select_qt)
        l.addWidget(QLabel("Project images"))
        l.addWidget(self.ann_list_qt, 1)

        nav = QHBoxLayout()
        prev = QPushButton("<"); prev.clicked.connect(lambda: self.nextann(-1)); nav.addWidget(prev)
        nxt = QPushButton(">"); nxt.clicked.connect(lambda: self.nextann(1)); nav.addWidget(nxt)
        pos = QPushButton("Image = POSITIVE"); pos.clicked.connect(lambda: self.set_image_class_qt(0)); nav.addWidget(pos)
        neg = QPushButton("Image = FALSE"); neg.clicked.connect(lambda: self.set_image_class_qt(1)); nav.addWidget(neg)
        l.addLayout(nav)

        ann_box = QGroupBox("Annotations for current image")
        ann_lay = QVBoxLayout(ann_box)
        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Class for selected/new box"))
        self.ann_class_combo = QComboBox()
        self.ann_class_combo.addItems(["0 - POSITIVE / mustatil", "1 - FALSE positive"])
        self.ann_class_combo.currentIndexChanged.connect(lambda ix: self.ann_cls.set(int(ix)))
        class_row.addWidget(self.ann_class_combo, 1)
        ann_lay.addLayout(class_row)
        self.ann_box_list_qt = QListWidget()
        self.ann_box_list_qt.currentRowChanged.connect(self.on_ann_box_select_qt)
        ann_lay.addWidget(self.ann_box_list_qt, 1)
        btn_grid = QGridLayout()
        buttons = [
            ("Set selected class", self.set_selected_cls_qt, 0, 0),
            ("Delete selected box", self.delete_selected_box_qt, 0, 1),
            ("Delete all / neutral", self.clear_current_ann_qt, 1, 0),
            ("Reset from manifest box", self.reset_from_manifest_qt, 1, 1),
            ("Save annotation", self.saveann, 2, 0),
            ("Reload current", self.showann, 2, 1),
        ]
        for text, func, row, col in buttons:
            b = QPushButton(text); b.clicked.connect(func); btn_grid.addWidget(b, row, col)
        ann_lay.addLayout(btn_grid)
        l.addWidget(ann_box, 2)

        sam = QPushButton("SAM2 segment current crop")
        sam.clicked.connect(lambda: self.run_task("SAM2 current crop", self.sam2_current_annotation))
        l.addWidget(sam)
        sam_all_annotator = QPushButton("SAM2 segment all annotated images")
        sam_all_annotator.clicked.connect(lambda: self.run_task("SAM2 all annotated images", self.sam2_all_annotation_project))
        l.addWidget(sam_all_annotator)
        self.ann_status_qt = QLabel("No annotation loaded"); self.ann_status_qt.setWordWrap(True); l.addWidget(self.ann_status_qt)
        self.annotator_view = AnnotatorCanvas()
        self.annotator_view.box_drawn.connect(self.add_annotation_box_from_scene_qt)
        spl.addWidget(left); spl.addWidget(self.annotator_view); spl.setSizes([500, 900])
        self.tabs.addTab(page, "Annotator")

    def _build_trainer_tab(self):
        page = QWidget(); form = QVBoxLayout(page)
        gb = QGroupBox("YOLO Trainer")
        f = QFormLayout(gb)
        f.addRow("Project", self._var_line(self.project, self.browse_dir))
        f.addRow("Base model", self._var_line(self.trainmodel, self.browse_file, "YOLO models (*.pt);;All files (*)"))
        f.addRow("Model output folder", self._var_line(self.train_output_dir, self.browse_dir))
        f.addRow("Epochs", self._spin(self.epochs, 1, 10000, 1))
        f.addRow("Image size", self._spin(self.imgsz, 64, 4096, 32))
        f.addRow("Batch", self._spin(self.batch, 1, 256, 1))
        device = QComboBox(); device.setEditable(True); device.addItems(["cpu", "directml", "cuda", "0"]); self.device.bind_widget(device); device.currentTextChanged.connect(self.device.set); f.addRow("Device", device)
        f.addRow("", self._check("Low-RAM stable mode", self.low_ram_mode))
        f.addRow("", self._check("Auto-resume", self.auto_resume))
        form.addWidget(gb)
        chunk = QGroupBox("Training image chunking")
        cf = QFormLayout(chunk)
        cf.addRow("", self._check("Cut huge maps into chunks", self.train_chunk_enabled))
        cf.addRow("Chunk size", self._spin(self.train_chunk_size, 64, 8192, 64))
        cf.addRow("Overlap", self._spin(self.train_chunk_overlap, 0, 4096, 16))
        cf.addRow("Min visible label fraction", self._dspin(self.train_chunk_min_visible, 0, 1, 0.05))
        cf.addRow("", self._check("Keep empty/negative chunks", self.train_keep_negative_chunks))
        form.addWidget(chunk)
        buttons = QHBoxLayout()
        for text, func in [("Prepare YOLO Dataset", self.prepare_yolo_dataset), ("Train YOLO", self.train), ("Resume", lambda: self.train(resume=True)), ("Export ONNX", self.export_onnx)]:
            b = QPushButton(text); b.clicked.connect(lambda _=False, fn=func: self.run_task(text, fn)); buttons.addWidget(b)
        form.addLayout(buttons)
        self.train_log = QTextEdit(); self.train_log.setReadOnly(True); form.addWidget(self.train_log, 1)
        self.tabs.addTab(page, "YOLO Trainer")

    def _build_sam_tab(self):
        page = QWidget(); spl = QSplitter(Qt.Horizontal); lay = QHBoxLayout(page); lay.addWidget(spl)
        left = QWidget(); l = QVBoxLayout(left)
        gb = QGroupBox("SAM2")
        f = QFormLayout(gb)
        f.addRow("SAM2 model", self._var_line(self.sammodel, self.browse_file, "PyTorch models (*.pt);;All files (*)"))
        f.addRow("Image folder", self._var_line(self.sam_source_dir, self.browse_dir))
        f.addRow("Output JSON", self._var_line(self.sam_out, self.save_file, "JSON (*.json);;All files (*)"))
        f.addRow("Padding", self._spin(self.sam_padding, 0, 4096, 16))
        f.addRow("Max crop", self._spin(self.sam_max_crop, 64, 8192, 64))
        f.addRow("", self._check("Use boxes/annotations as prompts", self.sam_use_ann_boxes))
        f.addRow("", self._check("Skip existing .sam2.json", self.sam_skip_existing))
        l.addWidget(gb)
        bload = QPushButton("Load image list"); bload.clicked.connect(self.load_sam_images); l.addWidget(bload)
        self.sam_list_qt = QListWidget(); self.sam_list_qt.currentRowChanged.connect(self.on_sam_select_qt); l.addWidget(self.sam_list_qt, 1)
        brow = QHBoxLayout()
        one = QPushButton("Segment selected"); one.clicked.connect(lambda: self.run_task("SAM2 selected", self.sam2_selected)); brow.addWidget(one)
        allb = QPushButton("Segment all"); allb.clicked.connect(lambda: self.run_task("SAM2 all", self.sam2_all)); brow.addWidget(allb)
        l.addLayout(brow)
        self.sam_view = ImageCanvas(); spl.addWidget(left); spl.addWidget(self.sam_view); spl.setSizes([420, 900])
        self.tabs.addTab(page, "SAM2")

    def _build_formtrainer_tab(self):
        page = QWidget(); form = QVBoxLayout(page)
        gb = QGroupBox("FormTrainer")
        f = QFormLayout(gb)
        f.addRow("Training project", self._var_line(self.form_project, self.browse_dir))
        f.addRow("Output model JSON", self._var_line(self.form_model_path, self.save_file, "JSON (*.json);;All files (*)"))
        f.addRow("Epochs", self._spin(self.form_epochs, 10, 20000, 10))
        form.addWidget(gb)
        b = QPushButton("Start FormTrainer"); b.clicked.connect(lambda: self.run_task("FormTrainer", self.train_formlearner)); form.addWidget(b)
        self.form_log = QTextEdit(); self.form_log.setReadOnly(True); form.addWidget(self.form_log, 1)
        self.tabs.addTab(page, "FormTrainer")

    def _build_formdetect_tab(self):
        page = QWidget(); spl = QSplitter(Qt.Horizontal); lay = QHBoxLayout(page); lay.addWidget(spl)
        left = QWidget(); l = QVBoxLayout(left)
        gb = QGroupBox("Detection with FormLearner")
        f = QFormLayout(gb)
        f.addRow("FormLearner model", self._var_line(self.fl_model_path, self.browse_file, "JSON (*.json);;All files (*)"))
        f.addRow("Minimum FormScore", self._dspin(self.fl_threshold, 0, 1, 0.01))
        f.addRow("Output", self._var_line(self.fl_output, self.save_file, "GeoPackage (*.gpkg);;GeoJSON (*.geojson);;All files (*)"))
        l.addWidget(gb)
        b = QPushButton("Start detection with FormLearner"); b.clicked.connect(lambda: self.run_task("Detection with FormLearner", self.detect_with_formlearner)); l.addWidget(b)
        b2 = QPushButton("Export current FormScore filter"); b2.clicked.connect(lambda: self.run_task("Export FormLearner", self.export_formlearner_current)); l.addWidget(b2)
        self.fl_log = QTextEdit(); self.fl_log.setReadOnly(True); l.addWidget(self.fl_log, 1)
        self.fl_view = ImageCanvas(); spl.addWidget(left); spl.addWidget(self.fl_view); spl.setSizes([420, 900])
        self.tabs.addTab(page, "FormLearner Detection")

    def _build_docks(self):
        self.project_tree = QTreeWidget(); self.project_tree.setHeaderLabels(["Project"]); self.project_tree.itemDoubleClicked.connect(self._project_tree_open)
        d = QDockWidget("Project Browser", self); d.setWidget(self.project_tree); self.addDockWidget(Qt.LeftDockWidgetArea, d)
        self.layer_tree = QTreeWidget(); self.layer_tree.setHeaderLabels(["Layers"])
        ld = QDockWidget("Layers", self); ld.setWidget(self.layer_tree); self.addDockWidget(Qt.LeftDockWidgetArea, ld)
        self.general_log = QTextEdit(); self.general_log.setReadOnly(True)
        logdock = QDockWidget("Console / Logs", self); logdock.setWidget(self.general_log); self.addDockWidget(Qt.BottomDockWidgetArea, logdock)
        self.task_progress = QProgressBar(); self.task_progress.setRange(0, 0); self.task_progress.hide()
        self.statusBar().addPermanentWidget(self.task_progress)
        self.refresh_project_browser()
        self.refresh_layers()

    def _build_menus(self):
        mb = self.menuBar()
        filem = mb.addMenu("File")
        self._act(filem, "New", self.new_project, QKeySequence.New)
        self._act(filem, "Open Project…", self.open_project, QKeySequence.Open)
        self._act(filem, "Save Project", self.save_project, QKeySequence.Save)
        self._act(filem, "Save Project As…", self.save_project_as, QKeySequence.SaveAs)
        filem.addSeparator(); self._act(filem, "Exit", self.close)
        pm = mb.addMenu("Projekt")
        self._act(pm, "Create Project Folder…", self.create_new_project_folder_dialog)
        self._act(pm, "Refresh Project Browser", self.refresh_project_browser)
        dm = mb.addMenu("Detection")
        self._act(dm, "Load Preview", self.loadprev)
        self._act(dm, "Start Tiled Detection", lambda: self.run_task("Detection", self.detect))
        self._act(dm, "Start FormLearner Detection", lambda: self.run_task("FormLearner Detection", self.detect_with_formlearner))
        self._act(dm, "Export Crops", lambda: self.run_task("Crops", self.export_crops))
        sm = mb.addMenu("SAM2")
        self._act(sm, "Load Image List", self.load_sam_images)
        self._act(sm, "Segment Selected Image", lambda: self.run_task("SAM2 selected", self.sam2_selected))
        self._act(sm, "Segment All Images", lambda: self.run_task("SAM2 all", self.sam2_all))
        tm = mb.addMenu("Training")
        self._act(tm, "Prepare YOLO Dataset", lambda: self.run_task("Prepare YOLO", self.prepare_yolo_dataset))
        self._act(tm, "Train YOLO", lambda: self.run_task("Train YOLO", self.train))
        self._act(tm, "Start FormTrainer", lambda: self.run_task("FormTrainer", self.train_formlearner))
        em = mb.addMenu("Export")
        self._act(em, "Visible GeoJSON/GPKG", lambda: self.run_task("Export", self.export_geo))
        self._act(em, "Per Model", lambda: self.run_task("Export by model", self.export_geo_by_model))
        hm = mb.addMenu("Help")
        self._act(hm, "Dependency Check", lambda: QMessageBox.information(self, "Dependency Check", backend.deps()))

    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    # ------------------------------------------------------------------
    # Project handling
    # ------------------------------------------------------------------
    def project_from_vars(self) -> MustatilProject:
        root = Path(self.project.get()).expanduser() if self.project.get() else Path("")
        p = self.project_state
        p.project_root = str(root) if str(root) != "." else ""
        p.project_name = root.name if p.project_root else p.project_name
        p.yolo_models = [str(v.get() or "") for v in self.models]
        p.image = str(self.image.get() or "")
        p.output = str(self.output.get() or "")
        p.crop_folder = str(self.cropdir.get() or (root / "crops" if p.project_root else ""))
        p.image_folder = str(root / "images") if p.project_root and not p.image_folder else str(p.image_folder or "")
        p.label_folder = str(root / "labels") if p.project_root and not p.label_folder else str(p.label_folder or "")
        p.export_folder = str(root / "exports") if p.project_root and not p.export_folder else str(p.export_folder or "")
        p.weights_folder = str(root / "weights") if p.project_root and not p.weights_folder else str(p.weights_folder or "")
        p.sam_folder = str(root / "sam2") if p.project_root and not p.sam_folder else str(p.sam_folder or "")
        p.runs_folder = str(root / "runs") if p.project_root and not p.runs_folder else str(p.runs_folder or "")
        p.train_output_folder = str(self.train_output_dir.get() or p.runs_folder or (root / "runs" if p.project_root else ""))
        p.sam_model = str(self.sammodel.get() or "")
        p.train_model = str(self.trainmodel.get() or "")
        p.formlearner_model = str(self.fl_model_path.get() or self.form_model_path.get() or "")
        p.tile_size = int(self.tile.get())
        p.overlap = int(self.overlap.get())
        p.shifted_tiles = bool(self.shift.get())
        p.yolo_confidence = float(self.conf.get())
        p.preview_confidence = float(self.showconf.get())
        p.formscore_preview = float(self.minscore.get())
        p.detection_min_score = float(self.filter_score.get())
        p.detection_min_consensus = int(self.filter_consensus.get())
        p.formscore_threshold = float(self.fl_threshold.get())
        p.crop_size = int(self.cropsize.get())
        p.crop_padding = int(self.pad.get())
        p.epochs = int(self.epochs.get())
        p.imgsz = int(self.imgsz.get())
        p.batch = int(self.batch.get())
        p.device = str(self.device.get())
        return p

    def apply_project(self, p: MustatilProject):
        self.project_state = p
        self._current_project_file = Path(p.project_file) if p.project_file else None
        self.project.set(p.project_root)
        self.project_create_dir.set(p.project_root)
        self.form_project.set(p.project_root)
        self.image.set(p.image)
        self.output.set(p.output or (str(Path(p.project_root) / "exports" / "detections.gpkg") if p.project_root else ""))
        self.cropdir.set(p.crop_folder or (str(Path(p.project_root) / "crops") if p.project_root else ""))
        for i, m in enumerate(p.yolo_models[:3]):
            self.models[i].set(m)
        self.sammodel.set(p.sam_model)
        self.trainmodel.set(p.train_model)
        self.train_output_dir.set(getattr(p, "train_output_folder", "") or getattr(p, "runs_folder", "") or (str(Path(p.project_root) / "runs") if p.project_root else ""))
        self.fl_model_path.set(p.formlearner_model)
        self.form_model_path.set(p.formlearner_model)
        self.tile.set(p.tile_size); self.overlap.set(p.overlap); self.shift.set(p.shifted_tiles)
        self.conf.set(p.yolo_confidence); self.showconf.set(p.preview_confidence); self.minscore.set(p.formscore_preview)
        self.filter_score.set(getattr(p, "detection_min_score", 0.0)); self.filter_consensus.set(getattr(p, "detection_min_consensus", 1))
        self.fl_threshold.set(p.formscore_threshold); self.cropsize.set(p.crop_size); self.pad.set(p.crop_padding)
        self.epochs.set(p.epochs); self.imgsz.set(p.imgsz); self.batch.set(p.batch); self.device.set(p.device)
        if p.project_root:
            self.sam_source_dir.set(str(Path(p.project_root) / "images"))
        self.refresh_project_browser(); self.refresh_layers(); self.statusBar().showMessage(f"Project loaded: {p.project_name}", 5000)

    def new_project(self):
        parent = QFileDialog.getExistingDirectory(self, "Choose parent folder")
        if not parent:
            return
        name, ok = QInputDialog.getText(self, "New Mustatil project", "Project folder name:", text="Mustatil_Project")
        if not ok or not name.strip():
            return
        safe = "".join(ch for ch in name.strip() if ch not in '<>:"/\\|?*').strip() or "Mustatil_Project"
        root = Path(parent) / safe
        self._ensure_project_folders(root)
        p = MustatilProject(project_name=safe, project_root=str(root), project_file=str(root / f"{safe}{PROJECT_EXT}"))
        p.image_folder = str(root / "images"); p.label_folder = str(root / "labels"); p.crop_folder = str(root / "crops")
        p.export_folder = str(root / "exports"); p.weights_folder = str(root / "weights"); p.sam_folder = str(root / "sam2"); p.runs_folder = str(root / "runs")
        p.output = str(root / "exports" / "detections.gpkg")
        p.train_output_folder = str(root / "runs")
        p.save(Path(p.project_file))
        self.apply_project(p)
        self.log(f"New project created: {p.project_file}")

    def open_project(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open Mustatil project", "", f"Mustatil Project (*{PROJECT_EXT} *.json);;All files (*)")
        if not fn:
            return
        try:
            self.apply_project(MustatilProject.from_file(Path(fn)))
            self.log(f"Opened project: {fn}")
        except Exception as exc:
            self.show_error(APP_NAME, str(exc))

    def save_project(self):
        if not self._current_project_file:
            return self.save_project_as()
        p = self.project_from_vars()
        p.save(self._current_project_file)
        self.log(f"Project saved: {self._current_project_file}")
        self.refresh_project_browser()

    def save_project_as(self):
        start = str(Path(self.project.get() or ".") / f"{Path(self.project.get() or 'Mustatil_Project').name}{PROJECT_EXT}")
        fn, _ = QFileDialog.getSaveFileName(self, "Save Mustatil project", start, f"Mustatil Project (*{PROJECT_EXT});;JSON (*.json);;All files (*)")
        if not fn:
            return
        if not fn.lower().endswith((PROJECT_EXT, ".json")):
            fn += PROJECT_EXT
        self._current_project_file = Path(fn)
        self.project_state.project_file = fn
        self.save_project()

    def _start_autosave_timer(self):
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(120_000)
        self.autosave_timer.timeout.connect(self.autosave_project)
        self.autosave_timer.start()

    def autosave_project(self):
        if not self._autosave_enabled:
            return
        try:
            root = Path(self.project.get()) if self.project.get() else None
            if not root:
                return
            p = self.project_from_vars()
            p.save(root / ".autosave.mustatil")
            self.log("Autosave written: .autosave.mustatil")
        except Exception as exc:
            self.log("Autosave failed: " + str(exc))

    # ------------------------------------------------------------------
    # Qt replacements for UI-specific legacy methods
    # ------------------------------------------------------------------
    def visible(self):
        out = []
        try:
            conf_thr = float(self.showconf.get())
        except Exception:
            conf_thr = 0.0
        try:
            form_thr = float(self.minscore.get())
        except Exception:
            form_thr = 0.0
        try:
            score_thr = float(self.filter_score.get())
        except Exception:
            score_thr = 0.0
        try:
            consensus_thr = int(self.filter_consensus.get())
        except Exception:
            consensus_thr = 1
        for d in self.dets:
            if float(getattr(d, "conf", 0.0)) < conf_thr:
                continue
            if float(getattr(d, "score", 0.0)) < score_thr:
                continue
            if int(getattr(d, "consensus", 1)) < consensus_thr:
                continue
            fs = getattr(d, "form_score", None)
            if fs is not None and float(fs) < form_thr:
                continue
            out.append(d)
        return out

    def computed_candidates(self):
        return list(self.dets)

    def loadprev(self):
        try:
            path = str(self.image.get() or "").strip()
            if not path:
                raise RuntimeError("No image selected.")
            # Normal overview preview: one fixed-size image, no pyramid reload while zooming.
            # For large TIFF/GeoTIFF files, backend.load_preview creates a display-sized
            # overview only once; wheel zoom then scales this preview in Qt.
            self.preview, self.origW, self.origH = backend.load_preview(path, maxs=2400)
            self.det_preview_source = path
            self.last_img = Path(path)
            self.redraw(fit=True)
            self.log(f"Overview preview loaded: {self.preview.size}; original {self.origW}x{self.origH}")
        except Exception as exc:
            self.show_error(APP_NAME, str(exc))

    def redraw(self, fit: bool = False):
        try:
            if self.preview is None:
                return
            # Preserve current view transform during filter changes; fit only after an explicit reload.
            old_transform = self.image_view.transform() if not fit else None
            self.image_view.set_pil_image(self.preview)
            if old_transform is not None:
                self.image_view.setTransform(old_transform)
            sx = self.preview.width / max(1, self.origW)
            sy = self.preview.height / max(1, self.origH)
            colors = ["lime", "cyan", "magenta", "yellow", "orange", "red"]
            for d in self.visible():
                col = "white" if getattr(d, "consensus", 1) > 1 else colors[getattr(d, "slot", 0) % len(colors)]
                label = f"M{getattr(d,'slot',0)+1} C{getattr(d,'conf',0):.2f}"
                if getattr(d, "form_score", None) is not None:
                    label += f" F{float(getattr(d, 'form_score')):.2f}"
                self.image_view.add_box(d.x1 * sx, d.y1 * sy, d.x2 * sx, d.y2 * sy, col, label)
            self.refresh_layers()
        except Exception as exc:
            self.log("Redraw error: " + str(exc))

    def loadproj(self):
        try:
            root = Path(self.project.get())
            (root / "labels").mkdir(parents=True, exist_ok=True)
            imgdir = root / "images"
            if not imgdir.exists():
                imgdir = root
            self.ann_imgs = self._image_files_recursive(imgdir)
            self.ann_i = 0
            self.ann_selected = -1
            self.ann_manifest = {}
            mf = root / "crop_manifest.json"
            if mf.exists():
                try:
                    data = json.loads(mf.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        self.ann_manifest = {d.get("image"): d for d in data if isinstance(d, dict) and d.get("image")}
                except Exception as exc:
                    self.log("Manifest could not be read: " + str(exc))
            self.ann_list_qt.blockSignals(True)
            self.ann_list_qt.clear()
            for p in self.ann_imgs:
                self.ann_list_qt.addItem(p.name)
            self.ann_list_qt.blockSignals(False)
            self.showann()
            self.log(f"Loaded annotator project images: {len(self.ann_imgs)}")
        except Exception as exc:
            self.show_error(APP_NAME, str(exc))

    def on_ann_image_select_qt(self, row: int):
        if 0 <= row < len(self.ann_imgs) and row != self.ann_i:
            try:
                self.saveann()
            except Exception:
                pass
            self.ann_i = row
            self.ann_selected = -1
            self.showann()

    def manifest_bbox_for_current_qt(self):
        if not self.ann_imgs:
            return None
        img = Path(self.ann_imgs[self.ann_i])
        m = getattr(self, "ann_manifest", {}).get(img.name, {}) or {}
        bb = m.get("bbox_crop") or m.get("bbox")
        if bb and len(bb) >= 4:
            return [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
        try:
            W, H = Image.open(img).size
        except Exception:
            W = H = 1024
        pad = max(20, int(min(W, H) * 0.12))
        return [pad, pad, W - pad, H - pad]

    def refresh_ann_list_qt(self):
        if not hasattr(self, "ann_box_list_qt"):
            return
        self.ann_box_list_qt.blockSignals(True)
        self.ann_box_list_qt.clear()
        for i, b in enumerate(getattr(self, "ann_boxes", [])):
            c, x1, y1, x2, y2 = b
            name = "POSITIVE / mustatil" if int(c) == 0 else "FALSE positive"
            self.ann_box_list_qt.addItem(f"{i+1:02d} | {name} | x={x1:.0f} y={y1:.0f} w={x2-x1:.0f} h={y2-y1:.0f}")
        if 0 <= getattr(self, "ann_selected", -1) < len(getattr(self, "ann_boxes", [])):
            self.ann_box_list_qt.setCurrentRow(self.ann_selected)
        self.ann_box_list_qt.blockSignals(False)

    def on_ann_box_select_qt(self, row: int):
        self.ann_selected = int(row) if 0 <= row < len(getattr(self, "ann_boxes", [])) else -1
        self.draw_current_annotation_qt()

    def draw_current_annotation_qt(self):
        if not getattr(self, "ann_imgs", None):
            return
        try:
            img = Path(self.ann_imgs[self.ann_i])
            pil, W, H = backend.load_preview(str(img), maxs=2400)
            self.annotator_view.set_pil_image(pil)
            sx = pil.width / max(1, W); sy = pil.height / max(1, H)
            self.ann_preview_sx = sx
            self.ann_preview_sy = sy
            self.ann_preview_w = pil.width
            self.ann_preview_h = pil.height
            self.ann_orig_w = W
            self.ann_orig_h = H
            for idx, b in enumerate(getattr(self, "ann_boxes", [])):
                c, x1, y1, x2, y2 = b
                selected = idx == getattr(self, "ann_selected", -1)
                col = "orange" if selected else ("lime" if int(c) == 0 else "red")
                label = f"{idx+1}: {'POS' if int(c) == 0 else 'FALSE'}"
                self.annotator_view.add_box(x1*sx, y1*sy, x2*sx, y2*sy, col, label)
        except Exception as exc:
            self.log("Annotator draw error: " + str(exc))

    def showann(self):
        try:
            if not self.ann_imgs:
                self.ann_status_qt.setText("No images loaded")
                if hasattr(self, "ann_box_list_qt"):
                    self.ann_box_list_qt.clear()
                return
            img = Path(self.ann_imgs[self.ann_i])
            self.ann_list_qt.blockSignals(True)
            self.ann_list_qt.setCurrentRow(self.ann_i)
            self.ann_list_qt.blockSignals(False)
            root = Path(self.project.get()) if self.project.get() else img.parent.parent
            label = self._find_existing_label_for_image(root, img)
            self.ann_boxes = backend.read_boxes(str(img), str(label)) if label and Path(label).exists() else []
            if self.ann_boxes and not (0 <= getattr(self, "ann_selected", -1) < len(self.ann_boxes)):
                self.ann_selected = 0
            elif not self.ann_boxes:
                self.ann_selected = -1
            self.draw_current_annotation_qt()
            self.refresh_ann_list_qt()
            status = "NEUTRAL / unchecked" if not self.ann_boxes else ("POSITIVE" if any(int(b[0]) == 0 for b in self.ann_boxes) else "FALSE")
            self.ann_status_qt.setText(f"{self.ann_i+1}/{len(self.ann_imgs)}  {img.name}\nStatus: {status}\nBoxes: {len(self.ann_boxes)}")
        except Exception as exc:
            self.log("Annotator preview error: " + str(exc))

    def add_annotation_box_from_scene_qt(self, x1: float, y1: float, x2: float, y2: float):
        """Add a mouse-drawn rectangle to the current annotation list."""
        try:
            if not self.ann_imgs:
                return
            sx = float(getattr(self, "ann_preview_sx", 1.0) or 1.0)
            sy = float(getattr(self, "ann_preview_sy", 1.0) or 1.0)
            pw = float(getattr(self, "ann_preview_w", 1.0) or 1.0)
            ph = float(getattr(self, "ann_preview_h", 1.0) or 1.0)
            ow = float(getattr(self, "ann_orig_w", pw) or pw)
            oh = float(getattr(self, "ann_orig_h", ph) or ph)
            # Clamp scene coordinates to the visible preview image.
            x1, x2 = sorted((max(0.0, min(pw, float(x1))), max(0.0, min(pw, float(x2)))))
            y1, y2 = sorted((max(0.0, min(ph, float(y1))), max(0.0, min(ph, float(y2)))))
            if (x2 - x1) < 3 or (y2 - y1) < 3:
                return
            ox1 = max(0.0, min(ow, x1 / sx))
            oy1 = max(0.0, min(oh, y1 / sy))
            ox2 = max(0.0, min(ow, x2 / sx))
            oy2 = max(0.0, min(oh, y2 / sy))
            cls = int(self.ann_cls.get())
            self.ann_boxes.append([cls, ox1, oy1, ox2, oy2])
            self.ann_selected = len(self.ann_boxes) - 1
            self.refresh_ann_list_qt()
            self.draw_current_annotation_qt()
            # Save immediately so annotations survive navigation/crash.
            self.saveann()
            img = Path(self.ann_imgs[self.ann_i])
            self.log(f"Mouse annotation added: {img.name} class={cls} box=({ox1:.0f},{oy1:.0f},{ox2:.0f},{oy2:.0f})")
        except Exception as exc:
            self.show_error(APP_NAME, str(exc))

    def set_image_class_qt(self, cls: int):
        try:
            if not self.ann_imgs:
                return
            bb = self.manifest_bbox_for_current_qt()
            self.ann_boxes = [[int(cls), bb[0], bb[1], bb[2], bb[3]]]
            self.ann_selected = 0
            self.saveann()
            self.showann()
            img = Path(self.ann_imgs[self.ann_i])
            self.log(f"Image class set: {img.name} -> {cls}")
        except Exception as exc:
            self.show_error(APP_NAME, str(exc))

    def set_selected_cls_qt(self):
        try:
            if 0 <= getattr(self, "ann_selected", -1) < len(getattr(self, "ann_boxes", [])):
                self.ann_boxes[self.ann_selected][0] = int(self.ann_cls.get())
                self.saveann(); self.showann()
        except Exception as exc:
            self.show_error(APP_NAME, str(exc))

    def delete_selected_box_qt(self):
        try:
            if 0 <= getattr(self, "ann_selected", -1) < len(getattr(self, "ann_boxes", [])):
                del self.ann_boxes[self.ann_selected]
                self.ann_selected = min(self.ann_selected, len(self.ann_boxes) - 1)
                self.saveann(); self.showann()
        except Exception as exc:
            self.show_error(APP_NAME, str(exc))

    def clear_current_ann_qt(self):
        try:
            self.ann_boxes = []
            self.ann_selected = -1
            self.saveann(); self.showann()
        except Exception as exc:
            self.show_error(APP_NAME, str(exc))

    def reset_from_manifest_qt(self):
        try:
            if not self.ann_imgs:
                return
            bb = self.manifest_bbox_for_current_qt()
            cls = int(self.ann_cls.get())
            self.ann_boxes = [[cls, bb[0], bb[1], bb[2], bb[3]]]
            self.ann_selected = 0
            self.saveann(); self.showann()
        except Exception as exc:
            self.show_error(APP_NAME, str(exc))

    def load_sam_images(self):
        try:
            folder = Path(self.sam_source_dir.get() or self.project.get() or ".")
            if folder.name.lower() != "images" and (folder / "images").exists():
                folder = folder / "images"
            self.sam_images = self._image_files_recursive(folder)
            self.sam_list_qt.clear()
            for p in self.sam_images:
                self.sam_list_qt.addItem(p.name)
            if self.sam_images:
                self.sam_list_qt.setCurrentRow(0)
                self.sam_selected_img = self.sam_images[0]
                self.draw_sam_preview()
            self.log(f"SAM2 image list loaded: {len(self.sam_images)}")
        except Exception as exc:
            self.show_error(APP_NAME, str(exc))

    def on_sam_select_qt(self, row: int):
        if 0 <= row < len(self.sam_images):
            self.sam_selected_img = self.sam_images[row]
            self.draw_sam_preview()

    def _read_yolo_prompt_boxes_for_image(self, img: Path, W: int, H: int) -> List[List[float]]:
        """Read every YOLO box available for this image from current annotator state or label files."""
        boxes: List[List[float]] = []
        try:
            if bool(self.sam_use_ann_boxes.get()) and self.ann_imgs:
                cur = Path(self.ann_imgs[self.ann_i])
                if cur.resolve() == img.resolve() and self.ann_boxes:
                    return [[float(b[1]), float(b[2]), float(b[3]), float(b[4])] for b in self.ann_boxes]
        except Exception:
            pass

        label_paths: List[Path] = []
        try:
            root = Path(self.project.get()) if self.project.get() else img.parent.parent
            found = self._find_existing_label_for_image(root, img)
            if found:
                label_paths.append(Path(found))
            label_paths.extend(self._label_candidates_for_image(root, img))
        except Exception:
            pass
        label_paths.extend([
            img.with_suffix('.txt'),
            img.parent / 'labels' / (img.stem + '.txt'),
            img.parent.parent / 'labels' / (img.stem + '.txt'),
        ])

        seen = set()
        for lab in label_paths:
            try:
                lab = Path(lab)
                key = str(lab.resolve()) if lab.exists() else str(lab)
                if key in seen:
                    continue
                seen.add(key)
                if not lab.exists():
                    continue
                for line in lab.read_text(encoding='utf-8', errors='ignore').splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = max(0.0, (cx - bw / 2.0) * W)
                    y1 = max(0.0, (cy - bh / 2.0) * H)
                    x2 = min(float(W - 1), (cx + bw / 2.0) * W)
                    y2 = min(float(H - 1), (cy + bh / 2.0) * H)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                if boxes:
                    return boxes
            except Exception as exc:
                self.log(f"SAM2 label read warning for {lab}: {exc}")
        return boxes

    def sam_boxes_for_image(self, img: Path, W: int, H: int):
        """Prompt SAM2 with all annotation/label boxes; fallback to visible detections or full image."""
        img = Path(img)
        boxes = self._read_yolo_prompt_boxes_for_image(img, W, H)
        if boxes:
            return boxes
        try:
            cur = Path(self.image.get().strip())
            if cur.exists() and cur.resolve() == img.resolve() and self.dets:
                det_boxes = [list(d.bbox()) for d in self.visible()]
                if det_boxes:
                    return det_boxes
        except Exception:
            pass
        return [[0.0, 0.0, float(W - 1), float(H - 1)]]

    def sam2_segment_image(self, img: Path, sam=None, save=True):
        """Segment one image with SAM2 using every box from the annotator/YOLO labels."""
        import numpy as np
        img = Path(img)
        if sam is None:
            sam = backend.load_sam2_model_safe(self.sammodel.get().strip() or 'sam2_b.pt', log_fn=self.sammsg)
        W, H, mode, reader = backend.open_img(img, self.log)
        boxes = self.sam_boxes_for_image(img, W, H)
        self.sam_prompt_boxes = boxes
        polys = []
        self.sammsg(f"SAM2 start: {img.name} | prompt boxes={len(boxes)} | size={W}x{H}")
        try:
            for i, bb in enumerate(boxes, 1):
                x1, y1, x2, y2 = map(float, bb)
                pad = int(self.sam_padding.get())
                left = max(0, int(x1 - pad)); top = max(0, int(y1 - pad))
                right = min(W, int(x2 + pad)); bottom = min(H, int(y2 + pad))
                tw = max(1, right - left); th = max(1, bottom - top)
                crop = backend.read_tile(reader, mode, left, top, max(tw, th), W, H)
                rb = [x1 - left, y1 - top, x2 - left, y2 - top]
                scale = 1.0
                max_crop = max(128, int(self.sam_max_crop.get()))
                max_side = max(crop.size)
                if max_side > max_crop:
                    scale = max_crop / float(max_side)
                    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
                    rb = [v * scale for v in rb]
                try:
                    res = sam.predict(np.asarray(crop), bboxes=[rb], verbose=False)
                    if res and res[0].masks is not None and getattr(res[0].masks, 'xy', None):
                        inv = 1.0 / scale
                        poly = [(float(x) * inv + left, float(y) * inv + top) for x, y in res[0].masks.xy[0]]
                        polys.append({'image': img.name, 'bbox': [x1, y1, x2, y2], 'polygon': poly, 'prompt_index': i, 'padding': pad, 'max_crop': max_crop})
                except Exception as exc:
                    self.sammsg(f"SAM2 error {img.name} prompt {i}: {exc}")
                if i % 5 == 0 or i == len(boxes):
                    self.sammsg(f"SAM2 progress {img.name}: {i}/{len(boxes)} | masks={len(polys)}")
        finally:
            try:
                if mode == 'rasterio':
                    reader.close()
            except Exception:
                pass
        if save:
            side = img.with_suffix('.sam2.json')
            side.write_text(json.dumps(polys, indent=2), encoding='utf-8')
            self.sammsg(f"SAM2 saved: {side}")
        return polys

    def sam2_selected(self):
        if not getattr(self, 'sam_selected_img', None):
            self.load_sam_images()
        if not getattr(self, 'sam_selected_img', None):
            raise RuntimeError('No image selected.')
        img = Path(self.sam_selected_img)
        polys = self.sam2_segment_image(img, sam=None, save=True)
        self.sam_polys = polys; self.sam_current_polys = polys; self.sam_preview_img = img
        self.sam_out.set(str(Path(self.sam_out.get().strip() or img.with_suffix('.sam2.json'))))
        QTimer.singleShot(0, self.draw_sam_preview)
        self.sammsg(f"SAM2 finished for selected image: {img.name}")

    def sam2_all(self):
        if not getattr(self, 'sam_images', None):
            self.load_sam_images()
        if not self.sam_images:
            raise RuntimeError('No images in the SAM2 list.')
        sam = backend.load_sam2_model_safe(self.sammodel.get().strip() or 'sam2_b.pt', log_fn=self.sammsg)
        all_polys = []
        skipped = 0
        for idx, img in enumerate(self.sam_images, 1):
            imgp = Path(img)
            if self.sam_skip_existing.get() and imgp.with_suffix('.sam2.json').exists():
                skipped += 1
                self.sammsg(f"SAM2 skipped, already exists: {idx}/{len(self.sam_images)} {imgp.name}")
                continue
            self.sammsg(f"SAM2 all images: {idx}/{len(self.sam_images)} {imgp.name}")
            polys = self.sam2_segment_image(imgp, sam=sam, save=True)
            all_polys.extend(polys)
            self.sam_polys = polys; self.sam_current_polys = polys; self.sam_preview_img = imgp
            QTimer.singleShot(0, self.draw_sam_preview)
        if skipped:
            self.sammsg(f"SAM2 batch skipped already segmented images: {skipped}")
        out = Path(self.sam_out.get().strip() or str(Path(self.sam_source_dir.get()).joinpath('sam2_all_polygons.json')))
        self.sam_out.set(str(out)); out.write_text(json.dumps(all_polys, indent=2), encoding='utf-8')
        self.sammsg(f"SAM2 all images finished: {len(all_polys)} masks | combined file: {out}")

    def sam2_current_annotation(self):
        if not self.ann_imgs:
            self.loadproj()
        if not self.ann_imgs:
            raise RuntimeError('No annotator project loaded.')
        self.saveann()
        img = Path(self.ann_imgs[self.ann_i])
        self.sam_selected_img = img; self.sam_preview_img = img; self.sam_source_dir.set(str(img.parent))
        polys = self.sam2_segment_image(img, sam=None, save=True)
        self.sam_polys = polys; self.sam_current_polys = polys
        QTimer.singleShot(0, self.draw_sam_preview)
        QTimer.singleShot(0, self.draw_current_annotation_qt)
        self.sammsg(f"SAM2 current annotator image finished: {img.name}")

    def sam2_all_annotation_project(self):
        if not self.ann_imgs:
            self.loadproj()
        if not self.ann_imgs:
            raise RuntimeError('No annotator images loaded.')
        self.saveann()
        sam = backend.load_sam2_model_safe(self.sammodel.get().strip() or 'sam2_b.pt', log_fn=self.sammsg)
        skipped = 0
        for idx, img in enumerate(self.ann_imgs, 1):
            imgp = Path(img)
            if self.sam_skip_existing.get() and imgp.with_suffix('.sam2.json').exists():
                skipped += 1
                self.sammsg(f"SAM2 annotator image skipped, already exists: {idx}/{len(self.ann_imgs)} {imgp.name}")
                continue
            self.sammsg(f"SAM2 annotator images: {idx}/{len(self.ann_imgs)} {imgp.name}")
            polys = self.sam2_segment_image(imgp, sam=sam, save=True)
            self.sam_polys = polys; self.sam_current_polys = polys; self.sam_preview_img = imgp
            QTimer.singleShot(0, self.draw_sam_preview)
        if skipped:
            self.sammsg(f"SAM2 annotator images skipped: {skipped}")
        QTimer.singleShot(0, self.showann)
        self.sammsg('SAM2 finished for all annotator images.')

    def draw_sam_preview(self):
        try:
            img = self.sam_selected_img
            if not img:
                return
            pil, W, H = backend.load_preview(str(img), maxs=2400)
            self.sam_view.set_pil_image(pil)
            sx = pil.width / max(1, W); sy = pil.height / max(1, H)
            # draw all available prompt boxes from annotator/labels
            for bb in self.sam_boxes_for_image(Path(img), W, H):
                try:
                    x1, y1, x2, y2 = bb
                    self.sam_view.add_box(x1*sx, y1*sy, x2*sx, y2*sy, "yellow", "prompt")
                except Exception:
                    pass
            # draw saved SAM2 masks, if present
            side = Path(img).with_suffix('.sam2.json')
            if side.exists():
                try:
                    data = json.loads(side.read_text(encoding='utf-8'))
                    items = data if isinstance(data, list) else data.get('polygons', [])
                    for item in items:
                        pts = item.get('polygon') if isinstance(item, dict) else None
                        if pts and len(pts) >= 2:
                            last = None
                            first = None
                            for px, py in pts:
                                cur = (float(px)*sx, float(py)*sy)
                                if first is None:
                                    first = cur
                                if last is not None:
                                    pen = QPen(QColor("cyan")); pen.setWidth(2)
                                    self.sam_view.scene().addLine(last[0], last[1], cur[0], cur[1], pen)
                                last = cur
                            if last is not None and first is not None:
                                pen = QPen(QColor("cyan")); pen.setWidth(2)
                                self.sam_view.scene().addLine(last[0], last[1], first[0], first[1], pen)
                except Exception as exc:
                    self.log("SAM2 mask preview warning: " + str(exc))
        except Exception as exc:
            self.log("SAM preview error: " + str(exc))

    def draw_fl_preview(self):
        try:
            if self.preview is not None:
                self.fl_view.set_pil_image(self.preview)
                sx = self.preview.width / max(1, self.origW); sy = self.preview.height / max(1, self.origH)
                thr = float(self.fl_threshold.get())
                for d in self.dets:
                    fs = getattr(d, "form_score", None)
                    if fs is None or float(fs) < thr:
                        continue
                    self.fl_view.add_box(d.x1*sx, d.y1*sy, d.x2*sx, d.y2*sy, "lime", f"F{float(fs):.2f}")
        except Exception as exc:
            self.log("FormLearner preview error: " + str(exc))

    def create_new_project_folder_dialog(self):
        self.new_project()

    def create_project_structure_from_detection(self):
        self.new_project()

    def create_crop_annotator_image_folder(self):
        root = Path(self.project.get())
        self._ensure_project_folders(root)
        self.log(f"Image folder ready: {root / 'images'}")

    def sam_use_cropdir(self):
        self.sam_source_dir.set(str(Path(self.cropdir.get()) / "images" if self.cropdir.get() else ""))
        self.load_sam_images()

    def sam_use_project_images(self):
        self.sam_source_dir.set(str(Path(self.project.get()) / "images" if self.project.get() else ""))
        self.load_sam_images()

    def pickdir(self, var: Var):
        self.browse_dir(var)

    def pick_form_out(self):
        self.save_file(self.form_model_path, "JSON (*.json);;All files (*)")

    # ------------------------------------------------------------------
    # File dialogs and browser/layer UI
    # ------------------------------------------------------------------
    def browse_file(self, var: Var, file_filter="All files (*)"):
        fn, _ = QFileDialog.getOpenFileName(self, "Choose file", "", file_filter)
        if fn:
            var.set(fn)

    def browse_dir(self, var: Var, _filter=""):
        fn = QFileDialog.getExistingDirectory(self, "Choose folder")
        if fn:
            var.set(fn)

    def save_file(self, var: Var, file_filter="All files (*)"):
        fn, _ = QFileDialog.getSaveFileName(self, "Choose output file", "", file_filter)
        if fn:
            var.set(fn)

    def refresh_project_browser(self):
        self.project_tree.clear()
        root_txt = self.project.get()
        root_item = QTreeWidgetItem([Path(root_txt).name if root_txt else "No project"])
        self.project_tree.addTopLevelItem(root_item)
        if root_txt and Path(root_txt).exists():
            for name in ["images", "labels", "crops", "sam2", "exports", "weights", "runs", "trained_form_models", "logs"]:
                p = Path(root_txt) / name
                item = QTreeWidgetItem([name])
                item.setData(0, Qt.UserRole, str(p))
                root_item.addChild(item)
                if p.exists():
                    for child in sorted(list(p.iterdir()))[:80]:
                        ci = QTreeWidgetItem([child.name])
                        ci.setData(0, Qt.UserRole, str(child))
                        item.addChild(ci)
        root_item.setExpanded(True)

    def _project_tree_open(self, item, col):
        p = item.data(0, Qt.UserRole)
        if not p:
            return
        path = Path(p)
        if path.is_dir():
            try:
                os.startfile(str(path))
            except Exception:
                pass
        elif path.suffix.lower() in IMG_EXT:
            self.image.set(str(path)); self.loadprev(); self.tabs.setCurrentIndex(0)

    def refresh_layers(self):
        self.layer_tree.clear()
        base = QTreeWidgetItem(["Layers"]); self.layer_tree.addTopLevelItem(base)
        for text in [
            f"Image: {Path(self.image.get()).name if self.image.get() else '-'}",
            f"YOLO detections: {len(self.dets)}",
            f"Visible detections: {len(self.visible()) if hasattr(self, 'visible') else 0}",
            f"FormLearner kept: {len(getattr(self, 'fl_kept', []))}",
            f"SAM2 images: {len(getattr(self, 'sam_images', []))}",
        ]:
            base.addChild(QTreeWidgetItem([text]))
        base.setExpanded(True)

    # ------------------------------------------------------------------
    # Threading/logging compatibility
    # ------------------------------------------------------------------
    def run_task(self, name: str, func):
        def worker():
            self.task_started(name)
            try:
                func()
                self.log(f"Task finished: {name}")
            except Exception as exc:
                self.log(f"Task error: {name}: {exc}\n{traceback.format_exc()}")
                self.show_error(APP_NAME, str(exc))
            finally:
                self.task_finished(name)
        threading.Thread(target=worker, daemon=True).start()

    def _training_output_root(self) -> Path:
        """Return the selected YOLO training output folder, defaulting to <project>/runs."""
        project_root = Path(self.project.get().strip() or ".").expanduser()
        raw = str(self.train_output_dir.get() or "").strip()
        if raw:
            out = Path(raw).expanduser()
        else:
            out = project_root / "runs"
            self.train_output_dir.set(str(out))
        out.mkdir(parents=True, exist_ok=True)
        return out

    def export_onnx(self):
        try:
            root = Path(self.project.get().strip() or ".").expanduser()
            out_root = self._training_output_root()
            best = out_root / "train_mustatil" / "weights" / "best.pt"
            if not best.exists():
                # Backward compatibility with older projects that trained into <project>/runs.
                fallback = root / "runs" / "train_mustatil" / "weights" / "best.pt"
                if fallback.exists():
                    best = fallback
                else:
                    raise RuntimeError(f"best.pt not found: {best}")
            cmd = [sys.executable, "-c", "from ultralytics import YOLO; import sys; YOLO(sys.argv[1]).export(format='onnx', imgsz=int(sys.argv[2]))", str(best), str(self.imgsz.get())]
            backend.run_live(cmd, self.tmsg, best.parent)
        except Exception as e:
            self.tmsg("ONNX EXPORT ERROR " + str(e)); self.show_error(APP_NAME, str(e))

    def train(self, resume=False):
        try:
            root = Path(self.project.get().strip() or ".").expanduser()
            out_root = self._training_output_root()
            self.tmsg(f"YOLO model output folder: {out_root}")
            y = self.prepare_yolo_dataset()
            imgsz = int(self.imgsz.get()); batch = int(self.batch.get()); device = (self.device.get().strip() or "cpu").lower()
            if self.low_ram_mode.get():
                imgsz = min(imgsz, 640); batch = min(batch, 2)
                self.tmsg("Low-RAM stable mode active: imgsz<=640, batch<=2, workers=0, cache=False, plots=False.")
            extra = ""
            if resume:
                last = out_root / "train_mustatil" / "weights" / "last.pt"
                if last.exists():
                    model_arg = str(last); extra = ", resume=True"
                else:
                    # Backward compatibility with older default output folder.
                    fallback = root / "runs" / "train_mustatil" / "weights" / "last.pt"
                    if fallback.exists():
                        model_arg = str(fallback); extra = ", resume=True"
                    else:
                        model_arg = self.trainmodel.get(); self.tmsg("last.pt not found; starting normal training.")
            else:
                model_arg = self.trainmodel.get()
            code = ("from ultralytics import YOLO\nimport sys\n"
                    "model=YOLO(sys.argv[1])\n"
                    f"model.train(data=r'{str(y)}',epochs={int(self.epochs.get())},imgsz={imgsz},batch={batch},device=r'{device}',project=r'{str(out_root)}',name='train_mustatil',exist_ok=True,workers=0,cache=False,plots=False{extra})\n")
            backend.run_live([sys.executable, "-c", code, model_arg], self.tmsg, root)
            self.tmsg(f"Training complete. Best model is usually: {out_root / 'train_mustatil' / 'weights' / 'best.pt'}")
            self.project_from_vars()
        except Exception as e:
            self.tmsg("TRAIN ERROR " + str(e)); self.show_error(APP_NAME, str(e))

    def task_started(self, name: str):
        self.signals.log.emit("General", f"Task started: {name}")
        QTimer.singleShot(0, lambda: (self.task_progress.show(), self.statusBar().showMessage(f"Running: {name}")))

    def task_finished(self, name: str):
        QTimer.singleShot(0, lambda: (self.task_progress.hide(), self.statusBar().showMessage(f"Finished: {name}", 5000), self.refresh_project_browser()))

    def _in_ui_thread(self):
        return threading.current_thread() is threading.main_thread()

    def _ui(self, func, *args, **kwargs):
        QTimer.singleShot(0, lambda: func(*args, **kwargs))

    def log(self, s=""):
        self.signals.log.emit("General", str(s))

    def tmsg(self, s=""):
        self.signals.log.emit("Training", str(s))
        self.signals.log.emit("General", str(s))

    def flog(self, s=""):
        self.signals.log.emit("FormTrainer", str(s))
        self.signals.log.emit("General", str(s))

    def fllogmsg(self, s=""):
        self.signals.log.emit("FormLearner", str(s))
        self.signals.log.emit("General", str(s))

    def sammsg(self, s=""):
        self.signals.log.emit("SAM2", str(s))
        self.signals.log.emit("General", str(s))

    def show_error(self, title, text):
        self.signals.error.emit(str(title), str(text))

    def show_info(self, title, text):
        self.signals.info.emit(str(title), str(text))

    def _append_log(self, channel: str, text: str):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        self.general_log.append(line)
        if channel == "Training" and hasattr(self, "train_log"):
            self.train_log.append(line)
        elif channel == "FormTrainer" and hasattr(self, "form_log"):
            self.form_log.append(line)
        elif channel == "FormLearner" and hasattr(self, "fl_log"):
            self.fl_log.append(line)

    def closeEvent(self, event):
        try:
            self.save_project()
        except Exception:
            pass
        super().closeEvent(event)



def apply_light_square_theme(app: QApplication):
    """Apply a bright, square-edged Office/QGIS-like Qt theme."""
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    app.setStyleSheet(r"""
        * {
            font-family: Segoe UI, Arial, sans-serif;
            font-size: 9.5pt;
        }
        QMainWindow, QWidget, QDialog {
            background: #f5f5f5;
            color: #202020;
        }
        QMenuBar {
            background: #ffffff;
            color: #202020;
            border-bottom: 1px solid #c8c8c8;
            padding: 0px;
        }
        QMenuBar::item {
            background: transparent;
            padding: 5px 10px;
            border-radius: 0px;
        }
        QMenuBar::item:selected {
            background: #e6f0fb;
            border: 1px solid #99c2ee;
        }
        QMenu {
            background: #ffffff;
            color: #202020;
            border: 1px solid #b8b8b8;
        }
        QMenu::item {
            padding: 5px 24px 5px 24px;
        }
        QMenu::item:selected {
            background: #dbeeff;
            color: #000000;
        }
        QPushButton:hover {
            background: #e6f0fb;
            border: 1px solid #7da9d8;
        }
        QPushButton:pressed {
            background: #cfe4fb;
            border: 1px solid #5f94c8;
        }
        QPushButton {
            background: #f8f8f8;
            color: #202020;
            border: 1px solid #b7b7b7;
            border-radius: 0px;
            padding: 5px 10px;
            min-height: 22px;
        }
        QPushButton:disabled {
            color: #8a8a8a;
            background: #eeeeee;
            border: 1px solid #d0d0d0;
        }
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #ffffff;
            color: #202020;
            border: 1px solid #b8b8b8;
            border-radius: 0px;
            padding: 3px;
            selection-background-color: #0078d7;
            selection-color: #ffffff;
        }
        QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #0078d7;
        }
        QTabWidget::pane {
            background: #ffffff;
            border: 1px solid #b8b8b8;
            top: -1px;
        }
        QTabBar::tab {
            background: #e8e8e8;
            color: #202020;
            border: 1px solid #b8b8b8;
            border-bottom: 1px solid #b8b8b8;
            border-radius: 0px;
            padding: 6px 12px;
            margin-right: 0px;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            border-bottom: 1px solid #ffffff;
        }
        QTabBar::tab:hover {
            background: #f3f8ff;
        }
        QGroupBox {
            background: #ffffff;
            color: #202020;
            border: 1px solid #b8b8b8;
            border-radius: 0px;
            margin-top: 16px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0px 4px;
            background: #f5f5f5;
            color: #202020;
        }
        QDockWidget {
            background: #f5f5f5;
            color: #202020;
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }
        QDockWidget::title {
            background: #e6e6e6;
            border: 1px solid #b8b8b8;
            padding: 4px;
            text-align: left;
        }
        QTreeWidget, QListWidget, QTableWidget {
            background: #ffffff;
            color: #202020;
            border: 1px solid #b8b8b8;
            alternate-background-color: #f2f2f2;
        }
        QTreeWidget::item:selected, QListWidget::item:selected {
            background: #cfe8ff;
            color: #000000;
        }
        QHeaderView::section {
            background: #e6e6e6;
            color: #202020;
            border: 1px solid #b8b8b8;
            padding: 4px;
        }
        QSplitter::handle {
            background: #d0d0d0;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background: #f0f0f0;
            border: 1px solid #c8c8c8;
            width: 14px;
            height: 14px;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: #c0c0c0;
            border: 1px solid #9f9f9f;
            border-radius: 0px;
            min-height: 24px;
            min-width: 24px;
        }
        QStatusBar {
            background: #eeeeee;
            color: #202020;
            border-top: 1px solid #c8c8c8;
        }
        QProgressBar {
            background: #ffffff;
            border: 1px solid #b8b8b8;
            border-radius: 0px;
            text-align: center;
        }
        QProgressBar::chunk {
            background: #0078d7;
            border-radius: 0px;
        }
    """)

def main():
    try:
        import multiprocessing as _mp
        _mp.freeze_support()
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    apply_light_square_theme(app)
    w = MustatilQtWorkspace()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
