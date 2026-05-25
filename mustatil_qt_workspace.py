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
import io
import math
import random
import tempfile
import shutil
import concurrent.futures as cf
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from PySide6.QtCore import Qt, QTimer, Signal, QObject, QRectF, QPointF
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
import satellite_preview_service

APP_NAME = "Mustatil Qt Workspace"
PROJECT_EXT = ".mustatil"
IMG_EXT = getattr(backend, "IMG_EXT", {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})

CLASSES = getattr(backend, "CLASSES", ["mustatil", "false_positive"])

YOLO_MODEL_PRESETS = {
    "Custom": ["", "", ""],
    "Houses": ["Houses300.pt", "", ""],
    "Trees": ["Tree50.pt", "", ""],
    "Cars": ["", "", ""],
}

# ------------------------------------------------------------------
# Satellite map helpers, adapted from the Py Map Stitcher workflow.
# Keep this code self-contained so the Qt workspace does not need to import
# the standalone Tkinter app.
# ------------------------------------------------------------------
WEB_TILE_SIZE = 256
USER_AGENT = "MustatilQtSatelliteDetection/1.0 (+local user tool)"
SATELLITE_MAP_PRESETS = {
    "Custom": {
        "url": "https://your-tile-server.example/{z}/{x}/{y}.png",
        "note": "Enter a custom URL template manually.",
    },
    "Google Satellite": {
        "url": "https://mt{rnd}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&hl=de",
        "note": "Google Satellite. Respect provider terms; avoid unauthorized bulk download.",
    },
    "Google Hybrid": {
        "url": "https://mt{rnd}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}&hl=de",
        "note": "Google Hybrid. Respect provider terms; avoid unauthorized bulk download.",
    },
    "Bing Satellite": {
        "url": "https://ecn.t{snum}.tiles.virtualearth.net/tiles/a{q}.jpeg?g=14574&mkt=de-DE&n=z",
        "note": "Bing aerial via QuadKey. Respect provider terms.",
    },
    "Bing Hybrid": {
        "url": "https://ecn.t{snum}.tiles.virtualearth.net/tiles/h{q}.jpeg?g=14574&mkt=de-DE&n=z",
        "note": "Bing hybrid via QuadKey. Respect provider terms.",
    },
    "Esri World Imagery": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "note": "Esri World Imagery. Respect Esri terms of use.",
    },
    "OpenStreetMap Mapnik": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "note": "OSM standard map. Not satellite; respect tile usage policy.",
    },
    "OpenTopoMap": {
        "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "note": "Topographic map. Not satellite; respect tile usage policy.",
    },
    "CartoDB Positron": {
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "note": "Light basemap. Not satellite; respect provider terms.",
    },
}


def sat_clamp_lat(lat: float) -> float:
    return max(min(float(lat), 85.05112878), -85.05112878)


def sat_lonlat_to_tile(lon: float, lat: float, z: int):
    lat = sat_clamp_lat(lat)
    n = 2 ** int(z)
    x = int((float(lon) + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def sat_tile_to_lonlat(x: float, y: float, z: int):
    n = 2 ** int(z)
    lon = float(x) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * float(y) / n)))
    return lon, math.degrees(lat_rad)


def sat_world_px(lon: float, lat: float, z: int):
    lat = sat_clamp_lat(lat)
    n = (2 ** int(z)) * WEB_TILE_SIZE
    x = (float(lon) + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def sat_lonlat_from_world_px(px: float, py: float, z: int):
    return sat_tile_to_lonlat(float(px) / WEB_TILE_SIZE, float(py) / WEB_TILE_SIZE, int(z))


def sat_webmercator_resolution(z: int) -> float:
    """Meters per pixel in global Web Mercator for XYZ tiles."""
    return (2.0 * math.pi * 6378137.0) / (WEB_TILE_SIZE * (2 ** int(z)))


def sat_webmercator_origin_for_tile(x_min: int, y_min: int, z: int):
    """Upper-left Web Mercator corner of an XYZ tile range."""
    origin_shift = math.pi * 6378137.0
    res = sat_webmercator_resolution(z)
    west = int(x_min) * WEB_TILE_SIZE * res - origin_shift
    north = origin_shift - int(y_min) * WEB_TILE_SIZE * res
    return west, north, res


def sat_webmercator_wkt() -> str:
    return 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","3857"]]'


def sat_tile_bounds_for_bbox(min_lat: float, min_lon: float, max_lat: float, max_lon: float, z: int):
    x1, y1 = sat_lonlat_to_tile(min_lon, max_lat, z)
    x2, y2 = sat_lonlat_to_tile(max_lon, min_lat, z)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def sat_tile_to_quadkey(x: int, y: int, z: int) -> str:
    q = []
    for i in range(int(z), 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if int(x) & mask:
            digit += 1
        if int(y) & mask:
            digit += 2
        q.append(str(digit))
    return "".join(q)


def sat_expand_url(template: str, x: int, y: int, z: int) -> str:
    rnd = random.randint(0, 3)
    sub = ["a", "b", "c"][rnd % 3]
    snum = str(rnd % 4)
    q = sat_tile_to_quadkey(x, y, z)
    return (str(template).replace("{x}", str(x))
                         .replace("{y}", str(y))
                         .replace("{z}", str(z))
                         .replace("{q}", q)
                         .replace("{quadkey}", q)
                         .replace("{rnd}", str(rnd))
                         .replace("{snum}", snum)
                         .replace("{s}", sub)
                         .replace("*GMX*", str(x))
                         .replace("*GMY*", str(y))
                         .replace("*ZM1*", str(z))
                         .replace("*IZM*", str(z))
                         .replace("*RND*", str(rnd))
                         .replace("*LAN*", "de")
                         .replace("*LAN-LAN*", "de-DE"))


def sat_cache_path(cache_dir: Path, z: int, x: int, y: int) -> Path:
    return Path(cache_dir) / str(int(z)) / f"z{int(z)}_x{int(x)}_y{int(y)}.tile"


def sat_blank_tile():
    return Image.new("RGB", (WEB_TILE_SIZE, WEB_TILE_SIZE), (245, 245, 245))


def sat_decode_tile(data: Optional[bytes]):
    if not data:
        return sat_blank_tile()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB").resize((WEB_TILE_SIZE, WEB_TILE_SIZE))
    except Exception:
        return sat_blank_tile()


class QtSignals(QObject):
    log = Signal(str, str)
    error = Signal(str, str)
    info = Signal(str, str)
    redraw = Signal()
    sam_redraw = Signal()
    form_redraw = Signal()
    sat_preview_ready = Signal(object, float, float, int, int, int, str)
    sat_preview_reset = Signal(float, float, int, int, int, str)
    sat_preview_tile_ready = Signal(object, int, int, float, float, int, int, int, str)
    sat_status = Signal(str)


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
    detection_model_preset: str = "Custom"
    satellite_detection_model_preset: str = "Custom"
    image: str = ""
    output: str = ""
    sam_model: str = "sam2_b.pt"
    sam_device: str = "cpu"
    satellite_map_preset: str = "Google Satellite"
    satellite_url_template: str = "https://mt{rnd}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&hl=de"
    satellite_zoom: int = 18
    satellite_cache_dir: str = ""
    satellite_output_gpkg: str = ""
    satellite_output_tif: str = ""
    train_model: str = "yolov8n.pt"
    formlearner_model: str = ""
    auto_annotate_yolo_model: str = ""
    auto_annotate_yolo_confidence: float = 0.001
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


class SatelliteMapCanvas(ImageCanvas):
    """Tile-map preview canvas with pan, wheel zoom and drag selection.

    The interaction model intentionally mirrors py_map_stitcher: left-drag
    changes the logical map center and asks the preview loader for the new
    visible tiles. It does not temporarily move QGraphicsItems around, because
    mixing item offsets with later full-image refreshes can make the map jump.
    """
    moved = Signal(float, float)
    zoom_delta = Signal(int, float, float)
    area_selected = Signal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.NoDrag)
        self._pan_start = None
        self._pan_start_world_center = None
        self._select_start = None
        self._select_item = None
        self._preview_w = 1
        self._preview_h = 1
        self._left_world = 0.0
        self._top_world = 0.0
        self._view_z = 3

    def set_map_transform(self, left_world: float, top_world: float, z: int, w: int, h: int):
        self._left_world = float(left_world)
        self._top_world = float(top_world)
        self._view_z = int(z)
        self._preview_w = max(1, int(w))
        self._preview_h = max(1, int(h))

    def scene_to_lonlat(self, p: QPointF):
        wx = self._left_world + float(p.x())
        wy = self._top_world + float(p.y())
        return sat_lonlat_from_world_px(wx, wy, self._view_z)

    def mousePressEvent(self, event):
        if self.pixmap_item is None:
            return super().mousePressEvent(event)
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier:
            self._select_start = self.mapToScene(event.position().toPoint())
            pen = QPen(QColor("red")); pen.setWidth(2); pen.setStyle(Qt.DashLine)
            self._select_item = self.scene().addRect(QRectF(self._select_start, self._select_start), pen)
            event.accept(); return
        if event.button() == Qt.RightButton:
            self._select_start = self.mapToScene(event.position().toPoint())
            pen = QPen(QColor("red")); pen.setWidth(2); pen.setStyle(Qt.DashLine)
            self._select_item = self.scene().addRect(QRectF(self._select_start, self._select_start), pen)
            event.accept(); return
        if event.button() == Qt.LeftButton:
            self._pan_start = (float(event.position().x()), float(event.position().y()))
            self._pan_start_world_center = (
                self._left_world + self._preview_w / 2.0,
                self._top_world + self._preview_h / 2.0,
            )
            self.setCursor(Qt.ClosedHandCursor)
            event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._select_start is not None and self._select_item is not None:
            cur = self.mapToScene(event.position().toPoint())
            self._select_item.setRect(QRectF(self._select_start, cur).normalized())
            event.accept(); return
        if self._pan_start is not None and self._pan_start_world_center is not None:
            dx = float(event.position().x()) - float(self._pan_start[0])
            dy = float(event.position().y()) - float(self._pan_start[1])
            cx = float(self._pan_start_world_center[0]) - dx
            cy = float(self._pan_start_world_center[1]) - dy
            lon, lat = sat_lonlat_from_world_px(cx, cy, self._view_z)
            self.moved.emit(lon, lat)
            event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._select_start is not None and event.button() in (Qt.RightButton, Qt.LeftButton):
            cur = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._select_start, cur).normalized()
            if self._select_item is not None:
                self.scene().removeItem(self._select_item)
            self._select_start = None
            self._select_item = None
            if rect.width() >= 5 and rect.height() >= 5:
                lon1, lat1 = self.scene_to_lonlat(rect.topLeft())
                lon2, lat2 = self.scene_to_lonlat(rect.bottomRight())
                self.area_selected.emit(min(lat1, lat2), min(lon1, lon2), max(lat1, lat2), max(lon1, lon2))
            event.accept(); return
        if self._pan_start is not None and event.button() == Qt.LeftButton:
            if self._pan_start_world_center is not None:
                dx = float(event.position().x()) - float(self._pan_start[0])
                dy = float(event.position().y()) - float(self._pan_start[1])
                cx = float(self._pan_start_world_center[0]) - dx
                cy = float(self._pan_start_world_center[1]) - dy
                lon, lat = sat_lonlat_from_world_px(cx, cy, self._view_z)
                self.moved.emit(lon, lat)
            self._pan_start = None
            self._pan_start_world_center = None
            self.unsetCursor()
            event.accept(); return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = 1 if event.angleDelta().y() > 0 else -1
        p = self.mapToScene(event.position().toPoint())
        self.zoom_delta.emit(delta, float(p.x()), float(p.y()))
        event.accept()


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
        self.signals.sat_preview_ready.connect(self._show_satellite_preview)
        self.signals.sat_preview_reset.connect(self._reset_satellite_preview_tiles)
        self.signals.sat_preview_tile_ready.connect(self._show_satellite_preview_tile)
        self.signals.sat_status.connect(self._set_satellite_status)

        self.project_state = MustatilProject()
        self._autosave_enabled = True
        self._current_project_file: Optional[Path] = None

        self._init_legacy_state()
        self._bind_legacy_backend_methods()
        self._build_ui()
        self._init_satellite_preview_timers()
        self._build_menus()
        self._start_autosave_timer()
        self.log(backend.deps())

    # ------------------------------------------------------------------
    # Legacy-compatible state
    # ------------------------------------------------------------------
    def _init_legacy_state(self):
        self.models = [Var(""), Var(""), Var("")]
        self.detection_model_preset = Var("Custom")
        self.sat_detection_model_preset = Var("Custom")
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
        self.sam_device = Var("cpu")
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
        self.auto_annotate_yolo_model = Var("")
        self.auto_annotate_yolo_confidence = Var(0.001)
        self.auto_annotate_use_sam2 = Var(False)
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

        # Satellite Detection tab state. These must be initialized before _build_ui(),
        # because the Satellite tab binds widgets directly to these Var objects.
        default_sat = SATELLITE_MAP_PRESETS.get("Google Satellite", SATELLITE_MAP_PRESETS.get("Custom", {}))
        self.sat_map_preset = Var("Google Satellite")
        self.sat_url_template = Var(default_sat.get("url", ""))
        self.sat_zoom = Var(18)
        self.sat_cache_dir = Var(str(Path.home() / "mustatil_satellite_cache"))
        self.sat_min_lat = Var("")
        self.sat_min_lon = Var("")
        self.sat_max_lat = Var("")
        self.sat_max_lon = Var("")
        self.sat_output_gpkg = Var("")
        self.sat_output_tif = Var("")
        self.sat_preview_z = 3
        self.sat_center_lon = 10.0
        self.sat_center_lat = 51.0
        self.sat_preview_left = 0.0
        self.sat_preview_top = 0.0
        self.sat_last_preview_size = (1, 1)
        self.sat_last_records: List[Dict[str, Any]] = []
        self.sat_preview_generation = 0
        self.satellite_detections: List[Dict[str, Any]] = []
        self.sat_last_x_min = None
        self.sat_last_y_min = None
        self.sat_last_z = None
        self.sat_last_crop_base = ""
        self.sat_preview_image = None
        self.sat_preview_tile_cache: Dict[Any, Image.Image] = {}
        self.sat_preview_tile_cache_order: List[Any] = []
        self.sat_preview_cache_limit = 2200  # ~430 MB raw RGB tiles; keeps zooming/panning much smoother.
        self.sat_preview_last_key = None
        self.sat_preview_last_emit = 0.0
        self.sat_preview_pending_args = None
        self.sat_preview_fast_zoom_timer = None

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
            try:
                if hasattr(self, "satellite_view"):
                    self._satellite_slider_changed_reload_map(120)
            except Exception:
                pass
            try:
                self.satellite_redraw_detection_overlay()
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
            try:
                if hasattr(self, "satellite_view"):
                    self._satellite_slider_changed_reload_map(120)
            except Exception:
                pass
            try:
                self.satellite_redraw_detection_overlay()
            except Exception:
                pass
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

    def _resolve_preset_model_path(self, model_name: str) -> str:
        """Return a usable path for a bundled preset model.

        Preset models are searched next to this Python file first, then in the
        current working directory and the project weights folder. Returning the
        script-local path even before the file exists keeps saved projects
        portable when the .py and .pt are copied together later.
        """
        name = str(model_name or "").strip()
        if not name:
            return ""
        candidates = []
        try:
            candidates.append(Path(__file__).resolve().parent / name)
        except Exception:
            pass
        candidates.append(Path.cwd() / name)
        try:
            root = Path(self.project.get() or "").expanduser()
            if str(root) and str(root) != ".":
                candidates.append(root / "weights" / name)
                candidates.append(root / name)
        except Exception:
            pass
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return str(candidate)
            except Exception:
                continue
        return str(candidates[0]) if candidates else name

    def _apply_yolo_model_preset(self, preset_name: str, preset_var: Var, context: str = "Detection"):
        preset_name = str(preset_name or "Custom")
        preset_var.set(preset_name)
        if preset_name == "Custom":
            return
        preset = YOLO_MODEL_PRESETS.get(preset_name)
        if not preset:
            self.log(f"{context}: unknown YOLO model preset: {preset_name}")
            return
        for i in range(3):
            raw = preset[i] if i < len(preset) else ""
            self.models[i].set(self._resolve_preset_model_path(raw) if raw else "")
        self.log(f"{context}: YOLO model preset applied: {preset_name}")

    def _yolo_preset_combo(self, preset_var: Var, context: str):
        combo = QComboBox()
        combo.addItems(list(YOLO_MODEL_PRESETS.keys()))
        preset_var.bind_widget(combo)
        combo.currentTextChanged.connect(lambda text: self._apply_yolo_model_preset(text, preset_var, context))
        return combo

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_detection_tab()
        self._build_satellite_detection_tab()
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
        g.addWidget(QLabel("Model preset"), 0, 0)
        g.addWidget(self._yolo_preset_combo(self.detection_model_preset, "Detection"), 0, 1)
        for i, v in enumerate(self.models):
            row_i = i + 1
            g.addWidget(QLabel(f"Model {i+1}"), row_i, 0)
            g.addWidget(self._var_line(v, self.browse_file, "YOLO models (*.pt *.onnx);;All files (*)"), row_i, 1)
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

    def _build_satellite_detection_tab(self):
        page = QWidget(); root = QSplitter(Qt.Horizontal); layout = QHBoxLayout(page); layout.addWidget(root)
        controls = QWidget(); form = QVBoxLayout(controls)

        mapbox = QGroupBox("Satellite map service / preview")
        mf = QFormLayout(mapbox)
        preset = QComboBox(); preset.addItems(list(SATELLITE_MAP_PRESETS.keys())); self.sat_map_preset.bind_widget(preset)
        preset.currentTextChanged.connect(self._satellite_preset_changed)
        mf.addRow("Map service", preset)
        mf.addRow("URL template", self._var_line(self.sat_url_template))
        mf.addRow("Detection zoom", self._spin(self.sat_zoom, 0, 22, 1))
        mf.addRow("Cache folder", self._var_line(self.sat_cache_dir, self.browse_dir))
        self.sat_note_label = QLabel(SATELLITE_MAP_PRESETS["Google Satellite"].get("note", "")); self.sat_note_label.setWordWrap(True)
        mf.addRow("Note", self.sat_note_label)
        form.addWidget(mapbox)

        bboxbox = QGroupBox("Selected map extent")
        bf = QFormLayout(bboxbox)
        bf.addRow("South / min lat", self._var_line(self.sat_min_lat))
        bf.addRow("West / min lon", self._var_line(self.sat_min_lon))
        bf.addRow("North / max lat", self._var_line(self.sat_max_lat))
        bf.addRow("East / max lon", self._var_line(self.sat_max_lon))
        form.addWidget(bboxbox)

        modelbox = QGroupBox("YOLO models and detection settings")
        mg = QGridLayout(modelbox)
        mg.addWidget(QLabel("Model preset"), 0, 0)
        mg.addWidget(self._yolo_preset_combo(self.sat_detection_model_preset, "Satellite Detection"), 0, 1)
        for i, v in enumerate(self.models):
            row_i = i + 1
            mg.addWidget(QLabel(f"Model {i+1}"), row_i, 0)
            mg.addWidget(self._var_line(v, self.browse_file, "YOLO models (*.pt *.onnx);;All files (*)"), row_i, 1)
        r = 4
        mg.addWidget(QLabel("YOLO compute confidence"), r, 0); mg.addWidget(self._dspin(self.conf, 0.001, 1, 0.01), r, 1); r += 1
        mg.addWidget(QLabel("Display confidence"), r, 0); mg.addWidget(self._dspin(self.showconf, 0.001, 1, 0.01), r, 1); r += 1
        mg.addWidget(QLabel("Preview FormScore"), r, 0); mg.addWidget(self._dspin(self.minscore, 0, 1, 0.01), r, 1); r += 1
        mg.addWidget(QLabel("Tile/chunk size"), r, 0); mg.addWidget(self._spin(self.tile, 64, 8192, 64), r, 1); r += 1
        mg.addWidget(QLabel("Overlap"), r, 0); mg.addWidget(self._spin(self.overlap, 0, 4096, 16), r, 1); r += 1
        mg.addWidget(self._check("shifted tiles", self.shift), r, 1); r += 1
        form.addWidget(modelbox)

        sat_filter_box = QGroupBox("Post-detection filter sliders")
        sf = QFormLayout(sat_filter_box)
        sf.addRow("Confidence filter", self._filter_slider_float(self.showconf, 0.0, 1.0, 0.01, 2))
        sf.addRow("FormScore filter", self._filter_slider_float(self.minscore, 0.0, 1.0, 0.01, 2))
        sf.addRow("Ensemble score filter", self._filter_slider_float(self.filter_score, 0.0, 4.0, 0.05, 2))
        sf.addRow("Minimum consensus", self._filter_slider_int(self.filter_consensus, 1, 3))
        sat_filter_hint = QLabel("These sliders filter the already computed satellite detections and redraw/export the visible subset without rerunning YOLO.")
        sat_filter_hint.setWordWrap(True)
        sf.addRow("", sat_filter_hint)
        form.addWidget(sat_filter_box)

        sat_fl_box = QGroupBox("Satellite FormLearner Detection")
        sff = QFormLayout(sat_fl_box)
        sff.addRow("FormLearner model", self._var_line(self.fl_model_path, self.browse_file, "JSON (*.json);;All files (*)"))
        sff.addRow("Minimum FormScore", self._dspin(self.fl_threshold, 0, 1, 0.01))
        sff.addRow("FormScore preview filter", self._dspin(self.minscore, 0, 1, 0.01))
        form.addWidget(sat_fl_box)

        outbox = QGroupBox("Outputs")
        of = QFormLayout(outbox)
        of.addRow("Output GeoPackage", self._var_line(self.sat_output_gpkg, self.save_file, "GeoPackage (*.gpkg);;GeoJSON (*.geojson);;All files (*)"))
        of.addRow("Output BigTIFF / GeoTIFF", self._var_line(self.sat_output_tif, self.save_file, "TIFF (*.tif *.tiff);;All files (*)"))
        of.addRow("Detection crop folder", self._var_line(self.cropdir, self.browse_dir))
        of.addRow("Detection crop size", self._spin(self.cropsize, 64, 8192, 64))
        form.addWidget(outbox)

        btns = QGridLayout()
        b = QPushButton("Calculate selected tiles"); b.clicked.connect(self.satellite_calculate_selection); btns.addWidget(b, 0, 0)
        b = QPushButton("Detect on satellite map"); b.clicked.connect(lambda: self.run_task("Satellite Detection", self.satellite_detect_selected)); btns.addWidget(b, 0, 1, 1, 2)
        b = QPushButton("Save selected map as BigTIFF"); b.clicked.connect(lambda: self.run_task("Satellite BigTIFF", self.satellite_save_selected_bigtiff)); btns.addWidget(b, 1, 0, 1, 3)
        b = QPushButton("Apply FormLearner to satellite crops"); b.clicked.connect(lambda: self.run_task("Satellite FormLearner", self.satellite_apply_formlearner_to_last)); btns.addWidget(b, 2, 0, 1, 3)
        b = QPushButton("Generate crops from last satellite detections"); b.clicked.connect(lambda: self.run_task("Satellite crops", self.satellite_generate_crops_from_last)); btns.addWidget(b, 3, 0, 1, 3)
        b = QPushButton("Export visible satellite detections"); b.clicked.connect(lambda: self.run_task("Satellite visible export", self.satellite_export_visible)); btns.addWidget(b, 4, 0, 1, 3)
        form.addLayout(btns)
        hint = QLabel("Mouse wheel: preview zoom. Left drag: move map. Right drag or Shift+Left drag: select detection/download area.")
        hint.setWordWrap(True); form.addWidget(hint)
        form.addStretch(1)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_toolbar = QWidget()
        preview_toolbar_layout = QHBoxLayout(preview_toolbar)
        preview_toolbar_layout.setContentsMargins(4, 4, 4, 4)
        b = QPushButton("Reload map")
        b.clicked.connect(self.satellite_refresh_map)
        preview_toolbar_layout.addWidget(b)
        b = QPushButton("+")
        b.setMaximumWidth(42)
        b.clicked.connect(lambda: self.satellite_zoom_preview(1))
        preview_toolbar_layout.addWidget(b)
        b = QPushButton("-")
        b.setMaximumWidth(42)
        b.clicked.connect(lambda: self.satellite_zoom_preview(-1))
        preview_toolbar_layout.addWidget(b)
        self.sat_view_status_label = QLabel("Mouse wheel: zoom | Left drag: move map | Right drag or Shift+Left: select area")
        self.sat_view_status_label.setWordWrap(False)
        preview_toolbar_layout.addWidget(self.sat_view_status_label, 1)
        preview_layout.addWidget(preview_toolbar)

        self.satellite_view = SatelliteMapCanvas()
        self.satellite_view.moved.connect(self.satellite_set_center)
        self.satellite_view.zoom_delta.connect(self.satellite_zoom_preview_at)
        self.satellite_view.area_selected.connect(self.satellite_area_selected)
        preview_layout.addWidget(self.satellite_view, 1)
        root.addWidget(self._scroll(controls)); root.addWidget(preview_panel); root.setSizes([500, 900])
        self.tabs.addTab(page, "Satellite Detection")
        QTimer.singleShot(300, self.satellite_refresh_map)

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

        auto_box = QGroupBox("Auto annotate crops")
        auto_form = QFormLayout(auto_box)
        auto_form.addRow("YOLO model for crops", self._var_line(self.auto_annotate_yolo_model, self.browse_file, "YOLO models (*.pt *.onnx);;All files (*)"))
        auto_form.addRow("Minimum YOLO confidence", self._dspin(self.auto_annotate_yolo_confidence, 0, 1, 0.001, 3))
        auto_form.addRow("", self._check("Skip SAM2 segmentation (YOLO + FormLearner only)", self.auto_annotate_use_sam2))
        auto_device = QComboBox(); auto_device.addItems(["cpu", "cuda", "directml"]); self.sam_device.bind_widget(auto_device); auto_device.currentTextChanged.connect(self.sam_device.set)
        auto_form.addRow("SAM2 device", auto_device)
        auto_form.addRow("FormLearner model", self._var_line(self.fl_model_path, self.browse_file, "JSON (*.json);;All files (*)"))
        auto_form.addRow("Minimum FormScore", self._dspin(self.fl_threshold, 0, 1, 0.01))
        auto_btn = QPushButton("Auto annotate crops")
        auto_btn.clicked.connect(lambda: self.run_task("Auto annotate crops", self.auto_annotate_crops))
        auto_form.addRow(auto_btn)
        l.addWidget(auto_box)

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
        sam_device_combo = QComboBox(); sam_device_combo.addItems(["cpu", "cuda", "directml"]); self.sam_device.bind_widget(sam_device_combo); sam_device_combo.currentTextChanged.connect(self.sam_device.set)
        f.addRow("Device", sam_device_combo)
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
        satm = mb.addMenu("Satellite")
        self._act(satm, "Reload Map Preview", self.satellite_refresh_map)
        self._act(satm, "Detect on Satellite Map", lambda: self.run_task("Satellite Detection", self.satellite_detect_selected))
        self._act(satm, "Save Selected Map as BigTIFF", lambda: self.run_task("Satellite BigTIFF", self.satellite_save_selected_bigtiff))
        self._act(satm, "Apply FormLearner to Satellite Crops", lambda: self.run_task("Satellite FormLearner", self.satellite_apply_formlearner_to_last))
        self._act(satm, "Generate Crops from Last Satellite Detections", lambda: self.run_task("Satellite crops", self.satellite_generate_crops_from_last))
        self._act(satm, "Export Visible Satellite Detections", lambda: self.run_task("Satellite visible export", self.satellite_export_visible))
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
        p.detection_model_preset = str(self.detection_model_preset.get() or "Custom")
        p.satellite_detection_model_preset = str(self.sat_detection_model_preset.get() or "Custom")
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
        p.sam_device = str(self.sam_device.get() or "cpu")
        p.satellite_map_preset = str(self.sat_map_preset.get() or "Google Satellite")
        p.satellite_url_template = str(self.sat_url_template.get() or "")
        p.satellite_zoom = int(self.sat_zoom.get() or 18)
        p.satellite_cache_dir = str(self.sat_cache_dir.get() or "")
        p.satellite_output_gpkg = str(self.sat_output_gpkg.get() or "")
        p.satellite_output_tif = str(self.sat_output_tif.get() or "")
        p.train_model = str(self.trainmodel.get() or "")
        p.formlearner_model = str(self.fl_model_path.get() or self.form_model_path.get() or "")
        p.auto_annotate_yolo_model = str(self.auto_annotate_yolo_model.get() or "")
        p.auto_annotate_yolo_confidence = float(self.auto_annotate_yolo_confidence.get())
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
        self.detection_model_preset.set(getattr(p, "detection_model_preset", "Custom") or "Custom")
        self.sat_detection_model_preset.set(getattr(p, "satellite_detection_model_preset", "Custom") or "Custom")
        self.sammodel.set(p.sam_model)
        self.sam_device.set(getattr(p, "sam_device", "cpu") or "cpu")
        self.sat_map_preset.set(getattr(p, "satellite_map_preset", "Google Satellite") or "Google Satellite")
        self.sat_url_template.set(getattr(p, "satellite_url_template", "") or SATELLITE_MAP_PRESETS.get("Google Satellite", {}).get("url", ""))
        self.sat_zoom.set(getattr(p, "satellite_zoom", 18) or 18)
        self.sat_cache_dir.set(getattr(p, "satellite_cache_dir", "") or str(Path.home() / "mustatil_satellite_cache"))
        self.sat_output_gpkg.set(getattr(p, "satellite_output_gpkg", ""))
        self.sat_output_tif.set(getattr(p, "satellite_output_tif", ""))
        self.trainmodel.set(p.train_model)
        self.train_output_dir.set(getattr(p, "train_output_folder", "") or getattr(p, "runs_folder", "") or (str(Path(p.project_root) / "runs") if p.project_root else ""))
        self.fl_model_path.set(p.formlearner_model)
        self.auto_annotate_yolo_model.set(getattr(p, "auto_annotate_yolo_model", ""))
        self.auto_annotate_yolo_confidence.set(getattr(p, "auto_annotate_yolo_confidence", 0.001))
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

    def _auto_annotate_yolo_models(self):
        """Load the annotator-selected YOLO model, with legacy Detection slots as fallback."""
        from ultralytics import YOLO
        candidates = []
        chosen = str(self.auto_annotate_yolo_model.get() or "").strip().strip('"')
        if chosen:
            candidates.append((0, "Annotator YOLO model", chosen))
        else:
            self.log("Auto annotate: no annotator YOLO model selected; falling back to Detection tab model slots.")
            for i, v in enumerate(self.models):
                raw = str(v.get() or "").strip().strip('"')
                if raw:
                    candidates.append((i, f"Detection model slot {i+1}", raw))

        models = []
        for i, label, raw in candidates:
            path = Path(raw).expanduser()
            if not path.is_file():
                self.log(f"Auto annotate: {label} skipped, file not found: {raw}")
                continue
            if path.suffix.lower() not in {".pt", ".onnx"}:
                self.log(f"Auto annotate: {label} skipped, unsupported file type: {path.name}")
                continue
            self.log(f"Auto annotate: loading {label}: {path}")
            try:
                models.append((i, path.name, YOLO(str(path))))
                self.log(f"Auto annotate: loaded {path.name} successfully")
            except Exception as exc:
                self.log(f"Auto annotate: failed loading {path.name}: {exc}")
                raise
        if not models:
            raise RuntimeError("Auto annotate needs a valid YOLO .pt or .onnx model. Choose it in the Annotator tab under 'YOLO model for crops'.")
        return models

    @staticmethod
    def _deduplicate_crop_detections(detections, iou_threshold: float = 0.50):
        """Keep strongest non-overlapping crop detections across model slots."""
        ordered = sorted(detections, key=lambda d: float(getattr(d, "conf", 0.0)), reverse=True)
        kept = []
        for d in ordered:
            try:
                if any(backend.iou(d.bbox(), k.bbox()) >= iou_threshold for k in kept):
                    continue
            except Exception:
                pass
            kept.append(d)
        return kept

    @staticmethod
    def _count_without_heavy_overlap(detections, iou_threshold: float = 0.90):
        """Count detections after removing boxes that overlap an already kept box by at least 90%."""
        try:
            return len(MustatilQtWorkspace._deduplicate_crop_detections(list(detections), iou_threshold=iou_threshold))
        except Exception:
            return len(list(detections or []))

    def _write_auto_annotation_sidecar(self, img: Path, records: List[Dict[str, Any]], threshold: float, yolo_confidence: float):
        side = img.with_suffix(".auto_annotate.json")
        side.write_text(json.dumps({
            "image": img.name,
            "formscore_threshold": float(threshold),
            "yolo_confidence": float(yolo_confidence),
            "records": records,
        }, indent=2), encoding="utf-8")

    def _selected_sam_device_name(self) -> str:
        raw = str(getattr(self, "sam_device", Var("cpu")).get() or "cpu").strip().lower()
        aliases = {"direct ml": "directml", "dml": "directml", "cuda:0": "cuda", "gpu": "cuda", "cude": "cuda"}
        return aliases.get(raw, raw if raw in {"cpu", "cuda", "directml"} else "cpu")

    def _move_sam_model_to_device(self, sam, device_name: str):
        """Best-effort device placement for Ultralytics/SAM2 with safe CPU fallback."""
        device_name = (device_name or "cpu").lower()
        if device_name == "cpu":
            self.sammsg("SAM2 device selected: cpu")
            try:
                if hasattr(sam, "to"):
                    sam.to("cpu")
                elif hasattr(sam, "model") and hasattr(sam.model, "to"):
                    sam.model.to("cpu")
            except Exception as exc:
                self.sammsg(f"SAM2 CPU placement warning: {exc}")
            return sam, "cpu", None

        if device_name == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self.sammsg("SAM2 CUDA requested, but torch.cuda.is_available() is false. Falling back to CPU.")
                    return self._move_sam_model_to_device(sam, "cpu")
                self.sammsg(f"SAM2 device selected: cuda ({torch.cuda.get_device_name(0)})")
                if hasattr(sam, "to"):
                    sam.to("cuda")
                elif hasattr(sam, "model") and hasattr(sam.model, "to"):
                    sam.model.to("cuda")
                return sam, "cuda", None
            except Exception as exc:
                self.sammsg(f"SAM2 CUDA setup failed: {exc}. Falling back to CPU.")
                return self._move_sam_model_to_device(sam, "cpu")

        if device_name == "directml":
            try:
                import torch_directml
                dml_device = torch_directml.device()
                self.sammsg(f"SAM2 device selected: directml ({dml_device})")
                if hasattr(sam, "to"):
                    sam.to(dml_device)
                elif hasattr(sam, "model") and hasattr(sam.model, "to"):
                    sam.model.to(dml_device)
                else:
                    self.sammsg("SAM2 DirectML warning: model object has no .to(...) method; prediction may still use its default device.")
                return sam, "directml", dml_device
            except Exception as exc:
                self.sammsg(f"SAM2 DirectML setup failed: {exc}. Install compatible torch-directml or use CPU/CUDA. Falling back to CPU.")
                return self._move_sam_model_to_device(sam, "cpu")

        self.sammsg(f"Unknown SAM2 device '{device_name}', using CPU.")
        return self._move_sam_model_to_device(sam, "cpu")

    def load_sam2_selected_device(self):
        """Load SAM2 and apply the device chosen in the Annotator/SAM2 tabs."""
        model_path = self.sammodel.get().strip() or "sam2_b.pt"
        requested = self._selected_sam_device_name()
        self.sammsg(f"SAM2 loading model: {model_path}")
        self.sammsg(f"SAM2 requested device: {requested}")
        sam = backend.load_sam2_model_safe(model_path, log_fn=self.sammsg)
        sam, active, device_obj = self._move_sam_model_to_device(sam, requested)
        self._sam_active_device = active
        self._sam_device_obj = device_obj
        self._sam_runtime_fallback_model = None
        self.sammsg(f"SAM2 active device: {active}")
        return sam

    def _sam_predict(self, sam, image_array, bboxes, verbose=False):
        """Predict with SAM2 and retry on CPU if CUDA/DirectML fails at runtime."""
        active = getattr(self, "_sam_active_device", self._selected_sam_device_name())
        fallback = getattr(self, "_sam_runtime_fallback_model", None)
        if active == "cpu" and fallback is not None:
            sam = fallback
        try:
            if active == "cuda":
                return sam.predict(image_array, bboxes=bboxes, verbose=verbose, device="cuda")
            # Ultralytics device= does not reliably accept torch_directml.device(); for DirectML,
            # keep the model on the DirectML device and call predict without a device string.
            return sam.predict(image_array, bboxes=bboxes, verbose=verbose)
        except Exception as exc:
            if active in {"cuda", "directml"}:
                self.sammsg(f"SAM2 prediction failed on {active}: {exc}. Retrying once on CPU.")
                cpu_sam = backend.load_sam2_model_safe(self.sammodel.get().strip() or "sam2_b.pt", log_fn=self.sammsg)
                cpu_sam, _, _ = self._move_sam_model_to_device(cpu_sam, "cpu")
                self._sam_active_device = "cpu"
                self._sam_device_obj = None
                self._sam_runtime_fallback_model = cpu_sam
                return cpu_sam.predict(image_array, bboxes=bboxes, verbose=verbose)
            raise

    def _segment_auto_boxes_for_image(self, img: Path, boxes: List[List[float]], sam=None):
        """Segment exactly the auto-annotation detections instead of rereading old labels."""
        import numpy as np
        img = Path(img)
        if sam is None:
            sam = self.load_sam2_selected_device()
        W, H, mode, reader = backend.open_img(img, self.log)
        polys = []
        total_boxes = len(boxes)
        self.log(f"Auto annotate SAM2 progress {img.name}: 0/{total_boxes} boxes processed, 0 segments created")
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
                    res = self._sam_predict(sam, np.asarray(crop), bboxes=[rb], verbose=False)
                    if res and res[0].masks is not None and getattr(res[0].masks, "xy", None):
                        inv = 1.0 / scale
                        poly = [(float(x) * inv + left, float(y) * inv + top) for x, y in res[0].masks.xy[0]]
                        polys.append({
                            "image": img.name,
                            "bbox": [x1, y1, x2, y2],
                            "polygon": poly,
                            "prompt_index": i,
                            "padding": pad,
                            "max_crop": max_crop,
                        })
                        self.log(f"Auto annotate SAM2 progress {img.name}: {i}/{total_boxes} boxes processed, {len(polys)} segments created")
                    else:
                        self.log(f"Auto annotate SAM2 progress {img.name}: {i}/{total_boxes} boxes processed, {len(polys)} segments created (no mask for this box)")
                except Exception as exc:
                    self.sammsg(f"Auto annotate SAM2 error {img.name} prompt {i}: {exc}")
                    self.log(f"Auto annotate SAM2 progress {img.name}: {i}/{total_boxes} boxes processed, {len(polys)} segments created (error on this box)")
        finally:
            try:
                if mode == "rasterio":
                    reader.close()
            except Exception:
                pass
        side = img.with_suffix(".sam2.json")
        side.write_text(json.dumps(polys, indent=2), encoding="utf-8")
        self.log(f"Auto annotate SAM2 finished {img.name}: segments={len(polys)}/{total_boxes} saved -> {side.name}")
        return polys

    def auto_annotate_crops(self):
        """YOLO -> SAM2 -> FormLearner review pass for every crop in the Annotator project.

        For each crop image, YOLO is rerun with the Minimum YOLO confidence chosen in the Auto annotate crops box.
        Each resulting detection is scored by the selected FormLearner model. Detections
        with FormScore >= the chosen threshold are written as class 0 (positive); lower
        scores are written as class 1 (false/negative). The same detections are also
        segmented with SAM2 and saved to the crop sidecar .sam2.json file.
        """
        self.log("Auto annotate crops: requested")
        if not self.ann_imgs:
            self.log("Auto annotate crops: no images currently loaded; loading project images now...")
            self.loadproj()
        if not self.ann_imgs:
            raise RuntimeError("No annotator crop project loaded. Choose the crop project and click Load project images first.")

        root = Path(self.project.get()).expanduser()
        self.log(f"Auto annotate crops: project root = {root}")
        self.log(f"Auto annotate crops: image count = {len(self.ann_imgs)}")
        (root / "labels").mkdir(parents=True, exist_ok=True)
        threshold = float(self.fl_threshold.get())
        yolo_conf = max(0.0, min(1.0, float(self.auto_annotate_yolo_confidence.get())))
        self.auto_annotate_yolo_confidence.set(yolo_conf)
        self.log(f"Auto annotate crops: YOLO minimum confidence = {yolo_conf:.3f}")
        self.log(f"Auto annotate crops: FormScore threshold = {threshold:.3f}")
        skip_sam2 = bool(getattr(self, "auto_annotate_use_sam2", Var(False)).get())
        use_sam2 = not skip_sam2
        self.log(f"Auto annotate crops: Skip SAM2 checkbox = {'checked' if skip_sam2 else 'unchecked'}")
        self.log(f"Auto annotate crops: SAM2 segmentation = {'disabled' if skip_sam2 else 'enabled'}")
        fl_path = Path(str(self.fl_model_path.get() or "").strip().strip('"')).expanduser()
        self.log(f"Auto annotate crops: FormLearner model = {fl_path}")
        if not fl_path.is_file():
            raise RuntimeError("Choose a valid FormLearner .json model before auto-annotating crops.")

        self.log("Auto annotate crops: loading FormLearner model...")
        form_model = backend.SimpleFormLearner.load(fl_path)
        self.log("Auto annotate crops: FormLearner model loaded")
        self.log("Auto annotate crops: loading YOLO model(s)...")
        yolo_models = self._auto_annotate_yolo_models()
        self.log(f"Auto annotate crops: YOLO models ready = {len(yolo_models)}")
        if use_sam2:
            self.log(f"Auto annotate crops: SAM2 model path = {self.sammodel.get().strip() or 'sam2_b.pt'}")
            self.log("Auto annotate crops: SAM2 will load before the first segmentation; if SAM2 is slow, this is the step to watch in the log.")
        else:
            self.log("Auto annotate crops: SAM2 disabled; using fast YOLO + FormLearner-only mode.")
        sam = None

        old_conf = self.conf.get()
        self.conf.set(yolo_conf)
        total_pos = total_neg = total_empty = total_det = total_masks = 0
        from mustatil_legacy_backend import Det
        try:
            import numpy as np
            for idx, img in enumerate(list(self.ann_imgs), 1):
                img = Path(img)
                self.log(f"Auto annotate crops: {idx}/{len(self.ann_imgs)} {img.name}")
                pil = Image.open(img).convert("RGB")
                W, H = pil.size
                arr = np.asarray(pil)
                detections = []
                for slot, name, model in yolo_models:
                    try:
                        res = model.predict(arr, conf=yolo_conf, imgsz=max(64, int(self.imgsz.get() or 640)), verbose=False)
                        if res and res[0].boxes is not None:
                            boxes = res[0].boxes
                            xy = boxes.xyxy.cpu().numpy(); cf = boxes.conf.cpu().numpy(); cl = boxes.cls.cpu().numpy()
                            for bb, conf, cls_id in zip(xy, cf, cl):
                                x1, y1, x2, y2 = map(float, bb[:4])
                                x1 = max(0.0, min(float(W), x1)); x2 = max(0.0, min(float(W), x2))
                                y1 = max(0.0, min(float(H), y1)); y2 = max(0.0, min(float(H), y2))
                                if x2 > x1 and y2 > y1:
                                    detections.append(Det(slot, name, int(cls_id), float(conf), x1, y1, x2, y2))
                    except Exception as exc:
                        self.log(f"Auto annotate YOLO error {img.name} model {slot+1}: {exc}")

                raw_count = len(detections)
                detections = self._deduplicate_crop_detections(detections, iou_threshold=0.50)
                self.log(f"Auto annotate YOLO result {img.name}: raw={raw_count}, after_iou_dedup={len(detections)}")
                label_boxes = []
                records = []
                for det in detections:
                    x1, y1, x2, y2 = det.bbox()
                    fs = float(form_model.predict(backend.crop_features(pil, (x1, y1, x2, y2))))
                    cls = 0 if fs >= threshold else 1
                    label_boxes.append([cls, x1, y1, x2, y2])
                    records.append({
                        "class": int(cls),
                        "status": "positive" if cls == 0 else "false_positive",
                        "form_score": fs,
                        "threshold": threshold,
                        "confidence": float(det.conf),
                        "model": det.name,
                        "model_slot": int(det.slot) + 1,
                        "bbox": [x1, y1, x2, y2],
                    })

                label_path = self._preferred_label_path_for_image(root, img)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                if label_boxes:
                    self.log(f"Auto annotate labels: writing {len(label_boxes)} boxes -> {label_path}")
                    backend.save_boxes(str(img), str(label_path), label_boxes)
                    self.update_crop_manifest_status(img.name, "positive" if any(int(b[0]) == 0 for b in label_boxes) else "false_positive", label_boxes)
                    if use_sam2:
                        if sam is None:
                            self.log("Auto annotate SAM2: loading model now because detections exist...")
                            sam = self.load_sam2_selected_device()
                            self.log("Auto annotate SAM2: model loaded")
                        self.log(f"Auto annotate SAM2: segmenting {len(label_boxes)} boxes for {img.name}")
                        polys = self._segment_auto_boxes_for_image(img, [[b[1], b[2], b[3], b[4]] for b in label_boxes], sam=sam)
                        total_masks += len(polys)
                    else:
                        self.log(f"Auto annotate SAM2: skipped for {img.name}; keeping YOLO + FormLearner labels only.")
                        polys = []
                else:
                    self.log(f"Auto annotate labels: no detections for {img.name}; clearing label if present -> {label_path}")
                    if label_path.exists():
                        label_path.unlink()
                    self.update_crop_manifest_status(img.name, "neutral", [])
                    if use_sam2:
                        img.with_suffix(".sam2.json").write_text("[]\n", encoding="utf-8")
                    total_empty += 1
                    polys = []
                self._write_auto_annotation_sidecar(img, records, threshold, yolo_conf)
                pos = sum(1 for b in label_boxes if int(b[0]) == 0)
                neg = sum(1 for b in label_boxes if int(b[0]) == 1)
                total_pos += pos; total_neg += neg; total_det += len(label_boxes)
                self.log(f"Auto annotated {img.name}: detections={len(label_boxes)} positive={pos} negative={neg} masks={len(polys)}")
        finally:
            self.conf.set(old_conf)

        self.ann_i = min(max(0, self.ann_i), max(0, len(self.ann_imgs) - 1))
        QTimer.singleShot(0, self.showann)
        self.log(f"Auto annotate crops finished: images={len(self.ann_imgs)} detections={total_det} positive={total_pos} negative={total_neg} empty={total_empty} sam2_masks={total_masks} sam2={'on' if use_sam2 else 'off'} threshold={threshold:.3f} yolo_conf={yolo_conf:.3f}")

    def sam2_segment_image(self, img: Path, sam=None, save=True):
        """Segment one image with SAM2 using every box from the annotator/YOLO labels."""
        import numpy as np
        img = Path(img)
        if sam is None:
            sam = self.load_sam2_selected_device()
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
                    res = self._sam_predict(sam, np.asarray(crop), bboxes=[rb], verbose=False)
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
        sam = self.load_sam2_selected_device()
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
        sam = self.load_sam2_selected_device()
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
        """Update the FormLearner overlay without requiring the removed standalone FormLearner tab."""
        try:
            if self.preview is None:
                return
            view = getattr(self, "fl_view", None) or getattr(self, "image_view", None)
            if view is None:
                return
            # The Detection tab already redraws all boxes through redraw(). If the old standalone
            # fl_view still exists in a custom build, keep supporting it; otherwise avoid clearing
            # the integrated Detection preview.
            if view is getattr(self, "image_view", None):
                return
            view.set_pil_image(self.preview)
            sx = self.preview.width / max(1, self.origW); sy = self.preview.height / max(1, self.origH)
            thr = float(self.fl_threshold.get())
            for d in self.dets:
                fs = getattr(d, "form_score", None)
                if fs is None or float(fs) < thr:
                    continue
                view.add_box(d.x1*sx, d.y1*sy, d.x2*sx, d.y2*sy, "lime", f"F{float(fs):.2f}")
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


    def _init_satellite_preview_timers(self):
        """Timers used to keep the satellite preview responsive during zoom/pan."""
        try:
            self.sat_preview_refresh_timer = QTimer(self)
            self.sat_preview_refresh_timer.setSingleShot(True)
            self.sat_preview_refresh_timer.timeout.connect(self._satellite_refresh_map_now)
        except Exception:
            self.sat_preview_refresh_timer = None

    def _schedule_satellite_preview_refresh(self, delay_ms: int = 90):
        """Debounce expensive tile reloads while the user spins the mouse wheel."""
        try:
            timer = getattr(self, "sat_preview_refresh_timer", None)
            if timer is not None:
                timer.start(max(0, int(delay_ms)))
                return
        except Exception:
            pass
        self._satellite_refresh_map_now()

    def _satellite_cache_put(self, key, tile):
        """Bounded RAM cache for preview tiles."""
        if tile is None:
            return
        cache = getattr(self, "sat_preview_tile_cache", {})
        order = getattr(self, "sat_preview_tile_cache_order", [])
        cache[key] = tile
        try:
            if key in order:
                order.remove(key)
        except Exception:
            pass
        order.append(key)
        limit = int(getattr(self, "sat_preview_cache_limit", 2200) or 2200)
        while len(order) > limit:
            old_key = order.pop(0)
            cache.pop(old_key, None)
        self.sat_preview_tile_cache = cache
        self.sat_preview_tile_cache_order = order

    # ------------------------------------------------------------------
    def _set_satellite_status(self, text: str):
        """Update satellite status safely on the Qt GUI thread."""
        try:
            txt = str(text)
            if hasattr(self, "sat_view_status_label"):
                self.sat_view_status_label.setText(txt)
            try:
                self.statusBar().showMessage(txt)
            except Exception:
                pass
        except Exception:
            pass

    def _satellite_progress(self, text: str, force: bool = False, min_seconds: float = 3.0):
        """Throttled feedback for long satellite operations.

        This keeps the UI/log visibly alive during slow tile downloads, crop export
        and YOLO processing without flooding the console.
        """
        try:
            now = time.time()
            last = float(getattr(self, "_satellite_progress_last", 0.0) or 0.0)
            if force or (now - last) >= float(min_seconds):
                self._satellite_progress_last = now
                msg = str(text)
                self.log(msg)
                try:
                    self.signals.sat_status.emit(msg)
                except Exception:
                    pass
        except Exception:
            try:
                self.log(str(text))
            except Exception:
                pass

    def _schedule_satellite_overlay_redraw(self, delay_ms: int = 160):
        """Debounce expensive red selection/detection overlay redraws.

        Individual map tiles can arrive very quickly. Redrawing hundreds of
        detection boxes after every single tile makes Qt look unresponsive.
        This schedules one redraw after a short quiet period instead.
        """
        try:
            if not hasattr(self, "sat_overlay_redraw_timer") or self.sat_overlay_redraw_timer is None:
                self.sat_overlay_redraw_timer = QTimer(self)
                self.sat_overlay_redraw_timer.setSingleShot(True)
                self.sat_overlay_redraw_timer.timeout.connect(self._satellite_overlay_redraw_now)
            self.sat_overlay_redraw_timer.start(max(20, int(delay_ms)))
        except Exception:
            try:
                self._satellite_overlay_redraw_now()
            except Exception:
                pass

    def _satellite_slider_changed_reload_map(self, delay_ms: int = 120):
        """Reload Satellite preview after filter/slider changes.

        The map tiles are reused from cache where possible; this mainly refreshes
        the view transform and redraws the red detection overlays after the
        current slider filters changed.
        """
        try:
            self._schedule_satellite_preview_refresh(max(20, int(delay_ms)))
        except Exception:
            try:
                self._schedule_satellite_overlay_redraw(80)
            except Exception:
                pass

    def _satellite_overlay_redraw_now(self):
        try:
            self._satellite_draw_selection_on_preview()
            self._satellite_draw_records_on_preview(self._satellite_visible_records())
        except Exception as exc:
            self.log("Satellite overlay redraw warning: " + str(exc))

    # Satellite Detection tab
    # ------------------------------------------------------------------
    def _satellite_preset_changed(self, name: str):
        preset = SATELLITE_MAP_PRESETS.get(name, SATELLITE_MAP_PRESETS["Custom"])
        self.sat_url_template.set(preset.get("url", ""))
        if hasattr(self, "sat_note_label"):
            self.sat_note_label.setText(preset.get("note", ""))
        self._schedule_satellite_preview_refresh(80)

    def satellite_set_center(self, lon: float, lat: float):
        self.sat_center_lon = float(lon)
        self.sat_center_lat = sat_clamp_lat(float(lat))
        self._schedule_satellite_preview_refresh(35)

    def satellite_zoom_preview(self, delta: int):
        old_z = int(self.sat_preview_z)
        new_z = max(0, min(22, old_z + int(delta)))
        if new_z == old_z:
            return
        self.sat_preview_z = new_z
        self._schedule_satellite_preview_refresh(25)

    def satellite_zoom_preview_at(self, delta: int, sx: float, sy: float):
        old_z = int(self.sat_preview_z)
        new_z = max(0, min(22, old_z + int(delta)))
        if new_z == old_z:
            return
        wx = float(getattr(self, "sat_preview_left", 0.0)) + float(sx)
        wy = float(getattr(self, "sat_preview_top", 0.0)) + float(sy)
        anchor_lon, anchor_lat = sat_lonlat_from_world_px(wx, wy, old_z)
        w, h = getattr(self, "sat_last_preview_size", (1, 1))
        ax, ay = sat_world_px(anchor_lon, anchor_lat, new_z)
        cx = ax - float(sx) + float(w) / 2.0
        cy = ay - float(sy) + float(h) / 2.0
        self.sat_center_lon, self.sat_center_lat = sat_lonlat_from_world_px(cx, cy, new_z)
        self.sat_preview_z = new_z
        self._schedule_satellite_preview_refresh(25)

    def satellite_area_selected(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float):
        self.sat_min_lat.set(f"{min_lat:.8f}")
        self.sat_min_lon.set(f"{min_lon:.8f}")
        self.sat_max_lat.set(f"{max_lat:.8f}")
        self.sat_max_lon.set(f"{max_lon:.8f}")
        self.log("Satellite map selection imported from preview.")
        self.satellite_calculate_selection()
        self._satellite_draw_selection_on_preview()


    def _satellite_apply_fast_zoom(self, delta: int, sx: Optional[float] = None, sy: Optional[float] = None):
        """Instant visual zoom of the existing preview pixmap while real tiles load."""
        try:
            view = getattr(self, "satellite_view", None)
            item = getattr(view, "pixmap_item", None)
            if item is None:
                return
            factor = 1.22 if int(delta) > 0 else (1.0 / 1.22)
            item.setTransformOriginPoint(float(sx) if sx is not None else item.boundingRect().width() / 2.0,
                                         float(sy) if sy is not None else item.boundingRect().height() / 2.0)
            item.setScale(float(item.scale()) * factor)
            if hasattr(self, "sat_view_status_label"):
                self.sat_view_status_label.setText("Zooming preview… loading exact tiles in background")
        except Exception:
            pass

    def _satellite_cache_dir(self) -> Path:
        raw = str(self.sat_cache_dir.get() or "").strip()
        return Path(raw).expanduser() if raw else Path.home() / "mustatil_satellite_cache"

    def _satellite_bbox(self):
        try:
            min_lat = float(str(self.sat_min_lat.get()).replace(",", "."))
            min_lon = float(str(self.sat_min_lon.get()).replace(",", "."))
            max_lat = float(str(self.sat_max_lat.get()).replace(",", "."))
            max_lon = float(str(self.sat_max_lon.get()).replace(",", "."))
        except Exception:
            raise RuntimeError("Select an area on the satellite map first, or enter South/West/North/East manually.")
        if max_lat <= min_lat or max_lon <= min_lon:
            raise RuntimeError("Invalid satellite extent: north/east must be larger than south/west.")
        return min_lat, min_lon, max_lat, max_lon

    def _download_sat_tile(self, x: int, y: int, z: int, preview: bool = False, cache_root_override: Optional[Path] = None):
        import requests
        if cache_root_override is not None:
            cache_root = Path(cache_root_override)
        else:
            cache_root = self._satellite_cache_dir() / ("_preview" if preview else "tiles")
        p = sat_cache_path(cache_root, z, x, y)
        if p.exists() and p.stat().st_size > 100:
            return p.read_bytes(), p
        url = sat_expand_url(str(self.sat_url_template.get() or ""), x, y, z)
        headers = {"User-Agent": USER_AGENT}
        timeout = 6 if preview else int(os.environ.get("MUSTATIL_SAT_TILE_TIMEOUT", "12") or "12")
        r = requests.get(url, headers=headers, timeout=(5, timeout))
        r.raise_for_status()
        data = r.content
        if len(data) < 50:
            raise RuntimeError(f"empty tile {z}/{x}/{y}")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        return data, p

    def _satellite_delete_tile_paths(self, paths):
        """Delete temporary detection tiles/chunks after YOLO has scanned them."""
        deleted = 0
        for p in paths or []:
            try:
                pp = Path(p)
                if pp.exists():
                    pp.unlink()
                    deleted += 1
            except Exception as exc:
                self.log(f"Satellite temporary tile delete warning: {p}: {exc}")
        return deleted

    def _satellite_save_tile_tif(self, data: Optional[bytes], out_path: Path):
        """Save one downloaded web tile as a temporary TIFF, like PyMapStitcher.

        Detection uses these TIFF files as the cached source material. They are
        intentionally written under the temporary detection cache and removed
        after the matching YOLO chunk has been processed.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im = sat_decode_tile(data)
        tmp = out_path.with_suffix(".tmp.tif")
        im.save(tmp, format="TIFF", compression="tiff_deflate")
        os.replace(tmp, out_path)
        try:
            im.close()
        except Exception:
            pass
        return out_path

    def _satellite_set_preview_image(self, pil_img: Image.Image):
        """Update the satellite preview pixmap without fitInView/reset jumps.

        PyMapStitcher paints visible tiles into a fixed canvas. For the Qt
        version we keep the same behavior by updating one pixmap at scene
        position 0/0 and never auto-fitting the QGraphicsView on every tile
        update. Calling ImageCanvas.set_pil_image() here would clear/refit the
        view repeatedly and causes the visible map to jump.
        """
        if pil_img is None or not hasattr(self, "satellite_view"):
            return
        im = pil_img.convert("RGBA")
        data = im.tobytes("raw", "RGBA")
        qimg = QImage(data, im.width, im.height, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg.copy())
        view = self.satellite_view
        scene = view.scene()
        try:
            view.resetTransform()
        except Exception:
            pass
        if getattr(view, "pixmap_item", None) is None:
            scene.clear()
            view.pixmap_item = scene.addPixmap(pix)
            view.pixmap_item.setPos(0, 0)
            view.pixmap_item.setScale(1.0)
        else:
            view.pixmap_item.setPixmap(pix)
            view.pixmap_item.setPos(0, 0)
            view.pixmap_item.setScale(1.0)
        self.sat_preview_canvas_pixmap = QPixmap(pix)
        scene.setSceneRect(0, 0, pix.width(), pix.height())
        try:
            view.setSceneRect(scene.sceneRect())
        except Exception:
            pass

    def _satellite_draw_tile_on_canvas(self, tile_img, sx: int, sy: int):
        """Paint one XYZ tile into the persistent preview canvas.

        QGIS/Google Maps style preview stays smooth when the map background is a
        single pixmap/canvas and incoming tiles are painted into that canvas,
        instead of creating one QGraphicsPixmapItem per tile. Overlays (red
        outline / detection boxes) remain separate vector items above it.
        """
        if tile_img is None or not hasattr(self, "satellite_view"):
            return
        view = self.satellite_view
        base = getattr(self, "sat_preview_canvas_pixmap", None)
        if base is None or base.isNull():
            item = getattr(view, "pixmap_item", None)
            if item is None:
                return
            base = QPixmap(item.pixmap())
        else:
            base = QPixmap(base)
        im = tile_img.convert("RGBA")
        data = im.tobytes("raw", "RGBA")
        qimg = QImage(data, im.width, im.height, QImage.Format_RGBA8888)
        tile_pix = QPixmap.fromImage(qimg.copy())
        painter = QPainter(base)
        painter.drawPixmap(int(sx), int(sy), tile_pix)
        painter.end()
        self.sat_preview_canvas_pixmap = base
        if getattr(view, "pixmap_item", None) is None:
            view.scene().clear()
            view.pixmap_item = view.scene().addPixmap(base)
            view.pixmap_item.setPos(0, 0)
            view.pixmap_item.setZValue(-1000)
        else:
            view.pixmap_item.setPixmap(base)
            view.pixmap_item.setPos(0, 0)
            view.pixmap_item.setZValue(-1000)

    def _reset_satellite_preview_tiles(self, left: float, top: float, z: int, cw: int, ch: int, status: str):
        """Reset the satellite preview to a Google-Maps-like tile canvas.

        This does not compose a large preview chunk/image. The scene is cleared,
        one persistent background pixmap is installed, and visible XYZ tiles are
        painted into that pixmap as they arrive.
        """
        if not hasattr(self, "satellite_view"):
            return
        try:
            self.sat_preview_left = float(left)
            self.sat_preview_top = float(top)
            self.sat_last_preview_size = (int(cw), int(ch))
            view = self.satellite_view
            scene = view.scene()
            try:
                view.resetTransform()
            except Exception:
                pass
            scene.clear()
            base = QPixmap(max(1, int(cw)), max(1, int(ch)))
            base.fill(QColor("#e8eef5"))
            self.sat_preview_canvas_pixmap = QPixmap(base)
            view.pixmap_item = scene.addPixmap(base)
            view.pixmap_item.setPos(0, 0)
            view.pixmap_item.setZValue(-1000)
            scene.setSceneRect(0, 0, int(cw), int(ch))
            try:
                view.setSceneRect(scene.sceneRect())
            except Exception:
                pass
            view.set_map_transform(float(left), float(top), int(z), int(cw), int(ch))
            self._satellite_draw_selection_on_preview()
            self._schedule_satellite_overlay_redraw(80)
            if hasattr(self, "sat_view_status_label"):
                self.sat_view_status_label.setText(status)
        except Exception as exc:
            self.log("Satellite preview reset error: " + str(exc))

    def _show_satellite_preview_tile(self, tile_img, sx: int, sy: int, left: float, top: float, z: int, cw: int, ch: int, status: str):
        """Draw one visible satellite XYZ tile on the Qt GUI thread."""
        if not hasattr(self, "satellite_view") or tile_img is None:
            return
        try:
            # Ignore stale tile workers from an older view.
            if int(z) != int(getattr(self, "sat_preview_z", z)):
                return
            if abs(float(left) - float(getattr(self, "sat_preview_left", left))) > 0.5:
                return
            if abs(float(top) - float(getattr(self, "sat_preview_top", top))) > 0.5:
                return
            self._satellite_draw_tile_on_canvas(tile_img, int(sx), int(sy))
            if hasattr(self, "sat_view_status_label"):
                self.sat_view_status_label.setText(status)
            # Keep overlays above newly added map tiles, but debounce the expensive
            # redraw so many arriving tiles do not make Qt appear frozen.
            self._schedule_satellite_overlay_redraw(180)
        except Exception as exc:
            self.log("Satellite preview tile draw error: " + str(exc))

    def _show_satellite_preview(self, canvas, left: float, top: float, z: int, cw: int, ch: int, status: str):
        """Compatibility path for older callers that still send a full preview image."""
        if not hasattr(self, "satellite_view"):
            return
        try:
            self.sat_preview_image = canvas
            self.sat_preview_left = float(left)
            self.sat_preview_top = float(top)
            self.sat_last_preview_size = (int(cw), int(ch))
            self._satellite_set_preview_image(canvas)
            self.satellite_view.set_map_transform(float(left), float(top), int(z), int(cw), int(ch))
            self._satellite_draw_selection_on_preview()
            if hasattr(self, "sat_view_status_label"):
                self.sat_view_status_label.setText(status)
            try:
                self._satellite_draw_records_on_preview(self._satellite_visible_records())
            except Exception as exc:
                self.log("Satellite preview overlay warning: " + str(exc))
        except Exception as exc:
            self.log("Satellite preview display error: " + str(exc))

    def _satellite_draw_selection_on_preview(self):
        """Draw the selected satellite extent as a persistent red outline."""
        if not hasattr(self, "satellite_view"):
            return
        try:
            for item in list(getattr(self, "satellite_selection_items", []) or []):
                try:
                    self.satellite_view.scene().removeItem(item)
                except Exception:
                    pass
            self.satellite_selection_items = []
        except Exception:
            pass
        try:
            min_lat, min_lon, max_lat, max_lon = self._satellite_bbox()
        except Exception:
            return
        try:
            z = int(getattr(self, "sat_preview_z", 3))
            left = float(getattr(self, "sat_preview_left", 0.0))
            top = float(getattr(self, "sat_preview_top", 0.0))
            wx1, wy1 = sat_world_px(min_lon, max_lat, z)
            wx2, wy2 = sat_world_px(max_lon, min_lat, z)
            x1, y1 = wx1 - left, wy1 - top
            x2, y2 = wx2 - left, wy2 - top
            rect = QRectF(QPointF(float(x1), float(y1)), QPointF(float(x2), float(y2))).normalized()
            if rect.width() < 1 or rect.height() < 1:
                return
            pen = QPen(QColor("red"))
            pen.setWidth(3)
            pen.setStyle(Qt.SolidLine)
            item = self.satellite_view.scene().addRect(rect, pen)
            try:
                item.setZValue(900)
            except Exception:
                pass
            self.satellite_selection_items = [item]
        except Exception as exc:
            self.log("Satellite selection outline warning: " + str(exc))

    def _satellite_preview_tile_from_cache(self, template: str, z: int, x: int, y: int):
        """Return a preview tile from RAM or disk cache without doing network I/O."""
        cache = getattr(self, "sat_preview_tile_cache", {})
        key = (str(template), int(z), int(x), int(y))
        tile = cache.get(key)
        if tile is not None:
            return tile
        try:
            p = sat_cache_path(self._satellite_cache_dir() / "_preview", int(z), int(x), int(y))
            if p.exists() and p.stat().st_size > 100:
                tile = sat_decode_tile(p.read_bytes())
                self._satellite_cache_put(key, tile)
                return tile
        except Exception:
            return None
        return None

    def _satellite_preview_pyramid_placeholder(self, template: str, z: int, x: int, y: int):
        """Build a child tile from already cached lower pyramid tiles only.

        Important for smooth zooming: this method must never do network I/O or
        disk reads on the GUI thread. Missing exact tiles are fetched later by
        the background worker.
        """
        z = int(z); x = int(x); y = int(y)
        if z <= 0:
            return None
        cache = getattr(self, "sat_preview_tile_cache", {})
        for dz in range(1, min(6, z + 1)):
            pz = z - dz
            scale = 2 ** dz
            px = x // scale
            py = y // scale
            parent = cache.get((str(template), int(pz), int(px), int(py)))
            if parent is None:
                continue
            step = WEB_TILE_SIZE / float(scale)
            lx = (x % scale) * step
            ly = (y % scale) * step
            try:
                crop = parent.crop((int(round(lx)), int(round(ly)), int(round(lx + step)), int(round(ly + step))))
                return crop.resize((WEB_TILE_SIZE, WEB_TILE_SIZE))
            except Exception:
                continue
        return None

    def satellite_refresh_map(self):
        self._schedule_satellite_preview_refresh(40)

    def _satellite_refresh_map_now(self):
        """Refresh satellite preview through the external preview service module."""
        try:
            satellite_preview_service.refresh_satellite_preview(self)
        except Exception as exc:
            self.log("Satellite preview service failed: " + str(exc))
            if hasattr(self, "sat_view_status_label"):
                self.sat_view_status_label.setText("Preview error - see log")

    def satellite_calculate_selection(self):
        try:
            min_lat, min_lon, max_lat, max_lon = self._satellite_bbox()
            z = int(self.sat_zoom.get())
            x_min, y_min, x_max, y_max = sat_tile_bounds_for_bbox(min_lat, min_lon, max_lat, max_lon, z)
            cols = x_max - x_min + 1
            rows = y_max - y_min + 1
            width = cols * WEB_TILE_SIZE
            height = rows * WEB_TILE_SIZE
            chunk = max(64, int(self.tile.get()))
            chunks_x = math.ceil(width / chunk)
            chunks_y = math.ceil(height / chunk)
            self.log(f"Satellite selection z={z}: tiles x={x_min}..{x_max}, y={y_min}..{y_max}")
            self.log(f"Satellite selection pixel size: {width:,} x {height:,}; YOLO chunks {chunks_x} x {chunks_y} = {chunks_x*chunks_y:,} at tile size {chunk}")
        except Exception as exc:
            self.show_error(APP_NAME, str(exc))


    def _satellite_chunk_image(self, x_min: int, y_min: int, z: int, chunk_left: int, chunk_top: int, chunk_w: int, chunk_h: int, cache_root_override: Optional[Path] = None):
        """Build one YOLO-sized satellite chunk from web tiles.

        Detection mode is intentionally restored to the previous in-memory/PIL path:
        downloaded raw tiles are stored briefly in the temporary cache, decoded into
        one PIL chunk, scanned by YOLO, and then deleted by the caller. No temporary
        TIFF chunk is used for YOLO here.
        """
        img = Image.new("RGB", (chunk_w, chunk_h), (255, 255, 255))
        used_paths = []
        gx0 = x_min * WEB_TILE_SIZE + int(chunk_left)
        gy0 = y_min * WEB_TILE_SIZE + int(chunk_top)
        tx0 = int(math.floor(gx0 / WEB_TILE_SIZE))
        ty0 = int(math.floor(gy0 / WEB_TILE_SIZE))
        tx1 = int(math.floor((gx0 + chunk_w - 1) / WEB_TILE_SIZE))
        ty1 = int(math.floor((gy0 + chunk_h - 1) / WEB_TILE_SIZE))
        tile_total = max(1, (ty1 - ty0 + 1) * (tx1 - tx0 + 1))
        tile_done = 0
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile_done += 1
                self._satellite_progress(f"Satellite chunk tile download: z={z} tile {tx},{ty} ({tile_done}/{tile_total})", min_seconds=4.0)
                try:
                    data, tile_path = self._download_sat_tile(tx, ty, z, preview=False, cache_root_override=cache_root_override)
                    if cache_root_override is not None:
                        used_paths.append(tile_path)
                    tile = sat_decode_tile(data)
                except Exception as exc:
                    self.log(f"Satellite tile warning {z}/{tx}/{ty}: {exc}")
                    tile = sat_blank_tile()
                px = tx * WEB_TILE_SIZE - gx0
                py = ty * WEB_TILE_SIZE - gy0
                img.paste(tile, (int(px), int(py)))
        return img, used_paths

    def _satellite_build_chunk_to_cache(self, x_min: int, y_min: int, z: int, chunk_left: int, chunk_top: int, chunk_w: int, chunk_h: int, chunk_id: int, cache_root: Path):
        """Download/build one satellite YOLO chunk in a worker thread and store it in the temporary cache.

        The main detection thread only reads the ready chunk file, runs YOLO, and
        then deletes both the chunk file and the temporary source tiles. This lets
        tile downloads run in parallel while YOLO scans already prepared chunks.
        """
        im, used_tile_paths = self._satellite_chunk_image(
            x_min, y_min, z, chunk_left, chunk_top, chunk_w, chunk_h,
            cache_root_override=cache_root,
        )
        chunk_dir = Path(cache_root) / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunk_dir / f"chunk_{int(chunk_id):06d}_x{int(chunk_left)}_y{int(chunk_top)}.tif"
        tmp = chunk_path.with_suffix(".tmp.tif")
        im.save(tmp, format="TIFF", compression="tiff_deflate")
        os.replace(tmp, chunk_path)
        try:
            im.close()
        except Exception:
            pass
        return {
            "chunk_id": int(chunk_id),
            "x": int(chunk_left),
            "y": int(chunk_top),
            "w": int(chunk_w),
            "h": int(chunk_h),
            "chunk_path": str(chunk_path),
            "tile_paths": [str(p) for p in (used_tile_paths or [])],
        }

    def _satellite_output_gpkg_path(self) -> Path:
        raw = str(self.sat_output_gpkg.get() or "").strip()
        if raw:
            p = Path(raw).expanduser()
        else:
            root = Path(self.project.get() or Path.home()).expanduser()
            p = root / "exports" / f"satellite_detections_z{int(self.sat_zoom.get())}.gpkg"
            self.sat_output_gpkg.set(str(p))
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _satellite_output_tif_path(self) -> Path:
        raw = str(self.sat_output_tif.get() or "").strip()
        if raw:
            p = Path(raw).expanduser()
        else:
            root = Path(self.project.get() or Path.home()).expanduser()
            p = root / "exports" / f"satellite_map_z{int(self.sat_zoom.get())}.tif"
            self.sat_output_tif.set(str(p))
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _satellite_load_yolo_models(self):
        from ultralytics import YOLO
        models = []
        for i, v in enumerate(self.models):
            raw = str(v.get() or "").strip().strip('"')
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_file():
                self.log(f"Satellite Detection: model slot {i+1} skipped, file not found: {raw}")
                continue
            self.log(f"Satellite Detection: loading YOLO model {i+1}: {path}")
            models.append((i, path.name, YOLO(str(path))))
        if not models:
            raise RuntimeError("Choose at least one YOLO .pt/.onnx model for Satellite Detection.")
        return models

    def _satellite_features_to_file(self, records: List[Dict[str, Any]], out_path: Path):
        if not records:
            raise RuntimeError("No satellite detections to write.")
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
        except Exception as exc:
            raise RuntimeError("GeoPackage export needs geopandas and shapely. Install them in this runtime first.") from exc
        geoms = [Polygon(r.pop("polygon_lonlat")) for r in records]
        gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
        if out_path.suffix.lower() == ".geojson":
            gdf.to_file(out_path, driver="GeoJSON")
        else:
            gdf.to_file(out_path, driver="GPKG", layer="satellite_detections")
        return gdf

    def _satellite_write_georef_sidecars(self, out_path: Path, x_min: int, y_min: int, z: int):
        """Write worldfile/projection sidecars for QGIS when rasterio is unavailable."""
        west, north, res = sat_webmercator_origin_for_tile(x_min, y_min, z)
        tfw = out_path.with_suffix(".tfw")
        prj = out_path.with_suffix(".prj")
        # Worldfile stores pixel size and the center of the upper-left pixel.
        tfw.write_text(
            f"{res:.12f}\n0.000000000000\n0.000000000000\n{-res:.12f}\n{west + res / 2.0:.12f}\n{north - res / 2.0:.12f}\n",
            encoding="utf-8",
        )
        prj.write_text(sat_webmercator_wkt(), encoding="utf-8")
        self.log(f"Satellite georef sidecars written for QGIS: {tfw.name}, {prj.name}")

    def _satellite_crop_base_dir(self, z: int) -> Path:
        raw = str(self.cropdir.get() or "").strip()
        if raw:
            base = Path(raw).expanduser()
        else:
            root = Path(self.project.get() or Path.home()).expanduser()
            base = root / "crops"
            self.cropdir.set(str(base))
        out = base / f"satellite_crops_z{int(z)}"
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        return out

    def _satellite_world_crop_image(self, z: int, crop_left_world_px: float, crop_top_world_px: float, crop_w: int, crop_h: int):
        """Build an exact map crop from cached/downloaded XYZ tiles at the selected detection zoom."""
        img = Image.new("RGB", (int(crop_w), int(crop_h)), (255, 255, 255))
        tx0 = int(math.floor(float(crop_left_world_px) / WEB_TILE_SIZE))
        ty0 = int(math.floor(float(crop_top_world_px) / WEB_TILE_SIZE))
        tx1 = int(math.floor((float(crop_left_world_px) + int(crop_w) - 1) / WEB_TILE_SIZE))
        ty1 = int(math.floor((float(crop_top_world_px) + int(crop_h) - 1) / WEB_TILE_SIZE))
        ntiles = 2 ** int(z)
        for ty in range(max(0, ty0), min(ntiles - 1, ty1) + 1):
            for tx in range(max(0, tx0), min(ntiles - 1, tx1) + 1):
                try:
                    cache_override = getattr(self, "sat_detection_temp_cache_root", None)
                    data, _tile_path = self._download_sat_tile(tx, ty, z, preview=False, cache_root_override=cache_override)
                    tile = sat_decode_tile(data)
                except Exception as exc:
                    self.log(f"Satellite crop tile warning {z}/{tx}/{ty}: {exc}")
                    tile = sat_blank_tile()
                px = int(tx * WEB_TILE_SIZE - float(crop_left_world_px))
                py = int(ty * WEB_TILE_SIZE - float(crop_top_world_px))
                img.paste(tile, (px, py))
        return img

    def _satellite_export_detection_crops(self, records: List[Dict[str, Any]], x_min: int, y_min: int, z: int):
        """Create one crop image per detection, exactly crop-size pixels around the detection center.

        The crop is built at the detection zoom from the same tile cache used by detection. If the
        needed tiles are still cached from chunk detection, this does not download them again; if not,
        it downloads only the small crop area.

        Important: crop export deduplicates the selected satellite detections before any tile
        download happens. Strongly overlapping boxes would create nearly identical crops and cause
        redundant satellite tile requests, so detections with >=90% IoU to a stronger detection are
        skipped here as well.
        """
        if not records:
            return None

        before_dedup = len(records)
        try:
            records = self._satellite_deduplicate_records([dict(r) for r in records], 0.90)
        except Exception as exc:
            self.log(f"Satellite crop overlap filter warning: {exc}; using unfiltered records.")
            records = [dict(r) for r in records]
        if not records:
            self.log("Satellite crops: no detections left after >90% overlap filtering.")
            return None
        if len(records) != before_dedup:
            self.log(f"Satellite crops overlap filter: {before_dedup} -> {len(records)} crops; skipped {before_dedup - len(records)} with >90% overlap before downloading crop tiles.")

        crop_size = max(64, int(self.cropsize.get() or 1024))
        base = self._satellite_crop_base_dir(z)
        img_dir = base / "images"
        lab_dir = base / "labels"
        manifest = []
        selection_world_x = int(x_min) * WEB_TILE_SIZE
        selection_world_y = int(y_min) * WEB_TILE_SIZE
        for i, r in enumerate(records, 1):
            bx1 = float(r.get("bbox_px_x1", 0.0)); by1 = float(r.get("bbox_px_y1", 0.0))
            bx2 = float(r.get("bbox_px_x2", 0.0)); by2 = float(r.get("bbox_px_y2", 0.0))
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            crop_left = selection_world_x + cx - crop_size / 2.0
            crop_top = selection_world_y + cy - crop_size / 2.0
            crop_img = self._satellite_world_crop_image(z, crop_left, crop_top, crop_size, crop_size)
            stem = f"sat_z{int(z)}_det_{i:06d}"
            img_path = img_dir / f"{stem}.jpg"
            lab_path = lab_dir / f"{stem}.txt"

            gx1 = selection_world_x + bx1; gy1 = selection_world_y + by1
            gx2 = selection_world_x + bx2; gy2 = selection_world_y + by2
            lx1 = max(0.0, min(float(crop_size), gx1 - crop_left))
            ly1 = max(0.0, min(float(crop_size), gy1 - crop_top))
            lx2 = max(0.0, min(float(crop_size), gx2 - crop_left))
            ly2 = max(0.0, min(float(crop_size), gy2 - crop_top))
            cls = int(r.get("class_id", 0))
            if lx2 > lx1 and ly2 > ly1:
                yolo_cx = ((lx1 + lx2) / 2.0) / crop_size
                yolo_cy = ((ly1 + ly2) / 2.0) / crop_size
                yolo_w = (lx2 - lx1) / crop_size
                yolo_h = (ly2 - ly1) / crop_size
                lab_path.write_text(f"{cls} {yolo_cx:.8f} {yolo_cy:.8f} {yolo_w:.8f} {yolo_h:.8f}\n", encoding="utf-8")
            else:
                lab_path.write_text("", encoding="utf-8")

            # Mark the detection in exported satellite crop images with a red rectangle,
            # so wrong/over-dense detections can be reviewed immediately in the tile/crop folder.
            try:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(crop_img)
                if lx2 > lx1 and ly2 > ly1:
                    w = max(3, int(round(crop_size / 256)))
                    for off in range(w):
                        draw.rectangle([lx1 - off, ly1 - off, lx2 + off, ly2 + off], outline=(255, 0, 0))
            except Exception as draw_exc:
                self.log(f"Satellite crop marker warning: {img_path.name}: {draw_exc}")
            crop_img.save(img_path, quality=94)

            crop_lon_w, crop_lat_n = sat_lonlat_from_world_px(crop_left, crop_top, z)
            crop_lon_e, crop_lat_s = sat_lonlat_from_world_px(crop_left + crop_size, crop_top + crop_size, z)
            manifest.append({
                "image": img_path.name,
                "label": lab_path.name,
                "source": "satellite_detection",
                "map_service": str(self.sat_map_preset.get() or ""),
                "zoom": int(z),
                "crop_size": int(crop_size),
                "confidence": float(r.get("confidence", 0.0)),
                "class_id": cls,
                "bbox_crop_px": [lx1, ly1, lx2, ly2],
                "bbox_selection_px": [bx1, by1, bx2, by2],
                "crop_extent_lonlat": [min(crop_lon_w, crop_lon_e), min(crop_lat_s, crop_lat_n), max(crop_lon_w, crop_lon_e), max(crop_lat_s, crop_lat_n)],
                "detection_polygon_lonlat": r.get("polygon_lonlat", []),
            })
            if i % 5 == 0 or i == len(records):
                self._satellite_progress(f"Satellite crops written: {i}/{len(records)}", force=True)
        (base / "crop_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"Satellite detection crops ready: {len(records)} images in {img_dir}")
        return base

    def _satellite_record_conf(self, r: Dict[str, Any]) -> float:
        try:
            return float(r.get("confidence", r.get("conf", 0.0)) or 0.0)
        except Exception:
            return 0.0

    def _satellite_record_form_score(self, r: Dict[str, Any]):
        for key in ("form_score", "FormScore", "formscore"):
            if key in r and r.get(key) is not None:
                try:
                    return float(r.get(key))
                except Exception:
                    return None
        return None

    def _satellite_visible_records(self, records: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        records = list(records if records is not None else (getattr(self, "satellite_detections", None) or getattr(self, "sat_last_records", []) or []))
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
        out = []
        for r in records:
            if self._satellite_record_conf(r) < conf_thr:
                continue
            try:
                if float(r.get("score", 0.0) or 0.0) < score_thr:
                    continue
            except Exception:
                pass
            try:
                if int(r.get("consensus", 1) or 1) < consensus_thr:
                    continue
            except Exception:
                pass
            fs = self._satellite_record_form_score(r)
            if fs is not None and fs < form_thr:
                continue
            out.append(r)
        return out

    def satellite_redraw_detection_overlay(self):
        """Redraw satellite preview plus current filtered detection overlay."""
        try:
            canvas = getattr(self, "sat_preview_image", None)
            if canvas is None or not hasattr(self, "satellite_view"):
                return
            self.satellite_view.set_pil_image(canvas)
            cw, ch = getattr(self, "sat_last_preview_size", (canvas.width, canvas.height))
            self.satellite_view.set_map_transform(float(getattr(self, "sat_preview_left", 0.0)), float(getattr(self, "sat_preview_top", 0.0)), int(getattr(self, "sat_preview_z", 3)), int(cw), int(ch))
            self._satellite_draw_selection_on_preview()
            recs = self._satellite_visible_records()
            self._satellite_draw_records_on_preview(recs)
            if recs:
                self.log(f"Satellite preview filter: showing {len(recs)} / {len(getattr(self, 'satellite_detections', []) or getattr(self, 'sat_last_records', []) or [])} detections")
        except Exception as exc:
            self.log("Satellite overlay redraw error: " + str(exc))

    def satellite_generate_crops_from_last(self):
        records = self._satellite_visible_records()
        if not records:
            raise RuntimeError("No visible satellite detections available. Run Satellite Detection first or lower the filters.")
        x_min = getattr(self, "sat_last_x_min", None)
        y_min = getattr(self, "sat_last_y_min", None)
        z = getattr(self, "sat_last_z", None)
        if x_min is None or y_min is None or z is None:
            min_lat, min_lon, max_lat, max_lon = self._satellite_bbox()
            z = int(self.sat_zoom.get())
            x_min, y_min, _, _ = sat_tile_bounds_for_bbox(min_lat, min_lon, max_lat, max_lon, z)
        base = self._satellite_export_detection_crops([dict(r) for r in records], int(x_min), int(y_min), int(z))
        if base:
            self.sat_last_crop_base = str(base)
            self.project.set(str(base))
            self.log(f"Satellite crops regenerated from current visible detections: {base}")
        return base

    def satellite_export_visible(self):
        records = self._satellite_visible_records()
        if not records:
            raise RuntimeError("No visible satellite detections to export. Run Satellite Detection first or lower the filters.")
        out_path = self._satellite_output_gpkg_path()
        self._satellite_features_to_file([dict(r) for r in records], out_path)
        self.log(f"Visible satellite detections exported: {len(records)} -> {out_path}")
        return out_path

    def satellite_apply_formlearner_to_last(self):
        records = list(getattr(self, "satellite_detections", None) or getattr(self, "sat_last_records", []) or [])
        if not records:
            raise RuntimeError("No satellite detections available. Run Satellite Detection first.")
        fl_path = Path(str(self.fl_model_path.get() or "").strip().strip('"')).expanduser()
        if not fl_path.is_file():
            raise RuntimeError("Choose a valid FormLearner .json model first.")
        threshold = float(self.fl_threshold.get())
        self.log(f"Satellite FormLearner: loading model {fl_path}")
        form_model = backend.SimpleFormLearner.load(fl_path)

        x_min = getattr(self, "sat_last_x_min", None)
        y_min = getattr(self, "sat_last_y_min", None)
        z = getattr(self, "sat_last_z", None)
        if x_min is None or y_min is None or z is None:
            min_lat, min_lon, max_lat, max_lon = self._satellite_bbox()
            z = int(self.sat_zoom.get())
            x_min, y_min, _, _ = sat_tile_bounds_for_bbox(min_lat, min_lon, max_lat, max_lon, z)
        crop_base = self._satellite_export_detection_crops([dict(r) for r in records], int(x_min), int(y_min), int(z))
        self.sat_last_crop_base = str(crop_base or "")
        manifest_path = Path(crop_base) / "crop_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
        img_dir = Path(crop_base) / "images"
        lab_dir = Path(crop_base) / "labels"

        positives = negatives = 0
        for idx, (r, m) in enumerate(zip(records, manifest), 1):
            img_path = img_dir / str(m.get("image"))
            if not img_path.is_file():
                self.log(f"Satellite FormLearner warning: crop missing for record {idx}: {img_path}")
                continue
            pil = Image.open(img_path).convert("RGB")
            bbox = m.get("bbox_crop_px") or [0, 0, pil.width, pil.height]
            fs = float(form_model.predict(backend.crop_features(pil, tuple(map(float, bbox)))))
            cls = 0 if fs >= threshold else 1
            r["form_score"] = fs
            r["form_threshold"] = threshold
            r["form_status"] = "positive" if cls == 0 else "false_positive"
            r["class_id"] = cls
            m["form_score"] = fs
            m["status"] = r["form_status"]
            m["class_id"] = cls
            lab_path = lab_dir / str(m.get("label"))
            try:
                lx1, ly1, lx2, ly2 = map(float, bbox)
                cw = max(1, int(m.get("crop_size") or pil.width))
                ch = max(1, int(m.get("crop_size") or pil.height))
                if lx2 > lx1 and ly2 > ly1:
                    yolo_cx = ((lx1 + lx2) / 2.0) / cw
                    yolo_cy = ((ly1 + ly2) / 2.0) / ch
                    yolo_w = (lx2 - lx1) / cw
                    yolo_h = (ly2 - ly1) / ch
                    lab_path.write_text(f"{cls} {yolo_cx:.8f} {yolo_cy:.8f} {yolo_w:.8f} {yolo_h:.8f}\n", encoding="utf-8")
            except Exception as exc:
                self.log(f"Satellite FormLearner label warning {img_path.name}: {exc}")
            positives += int(cls == 0); negatives += int(cls == 1)
            if idx % 10 == 0 or idx == len(records):
                self.log(f"Satellite FormLearner progress: {idx}/{len(records)} scored")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self.satellite_detections = records
        self.sat_last_records = records
        try:
            self._schedule_satellite_overlay_redraw(80)
        except Exception:
            pass
        out_path = self._satellite_output_gpkg_path()
        self._satellite_features_to_file([dict(r) for r in records], out_path)
        self.log(f"Satellite FormLearner finished: positive={positives}, negative={negatives}, threshold={threshold:.3f}")
        self.log(f"Satellite detections rewritten with FormScore/classes: {out_path}")
        self.satellite_redraw_detection_overlay()
        self.refresh_layers()
        return records


    def _satellite_detection_box_is_valid(self, bx1: float, by1: float, bx2: float, by2: float, cw: int, ch: int):
        """Reject obviously broken/tiny satellite YOLO boxes before export/crop creation.

        This prevents hundreds of false positives on very small or invalid satellite tile/chunk
        inputs while keeping the filter conservative for real small objects. Override with
        MUSTATIL_SAT_MIN_DET_PX if needed.
        """
        try:
            w = float(bx2) - float(bx1)
            h = float(by2) - float(by1)
            if w <= 0 or h <= 0:
                return False, "invalid"
            min_px = max(1.0, float(os.environ.get("MUSTATIL_SAT_MIN_DET_PX", "8") or "8"))
            if w < min_px or h < min_px:
                return False, f"too small <{min_px:g}px"
            # Guard against pathological detections on very small/blank chunks.
            chunk_area = max(1.0, float(cw) * float(ch))
            if (w * h) < max(16.0, chunk_area * 0.00001):
                return False, "area too small"
            return True, ""
        except Exception:
            return False, "validation error"


    def satellite_detect_selected(self):
        min_lat, min_lon, max_lat, max_lon = self._satellite_bbox()
        z = int(self.sat_zoom.get())
        x_min, y_min, x_max, y_max = sat_tile_bounds_for_bbox(min_lat, min_lon, max_lat, max_lon, z)
        cols = x_max - x_min + 1; rows = y_max - y_min + 1
        width = cols * WEB_TILE_SIZE; height = rows * WEB_TILE_SIZE
        chunk = max(64, int(self.tile.get()))
        conf = max(0.001, min(1.0, float(self.conf.get())))
        out_path = self._satellite_output_gpkg_path()
        models = self._satellite_load_yolo_models()
        self._satellite_progress_last = 0.0
        self._satellite_progress(f"Satellite Detection started: z={z}, tiles={cols}x{rows}, pixels={width:,}x{height:,}, chunk={chunk}, conf={conf:.3f}", force=True)
        self.log("Satellite Detection restored: temporary raw tile cache -> PIL chunk -> YOLO -> delete temporary tiles. No TIFF chunk detection and no NumPy conversion are used.")
        records: List[Dict[str, Any]] = []
        chunk_id = 0
        total_chunks = math.ceil(width / chunk) * math.ceil(height / chunk)
        temp_cache_root = self._satellite_cache_dir() / "_detection_tmp" / f"z{z}_{int(time.time())}"
        old_temp_cache = getattr(self, "sat_detection_temp_cache_root", None)
        self.sat_detection_temp_cache_root = temp_cache_root
        try:
            temp_cache_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            for y in range(0, height, chunk):
                for x in range(0, width, chunk):
                    chunk_id += 1
                    cw = min(chunk, width - x); ch = min(chunk, height - y)
                    self._satellite_progress(f"Satellite chunk {chunk_id}/{total_chunks}: building image at pixel x={x} y={y} size={cw}x{ch}", force=True)
                    im, used_tile_paths = self._satellite_chunk_image(x_min, y_min, z, x, y, cw, ch, cache_root_override=temp_cache_root)
                    self._satellite_progress(f"Satellite chunk {chunk_id}/{total_chunks}: image ready, starting YOLO", force=True)
                    chunk_records_before = len(records)
                    chunk_small_skipped = 0
                    chunk_invalid_skipped = 0
                    for slot, model_name, model in models:
                        self._satellite_progress(f"Satellite chunk {chunk_id}/{total_chunks}: YOLO model {slot+1}/{len(models)} running", min_seconds=2.0)
                        try:
                            # Use the PIL image directly and convert tensors with .tolist().
                            # This avoids both the TIFF-source pipeline and the .numpy() calls
                            # that fail in environments where NumPy is not installed.
                            res = model.predict(source=im, conf=conf, imgsz=max(64, int(self.imgsz.get() or 640)), verbose=False)
                            if not res or res[0].boxes is None:
                                continue
                            boxes = res[0].boxes
                            xy = boxes.xyxy.detach().cpu().tolist()
                            cfv = boxes.conf.detach().cpu().tolist()
                            cl = boxes.cls.detach().cpu().tolist()
                            for bb, score, cls_id in zip(xy, cfv, cl):
                                bx1, by1, bx2, by2 = map(float, bb[:4])
                                bx1 = max(0.0, min(float(cw), bx1)); bx2 = max(0.0, min(float(cw), bx2))
                                by1 = max(0.0, min(float(ch), by1)); by2 = max(0.0, min(float(ch), by2))
                                valid_box, invalid_reason = self._satellite_detection_box_is_valid(bx1, by1, bx2, by2, cw, ch)
                                if not valid_box:
                                    if "small" in invalid_reason:
                                        chunk_small_skipped += 1
                                    else:
                                        chunk_invalid_skipped += 1
                                    continue
                                # Store every detection in zoom-independent geographic coordinates.
                                # bbox_px_* stays available for crop generation at the detection zoom,
                                # but preview rendering must use polygon_lonlat/bbox_lonlat and reproject
                                # to the current preview zoom on every redraw.
                                gx1 = x_min * WEB_TILE_SIZE + x + bx1
                                gy1 = y_min * WEB_TILE_SIZE + y + by1
                                gx2 = x_min * WEB_TILE_SIZE + x + bx2
                                gy2 = y_min * WEB_TILE_SIZE + y + by2
                                lon1, lat1 = sat_lonlat_from_world_px(gx1, gy1, z)
                                lon2, lat2 = sat_lonlat_from_world_px(gx2, gy2, z)
                                west, east = sorted((float(lon1), float(lon2)))
                                south, north = sorted((float(lat1), float(lat2)))
                                poly = [(west, north), (east, north), (east, south), (west, south), (west, north)]
                                records.append({
                                    "class_id": int(cls_id),
                                    "confidence": float(score),
                                    "model": str(model_name),
                                    "model_slot": int(slot) + 1,
                                    "zoom": int(z),
                                    "tile_x_min": int(x_min),
                                    "tile_y_min": int(y_min),
                                    "chunk_id": int(chunk_id),
                                    "chunk_px_x": int(x),
                                    "chunk_px_y": int(y),
                                    "bbox_px_x1": float(x + bx1),
                                    "bbox_px_y1": float(y + by1),
                                    "bbox_px_x2": float(x + bx2),
                                    "bbox_px_y2": float(y + by2),
                                    "bbox_lon_min": west,
                                    "bbox_lat_min": south,
                                    "bbox_lon_max": east,
                                    "bbox_lat_max": north,
                                    "world_px_z": int(z),
                                    "world_px_x1": float(gx1),
                                    "world_px_y1": float(gy1),
                                    "world_px_x2": float(gx2),
                                    "world_px_y2": float(gy2),
                                    "polygon_lonlat": poly,
                                })
                        except Exception as exc:
                            self.log(f"Satellite YOLO error chunk {chunk_id} model {slot+1}: {exc}")
                    found_here = len(records) - chunk_records_before
                    deleted_tiles = self._satellite_delete_tile_paths(used_tile_paths or [])
                    skipped_note = ""
                    if chunk_small_skipped or chunk_invalid_skipped:
                        skipped_note = f"; skipped tiny/invalid={chunk_small_skipped + chunk_invalid_skipped}"
                    self._satellite_progress(f"Satellite chunk {chunk_id}/{total_chunks}: detections={found_here}{skipped_note}; temp tiles deleted={deleted_tiles}; total detections={len(records)}", force=True)
                    try:
                        del im
                    except Exception:
                        pass
            self.log(f"Satellite Detection raw detections: {len(records)}")
            records = self._satellite_deduplicate_records(records, 0.90)
            self.log(f"Satellite Detection after >90% overlap dedup: {len(records)}")
            self.sat_last_x_min = int(x_min)
            self.sat_last_y_min = int(y_min)
            self.sat_last_z = int(z)
            self.sat_last_records = [dict(r) for r in records]
            self.satellite_detections = [dict(r) for r in records]
            if not records:
                self.log("Satellite Detection finished: YOLO found no detections. No GeoPackage was written, and the temporary tile cache was deleted.")
                self.satellite_redraw_detection_overlay()
                self.refresh_layers()
                return
            crop_base = self._satellite_export_detection_crops(records, x_min, y_min, z)
            if crop_base:
                self.sat_last_crop_base = str(crop_base)
                self.log(f"Satellite crop project folder: {crop_base}")
            self._satellite_features_to_file([dict(r) for r in records], out_path)
            self.satellite_output_last = str(out_path)
            self.log(f"Satellite GeoPackage written: {out_path}")
            read_back = self._satellite_read_detections_from_file(out_path)
            self.satellite_detections = read_back or [dict(r) for r in records]
            self.sat_last_records = self.satellite_detections

            try:
                self._schedule_satellite_preview_refresh(80)
            except Exception:
                pass
            self.log(f"Satellite display detections read from output file: {len(read_back)}")
            self.satellite_redraw_detection_overlay()
            self.refresh_layers()
        finally:
            try:
                shutil.rmtree(temp_cache_root, ignore_errors=True)
                self.log(f"Satellite temporary tile cache deleted: {temp_cache_root}")
            except Exception as exc:
                self.log(f"Satellite temporary cache cleanup warning: {exc}")
            self.sat_detection_temp_cache_root = old_temp_cache

    def _satellite_read_detections_from_file(self, out_path: Path):
        """Read the just-written GeoPackage/GeoJSON and convert geometries back to preview records."""
        try:
            import geopandas as gpd
            gdf = gpd.read_file(out_path)
            records = []
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                try:
                    coords = list(geom.exterior.coords)
                except Exception:
                    minx, miny, maxx, maxy = geom.bounds
                    coords = [(minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny), (minx, maxy)]
                rec = {k: row[k] for k in gdf.columns if k != "geometry"}
                rec["polygon_lonlat"] = [(float(x), float(y)) for x, y in coords]
                records.append(rec)
            return records
        except Exception as exc:
            self.log(f"Satellite output read warning: {exc}; using in-memory records for preview.")
            return list(getattr(self, "satellite_detections", []) or [])

    def _satellite_deduplicate_records(self, records: List[Dict[str, Any]], iou_threshold: float):
        def box(r):
            return (float(r["bbox_px_x1"]), float(r["bbox_px_y1"]), float(r["bbox_px_x2"]), float(r["bbox_px_y2"]))
        ordered = sorted(records, key=lambda r: float(r.get("confidence", 0.0)), reverse=True)
        kept = []
        for r in ordered:
            if any(backend.iou(box(r), box(k)) >= iou_threshold for k in kept):
                continue
            kept.append(r)
        return kept

    def _satellite_draw_records_on_preview(self, records: List[Dict[str, Any]]):
        """Draw current satellite detections as red boxes on the satellite preview."""
        if not hasattr(self, "satellite_view"):
            return
        try:
            for item in list(getattr(self, "satellite_detection_items", []) or []):
                try:
                    self.satellite_view.scene().removeItem(item)
                except Exception:
                    pass
            self.satellite_detection_items = []

            if not records:
                return

            z = int(getattr(self, "sat_preview_z", 3))
            left = float(getattr(self, "sat_preview_left", 0.0))
            top = float(getattr(self, "sat_preview_top", 0.0))
            cw, ch = getattr(self, "sat_last_preview_size", (1, 1))
            cw = int(cw or 1)
            ch = int(ch or 1)

            try:
                max_records = int(os.environ.get("MUSTATIL_SAT_PREVIEW_MAX_BOXES", "1200") or "1200")
            except Exception:
                max_records = 1200
            try:
                max_labels = int(os.environ.get("MUSTATIL_SAT_PREVIEW_MAX_LABELS", "150") or "150")
            except Exception:
                max_labels = 150

            try:
                draw_records = sorted(
                    list(records),
                    key=lambda r: float(r.get("confidence", r.get("conf", 0.0)) or 0.0),
                    reverse=True,
                )
            except Exception:
                draw_records = list(records)

            total = len(draw_records)
            draw_records = draw_records[:max(0, max_records)]

            pen = QPen(QColor("red"))
            pen.setWidth(2)
            shown = 0
            labels = 0

            for r in draw_records:
                try:
                    # Draw detections zoom-independently. Prefer lon/lat geometry because
                    # world/tile pixels are only valid at the zoom where YOLO was run.
                    if "polygon_lonlat" in r and r.get("polygon_lonlat"):
                        pts = r.get("polygon_lonlat") or []
                        worlds = [sat_world_px(float(lon), float(lat), z) for lon, lat in pts]
                        wx1 = min(p[0] for p in worlds); wy1 = min(p[1] for p in worlds)
                        wx2 = max(p[0] for p in worlds); wy2 = max(p[1] for p in worlds)
                    elif all(k in r for k in ("bbox_lon_min", "bbox_lat_min", "bbox_lon_max", "bbox_lat_max")):
                        lon_min = float(r.get("bbox_lon_min")); lat_min = float(r.get("bbox_lat_min"))
                        lon_max = float(r.get("bbox_lon_max")); lat_max = float(r.get("bbox_lat_max"))
                        w1 = sat_world_px(lon_min, lat_max, z)
                        w2 = sat_world_px(lon_max, lat_min, z)
                        wx1, wy1 = w1; wx2, wy2 = w2
                    elif all(k in r for k in ("world_px_x1", "world_px_y1", "world_px_x2", "world_px_y2")):
                        # Backward compatibility: convert stored world pixels from their
                        # original zoom to lon/lat, then project to the current preview zoom.
                        src_z = int(r.get("world_px_z", r.get("zoom", z)) or z)
                        lon1, lat1 = sat_lonlat_from_world_px(float(r.get("world_px_x1")), float(r.get("world_px_y1")), src_z)
                        lon2, lat2 = sat_lonlat_from_world_px(float(r.get("world_px_x2")), float(r.get("world_px_y2")), src_z)
                        w1 = sat_world_px(lon1, lat1, z)
                        w2 = sat_world_px(lon2, lat2, z)
                        wx1, wy1 = w1; wx2, wy2 = w2
                    elif all(k in r for k in ("bbox_world_x1", "bbox_world_y1", "bbox_world_x2", "bbox_world_y2")):
                        src_z = int(r.get("world_px_z", r.get("zoom", z)) or z)
                        lon1, lat1 = sat_lonlat_from_world_px(float(r.get("bbox_world_x1")), float(r.get("bbox_world_y1")), src_z)
                        lon2, lat2 = sat_lonlat_from_world_px(float(r.get("bbox_world_x2")), float(r.get("bbox_world_y2")), src_z)
                        w1 = sat_world_px(lon1, lat1, z)
                        w2 = sat_world_px(lon2, lat2, z)
                        wx1, wy1 = w1; wx2, wy2 = w2
                    elif all(k in r for k in ("bbox_px_x1", "bbox_px_y1", "bbox_px_x2", "bbox_px_y2")):
                        sel_xmin = r.get("tile_x_min", r.get("x_min", None))
                        sel_ymin = r.get("tile_y_min", r.get("y_min", None))
                        src_z = int(r.get("zoom", self.sat_zoom.get()) or self.sat_zoom.get())
                        if sel_xmin is None or sel_ymin is None:
                            try:
                                min_lat, min_lon, max_lat, max_lon = self._satellite_bbox()
                                sel_xmin, sel_ymin, _, _ = sat_tile_bounds_for_bbox(min_lat, min_lon, max_lat, max_lon, src_z)
                            except Exception:
                                sel_xmin = int(left // WEB_TILE_SIZE)
                                sel_ymin = int(top // WEB_TILE_SIZE)
                        src_wx_off = int(sel_xmin) * WEB_TILE_SIZE
                        src_wy_off = int(sel_ymin) * WEB_TILE_SIZE
                        src_wx1 = src_wx_off + float(r.get("bbox_px_x1"))
                        src_wy1 = src_wy_off + float(r.get("bbox_px_y1"))
                        src_wx2 = src_wx_off + float(r.get("bbox_px_x2"))
                        src_wy2 = src_wy_off + float(r.get("bbox_px_y2"))
                        lon1, lat1 = sat_lonlat_from_world_px(src_wx1, src_wy1, src_z)
                        lon2, lat2 = sat_lonlat_from_world_px(src_wx2, src_wy2, src_z)
                        w1 = sat_world_px(lon1, lat1, z)
                        w2 = sat_world_px(lon2, lat2, z)
                        wx1, wy1 = w1; wx2, wy2 = w2
                    else:
                        continue

                    sx1 = wx1 - left
                    sy1 = wy1 - top
                    sx2 = wx2 - left
                    sy2 = wy2 - top
                    x1, x2 = sorted((float(sx1), float(sx2)))
                    y1, y2 = sorted((float(sy1), float(sy2)))

                    if x2 < 0 or y2 < 0 or x1 > cw or y1 > ch:
                        continue
                    if (x2 - x1) < 2 or (y2 - y1) < 2:
                        continue

                    rect = self.satellite_view.scene().addRect(x1, y1, x2 - x1, y2 - y1, pen)
                    rect.setZValue(500)
                    self.satellite_detection_items.append(rect)
                    shown += 1

                    if labels < max_labels:
                        conf = float(r.get("confidence", r.get("conf", 0.0)) or 0.0)
                        label = f"{conf:.2f}"
                        fs = r.get("form_score", None)
                        if fs is not None:
                            try:
                                label += f" F{float(fs):.2f}"
                            except Exception:
                                pass
                        txt = self.satellite_view.scene().addText(label)
                        txt.setDefaultTextColor(QColor("red"))
                        txt.setPos(x1, max(0.0, y1 - 18.0))
                        txt.setZValue(501)
                        self.satellite_detection_items.append(txt)
                        labels += 1
                except Exception:
                    continue

            if total > max_records:
                self._satellite_progress(
                    f"Satellite preview overlay limited: showing {shown} / {total} red detection boxes. Set MUSTATIL_SAT_PREVIEW_MAX_BOXES to change.",
                    min_seconds=10.0,
                )
        except Exception as exc:
            self.log("Satellite detection overlay warning: " + str(exc))


    def satellite_save_selected_bigtiff(self):
        """Save the selected satellite map as a georeferenced GeoTIFF/BigTIFF.

        Preferred path uses rasterio so QGIS reads CRS/transform directly. If rasterio is not
        available, the TIFF is still streamed with tifffile and a .tfw + .prj sidecar is written.
        """
        min_lat, min_lon, max_lat, max_lon = self._satellite_bbox()
        z = int(self.sat_zoom.get())
        x_min, y_min, x_max, y_max = sat_tile_bounds_for_bbox(min_lat, min_lon, max_lat, max_lon, z)
        cols = x_max - x_min + 1; rows = y_max - y_min + 1
        width = cols * WEB_TILE_SIZE; height = rows * WEB_TILE_SIZE
        out_path = self._satellite_output_tif_path()
        west, north, res = sat_webmercator_origin_for_tile(x_min, y_min, z)
        self.log(f"Satellite GeoTIFF started: z={z}, tiles={cols}x{rows}, pixels={width:,}x{height:,}, EPSG:3857")

        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("Satellite BigTIFF export needs numpy. Install it first: pip install numpy") from exc

        try:
            import rasterio
            from rasterio.transform import from_origin
            transform = from_origin(west, north, res, res)
            profile = {
                "driver": "GTiff",
                "height": int(height),
                "width": int(width),
                "count": 3,
                "dtype": "uint8",
                "crs": "EPSG:3857",
                "transform": transform,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "IF_SAFER",
            }
            written = 0; total = cols * rows
            with rasterio.open(out_path, "w", **profile) as dst:
                for row in range(rows):
                    for col in range(cols):
                        tx = x_min + col; ty = y_min + row
                        data, _tile_path = self._download_sat_tile(tx, ty, z, preview=False)
                        tile = sat_decode_tile(data)
                        arr = np.asarray(tile, dtype="uint8")
                        window = rasterio.windows.Window(col * WEB_TILE_SIZE, row * WEB_TILE_SIZE, WEB_TILE_SIZE, WEB_TILE_SIZE)
                        dst.write(arr[:, :, 0], 1, window=window)
                        dst.write(arr[:, :, 1], 2, window=window)
                        dst.write(arr[:, :, 2], 3, window=window)
                        written += 1
                        if written % 10 == 0 or written == total:
                            self.log(f"Satellite GeoTIFF write: {written}/{total} tiles")
            self.log(f"Satellite georeferenced GeoTIFF written: {out_path}")
            return
        except Exception as exc:
            self.log(f"Rasterio GeoTIFF writer unavailable/failed ({exc}); falling back to tifffile + QGIS sidecars.")

        try:
            import tifffile
        except Exception as exc:
            raise RuntimeError("Fallback BigTIFF export needs tifffile. Install it first: pip install tifffile rasterio") from exc
        bigtiff = width * height * 3 > 3_800_000_000
        mem = tifffile.memmap(str(out_path), shape=(height, width, 3), dtype="uint8", bigtiff=bigtiff)
        written = 0; total = cols * rows
        try:
            for row in range(rows):
                for col in range(cols):
                    tx = x_min + col; ty = y_min + row
                    data, _tile_path = self._download_sat_tile(tx, ty, z, preview=False)
                    tile = sat_decode_tile(data)
                    mem[row*WEB_TILE_SIZE:(row+1)*WEB_TILE_SIZE, col*WEB_TILE_SIZE:(col+1)*WEB_TILE_SIZE, :] = np.asarray(tile, dtype="uint8")
                    written += 1
                    if written % 10 == 0 or written == total:
                        self.log(f"Satellite BigTIFF write: {written}/{total} tiles")
            mem.flush()
        finally:
            del mem
        self._satellite_write_georef_sidecars(out_path, x_min, y_min, z)
        self.log(f"Satellite BigTIFF written with QGIS georef sidecars: {out_path}")

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
        all_dets = list(getattr(self, "dets", []) or [])
        visible_dets = list(self.visible()) if hasattr(self, "visible") else []
        all_no_90 = self._count_without_heavy_overlap(all_dets, iou_threshold=0.90)
        visible_no_90 = self._count_without_heavy_overlap(visible_dets, iou_threshold=0.90)
        sat_all = list(getattr(self, "satellite_detections", []) or getattr(self, "sat_last_records", []) or [])
        try:
            sat_visible = self._satellite_visible_records(sat_all)
        except Exception:
            sat_visible = list(sat_all)
        try:
            sat_all_no_90 = len(self._satellite_deduplicate_records(list(sat_all), 0.90))
            sat_visible_no_90 = len(self._satellite_deduplicate_records(list(sat_visible), 0.90))
        except Exception:
            sat_all_no_90 = len(sat_all)
            sat_visible_no_90 = len(sat_visible)
        for text in [
            f"Image: {Path(self.image.get()).name if self.image.get() else '-'}",
            f"YOLO detections: {len(all_dets)}",
            f"YOLO detections without >90% overlap: {all_no_90}",
            f"Visible detections: {len(visible_dets)}",
            f"Visible detections without >90% overlap: {visible_no_90}",
            f"FormLearner kept: {len(getattr(self, 'fl_kept', []))}",
            f"SAM2 images: {len(getattr(self, 'sam_images', []))}",
            f"Satellite detections: {len(sat_all)}",
            f"Satellite detections without >90% overlap: {sat_all_no_90}",
            f"Visible satellite detections: {len(sat_visible)}",
            f"Visible satellite detections without >90% overlap: {sat_visible_no_90}",
            f"Satellite output: {Path(getattr(self, 'satellite_output_last', '')).name if getattr(self, 'satellite_output_last', '') else '-'}",
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
