#!/usr/bin/env python3
"""
Mustatil Trainer & Detector
One-file GUI for annotating satellite/map imagery, training a YOLO detector,
and running tiled inference on very large maps. Exports detections as GeoJSON
when rasterio can read georeferencing; otherwise exports pixel-coordinate GeoJSON.

Install CPU-safe dependencies:
    python -m pip install --upgrade pip
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    python -m pip install ultralytics pillow opencv-python numpy pyyaml tqdm pandas matplotlib

Optional GeoTIFF georeferencing:
    python -m pip install rasterio shapely

Run:
    python mustatil_trainer_detector.py
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import importlib.util
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
except Exception as exc:
    raise SystemExit(f"Tkinter is required: {exc}")

try:
    from PIL import Image, ImageTk
    Image.MAX_IMAGE_PIXELS = None
except Exception:
    raise SystemExit("Missing Pillow. Install with: python -m pip install pillow")

try:
    import yaml
except Exception:
    yaml = None

APP_TITLE = "Mustatil Trainer - GeoPackage Export"
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def resolve_model_path(model_name: str, project_root: Path, log_fn=None) -> Path:
    """
    Keeps YOLO weights inside the project folder so Ultralytics never tries to
    download into a protected/current working directory.
    """
    model_name = (model_name or "yolov8n.pt").strip()
    p = Path(model_name)

    if p.exists():
        return p.resolve()

    weights_dir = project_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # If user entered only yolov8n.pt, store it in project/weights/yolov8n.pt
    target = weights_dir / p.name

    if target.exists() and target.stat().st_size > 100_000:
        if log_fn:
            log_fn(f"Using existing local model: {target}")
        return target

    known = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt",
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt",
        "yolov5nu.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov5nu.pt",
    }

    if p.name not in known:
        # Unknown model name: return project-local path. Ultralytics may still download if supported.
        return target

    url = known[p.name]
    if log_fn:
        log_fn(f"Downloading base model to writable project folder:")
        log_fn(f"{target}")
        log_fn(url)

    tmp = target.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size < 100_000:
        raise RuntimeError(f"Downloaded model is too small/invalid: {tmp}")
    tmp.replace(target)
    return target.resolve()



def resolve_sam_model_path(model_name: str, project_root: Optional[Path] = None, log_fn=None) -> str:
    """
    Resolve SAM/SAM2 model to an existing local file when possible.
    If a local file is selected via Browse, Ultralytics will not auto-download.
    """
    name = (model_name or "").strip() or "sam2_b.pt"
    p = Path(name)

    if p.exists() and p.stat().st_size > 100_000:
        rp = str(p.resolve())
        if log_fn:
            log_fn(f"SAM/SAM2 local file selected: {rp}")
        return rp

    search_dirs = []
    if project_root is not None:
        search_dirs += [project_root / "weights", project_root]
    search_dirs += [
        Path.cwd(),
        Path.home() / "Desktop" / "sam_models",
        Path.home() / "Downloads",
        Path.home(),
    ]

    for d in search_dirs:
        try:
            candidate = d / p.name
            if candidate.exists() and candidate.stat().st_size > 100_000:
                rp = str(candidate.resolve())
                if log_fn:
                    log_fn(f"SAM/SAM2 model found locally: {rp}")
                return rp
        except Exception:
            pass

    if log_fn:
        log_fn(f"SAM/SAM2 model not found locally: {name}")
        log_fn("If download fails, use Browse SAM/SAM2 and choose the local .pt file.")
    return name


def sam_masks_to_global_polygons(sam_results, offset_x: int, offset_y: int):
    """
    Converts Ultralytics SAM/SAM2 result masks to global pixel polygons.
    Returns list of polygon point lists.
    """
    polygons = []
    if not sam_results:
        return polygons

    try:
        res = sam_results[0]
        if res.masks is None:
            return polygons

        # Ultralytics masks.xy gives pixel-space contours.
        masks_xy = getattr(res.masks, "xy", None)
        if masks_xy is None:
            return polygons

        for poly in masks_xy:
            pts = []
            for p in poly:
                if len(p) >= 2:
                    pts.append((float(p[0]) + offset_x, float(p[1]) + offset_y))
            if len(pts) >= 3:
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                polygons.append(pts)
    except Exception:
        return polygons

    return polygons



def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_report() -> Tuple[bool, str]:
    """
    Checks runtime dependencies without crashing the GUI.
    """
    missing = []
    for mod in ["torch", "ultralytics", "yaml", "PIL", "cv2", "numpy"]:
        if not have_module(mod):
            missing.append(mod)

    py = sys.executable
    if missing:
        msg = (
            "Missing dependencies: " + ", ".join(missing) + "\n\n"
            "Install with the SAME Python used to start this GUI:\n\n"
            f'"{py}" -m pip install --upgrade pip\n'
            f'"{py}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n'
            f'"{py}" -m pip install ultralytics pillow opencv-python numpy pyyaml tqdm pandas matplotlib\n\n'
            "Then restart this app."
        )
        return False, msg

    try:
        import torch
        import ultralytics
        msg = (
            f"Python: {sys.executable}\n"
            f"torch: {getattr(torch, '__version__', 'unknown')}\n"
            f"ultralytics: {getattr(ultralytics, '__file__', 'unknown')}\n"
            f"CUDA available: {torch.cuda.is_available()}\n\n"
            "AMD R9 390X is not supported by PyTorch CUDA. Use CPU mode."
        )
        return True, msg
    except Exception as exc:
        return False, f"Dependency import failed:\n{exc}"


def show_dependency_help(parent=None):
    ok, msg = dependency_report()
    if parent is not None:
        messagebox.showinfo("Dependency Check" if ok else "Missing Dependencies", msg)
    return ok, msg



@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized_yolo(self, w: int, h: int) -> Tuple[int, float, float, float, float]:
        x1, x2 = sorted((max(0, self.x1), min(w - 1, self.x2)))
        y1, y2 = sorted((max(0, self.y1), min(h - 1, self.y2)))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = x1 + bw / 2
        cy = y1 + bh / 2
        return self.cls, cx / w, cy / h, bw / w, bh / h

    @staticmethod
    def from_yolo(cls: int, cx: float, cy: float, bw: float, bh: float, w: int, h: int) -> "Box":
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        return Box(cls, x1, y1, x2, y2)


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.images_dir = root / "images"
        self.labels_dir = root / "labels"
        self.runs_dir = root / "runs"
        self.exports_dir = root / "exports"
        self.config_path = root / "project.json"
        self.classes = ["mustatil", "false_positive"]
        for d in [self.images_dir, self.labels_dir, self.runs_dir, self.exports_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        if self.config_path.exists():
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.classes = data.get("classes", ["mustatil", "false_positive"])
            if "false_positive" not in self.classes:
                self.classes.append("false_positive")
                self.save()
        else:
            self.save()

    def save(self):
        self.config_path.write_text(json.dumps({"classes": self.classes}, indent=2), encoding="utf-8")

    def image_files(self) -> List[Path]:
        return sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGES])

    def label_path(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def load_boxes(self, image_path: Path) -> List[Box]:
        lp = self.label_path(image_path)
        if not lp.exists():
            return []
        with Image.open(image_path) as im:
            w, h = im.size
        boxes: List[Box] = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0])); cx, cy, bw, bh = map(float, parts[1:])
            boxes.append(Box.from_yolo(cls, cx, cy, bw, bh, w, h))
        return boxes

    def save_boxes(self, image_path: Path, boxes: List[Box]):
        with Image.open(image_path) as im:
            w, h = im.size
        lines = []
        for b in boxes:
            cls, cx, cy, bw, bh = b.normalized_yolo(w, h)
            if bw > 0 and bh > 0:
                lines.append(f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        self.label_path(image_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_cmd_live(cmd: List[str], log_fn, cwd: Optional[Path] = None):
    """
    Runs a subprocess and streams output into the GUI.
    Windows/Anaconda often emits UTF-8 progress characters that crash cp1252
    decoding with: 'charmap' codec can't decode byte ...
    This reader forces UTF-8 and replaces undecodable bytes.
    """
    log_fn("$ " + " ".join(map(str, cmd)))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert p.stdout is not None
    for line in p.stdout:
        # Remove common terminal control sequences from tqdm/Ultralytics output.
        clean = line.replace("\x1b[K", "").rstrip()
        log_fn(clean)
    code = p.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}")



def nms_xyxy_np(boxes, scores, iou_threshold=0.45):
    import numpy as np
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def prepare_onnx_input(pil_img, input_size):
    import numpy as np
    resized = pil_img.resize((input_size, input_size))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return arr[None, :, :, :]


def parse_yolo_onnx_output(outputs, crop_w, crop_h, input_size, conf_threshold, debug_fn=None):
    """
    ONNX parser using Ultralytics' own NMS when available.
    This is closer to YOLO(best.pt).predict() than a hand-written parser.

    Expected raw YOLOv8 output:
        (1, 5, N) for one class
        or (1, 4+nc, N)
    Returns:
        local detections: x1,y1,x2,y2,score,class_id
    """
    import numpy as np

    out = np.asarray(outputs[0])

    if debug_fn:
        try:
            debug_fn(f"ONNX raw output shape: {out.shape}, min={float(np.nanmin(out)):.6f}, max={float(np.nanmax(out)):.6f}")
            flat_scores = out.reshape(-1)
            debug_fn(f"ONNX raw global top values: " + ", ".join(f"{float(v):.4f}" for v in np.sort(flat_scores)[-10:][::-1]))
        except Exception:
            pass

    # Preferred path: use Ultralytics NMS directly.
    try:
        import torch
        from ultralytics.utils.ops import non_max_suppression

        pred = torch.from_numpy(out).float()

        # Ensure batch dimension.
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)

        # non_max_suppression expects (batch, 4+classes, anchors) for YOLOv8 raw output.
        # If model produced (batch, anchors, channels), transpose.
        if pred.ndim == 3 and pred.shape[1] > pred.shape[2]:
            pred = pred.transpose(1, 2)

        nc = max(1, int(pred.shape[1] - 4)) if pred.ndim == 3 else 1

        if debug_fn:
            debug_fn(f"Ultralytics NMS input shape: {tuple(pred.shape)}, nc={nc}, conf={conf_threshold}")

        nms_out = non_max_suppression(
            pred,
            conf_thres=float(conf_threshold),
            iou_thres=0.45,
            nc=nc,
            max_det=300,
        )

        dets = []
        if nms_out and len(nms_out[0]):
            arr = nms_out[0].cpu().numpy()
            sx = float(crop_w) / float(input_size)
            sy = float(crop_h) / float(input_size)
            for row in arr:
                x1, y1, x2, y2, score, cls = row[:6]
                x1, x2 = float(x1) * sx, float(x2) * sx
                y1, y2 = float(y1) * sy, float(y2) * sy
                x1 = max(0.0, min(float(crop_w - 1), x1))
                x2 = max(0.0, min(float(crop_w - 1), x2))
                y1 = max(0.0, min(float(crop_h - 1), y1))
                y2 = max(0.0, min(float(crop_h - 1), y2))
                if x2 > x1 and y2 > y1:
                    dets.append((x1, y1, x2, y2, float(score), int(cls)))

        if debug_fn:
            debug_fn(f"Ultralytics NMS detections on this tile: {len(dets)}")

        return dets

    except Exception as exc:
        if debug_fn:
            debug_fn(f"Ultralytics NMS path failed, fallback parser used: {exc}")

    # Fallback parser.
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))

    if out.ndim == 3:
        out = out[0]

    if out.ndim != 2:
        return []

    if out.shape[0] <= 200 and out.shape[0] < out.shape[1]:
        pred = out.T
    else:
        pred = out

    if pred.shape[1] < 5:
        return []

    xywh = pred[:, :4].astype(np.float32)
    scores_all = pred[:, 4:].astype(np.float32)

    if scores_all.shape[1] == 1:
        scores = scores_all[:, 0]
        cls_ids = np.zeros_like(scores, dtype=np.int64)
    else:
        cls_ids = scores_all.argmax(axis=1)
        scores = scores_all.max(axis=1)

    if np.nanmax(scores) > 1.0 or np.nanmin(scores) < 0.0:
        scores = sigmoid(scores)

    if debug_fn:
        try:
            debug_fn("Fallback ONNX top scores: " + ", ".join(f"{float(v):.4f}" for v in np.sort(scores)[-10:][::-1]))
        except Exception:
            pass

    mask = scores >= conf_threshold
    if not np.any(mask):
        return []

    xywh = xywh[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    if np.nanmax(xywh) <= 2.0:
        xywh = xywh * float(input_size)

    cx, cy, bw, bh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    boxes = np.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], axis=1)

    sx = float(crop_w) / float(input_size)
    sy = float(crop_h) / float(input_size)
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, max(1, crop_w-1))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, max(1, crop_h-1))

    keep = nms_xyxy_np(boxes, scores, 0.45) if "nms_xyxy_np" in globals() else list(range(len(boxes)))

    dets = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i].tolist()
        if x2 > x1 and y2 > y1:
            dets.append((x1, y1, x2, y2, float(scores[i]), int(cls_ids[i])))

    return dets

    # Case B: Ultralytics raw output. Usually channels x anchors.
    if out.ndim != 2:
        return []

    # Convert to anchors x channels if needed.
    if out.shape[0] <= 200 and out.shape[0] < out.shape[1]:
        pred = out.T
    else:
        pred = out

    if pred.shape[1] < 5:
        return []

    xywh = pred[:, :4].astype(np.float32)
    score_part = pred[:, 4:].astype(np.float32)

    # Some exports include objectness + classes, but YOLOv8 usually does not.
    # For one-class YOLOv8 shape is 5 channels: cx,cy,w,h,class_score.
    if score_part.shape[1] == 1:
        scores = score_part[:, 0]
        cls_ids = np.zeros_like(scores, dtype=np.int64)
    else:
        # If there are 2+ score columns, treat all as class scores and take max.
        cls_ids = score_part.argmax(axis=1)
        scores = score_part.max(axis=1)

    # If score values look like logits, apply sigmoid.
    raw_min = float(np.nanmin(scores)) if scores.size else 0.0
    raw_max = float(np.nanmax(scores)) if scores.size else 0.0
    if raw_max > 1.0 or raw_min < 0.0:
        scores = sigmoid(scores)
        if debug_fn:
            debug_fn(f"ONNX scores looked like logits; sigmoid applied. raw max={raw_max:.5f}")

    if debug_fn:
        try:
            top = np.sort(scores)[-10:][::-1]
            debug_fn("ONNX top scores: " + ", ".join(f"{float(v):.4f}" for v in top))
        except Exception:
            pass

    mask = scores >= conf_threshold
    if not np.any(mask):
        return []

    xywh = xywh[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    cx, cy, bw, bh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]

    # Ultralytics exported boxes are normally in input pixel coordinates.
    # If coordinates look normalized, scale by input_size first.
    if np.nanmax(xywh[:, :4]) <= 2.0:
        cx, bw = cx * input_size, bw * input_size
        cy, bh = cy * input_size, bh * input_size

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    sx = float(crop_w) / float(input_size)
    sy = float(crop_h) / float(input_size)

    boxes = np.stack([x1 * sx, y1 * sy, x2 * sx, y2 * sy], axis=1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, max(1, crop_w - 1))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, max(1, crop_h - 1))

    keep = nms_xyxy_np(boxes, scores, 0.45)

    for i in keep:
        bx = boxes[i].tolist()
        if bx[2] <= bx[0] or bx[3] <= bx[1]:
            continue
        dets.append((bx[0], bx[1], bx[2], bx[3], float(scores[i]), int(cls_ids[i])))

    return dets


def create_ort_session(model_path, backend_name, log_fn):
    import onnxruntime as ort

    available = ort.get_available_providers()
    log_fn(f"ONNX Runtime available providers: {available}")

    if backend_name == "ONNX Runtime DirectML GPU":
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        if "DmlExecutionProvider" not in available:
            raise RuntimeError("DmlExecutionProvider is not available. Install: pip install onnxruntime-directml")
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=providers)
    active = session.get_providers()
    log_fn(f"ONNX Runtime active providers: {active}")

    if backend_name == "ONNX Runtime DirectML GPU":
        if active and active[0] == "DmlExecutionProvider":
            log_fn("GPU STATUS: DirectML is ACTIVE. AMD GPU inference should be used.")
        else:
            log_fn("GPU STATUS: DirectML requested, but active provider order does not show it first.")

    return session





def get_onnx_fixed_input_size(model_path: Path, log_fn=None) -> Optional[int]:
    """
    Reads fixed ONNX input size from the model itself.
    For exports like best.onnx with input [1,3,640,640], returns 640.
    Returns None for dynamic or unreadable models.
    """
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = list(inp.shape)
        if len(shape) >= 4:
            h = shape[2]
            w = shape[3]
            if isinstance(h, int) and isinstance(w, int) and h == w and h > 0:
                if log_fn:
                    log_fn(f"ONNX fixed input size detected: {w}x{h}")
                return int(h)
            if log_fn:
                log_fn(f"ONNX input shape is dynamic or non-square: {shape}")
    except Exception as exc:
        if log_fn:
            log_fn(f"Could not read ONNX input size, using GUI tile size: {exc}")
    return None


def is_probably_geotiff(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def rasterio_available() -> bool:
    return importlib.util.find_spec("rasterio") is not None


def read_rasterio_window_as_pil(src_raster, x: int, y: int, w: int, h: int):
    """
    Reads only one raster window from a huge GeoTIFF and returns a PIL RGB image.
    """
    import numpy as np
    from rasterio.windows import Window

    win = Window(x, y, w, h)
    if src_raster.count >= 3:
        arr = src_raster.read([1, 2, 3], window=win, boundless=True, fill_value=0)
        arr = np.moveaxis(arr, 0, -1)
    else:
        arr = src_raster.read(1, window=win, boundless=True, fill_value=0)
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.dtype != np.uint8:
        arr = arr.astype("float32")
        mn = float(np.nanmin(arr)) if arr.size else 0.0
        mx = float(np.nanmax(arr)) if arr.size else 1.0
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        arr = np.clip(arr, 0, 255).astype("uint8")

    return Image.fromarray(arr, "RGB")


def open_large_image_info(path: Path, log_fn=None):
    """
    Returns (width, height, mode, reader).

    IMPORTANT STABILITY PATCH:
    For GeoTIFF/TIFF files this function refuses PIL full-image mode.
    Huge TIFFs must be streamed by rasterio windows. Otherwise PIL can load
    hundreds of MB/GB and the GUI appears to hang.
    """
    if is_probably_geotiff(path):
        if not rasterio_available():
            raise RuntimeError(
                "GeoTIFF/TIFF detection requires rasterio in this environment. "
                "Install it with: python -m pip install rasterio"
            )
        try:
            import rasterio
            src_r = rasterio.open(path)
            if log_fn:
                log_fn("Large image mode: rasterio streaming GeoTIFF")
                log_fn(f"Raster size: {src_r.width} x {src_r.height}; bands={src_r.count}; crs={src_r.crs}")
            return src_r.width, src_r.height, "rasterio", src_r
        except Exception as exc:
            raise RuntimeError(
                f"Rasterio could not open this TIFF. Copy it from network drive to local SSD "
                f"or convert/rebuild it as tiled GeoTIFF/BigTIFF. Details: {exc}"
            )

    im = Image.open(path)
    if log_fn:
        log_fn("Large image mode: PIL normal-image mode")
        log_fn(f"Image size: {im.width} x {im.height}")
    return im.width, im.height, "pil", im


def read_detection_tile(reader, mode: str, x: int, y: int, tile: int, W: int, H: int):
    """
    Reads one tile as PIL RGB.
    """
    w = min(tile, W - x)
    h = min(tile, H - y)
    if mode == "rasterio":
        return read_rasterio_window_as_pil(reader, x, y, w, h)
    return reader.crop((x, y, x + w, y + h)).convert("RGB")



def green_vegetation_ratio_rgb(crop_rgb) -> float:
    """
    Returns the ratio of pixels that look vegetation-like in an RGB crop.
    Tuned for satellite desert imagery: catches green/olive shrubs, but avoids
    rejecting most bright sand/stone.
    """
    try:
        import numpy as np
        import cv2

        arr = np.asarray(crop_rgb)
        if arr.size == 0:
            return 0.0
        if arr.ndim != 3 or arr.shape[2] < 3:
            return 0.0

        hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Main vegetation band in OpenCV HSV: green/olive/yellow-green.
        # The V lower bound is intentionally low because desert shrubs can be dark.
        mask_green = (h >= 25) & (h <= 95) & (s >= 28) & (v >= 22)

        # Extra olive/dark vegetation condition: green channel dominates slightly,
        # useful for brown-green bushes that are not very saturated.
        r = arr[:, :, 0].astype("int16")
        g = arr[:, :, 1].astype("int16")
        b = arr[:, :, 2].astype("int16")
        mask_olive = (g > r + 4) & (g > b + 4) & (v >= 18) & (s >= 18)

        mask = mask_green | mask_olive
        return float(mask.mean())
    except Exception:
        return 0.0


def detection_crop_green_ratio(tile_rgb_array, bbox) -> float:
    """
    bbox is local tile xyxy. Returns vegetation ratio for that box crop.
    """
    try:
        import numpy as np
        x1, y1, x2, y2 = map(float, bbox)
        h, w = tile_rgb_array.shape[:2]
        ix1 = max(0, min(w - 1, int(math.floor(x1))))
        iy1 = max(0, min(h - 1, int(math.floor(y1))))
        ix2 = max(0, min(w, int(math.ceil(x2))))
        iy2 = max(0, min(h, int(math.ceil(y2))))
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        crop = tile_rgb_array[iy1:iy2, ix1:ix2, :3]
        return green_vegetation_ratio_rgb(crop)
    except Exception:
        return 0.0


def load_preview_downsample(path: Path, max_size: int = 2400):
    """
    Loads a small preview image. For GeoTIFF uses rasterio downsample.
    Stability patch: huge TIFFs are not opened by PIL if rasterio is missing.
    Returns (preview_image, original_width, original_height).
    """
    if is_probably_geotiff(path):
        if not rasterio_available():
            raise RuntimeError(
                "Preview for TIFF requires rasterio. Install: python -m pip install rasterio. "
                "This prevents PIL from loading the whole TIFF and freezing the GUI."
            )
        try:
            import rasterio
            import numpy as np
            with rasterio.open(path) as src_r:
                scale = min(max_size / max(1, src_r.width), max_size / max(1, src_r.height), 1.0)
                out_w = max(1, int(src_r.width * scale))
                out_h = max(1, int(src_r.height * scale))
                if src_r.count >= 3:
                    arr = src_r.read([1, 2, 3], out_shape=(3, out_h, out_w))
                    arr = np.moveaxis(arr, 0, -1)
                else:
                    arr = src_r.read(1, out_shape=(out_h, out_w))
                    arr = np.stack([arr, arr, arr], axis=-1)

                if arr.dtype != np.uint8:
                    arr = arr.astype("float32")
                    mn = float(np.nanmin(arr)) if arr.size else 0.0
                    mx = float(np.nanmax(arr)) if arr.size else 1.0
                    if mx > mn:
                        arr = (arr - mn) / (mx - mn) * 255.0
                    arr = np.clip(arr, 0, 255).astype("uint8")
                return Image.fromarray(arr, "RGB"), src_r.width, src_r.height
        except Exception as exc:
            raise RuntimeError(f"Rasterio preview failed. Try copying TIFF to local SSD. Details: {exc}")

    im = Image.open(path).convert("RGB")
    orig_w, orig_h = im.size
    im.thumbnail((max_size, max_size))
    return im.copy(), orig_w, orig_h


def sam2_refine_box_safe(sam_model, tile_pil: Image.Image, local_bbox, global_offset_x: int, global_offset_y: int,
                         padding: int = 96, max_crop_size: int = 768, log_fn=None):
    """
    Safer SAM2 refinement:
    - crops around one YOLO box instead of sending the whole large tile
    - pads the crop so SAM2 has context
    - downsamples very large crops
    - returns global pixel polygon or None
    """
    import numpy as np

    lx1, ly1, lx2, ly2 = map(float, local_bbox)
    W, H = tile_pil.size

    pad = max(0, int(padding))
    cx1 = max(0, int(lx1 - pad))
    cy1 = max(0, int(ly1 - pad))
    cx2 = min(W, int(lx2 + pad))
    cy2 = min(H, int(ly2 + pad))

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = tile_pil.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    crop_w, crop_h = crop.size

    # BBox relative to the crop
    rb = [lx1 - cx1, ly1 - cy1, lx2 - cx1, ly2 - cy1]

    scale = 1.0
    max_side = max(crop_w, crop_h)
    max_crop_size = max(128, int(max_crop_size))
    if max_side > max_crop_size:
        scale = max_crop_size / float(max_side)
        new_w = max(1, int(crop_w * scale))
        new_h = max(1, int(crop_h * scale))
        crop = crop.resize((new_w, new_h))
        rb = [v * scale for v in rb]

    arr = np.array(crop)

    try:
        # One prompt per crop. Much safer than whole-tile SAM2.
        sam_results = sam_model.predict(arr, bboxes=[rb], verbose=False)
        polys = sam_masks_to_global_polygons(sam_results, 0, 0)
        if not polys:
            return None
        poly = max(polys, key=len)

        # Convert crop/downsample coordinates back to original global image coordinates.
        out = []
        inv = 1.0 / scale
        for px, py in poly:
            ox = float(px) * inv + cx1 + global_offset_x
            oy = float(py) * inv + cy1 + global_offset_y
            out.append((ox, oy))
        if len(out) >= 3 and out[0] != out[-1]:
            out.append(out[0])
        return out if len(out) >= 4 else None
    except Exception as exc:
        if log_fn:
            log_fn(f"SAM2 safe refine failed: {exc}")
        return None



def worldfile_candidates(image_path: Path):
    """
    Common worldfile names:
    .tfw for .tif/.tiff
    .jgw for .jpg/.jpeg
    .pgw for .png
    .bpw for .bmp
    .wld generic
    """
    suffix = image_path.suffix.lower()
    candidates = []

    if suffix in {".tif", ".tiff"}:
        candidates += [image_path.with_suffix(".tfw"), image_path.with_suffix(".tifw")]
    elif suffix in {".jpg", ".jpeg"}:
        candidates += [image_path.with_suffix(".jgw"), image_path.with_suffix(".jpgw")]
    elif suffix == ".png":
        candidates += [image_path.with_suffix(".pgw"), image_path.with_suffix(".pngw")]
    elif suffix == ".bmp":
        candidates += [image_path.with_suffix(".bpw"), image_path.with_suffix(".bmpw")]
    elif suffix == ".webp":
        candidates += [image_path.with_suffix(".wld")]

    candidates += [
        image_path.with_suffix(".wld"),
        image_path.with_name(image_path.name + ".wld"),
        image_path.with_name(image_path.name + "w"),
    ]

    # Keep order, remove duplicates
    seen = set()
    out = []
    for p in candidates:
        if str(p).lower() not in seen:
            seen.add(str(p).lower())
            out.append(p)
    return out


def read_worldfile_transform(image_path: Path, log_fn=None):
    """
    Reads an ESRI world file.

    World file lines:
    A: pixel size x
    D: rotation y
    B: rotation x
    E: pixel size y, usually negative
    C: x coordinate of center of upper-left pixel
    F: y coordinate of center of upper-left pixel

    For pixel corner coordinates used by GIS transforms:
    x_geo = A*x + B*y + C_corner
    y_geo = D*x + E*y + F_corner

    C_corner = C - A/2 - B/2
    F_corner = F - D/2 - E/2
    """
    for wf in worldfile_candidates(image_path):
        if wf.exists():
            vals = []
            for line in wf.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip().replace(",", ".")
                if line:
                    vals.append(float(line))
            if len(vals) < 6:
                raise RuntimeError(f"Invalid worldfile, expected 6 numbers: {wf}")

            A, D, B, E, C, F = vals[:6]
            C_corner = C - A / 2.0 - B / 2.0
            F_corner = F - D / 2.0 - E / 2.0

            if log_fn:
                log_fn(f"Worldfile georeference found: {wf}")
                log_fn(f"Worldfile transform: A={A}, B={B}, C_corner={C_corner}, D={D}, E={E}, F_corner={F_corner}")

            return (A, B, C_corner, D, E, F_corner), wf

    return None, None


def apply_affine_tuple(transform_tuple, px, py):
    """
    Applies a simple 6-number affine tuple:
    (A, B, C, D, E, F)
    x_geo = A*x + B*y + C
    y_geo = D*x + E*y + F
    """
    A, B, C, D, E, F = transform_tuple
    x_geo = A * float(px) + B * float(py) + C
    y_geo = D * float(px) + E * float(py) + F
    return x_geo, y_geo


def read_prj_or_default_crs(image_path: Path, default_crs: str = "EPSG:3857", log_fn=None):
    """
    GeoJSON CRS is not always honored by QGIS, but we write it for clarity.
    For worldfile sidecars, CRS is usually stored separately in .prj.
    If no .prj exists, we use default_crs.
    """
    prj_candidates = [
        image_path.with_suffix(".prj"),
        image_path.with_name(image_path.name + ".prj"),
    ]
    for prj in prj_candidates:
        if prj.exists():
            txt = prj.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                if log_fn:
                    log_fn(f"PRJ sidecar found: {prj}")
                # Keep WKT text in crs name field; QGIS usually still asks/infers, but it is informative.
                return txt

    if default_crs:
        if log_fn:
            log_fn(f"No .prj sidecar found. Using default CRS label: {default_crs}")
        return default_crs
    return None



def read_geotiff_tags_with_pillow(image_path: Path, log_fn=None):
    """
    Pure Pillow GeoTIFF georeference fallback.
    Reads common GeoTIFF tags without rasterio/tifffile:

    33550 ModelPixelScaleTag = (scale_x, scale_y, scale_z)
    33922 ModelTiepointTag   = (i, j, k, x, y, z, ...)
    34735 GeoKeyDirectoryTag = GeoKeys, used here to detect EPSG if possible

    Returns:
        (transform_tuple, crs_name) or (None, None)

    transform_tuple:
        (A, B, C, D, E, F)
        x_geo = A*x + B*y + C
        y_geo = D*x + E*y + F
    """
    try:
        im = Image.open(image_path)
        tags = getattr(im, "tag_v2", None)
        if tags is None:
            tags = getattr(im, "tag", None)
        if tags is None:
            return None, None

        pixel_scale = tags.get(33550)
        tiepoints = tags.get(33922)
        geokeys = tags.get(34735)

        if pixel_scale is None or tiepoints is None:
            if log_fn:
                log_fn("Pillow GeoTIFF fallback: required tags 33550/33922 not found.")
            return None, None

        ps = list(pixel_scale)
        tp = list(tiepoints)

        if len(ps) < 2 or len(tp) < 6:
            if log_fn:
                log_fn("Pillow GeoTIFF fallback: invalid GeoTIFF tag lengths.")
            return None, None

        scale_x = float(ps[0])
        scale_y = float(ps[1])

        # First tiepoint: raster pixel i,j,k maps to model x,y,z
        i = float(tp[0])
        j = float(tp[1])
        x_model = float(tp[3])
        y_model = float(tp[4])

        # North-up GeoTIFF convention:
        # x = x_model + (px - i) * scale_x
        # y = y_model - (py - j) * scale_y
        A = scale_x
        B = 0.0
        C = x_model - i * scale_x
        D = 0.0
        E = -scale_y
        F = y_model + j * scale_y

        crs_name = None

        # Try to extract projected/geographic EPSG from GeoKeyDirectoryTag.
        # GeoKeyDirectory structure:
        # [KeyDirectoryVersion, KeyRevision, MinorRevision, NumberOfKeys,
        #  keyid, tiffTagLocation, count, value_offset, ...]
        try:
            gk = list(geokeys) if geokeys is not None else []
            if len(gk) >= 4:
                nkeys = int(gk[3])
                for idx in range(nkeys):
                    base = 4 + idx * 4
                    if base + 3 >= len(gk):
                        break
                    key_id = int(gk[base])
                    loc = int(gk[base + 1])
                    count = int(gk[base + 2])
                    value = int(gk[base + 3])

                    # ProjectedCSTypeGeoKey = 3072
                    # GeographicTypeGeoKey = 2048
                    if loc == 0 and count == 1 and key_id in (3072, 2048):
                        if value > 0:
                            crs_name = f"EPSG:{value}"
                            break
        except Exception:
            crs_name = None

        # Detect obvious WebMercator coordinates.
        # Values in millions are definitely not EPSG:4326 degrees.
        if abs(C) > 10000 or abs(F) > 10000:
            crs_name = "EPSG:3857"

        if crs_name is None:
            crs_name = "EPSG:3857"

        if log_fn:
            log_fn("Georeference source: Pillow GeoTIFF tags")
            log_fn(f"Pillow GeoTIFF PixelScale: {pixel_scale}")
            log_fn(f"Pillow GeoTIFF Tiepoint: {tiepoints}")
            log_fn(f"Pillow GeoTIFF transform tuple: {(A, B, C, D, E, F)}")
            log_fn(f"Pillow GeoTIFF CRS: {crs_name}")

        return (A, B, C, D, E, F), crs_name

    except Exception as exc:
        if log_fn:
            log_fn(f"Pillow GeoTIFF tag fallback failed: {exc}")
        return None, None


def get_image_georeference_for_detection(img_path: Path, reader, mode: str, default_crs: str, log_fn=None):
    """
    Returns:
      transform_kind, transform_object, crs_name

    transform_kind:
      "rasterio"   -> use rasterio affine via transform * (x,y)
      "worldfile"  -> use tuple affine
      "pixel"      -> no georeference available
    """
    if mode == "rasterio" and reader is not None:
        try:
            transform = reader.transform
            crs_name = reader.crs.to_string() if reader.crs else None
            if transform is not None and not transform.is_identity:
                if log_fn:
                    log_fn("Georeference source: GeoTIFF/rasterio transform")
                    log_fn(f"GeoTIFF CRS: {crs_name}")
                    log_fn(f"GeoTIFF transform: {transform}")
                return "rasterio", transform, crs_name
            else:
                if log_fn:
                    log_fn("Rasterio opened the file, but transform is missing/identity.")
        except Exception as exc:
            if log_fn:
                log_fn(f"Rasterio georeference read failed: {exc}")

    # Pure Pillow GeoTIFF tag fallback. Works without rasterio and without tifffile.
    if img_path.suffix.lower() in {".tif", ".tiff"}:
        gt_transform, gt_crs = read_geotiff_tags_with_pillow(img_path, log_fn=log_fn)
        if gt_transform is not None:
            return "geotiff_pillow_tags", gt_transform, gt_crs

    wf_transform, wf_path = read_worldfile_transform(img_path, log_fn=log_fn)
    if wf_transform is not None:
        crs_name = read_prj_or_default_crs(img_path, default_crs=default_crs, log_fn=log_fn)
        return "worldfile", wf_transform, crs_name

    if log_fn:
        log_fn("WARNING: No GeoTIFF transform, Pillow GeoTIFF tags, or worldfile found.")
        log_fn("Output coordinates will be PIXEL coordinates and will NOT overlay correctly in QGIS.")
    return "pixel", None, None




def preview_geojson_path_for_output(out_path: Path) -> Path:
    """
    For QGIS-safe .gpkg outputs, create a sidecar GeoJSON used only by the app preview.
    Example:
        detections.gpkg -> detections.preview.geojson
    """
    if out_path.suffix.lower() == ".gpkg":
        return out_path.with_suffix(".preview.geojson")
    return out_path


def _epsg_int(crs_name):
    try:
        if not crs_name:
            return 3857
        s = str(crs_name).upper()
        if "EPSG:" in s:
            return int(s.split("EPSG:")[-1].split()[0].replace('"', '').replace("'", "").replace("}", "").replace(",", ""))
        return int(s)
    except Exception:
        return 3857


def _polygon_wkb(poly_coords):
    import struct

    rings = poly_coords
    if not rings:
        return b""

    clean_rings = []
    for ring in rings:
        pts = []
        for p in ring:
            if len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
        if len(pts) < 3:
            continue
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        if len(pts) >= 4:
            clean_rings.append(pts)

    if not clean_rings:
        return b""

    out = bytearray()
    out += struct.pack("<B", 1)
    out += struct.pack("<I", 3)
    out += struct.pack("<I", len(clean_rings))
    for pts in clean_rings:
        out += struct.pack("<I", len(pts))
        for x, y in pts:
            out += struct.pack("<dd", x, y)
    return bytes(out)


def _gpkg_geom_blob(wkb, srs_id):
    import struct
    return b"GP" + struct.pack("<BBi", 0, 1, int(srs_id)) + wkb


def _write_gpkg_sqlite(gpkg_path: Path, feature_collection: dict, srs_id: int, log_fn=None):
    import sqlite3
    import json as _json
    import time

    features = feature_collection.get("features", [])
    conn = sqlite3.connect(str(gpkg_path), timeout=60)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=DELETE")
        cur.execute("PRAGMA synchronous=FULL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA application_id = 1196437808")
        cur.execute("PRAGMA user_version = 10400")

        cur.execute("""
            CREATE TABLE gpkg_spatial_ref_sys (
                srs_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL PRIMARY KEY,
                organization TEXT NOT NULL,
                organization_coordsys_id INTEGER NOT NULL,
                definition TEXT NOT NULL,
                description TEXT
            )
        """)

        srs_rows = [
            ("Undefined Cartesian SRS", -1, "NONE", -1, "undefined", "undefined cartesian coordinate reference system"),
            ("Undefined Geographic SRS", 0, "NONE", 0, "undefined", "undefined geographic coordinate reference system"),
            ("WGS 84 geodetic", 4326, "EPSG", 4326, "EPSG:4326", "longitude/latitude coordinates in decimal degrees on WGS84"),
            ("WGS 84 / Pseudo-Mercator", 3857, "EPSG", 3857, "EPSG:3857", "Web Mercator meters"),
        ]
        cur.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)", srs_rows)

        if srs_id not in (-1, 0, 4326, 3857):
            cur.execute(
                "INSERT OR IGNORE INTO gpkg_spatial_ref_sys VALUES (?, ?, 'EPSG', ?, ?, ?)",
                (f"EPSG:{srs_id}", srs_id, srs_id, f"EPSG:{srs_id}", f"EPSG:{srs_id}"),
            )

        cur.execute("""
            CREATE TABLE gpkg_contents (
                table_name TEXT NOT NULL PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT UNIQUE,
                description TEXT DEFAULT '',
                last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                min_x DOUBLE,
                min_y DOUBLE,
                max_x DOUBLE,
                max_y DOUBLE,
                srs_id INTEGER
            )
        """)

        cur.execute("""
            CREATE TABLE gpkg_geometry_columns (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL,
                PRIMARY KEY (table_name, column_name)
            )
        """)

        cur.execute("""
            CREATE TABLE mustatil_detections (
                fid INTEGER PRIMARY KEY AUTOINCREMENT,
                geom BLOB NOT NULL,
                class TEXT,
                class_id INTEGER,
                confidence DOUBLE,
                geometry_source TEXT,
                backend TEXT,
                georef_source TEXT,
                image_read_mode TEXT,
                inference_engine TEXT,
                pixel_bbox TEXT,
                sam2_pixel_polygon TEXT,
                green_ratio DOUBLE,
                green_filter_threshold DOUBLE
            )
        """)

        cur.execute("INSERT INTO gpkg_geometry_columns VALUES ('mustatil_detections', 'geom', 'POLYGON', ?, 0, 0)", (srs_id,))

        minx = miny = float("inf")
        maxx = maxy = float("-inf")
        count = 0

        insert_sql = """
            INSERT INTO mustatil_detections
            (geom, class, class_id, confidence, geometry_source, backend, georef_source, image_read_mode, inference_engine, pixel_bbox, sam2_pixel_polygon, green_ratio, green_filter_threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for idx, feat in enumerate(features, start=1):
            geom = feat.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates", [])
            wkb = _polygon_wkb(coords)
            if not wkb:
                continue

            for ring in coords:
                for p in ring:
                    if len(p) >= 2:
                        x = float(p[0]); y = float(p[1])
                        minx = min(minx, x); miny = min(miny, y)
                        maxx = max(maxx, x); maxy = max(maxy, y)

            props = feat.get("properties", {})
            cur.execute(insert_sql, (
                sqlite3.Binary(_gpkg_geom_blob(wkb, srs_id)),
                props.get("class"),
                int(props.get("class_id", 0)) if props.get("class_id") is not None else None,
                float(props.get("confidence", 0.0)) if props.get("confidence") is not None else None,
                props.get("geometry_source"),
                props.get("backend"),
                props.get("georef_source"),
                props.get("image_read_mode"),
                props.get("inference_engine"),
                _json.dumps(props.get("pixel_bbox")) if props.get("pixel_bbox") is not None else None,
                _json.dumps(props.get("sam2_pixel_polygon")) if props.get("sam2_pixel_polygon") is not None else None,
                float(props.get("green_ratio", 0.0)) if props.get("green_ratio") is not None else None,
                float(props.get("green_filter_threshold")) if props.get("green_filter_threshold") is not None else None,
            ))
            count += 1

            if log_fn and idx % 100 == 0:
                log_fn(f"GeoPackage rows inserted: {idx}/{len(features)}")

        if count == 0:
            raise RuntimeError("No valid polygon features to write to GeoPackage.")

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cur.execute("""
            INSERT INTO gpkg_contents
            (table_name, data_type, identifier, description, last_change, min_x, min_y, max_x, max_y, srs_id)
            VALUES ('mustatil_detections', 'features', 'mustatil_detections', 'Mustatil detections', ?, ?, ?, ?, ?, ?)
        """, (now, minx, miny, maxx, maxy, srs_id))

        conn.commit()
        return count, (minx, miny, maxx, maxy)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()



def read_gpkg_preview_features(gpkg_path: Path, log_fn=None) -> List[dict]:
    """
    Reads just the attributes needed by the built-in preview from the GeoPackage.
    This avoids creating a GeoJSON sidecar file.
    """
    import sqlite3
    import json as _json

    if not gpkg_path.exists():
        return []

    features = []
    conn = sqlite3.connect(str(gpkg_path), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM mustatil_detections ORDER BY fid")
        rows = cur.fetchall()
        for row in rows:
            props = {
                "class": row["class"] if "class" in row.keys() else "mustatil",
                "class_id": row["class_id"] if "class_id" in row.keys() else 0,
                "confidence": row["confidence"] if "confidence" in row.keys() else None,
                "geometry_source": row["geometry_source"] if "geometry_source" in row.keys() else "bbox",
                "backend": row["backend"] if "backend" in row.keys() else None,
                "georef_source": row["georef_source"] if "georef_source" in row.keys() else None,
                "image_read_mode": row["image_read_mode"] if "image_read_mode" in row.keys() else None,
                "inference_engine": row["inference_engine"] if "inference_engine" in row.keys() else None,
            }
            try:
                props["pixel_bbox"] = _json.loads(row["pixel_bbox"]) if row["pixel_bbox"] else None
            except Exception:
                props["pixel_bbox"] = None
            if "sam2_pixel_polygon" in row.keys():
                try:
                    props["sam2_pixel_polygon"] = _json.loads(row["sam2_pixel_polygon"]) if row["sam2_pixel_polygon"] else None
                except Exception:
                    props["sam2_pixel_polygon"] = None
            if "green_ratio" in row.keys():
                props["green_ratio"] = row["green_ratio"]
            if "green_filter_threshold" in row.keys():
                props["green_filter_threshold"] = row["green_filter_threshold"]

            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": None,
            })
        if log_fn:
            log_fn(f"Loaded preview directly from GeoPackage: {gpkg_path} ({len(features)} features)")
        return features
    finally:
        conn.close()


def write_features_outputs(out_path: Path, feature_collection: dict, crs_name: Optional[str], log_fn=None):
    """
    Writes .gpkg and always creates a sidecar .preview.geojson first.
    For network/external drives like R:, the GeoPackage is created locally first,
    then copied to the target path. This avoids many SQLite lock/hang issues.
    """
    suffix = out_path.suffix.lower()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".gpkg":
        import shutil
        import tempfile
        import time

        features = feature_collection.get("features", [])
        if log_fn:
            log_fn(f"Writing output package: {out_path}")
            log_fn(f"Features to write: {len(features)}")

        if not features:
            raise RuntimeError("No features to write. Output was not created.")

        # 1) No GeoJSON sidecar is written. The app preview reads directly from the GeoPackage.
        srs_id = _epsg_int(crs_name)
        try:
            first = features[0]["geometry"]["coordinates"][0][0]
            if srs_id == 4326 and (abs(float(first[0])) > 1000 or abs(float(first[1])) > 1000):
                if log_fn:
                    log_fn("GeoPackage CRS auto-correction: meter coordinates detected -> EPSG:3857")
                srs_id = 3857
        except Exception:
            pass

        # 2) Build GeoPackage locally, not on R:/network drive.
        local_tmp_dir = Path(tempfile.gettempdir()) / "mustatil_gpkg_writer"
        local_tmp_dir.mkdir(parents=True, exist_ok=True)
        local_tmp = local_tmp_dir / f"{out_path.stem}_{int(time.time())}.gpkg"

        if local_tmp.exists():
            local_tmp.unlink()

        if log_fn:
            log_fn(f"Creating local temporary GeoPackage: {local_tmp}")

        count, bounds = _write_gpkg_sqlite(local_tmp, feature_collection, srs_id, log_fn=log_fn)

        if not local_tmp.exists() or local_tmp.stat().st_size < 1024:
            raise RuntimeError(f"Local GeoPackage was not created correctly: {local_tmp}")

        if log_fn:
            log_fn(f"Local GeoPackage ready: {local_tmp} ({local_tmp.stat().st_size} bytes)")

        # 3) Copy/replace target. If QGIS locks the target, write a timestamped alternative.
        final_target = out_path
        try:
            if out_path.exists():
                backup = out_path.with_suffix(out_path.suffix + ".bak")
                try:
                    if backup.exists():
                        backup.unlink()
                    out_path.replace(backup)
                    if log_fn:
                        log_fn(f"Old GeoPackage backed up: {backup}")
                except Exception as exc:
                    if log_fn:
                        log_fn(f"Target may be locked by QGIS, cannot backup/replace: {exc}")
                    final_target = out_path.with_name(out_path.stem + f"_new_{int(time.time())}" + out_path.suffix)

            shutil.copy2(local_tmp, final_target)
            if log_fn:
                log_fn(f"GeoPackage copied to: {final_target}")
        finally:
            try:
                local_tmp.unlink()
            except Exception:
                pass

        if log_fn:
            log_fn(f"GeoPackage written successfully: {final_target}")
            log_fn(f"GeoPackage features: {count}; EPSG:{srs_id}; bounds={bounds}")
            log_fn(f"GeoPackage file size: {final_target.stat().st_size} bytes")

        return final_target

    # Default GeoJSON
    if log_fn:
        log_fn(f"Writing GeoJSON output: {out_path}")
    out_path.write_text(json.dumps(feature_collection, indent=2), encoding="utf-8")
    if log_fn:
        log_fn(f"GeoJSON written successfully: {out_path}")
    return out_path






def yolo_box_intersection_for_tile(box: Box, tx: int, ty: int, tw: int, th: int, min_visible: float = 0.35) -> Optional[Box]:
    """
    Clips one global image-space Box into one training tile.
    Returns tile-local Box or None.

    min_visible is the minimum fraction of the original box area that must remain
    inside the tile. This prevents tiny clipped fragments from becoming bad labels.
    """
    bx1, bx2 = sorted((float(box.x1), float(box.x2)))
    by1, by2 = sorted((float(box.y1), float(box.y2)))
    orig_area = max(1.0, (bx2 - bx1) * (by2 - by1))

    ix1 = max(bx1, float(tx))
    iy1 = max(by1, float(ty))
    ix2 = min(bx2, float(tx + tw))
    iy2 = min(by2, float(ty + th))

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    if inter_area / orig_area < float(min_visible):
        return None

    # tile-local coordinates
    return Box(int(box.cls), ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty)


def save_yolo_boxes_for_tile(label_path: Path, boxes: List[Box], tile_w: int, tile_h: int):
    lines = []
    for b in boxes:
        cls, cx, cy, bw, bh = b.normalized_yolo(tile_w, tile_h)
        if bw > 0 and bh > 0:
            lines.append(f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_or_tile_image_for_yolo_dataset(
    image_path: Path,
    boxes: List[Box],
    out_img_dir: Path,
    out_lbl_dir: Path,
    chunk_enabled: bool,
    chunk_size: int,
    chunk_overlap: int,
    min_visible: float,
    keep_negative_tiles: bool,
    log_fn=None,
):
    """
    Creates either a normal image+label pair or many chunk image+label pairs.
    This is used only for the YOLO training dataset, not for detection.
    """
    if not chunk_enabled:
        shutil.copy2(image_path, out_img_dir / image_path.name)
        save_yolo_boxes_for_tile(out_lbl_dir / f"{image_path.stem}.txt", boxes, Image.open(image_path).width, Image.open(image_path).height)
        return 1, 0

    chunk_size = max(128, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    stride = max(1, chunk_size - chunk_overlap)
    min_visible = max(0.0, min(1.0, float(min_visible)))

    # Use rasterio streaming for TIFF when available. Otherwise PIL crop.
    reader = None
    mode = "pil"
    W = H = 0
    try:
        W, H, mode, reader = open_large_image_info(image_path, log_fn=log_fn)
    except Exception:
        im_tmp = Image.open(image_path)
        W, H = im_tmp.size
        reader = im_tmp
        mode = "pil"

    made = 0
    skipped_negative = 0
    try:
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                tw = min(chunk_size, W - x)
                th = min(chunk_size, H - y)
                if tw < 64 or th < 64:
                    continue

                tile_boxes: List[Box] = []
                for b in boxes:
                    clipped = yolo_box_intersection_for_tile(b, x, y, tw, th, min_visible=min_visible)
                    if clipped is not None:
                        tile_boxes.append(clipped)

                if not tile_boxes and not keep_negative_tiles:
                    skipped_negative += 1
                    continue

                crop = read_detection_tile(reader, mode, x, y, chunk_size, W, H).convert("RGB")
                tile_name = f"{image_path.stem}_x{x}_y{y}_w{tw}_h{th}.jpg"
                crop.save(out_img_dir / tile_name, quality=95, subsampling=0)
                save_yolo_boxes_for_tile(out_lbl_dir / f"{Path(tile_name).stem}.txt", tile_boxes, tw, th)
                made += 1

            if y + chunk_size >= H:
                break
    finally:
        try:
            if mode == "rasterio" and reader is not None:
                reader.close()
        except Exception:
            pass

    return made, skipped_negative

def directml_yolo_training_patch_code() -> str:
    """Return Python code injected into the training subprocess for DirectML YOLO compatibility."""
    return '\n# ---- DirectML YOLO training compatibility patch ----\n# torch-directml currently lacks unique(return_counts=True). Ultralytics YOLO\n# uses that op in the loss target preprocessor. We run only this specific op on\n# CPU and move its tensor outputs back to the original device.\n_orig_torch_unique = torch.unique\n\ndef _dml_safe_unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):\n    try:\n        return _orig_torch_unique(input, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)\n    except RuntimeError as exc:\n        msg = str(exc)\n        is_dml_tensor = getattr(getattr(input, "device", None), "type", "") == "privateuseone"\n        if is_dml_tensor and return_counts and ("DirectML" in msg or "unique" in msg or "not implemented" in msg):\n            cpu_input = input.detach().cpu()\n            result = _orig_torch_unique(cpu_input, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)\n            def _back(x):\n                return x.to(input.device) if torch.is_tensor(x) else x\n            if isinstance(result, tuple):\n                return tuple(_back(x) for x in result)\n            return _back(result)\n        raise\n\ntorch.unique = _dml_safe_unique\n\ndef _dml_tensor_unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):\n    return torch.unique(self, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)\n\ntorch.Tensor.unique = _dml_tensor_unique\nprint("DirectML YOLO patch active: torch.unique(return_counts=True) CPU fallback enabled")\n# ---- end DirectML patch ----\n'

class MustatilGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x820")
        self.project: Optional[Project] = None
        self.current_image: Optional[Path] = None
        self.boxes: List[Box] = []
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start: Optional[Tuple[int, int]] = None
        self.temp_rect = None
        self.photo = None
        self.det_preview_img = None
        self.det_preview_photo = None
        self.det_preview_features = []
        self.det_preview_orig_size = None
        self.det_preview_zoom = 1.0
        self.det_preview_pan_x = 0.0
        self.det_preview_pan_y = 0.0
        self.det_preview_drag_start = None
        self.annotation_class = tk.IntVar(value=0)  # 0=mustatil, 1=false_positive
        self.annotation_class_names = {0: "mustatil", 1: "false_positive"}
        # Annotate preview zoom/pan state
        self.annotate_zoom = 1.0
        self.annotate_pan_x = 0.0
        self.annotate_pan_y = 0.0
        self.annotate_drag_pan_start = None
        self.annotate_base_fit = 1.0
        self._build_ui()
        self.after(500, self._startup_dependency_log)

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)
        self.annotate_tab = ttk.Frame(nb)
        self.train_tab = ttk.Frame(nb)
        self.detect_tab = ttk.Frame(nb)
        nb.add(self.annotate_tab, text="1 Annotate")
        nb.add(self.train_tab, text="2 Train / Export")
        nb.add(self.detect_tab, text="3 Detect Maps")
        self._build_annotate_tab()
        self._build_train_tab()
        self._build_detect_tab()

    def _build_annotate_tab(self):
        left = ttk.Frame(self.annotate_tab, width=280)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self.annotate_tab)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Button(left, text="New / Open Project", command=self.open_project).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(left, text="Import Images", command=self.import_images).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(left, text="Save Labels", command=self.save_current).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(left, text="Delete Selected Box", command=self.delete_selected_box).pack(fill=tk.X, padx=8, pady=4)

        cls_frame = ttk.LabelFrame(left, text="Annotation class", padding=6)
        cls_frame.pack(fill=tk.X, padx=8, pady=6)
        ttk.Radiobutton(cls_frame, text="Mustatil (positive)", variable=self.annotation_class, value=0).pack(anchor="w")
        ttk.Radiobutton(cls_frame, text="False positive / bush", variable=self.annotation_class, value=1).pack(anchor="w")
        ttk.Button(cls_frame, text="Set selected box to Mustatil", command=lambda: self.set_selected_box_class(0)).pack(fill=tk.X, pady=(6, 2))
        ttk.Button(cls_frame, text="Set selected box to False positive", command=lambda: self.set_selected_box_class(1)).pack(fill=tk.X, pady=2)

        ttk.Label(left, text="Images").pack(anchor="w", padx=8, pady=(12, 0))
        self.img_list = tk.Listbox(left)
        self.img_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.img_list.bind("<<ListboxSelect>>", lambda e: self.select_image())
        ttk.Label(left, text="Boxes").pack(anchor="w", padx=8, pady=(12, 0))
        self.box_list = tk.Listbox(left, height=8)
        self.box_list.pack(fill=tk.X, padx=8, pady=4)

        self.canvas = tk.Canvas(right, bg="#222222", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.on_annotate_wheel)
        self.canvas.bind("<Button-4>", self.on_annotate_wheel)
        self.canvas.bind("<Button-5>", self.on_annotate_wheel)
        self.canvas.bind("<ButtonPress-2>", self.on_annotate_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_annotate_pan_move)
        self.canvas.bind("<ButtonPress-3>", self.on_annotate_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_annotate_pan_move)
        self.canvas.bind("<Double-Button-1>", self.reset_annotate_view)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        ttk.Label(right, text="Draw boxes. Mouse wheel = zoom, middle/right drag = pan, double-click = reset. Green/yellow = mustatil, red/orange = false_positive/bush.").pack(anchor="w")

    def _build_train_tab(self):
        frm = ttk.Frame(self.train_tab)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.epochs = tk.IntVar(value=80)
        self.imgsz = tk.IntVar(value=640)
        self.batch = tk.IntVar(value=2)
        self.device = tk.StringVar(value="cpu")
        self.model_name = tk.StringVar(value="yolov8n.pt")
        self.low_ram_mode = tk.BooleanVar(value=True)
        self.auto_resume = tk.BooleanVar(value=True)
        row = 0

        ttk.Label(frm, text="Base model").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=self.model_name).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1
        for label, var in [("Epochs", self.epochs), ("Image size", self.imgsz), ("Batch", self.batch)]:
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(frm, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
            row += 1

        ttk.Label(frm, text="Device").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(frm, textvariable=self.device, values=["cpu", "directml", "cuda", "0"], state="normal").grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Checkbutton(frm, text="Low-RAM stabil mode", variable=self.low_ram_mode).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        row += 1

        self.train_chunk_enabled = tk.BooleanVar(value=True)
        self.train_chunk_size = tk.IntVar(value=1024)
        self.train_chunk_overlap = tk.IntVar(value=128)
        self.train_chunk_min_visible = tk.DoubleVar(value=0.35)
        self.train_keep_negative_chunks = tk.BooleanVar(value=True)

        chunk_frame = ttk.LabelFrame(frm, text="Training image chunking for huge maps", padding=6)
        chunk_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        chunk_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(chunk_frame, text="Cut project images into chunks while preparing YOLO dataset", variable=self.train_chunk_enabled).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Label(chunk_frame, text="Chunk size px").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(chunk_frame, textvariable=self.train_chunk_size, width=10).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(chunk_frame, text="usually 1024 or 1280", foreground="#555").grid(row=1, column=2, sticky="w", padx=6)
        ttk.Label(chunk_frame, text="Overlap px").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(chunk_frame, textvariable=self.train_chunk_overlap, width=10).grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(chunk_frame, text="128-256 helps objects on borders", foreground="#555").grid(row=2, column=2, sticky="w", padx=6)
        ttk.Label(chunk_frame, text="Min visible label fraction").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(chunk_frame, textvariable=self.train_chunk_min_visible, width=10).grid(row=3, column=1, sticky="w", pady=2)
        ttk.Label(chunk_frame, text="0.35 avoids tiny clipped labels", foreground="#555").grid(row=3, column=2, sticky="w", padx=6)
        ttk.Checkbutton(chunk_frame, text="Keep empty/negative chunks too", variable=self.train_keep_negative_chunks).grid(row=4, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        ttk.Checkbutton(frm, text="Auto-resume after crash/interruption", variable=self.auto_resume).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        row += 1

        frm.columnconfigure(1, weight=1)
        ttk.Button(frm, text="Dependency Check", command=lambda: show_dependency_help(self)).grid(row=row, column=0, sticky="ew", pady=8)
        ttk.Button(frm, text="Prepare YOLO Dataset", command=self.prepare_dataset).grid(row=row, column=1, sticky="ew", pady=8)
        row += 1
        ttk.Button(frm, text="Train YOLO Model", command=self.train_model_thread).grid(row=row, column=0, sticky="ew", pady=8)
        ttk.Button(frm, text="Resume Training from last.pt", command=self.resume_training_thread).grid(row=row, column=1, sticky="ew", pady=8)
        row += 1
        ttk.Button(frm, text="Export Best Model to ONNX", command=self.export_onnx_thread).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        self.train_log = tk.Text(frm, height=28)
        self.train_log.grid(row=row, column=0, columnspan=2, sticky="nsew")
        frm.rowconfigure(row, weight=1)

    def _build_detect_tab(self):
        frm = ttk.Frame(self.detect_tab)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.det_model = tk.StringVar()
        self.det_image = tk.StringVar()
        self.conf = tk.DoubleVar(value=0.25)
        self.tile = tk.IntVar(value=1024)
        self.overlap = tk.IntVar(value=128)
        self.out_geojson = tk.StringVar()
        self.default_crs = tk.StringVar(value="EPSG:3857")
        self.det_backend = tk.StringVar(value="Ultralytics/PT CPU")
        self.use_sam2 = tk.BooleanVar(value=False)
        self.sam2_model = tk.StringVar(value="sam2_b.pt")
        self.sam2_max_per_tile = tk.IntVar(value=5)
        self.sam2_padding = tk.IntVar(value=96)
        self.sam2_max_crop = tk.IntVar(value=768)
        self.use_green_filter = tk.BooleanVar(value=False)
        self.green_filter_threshold = tk.DoubleVar(value=0.18)

        top = ttk.LabelFrame(frm, text="Detection Settings", padding=8)
        top.pack(fill=tk.X)

        rows = [
            ("YOLO model .pt/.onnx", self.det_model, self.pick_model),
            ("Map image/GeoTIFF", self.det_image, self.pick_detect_image),
            ("Output GeoPackage + Preview", self.out_geojson, self.pick_output_geojson),
        ]
        for r, (label, var, cmd) in enumerate(rows):
            ttk.Label(top, text=label).grid(row=r, column=0, sticky="w", pady=4)
            ttk.Entry(top, textvariable=var).grid(row=r, column=1, sticky="ew", pady=4)
            ttk.Button(top, text="Browse", command=cmd).grid(row=r, column=2, sticky="ew", pady=4)

        r = 3
        ttk.Label(top, text="Default CRS for worldfile").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(top, textvariable=self.default_crs).grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Label(top, text="used if .prj missing").grid(row=r, column=2, sticky="w", pady=4)
        r += 1

        for label, var in [("Confidence", self.conf), ("Tile size", self.tile), ("Overlap", self.overlap)]:
            ttk.Label(top, text=label).grid(row=r, column=0, sticky="w", pady=4)
            ttk.Entry(top, textvariable=var).grid(row=r, column=1, sticky="ew", pady=4)
            r += 1

        ttk.Checkbutton(top, text="Vegetation/Bush filter by green share", variable=self.use_green_filter).grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(top, textvariable=self.green_filter_threshold).grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Label(top, text="reject if green ratio is higher, e.g. 0.18", foreground="#555").grid(row=r, column=2, sticky="w", pady=4)
        r += 1

        ttk.Label(top, text="Inference backend").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(
            top,
            textvariable=self.det_backend,
            values=["Ultralytics/PT CPU", "Ultralytics/ONNX Runtime", "ONNX Runtime DirectML GPU"],
            state="readonly",
        ).grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Label(top, text="ONNX backends use best.onnx").grid(row=r, column=2, sticky="w", pady=4)
        r += 1

        ttk.Checkbutton(top, text="Refine YOLO boxes with SAM2 masks", variable=self.use_sam2).grid(row=r, column=0, columnspan=3, sticky="w", pady=4)
        r += 1

        ttk.Label(top, text="SAM/SAM2 model").grid(row=r, column=0, sticky="w", pady=4)
        sam_row = ttk.Frame(top)
        sam_row.grid(row=r, column=1, sticky="ew", pady=4)
        sam_row.columnconfigure(0, weight=1)
        self.sam2_combo = ttk.Combobox(
            sam_row,
            textvariable=self.sam2_model,
            values=[
                "sam2_t.pt",
                "sam2_s.pt",
                "sam2_b.pt",
                "sam2_l.pt",
                "sam2.1_t.pt",
                "sam2.1_s.pt",
                "sam2.1_b.pt",
                "sam2.1_l.pt",
                "sam_b.pt",
                "mobile_sam.pt",
            ],
            state="normal",
        )
        self.sam2_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text="Browse SAM/SAM2", command=self.pick_sam2_model).grid(row=r, column=2, sticky="ew", pady=4)
        r += 1

        ttk.Label(top, text="SAM2 max boxes/tile").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(top, textvariable=self.sam2_max_per_tile).grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Label(top, text="prevents crashes").grid(row=r, column=2, sticky="w", pady=4)
        r += 1

        ttk.Label(top, text="SAM2 padding px").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(top, textvariable=self.sam2_padding).grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Label(top, text="crop around box").grid(row=r, column=2, sticky="w", pady=4)
        r += 1

        ttk.Label(top, text="SAM2 max crop px").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(top, textvariable=self.sam2_max_crop).grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Label(top, text="downsample large crops").grid(row=r, column=2, sticky="w", pady=4)
        r += 1

        btnrow = ttk.Frame(top)
        btnrow.grid(row=r, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Button(btnrow, text="Run Tiled Detection", command=self.detect_thread).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btnrow, text="Show Image + Preview", command=self.load_detection_preview).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btnrow, text="Clear Preview", command=self.clear_detection_preview).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        top.columnconfigure(1, weight=1)

        middle = ttk.PanedWindow(frm, orient=tk.HORIZONTAL)
        middle.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        preview_frame = ttk.LabelFrame(middle, text="Image + Detection/SAM2 Preview", padding=4)
        log_frame = ttk.LabelFrame(middle, text="Detection Log", padding=4)
        middle.add(preview_frame, weight=3)
        middle.add(log_frame, weight=2)

        self.det_canvas = tk.Canvas(preview_frame, bg="#222222", highlightthickness=1, highlightbackground="#666")
        self.det_canvas.pack(fill=tk.BOTH, expand=True)
        self.det_canvas.bind("<Configure>", lambda e: self.redraw_detection_preview())
        self.det_canvas.bind("<MouseWheel>", self.on_detection_preview_wheel)
        self.det_canvas.bind("<Button-4>", self.on_detection_preview_wheel)
        self.det_canvas.bind("<Button-5>", self.on_detection_preview_wheel)
        self.det_canvas.bind("<ButtonPress-1>", self.on_detection_preview_pan_start)
        self.det_canvas.bind("<B1-Motion>", self.on_detection_preview_pan_move)
        self.det_canvas.bind("<Double-Button-1>", self.reset_detection_preview_view)

        ttk.Label(
            preview_frame,
            text="Mouse wheel = zoom, left-drag = pan, double-click = reset. Red = YOLO box fallback. Cyan = SAM2 mask polygon.",
            foreground="#555",
        ).pack(anchor="w")

        self.detect_log = tk.Text(log_frame, height=30)
        self.detect_log.pack(fill=tk.BOTH, expand=True)

    def log_train(self, s): self.train_log.insert(tk.END, s + "\n"); self.train_log.see(tk.END); self.update_idletasks()
    def log_detect(self, s): self.detect_log.insert(tk.END, s + "\n"); self.detect_log.see(tk.END); self.update_idletasks()

    def _startup_dependency_log(self):
        ok, msg = dependency_report()
        try:
            self.log_train(msg)
        except Exception:
            pass

    def open_project(self):
        path = filedialog.askdirectory(title="Choose/create project folder")
        if not path: return
        self.project = Project(Path(path))
        self.refresh_images()

    def import_images(self):
        if not self.project:
            self.open_project()
            if not self.project: return
        files = filedialog.askopenfilenames(title="Import images", filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp")])
        for f in files:
            src = Path(f)
            dst = self.project.images_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        self.refresh_images()

    def refresh_images(self):
        self.img_list.delete(0, tk.END)
        if not self.project: return
        for p in self.project.image_files():
            self.img_list.insert(tk.END, p.name)

    def select_image(self):
        if not self.project: return
        sel = self.img_list.curselection()
        if not sel: return
        if self.current_image:
            self.save_current(silent=True)
        self.current_image = self.project.images_dir / self.img_list.get(sel[0])
        self.boxes = self.project.load_boxes(self.current_image)
        self.annotate_zoom = 1.0
        self.annotate_pan_x = 0.0
        self.annotate_pan_y = 0.0
        self.redraw()

    def save_current(self, silent=False):
        if self.project and self.current_image:
            self.project.save_boxes(self.current_image, self.boxes)
            if not silent: messagebox.showinfo(APP_TITLE, "Labels saved.")

    def _image_to_canvas(self, x, y): return x * self.scale + self.offset_x, y * self.scale + self.offset_y
    def _canvas_to_image(self, x, y): return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale

    def redraw(self):
        self.canvas.delete("all")
        self.box_list.delete(0, tk.END)
        if not self.current_image: return
        im = Image.open(self.current_image).convert("RGB")
        cw = max(1, self.canvas.winfo_width()); ch = max(1, self.canvas.winfo_height())

        self.annotate_base_fit = min(cw / im.width, ch / im.height, 1.0)
        zoom = max(0.05, min(64.0, float(getattr(self, "annotate_zoom", 1.0))))
        self.scale = self.annotate_base_fit * zoom

        sw = max(1, int(im.width * self.scale))
        sh = max(1, int(im.height * self.scale))
        show = im.resize((sw, sh))

        base_x = (cw - sw) // 2 if zoom <= 1.0001 else (cw - int(im.width * self.annotate_base_fit)) // 2
        base_y = (ch - sh) // 2 if zoom <= 1.0001 else (ch - int(im.height * self.annotate_base_fit)) // 2
        self.offset_x = int(base_x + float(getattr(self, "annotate_pan_x", 0.0)))
        self.offset_y = int(base_y + float(getattr(self, "annotate_pan_y", 0.0)))

        self.photo = ImageTk.PhotoImage(show)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.photo)

        for i, b in enumerate(self.boxes):
            x1, y1 = self._image_to_canvas(b.x1, b.y1); x2, y2 = self._image_to_canvas(b.x2, b.y2)
            name = self.annotation_class_names.get(int(b.cls), f"class_{int(b.cls)}")
            color = "#00ff66" if int(b.cls) == 0 else "#ff3333"
            text_color = "#00ff66" if int(b.cls) == 0 else "#ffaaaa"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            self.canvas.create_text(x1 + 4, y1 + 4, anchor="nw", text=f"{i}:{name}", fill=text_color)
            self.box_list.insert(tk.END, f"{i}: {name} [{int(b.x1)},{int(b.y1)}]-[{int(b.x2)},{int(b.y2)}]")

        self.canvas.create_text(
            10, 10,
            anchor="nw",
            fill="white",
            text=f"zoom: {zoom:.2f}x | wheel=zoom | middle/right drag=pan | double-click=reset",
        )

    def reset_annotate_view(self, event=None):
        self.annotate_zoom = 1.0
        self.annotate_pan_x = 0.0
        self.annotate_pan_y = 0.0
        self.redraw()

    def on_annotate_pan_start(self, event):
        self.annotate_drag_pan_start = (event.x, event.y, self.annotate_pan_x, self.annotate_pan_y)

    def on_annotate_pan_move(self, event):
        if not self.annotate_drag_pan_start:
            return
        sx, sy, px, py = self.annotate_drag_pan_start
        self.annotate_pan_x = px + (event.x - sx)
        self.annotate_pan_y = py + (event.y - sy)
        self.redraw()

    def on_annotate_wheel(self, event):
        if not self.current_image:
            return
        old_zoom = max(0.05, float(getattr(self, "annotate_zoom", 1.0)))
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            factor = 1.15
        else:
            factor = 1.0 / 1.15
        new_zoom = max(0.05, min(64.0, old_zoom * factor))

        # Keep mouse position stable while zooming.
        mx, my = float(event.x), float(event.y)
        self.annotate_pan_x = mx - (mx - self.annotate_pan_x) * (new_zoom / old_zoom)
        self.annotate_pan_y = my - (my - self.annotate_pan_y) * (new_zoom / old_zoom)
        self.annotate_zoom = new_zoom
        self.redraw()

    def on_mouse_down(self, e): self.drag_start = (e.x, e.y)
    def on_mouse_drag(self, e):
        if not self.drag_start: return
        if self.temp_rect: self.canvas.delete(self.temp_rect)
        x0, y0 = self.drag_start
        self.temp_rect = self.canvas.create_rectangle(x0, y0, e.x, e.y, outline="red", width=2)
    def on_mouse_up(self, e):
        if not self.drag_start or not self.current_image: return
        x0, y0 = self.drag_start; x1, y1 = e.x, e.y
        ix0, iy0 = self._canvas_to_image(x0, y0); ix1, iy1 = self._canvas_to_image(x1, y1)
        if abs(ix1 - ix0) > 8 and abs(iy1 - iy0) > 8:
            self.boxes.append(Box(int(self.annotation_class.get()), ix0, iy0, ix1, iy1))
            self.save_current(silent=True)
        self.drag_start = None; self.temp_rect = None; self.redraw()

    def set_selected_box_class(self, cls_id: int):
        sel = self.box_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.boxes):
            self.boxes[idx].cls = int(cls_id)
            self.save_current(silent=True)
            self.redraw()

    def delete_selected_box(self):
        sel = self.box_list.curselection()
        if sel:
            del self.boxes[sel[0]]
            self.save_current(silent=True); self.redraw()

    def prepare_dataset(self):
        if not self.project: return messagebox.showerror(APP_TITLE, "Open a project first.")
        if yaml is None: return messagebox.showerror(APP_TITLE, "Install PyYAML: python -m pip install pyyaml")

        ds = self.project.root / "yolo_dataset"
        # Rebuild dataset cleanly, otherwise old chunks remain and poison training.
        if ds.exists():
            shutil.rmtree(ds)
        for split in ["train", "val"]:
            (ds / "images" / split).mkdir(parents=True, exist_ok=True)
            (ds / "labels" / split).mkdir(parents=True, exist_ok=True)

        imgs = self.project.image_files()
        if not imgs:
            messagebox.showerror(APP_TITLE, "No project images found.")
            return None

        random.seed(42); random.shuffle(imgs)
        cut = max(1, int(len(imgs) * 0.8))

        chunk_enabled = bool(getattr(self, "train_chunk_enabled", tk.BooleanVar(value=False)).get())
        chunk_size = int(getattr(self, "train_chunk_size", tk.IntVar(value=1024)).get())
        chunk_overlap = int(getattr(self, "train_chunk_overlap", tk.IntVar(value=128)).get())
        min_visible = float(getattr(self, "train_chunk_min_visible", tk.DoubleVar(value=0.35)).get())
        keep_negative = bool(getattr(self, "train_keep_negative_chunks", tk.BooleanVar(value=True)).get())

        self.log_train("=" * 70)
        self.log_train(f"Preparing YOLO dataset: {ds}")
        self.log_train(f"Training chunks active: {chunk_enabled}; chunk={chunk_size}; overlap={chunk_overlap}; min_visible={min_visible}; keep_negative={keep_negative}")

        total_tiles = 0
        total_skipped_negative = 0

        for idx, img in enumerate(imgs):
            split = "train" if idx < cut else "val"
            boxes = self.project.load_boxes(img)
            made, skipped = copy_or_tile_image_for_yolo_dataset(
                img,
                boxes,
                ds / "images" / split,
                ds / "labels" / split,
                chunk_enabled=chunk_enabled,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_visible=min_visible,
                keep_negative_tiles=keep_negative,
                log_fn=self.log_train,
            )
            total_tiles += made
            total_skipped_negative += skipped
            self.log_train(f"{split}: {img.name} -> {made} training image(s), skipped negatives={skipped}, labels={len(boxes)}")

        data = {"path": str(ds), "train": "images/train", "val": "images/val", "names": {0: "mustatil", 1: "false_positive"}}
        (ds / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        self.log_train(f"Dataset ready: {ds}")
        self.log_train(f"Total YOLO training images/chunks: {total_tiles}; skipped empty chunks: {total_skipped_negative}")
        return ds / "data.yaml"

    def train_model_thread(self): threading.Thread(target=lambda: self.train_model(resume=False), daemon=True).start()
    def resume_training_thread(self): threading.Thread(target=lambda: self.train_model(resume=True), daemon=True).start()

    def _last_checkpoint_path(self) -> Path:
        if not self.project:
            raise RuntimeError("Open a project first.")
        return self.project.runs_dir / "mustatil" / "weights" / "last.pt"

    def _train_code(self, data_yaml: Path, local_model: Path, device: str, resume: bool, force_cpu: bool = False) -> str:
        """Builds the isolated Ultralytics training code executed in a subprocess."""
        device = (device or "cpu").strip().lower()
        if force_cpu:
            device = "cpu"

        imgsz = int(self.imgsz.get())
        batch = int(self.batch.get())
        workers = 0
        cache = "False"
        plots = "False" if self.low_ram_mode.get() else "True"

        if self.low_ram_mode.get():
            imgsz = min(imgsz, 640)
            batch = min(batch, 2)

        if resume:
            last = self._last_checkpoint_path()
            if not last.exists():
                raise RuntimeError(f"No checkpoint found: {last}")
            return (
                "from ultralytics import YOLO\n"
                f"model = YOLO(r'{last}')\n"
                "model.train(resume=True)\n"
            )

        directml_prefix = ""
        train_device_expr = f"r'{device}'"
        if device == "directml":
            # Experimental: torch-directml sometimes works for PyTorch training, but Ultralytics may reject it.
            # The caller catches failure and falls back to CPU automatically.
            directml_patch = directml_yolo_training_patch_code()
            directml_prefix = (
                "import torch_directml\n"
                "dml_device = torch_directml.device()\n"
                f"{directml_patch}\n"
            )
            train_device_expr = "dml_device"

        return (
            "from ultralytics import YOLO\n"
            "import torch\n"
            f"{directml_prefix}"
            f"model = YOLO(r'{local_model}')\n"
            "model.train("
            f"data=r'{data_yaml}', "
            f"epochs={int(self.epochs.get())}, "
            f"imgsz={imgsz}, "
            f"batch={batch}, "
            f"device={train_device_expr}, "
            f"project=r'{self.project.runs_dir}', "
            "name='mustatil', "
            "exist_ok=True, "
            f"workers={workers}, "
            f"cache={cache}, "
            "patience=20, "
            "save=True, "
            "save_period=1, "
            "amp=False, "
            "deterministic=True, "
            f"plots={plots}, "
            "verbose=True)\n"
        )

    def train_model(self, resume: bool = False):
        try:
            if not self.project:
                raise RuntimeError("Open a project first.")

            ok, msg = dependency_report()
            self.log_train(msg)
            if not ok:
                messagebox.showerror(APP_TITLE, msg)
                return

            data_yaml = self.prepare_dataset()
            if not data_yaml:
                return
            device = (self.device.get().strip() or "cpu").lower()

            if self.low_ram_mode.get():
                self.log_train("Low-RAM mode active: imgsz<=640, batch<=2, workers=0, cache=False, plots=False.")

            if device == "directml":
                self.log_train("DirectML training requested. This is experimental; CPU fallback is automatic if Ultralytics/torch-directml rejects it.")
            elif device != "cpu":
                self.log_train("WARNING: Non-CPU device selected. AMD R9 390X is not supported by PyTorch CUDA.")

            local_model = resolve_model_path(self.model_name.get(), self.project.root, self.log_train)
            self.log_train(f"Model path: {local_model}")

            attempts = 2 if self.auto_resume.get() and not resume else 1
            last_error = None
            for attempt in range(1, attempts + 1):
                try:
                    use_resume = resume or (attempt > 1 and self._last_checkpoint_path().exists())
                    if use_resume:
                        self.log_train(f"Resume attempt {attempt}: using last.pt")
                    train_code = self._train_code(data_yaml, local_model, device, resume=use_resume, force_cpu=False)
                    cmd = [sys.executable, "-c", train_code]
                    run_cmd_live(cmd, self.log_train, cwd=self.project.root)
                    self.log_train("Training complete. Best model is usually runs/mustatil/weights/best.pt")
                    return
                except Exception as exc:
                    last_error = exc
                    self.log_train(f"Training attempt {attempt} failed: {exc}")
                    if device == "directml":
                        self.log_train("DirectML failed or is unsupported here. Retrying once on CPU.")
                        train_code = self._train_code(data_yaml, local_model, "cpu", resume=(self._last_checkpoint_path().exists()), force_cpu=True)
                        cmd = [sys.executable, "-c", train_code]
                        run_cmd_live(cmd, self.log_train, cwd=self.project.root)
                        self.log_train("Training complete on CPU after DirectML fallback.")
                        return

            raise RuntimeError(f"Training failed. Last error: {last_error}")
        except Exception as exc:
            self.log_train(f"ERROR: {exc}")
            messagebox.showerror(APP_TITLE, str(exc))

    def export_onnx_thread(self): threading.Thread(target=self.export_onnx, daemon=True).start()
    def export_onnx(self):
        try:
            if not self.project: raise RuntimeError("Open a project first.")
            best = self.project.runs_dir / "mustatil" / "weights" / "best.pt"
            if not best.exists():
                best = Path(filedialog.askopenfilename(title="Choose best.pt", filetypes=[("PyTorch model", "*.pt")]))
            export_code = (
                "from ultralytics import YOLO\n"
                f"model = YOLO(r'{best}')\n"
                f"model.export(format='onnx', imgsz={int(self.imgsz.get())})\n"
            )
            cmd = [sys.executable, "-c", export_code]
            run_cmd_live(cmd, self.log_train)
            self.log_train("ONNX export complete. Use .pt for this tool; use .onnx where QGIS/GeoAI supports ONNX.")
        except Exception as exc:
            self.log_train(f"ERROR: {exc}")

    def pick_model(self): self.det_model.set(filedialog.askopenfilename(filetypes=[("Models", "*.pt *.onnx"), ("PyTorch", "*.pt"), ("ONNX", "*.onnx")]))

    def pick_sam2_model(self):
        p = filedialog.askopenfilename(title="Choose SAM/SAM2 model", filetypes=[("SAM/SAM2 models", "*.pt")])
        if p:
            self.sam2_model.set(p)

    def pick_detect_image(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp *.webp")])
        if p:
            self.det_image.set(p)
            self.load_detection_preview(image_only=True)

    def pick_output_geojson(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".gpkg",
            filetypes=[("GeoPackage", "*.gpkg")],
        )
        if p:
            if not p.lower().endswith(".gpkg"):
                p += ".gpkg"
            self.out_geojson.set(p)


    def clear_detection_preview(self):
        self.det_preview_img = None
        self.det_preview_photo = None
        self.det_preview_features = []
        self.det_preview_zoom = 1.0
        self.det_preview_pan_x = 0.0
        self.det_preview_pan_y = 0.0
        if hasattr(self, "det_canvas"):
            self.det_canvas.delete("all")

    def reset_detection_preview_view(self, event=None):
        self.det_preview_zoom = 1.0
        self.det_preview_pan_x = 0.0
        self.det_preview_pan_y = 0.0
        self.redraw_detection_preview()

    def on_detection_preview_pan_start(self, event):
        self.det_preview_drag_start = (event.x, event.y, self.det_preview_pan_x, self.det_preview_pan_y)

    def on_detection_preview_pan_move(self, event):
        if not self.det_preview_drag_start:
            return
        sx, sy, px, py = self.det_preview_drag_start
        self.det_preview_pan_x = px + (event.x - sx)
        self.det_preview_pan_y = py + (event.y - sy)
        self.redraw_detection_preview()

    def on_detection_preview_wheel(self, event):
        if self.det_preview_img is None:
            return
        old_zoom = max(0.1, float(self.det_preview_zoom))
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            factor = 1.15
        else:
            factor = 1.0 / 1.15
        new_zoom = max(0.1, min(16.0, old_zoom * factor))

        # Keep the point under the mouse roughly stable while zooming.
        mx, my = float(event.x), float(event.y)
        self.det_preview_pan_x = mx - (mx - self.det_preview_pan_x) * (new_zoom / old_zoom)
        self.det_preview_pan_y = my - (my - self.det_preview_pan_y) * (new_zoom / old_zoom)
        self.det_preview_zoom = new_zoom
        self.redraw_detection_preview()

    def load_detection_preview(self, image_only=False):
        """
        Loads only a downsampled preview for huge images.
        The full image is not loaded into RAM for GeoTIFFs.
        """
        img_path = Path(self.det_image.get()) if self.det_image.get() else None
        if not img_path or not img_path.exists():
            messagebox.showerror(APP_TITLE, "Choose a map image first.")
            return

        try:
            preview, orig_w, orig_h = load_preview_downsample(img_path, max_size=2400)
            self.det_preview_img = preview
            self.det_preview_orig_size = (orig_w, orig_h)
            if image_only:
                self.det_preview_zoom = 1.0
                self.det_preview_pan_x = 0.0
                self.det_preview_pan_y = 0.0
            self.log_detect(f"Preview loaded downsampled: {preview.width}x{preview.height}; original: {orig_w}x{orig_h}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open preview image:\n{exc}")
            return

        self.det_preview_features = []

        if not image_only:
            raw_out = Path(self.out_geojson.get()) if self.out_geojson.get() else img_path.with_suffix(".detections.gpkg")
            if raw_out.suffix.lower() == ".gpkg" and raw_out.exists():
                try:
                    self.det_preview_features = read_gpkg_preview_features(raw_out, self.log_detect)
                except Exception as exc:
                    self.log_detect(f"Could not load GeoPackage preview: {exc}")
            else:
                self.log_detect(f"No GeoPackage preview found yet: {raw_out}")

        self.redraw_detection_preview()

    def redraw_detection_preview(self):
        if not hasattr(self, "det_canvas"):
            return
        self.det_canvas.delete("all")
        if self.det_preview_img is None:
            self.det_canvas.create_text(12, 12, anchor="nw", fill="white", text="No preview loaded.")
            return

        cw = max(1, self.det_canvas.winfo_width())
        ch = max(1, self.det_canvas.winfo_height())
        im = self.det_preview_img
        orig_w, orig_h = self.det_preview_orig_size or im.size

        fit_scale = min(cw / im.width, ch / im.height, 1.0)
        zoom = max(0.1, float(getattr(self, "det_preview_zoom", 1.0)))
        scale_canvas = fit_scale * zoom
        sw, sh = max(1, int(im.width * scale_canvas)), max(1, int(im.height * scale_canvas))
        show = im.resize((sw, sh))

        base_ox = (cw - sw) // 2 if zoom <= 1.0001 else (cw - int(im.width * fit_scale)) // 2
        base_oy = (ch - sh) // 2 if zoom <= 1.0001 else (ch - int(im.height * fit_scale)) // 2
        ox = int(base_ox + float(getattr(self, "det_preview_pan_x", 0.0)))
        oy = int(base_oy + float(getattr(self, "det_preview_pan_y", 0.0)))

        preview_scale_x = im.width / max(1, orig_w)
        preview_scale_y = im.height / max(1, orig_h)
        total_scale_x = preview_scale_x * scale_canvas
        total_scale_y = preview_scale_y * scale_canvas

        self.det_preview_photo = ImageTk.PhotoImage(show)
        self.det_canvas.create_image(ox, oy, anchor="nw", image=self.det_preview_photo)

        features_count = len(self.det_preview_features)
        sam_shown = 0
        bbox_shown = 0
        green_filtered_count = 0

        for feat in self.det_preview_features:
            props = feat.get("properties", {})
            source = props.get("geometry_source", "bbox")
            bbox = props.get("pixel_bbox")
            conf = props.get("confidence", None)
            green_ratio = props.get("green_ratio", None)
            if props.get("green_filter_rejected"):
                green_filtered_count += 1

            poly_px = props.get("sam2_pixel_polygon")
            if source == "sam2_mask" and poly_px and len(poly_px) >= 3:
                pts = []
                for pnt in poly_px:
                    if len(pnt) >= 2:
                        pts.extend([ox + float(pnt[0]) * total_scale_x, oy + float(pnt[1]) * total_scale_y])
                if len(pts) >= 6:
                    self.det_canvas.create_polygon(*pts, outline="cyan", fill="", width=3)
                    sam_shown += 1

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                sx1, sy1 = ox + float(x1) * total_scale_x, oy + float(y1) * total_scale_y
                sx2, sy2 = ox + float(x2) * total_scale_x, oy + float(y2) * total_scale_y
                color = "cyan" if source == "sam2_mask" else "red"
                self.det_canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=2)
                if isinstance(conf, (int, float)):
                    label = f"{float(conf):.2f} {source}"
                else:
                    label = source
                if isinstance(green_ratio, (int, float)):
                    label += f" g={float(green_ratio):.2f}"
                self.det_canvas.create_text(sx1 + 3, sy1 + 3, anchor="nw", text=label, fill=color)
                bbox_shown += 1

        self.det_canvas.create_text(
            10, 10,
            anchor="nw",
            fill="white",
            text=(
                f"Preview: {im.width}x{im.height} | original: {orig_w}x{orig_h} | "
                f"zoom: {zoom:.2f}x | features: {features_count} | "
                f"SAM2 masks: {sam_shown} | boxes: {bbox_shown}"
            ),
        )

    def detect_thread(self): threading.Thread(target=self.detect, daemon=True).start()
    def detect(self):
        backend = self.det_backend.get() if hasattr(self, "det_backend") else "Ultralytics/PT CPU"

        ok, msg = dependency_report()
        self.log_detect(msg)
        if not ok:
            messagebox.showerror(APP_TITLE, msg)
            return

        try:
            from ultralytics import YOLO
            import numpy as np
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Install dependencies first.\n\n{exc}")
            return

        use_sam2 = bool(self.use_sam2.get())
        sam_model = None

        if use_sam2:
            try:
                from ultralytics import SAM
                sam_name = resolve_sam_model_path(
                    self.sam2_model.get().strip() or "sam2_b.pt",
                    self.project.root if self.project else None,
                    self.log_detect,
                )
                self.sam2_model.set(str(sam_name))
                self.log_detect(f"SAM2 enabled. Loading SAM/SAM2 model: {sam_name}")
                sam_model = SAM(str(sam_name))
                self.log_detect("SAM2 model loaded successfully.")
            except Exception as exc:
                self.log_detect(f"SAM2 could not be loaded, falling back to YOLO boxes only: {exc}")
                messagebox.showwarning(APP_TITLE, f"SAM2 could not be loaded.\nFalling back to YOLO boxes only.\n\n{exc}")
                use_sam2 = False
                sam_model = None
        else:
            self.log_detect("SAM2 refinement is disabled. Only YOLO boxes will be exported.")

        model_path = Path(self.det_model.get())
        img_path = Path(self.det_image.get())
        out_path = Path(self.out_geojson.get() or (img_path.with_suffix(".detections.gpkg")))
        if out_path.suffix.lower() != ".gpkg":
            out_path = out_path.with_suffix(".gpkg")
            self.out_geojson.set(str(out_path))

        if not model_path.exists() or not img_path.exists():
            return messagebox.showerror(APP_TITLE, "Choose YOLO model and image.")

        if backend in {"Ultralytics/ONNX Runtime", "ONNX Runtime DirectML GPU"} and model_path.suffix.lower() != ".onnx":
            return messagebox.showerror(APP_TITLE, "For ONNX backend, choose best.onnx, not best.pt.")

        if backend == "ONNX Runtime DirectML GPU":
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                self.log_detect(f"ONNX Runtime available providers: {providers}")
                if "DmlExecutionProvider" in providers:
                    self.log_detect("GPU STATUS: DmlExecutionProvider is available.")
                    self.log_detect("Detection uses Ultralytics YOLO(best.onnx).predict() for correct ONNX preprocessing/NMS.")
                else:
                    self.log_detect("GPU STATUS: DmlExecutionProvider is NOT available in this Python environment.")
                    self.log_detect("Install in this env for ONNX GPU detection: python -m pip install onnxruntime-directml")
            except Exception as exc:
                self.log_detect(f"Could not check ONNX Runtime providers: {exc}")

        self.log_detect("=" * 70)
        self.log_detect(f"Detection backend: {backend}")
        self.log_detect(f"Model: {model_path}")
        self.log_detect(f"Image: {img_path}")

        model = YOLO(str(model_path))

        tile = int(self.tile.get())
        overlap = int(self.overlap.get())
        stride = max(1, tile - overlap)
        conf = float(self.conf.get())
        use_green_filter = bool(getattr(self, "use_green_filter", tk.BooleanVar(value=False)).get())
        try:
            green_threshold = float(getattr(self, "green_filter_threshold", tk.DoubleVar(value=0.18)).get())
        except Exception:
            green_threshold = 0.18
        green_threshold = max(0.0, min(1.0, green_threshold))

        # ONNX models may have a fixed input size, e.g. best.onnx exported at 640x640.
        # The map tile can still be 1024x1024, but Ultralytics must preprocess it to the ONNX size.
        infer_imgsz = tile
        if model_path.suffix.lower() == ".onnx":
            fixed = get_onnx_fixed_input_size(model_path, self.log_detect)
            if fixed:
                infer_imgsz = fixed
                if tile != fixed:
                    self.log_detect(
                        f"ONNX size fix active: tiles remain {tile}x{tile}, "
                        f"but model inference uses imgsz={infer_imgsz}."
                    )

        reader = None
        mode = None
        try:
            W, H, mode, reader = open_large_image_info(img_path, self.log_detect)
        except Exception as exc:
            return messagebox.showerror(APP_TITLE, f"Could not open image for streaming detection:\n\n{exc}")

        self.log_detect(f"Image size: {W} x {H}; tile={tile}; overlap={overlap}; stride={stride}")
        self.log_detect(f"Confidence={conf}")
        self.log_detect(f"Vegetation/Bush green filter active: {use_green_filter}; threshold={green_threshold:.3f}")
        self.log_detect(f"SAM2 refinement active: {use_sam2}")
        if use_sam2:
            try:
                self.log_detect(f"SAM2 safety: max boxes/tile={int(self.sam2_max_per_tile.get())}, padding={int(self.sam2_padding.get())}, max crop={int(self.sam2_max_crop.get())}")
            except Exception:
                pass

        default_crs = self.default_crs.get().strip() if hasattr(self, "default_crs") else "EPSG:3857"
        georef_kind, transform, crs_name = get_image_georeference_for_detection(
            img_path,
            reader,
            mode,
            default_crs=default_crs,
            log_fn=self.log_detect,
        )

        def geo_or_pixel(points_px):
            if georef_kind == "rasterio" and transform is not None:
                return [(transform * (float(px), float(py))) for px, py in points_px]
            if georef_kind in {"worldfile", "geotiff_pillow_tags"} and transform is not None:
                return [apply_affine_tuple(transform, px, py) for px, py in points_px]
            return [(float(px), float(py)) for px, py in points_px]

        features = []
        total = math.ceil(W / stride) * math.ceil(H / stride)
        n = 0
        yolo_detections = 0
        sam_success = 0
        sam_fallback = 0
        green_rejected = 0
        t0 = time.time()

        try:
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    n += 1
                    crop = read_detection_tile(reader, mode, x, y, tile, W, H)
                    arr = np.array(crop)

                    local_dets = []
                    try:
                        results = model.predict(arr, conf=conf, imgsz=infer_imgsz, verbose=False)
                        for rres in results:
                            if rres.boxes is None:
                                continue
                            for b in rres.boxes:
                                xyxy = b.xyxy.cpu().numpy()[0].tolist()
                                score = float(b.conf.cpu().numpy()[0])
                                cls = int(b.cls.cpu().numpy()[0])
                                # Class 1 is trained as false_positive/bush. Do not export it as a detection.
                                if cls != 0:
                                    continue

                                lx1, ly1, lx2, ly2 = map(float, xyxy)
                                lx1 = max(0.0, min(float(crop.width - 1), lx1))
                                ly1 = max(0.0, min(float(crop.height - 1), ly1))
                                lx2 = max(0.0, min(float(crop.width - 1), lx2))
                                ly2 = max(0.0, min(float(crop.height - 1), ly2))
                                if lx2 <= lx1 or ly2 <= ly1:
                                    continue

                                local_bbox = [lx1, ly1, lx2, ly2]
                                green_ratio = detection_crop_green_ratio(arr, local_bbox) if use_green_filter else 0.0
                                if use_green_filter and green_ratio > green_threshold:
                                    green_rejected += 1
                                    continue

                                local_dets.append({
                                    "cls": cls,
                                    "score": score,
                                    "local_bbox": local_bbox,
                                    "global_bbox": [lx1 + x, ly1 + y, lx2 + x, ly2 + y],
                                    "green_ratio": float(green_ratio),
                                })
                    except Exception as exc:
                        self.log_detect(f"Ultralytics prediction failed at tile ({x},{y}): {exc}")

                    yolo_detections += len(local_dets)
                    sam_polygons_by_index = [None] * len(local_dets)

                    if use_sam2 and sam_model is not None and local_dets:
                        try:
                            max_sam = max(0, int(self.sam2_max_per_tile.get()))
                        except Exception:
                            max_sam = 5
                        try:
                            sam_pad = max(0, int(self.sam2_padding.get()))
                        except Exception:
                            sam_pad = 96
                        try:
                            sam_max_crop = max(128, int(self.sam2_max_crop.get()))
                        except Exception:
                            sam_max_crop = 768

                        # Refine only the highest-confidence boxes per tile to avoid CPU/RAM crashes.
                        order = sorted(range(len(local_dets)), key=lambda k: local_dets[k]["score"], reverse=True)
                        refine_set = set(order[:max_sam]) if max_sam > 0 else set()

                        skipped = max(0, len(local_dets) - len(refine_set))
                        if skipped:
                            sam_fallback += skipped

                        for i in order:
                            if i not in refine_set:
                                continue
                            det = local_dets[i]
                            try:
                                best_poly = sam2_refine_box_safe(
                                    sam_model,
                                    crop,
                                    det["local_bbox"],
                                    x,
                                    y,
                                    padding=sam_pad,
                                    max_crop_size=sam_max_crop,
                                    log_fn=self.log_detect if sam_fallback < 5 else None,
                                )
                                if best_poly:
                                    sam_polygons_by_index[i] = best_poly
                                    sam_success += 1
                                else:
                                    sam_fallback += 1
                            except MemoryError:
                                sam_fallback += 1
                                self.log_detect("SAM2 MemoryError avoided: reduce SAM2 max boxes/tile or max crop px.")
                            except Exception as exc:
                                sam_fallback += 1
                                if sam_fallback <= 10:
                                    self.log_detect(f"SAM2 failed for one YOLO box at tile ({x},{y}): {exc}")

                    for i, det in enumerate(local_dets):
                        gx1, gy1, gx2, gy2 = det["global_bbox"]
                        sam_poly_px = sam_polygons_by_index[i]

                        if sam_poly_px and len(sam_poly_px) >= 4:
                            geom_source = "sam2_mask"
                            pts_px = sam_poly_px
                        else:
                            geom_source = "bbox"
                            pts_px = [(gx1, gy1), (gx2, gy1), (gx2, gy2), (gx1, gy2), (gx1, gy1)]

                        pts = geo_or_pixel(pts_px)
                        props = {
                            "class": "mustatil",
                            "class_id": int(det["cls"]),
                            "confidence": float(det["score"]),
                            "geometry_source": geom_source,
                            "backend": backend,
                            "image_read_mode": mode,
                            "georef_source": georef_kind,
                            "crs": crs_name,
                            "inference_engine": "ultralytics_predict_streaming",
                            "pixel_bbox": [float(gx1), float(gy1), float(gx2), float(gy2)],
                            "green_ratio": float(det.get("green_ratio", 0.0)),
                            "green_filter_threshold": float(green_threshold) if use_green_filter else None,
                        }
                        if geom_source == "sam2_mask":
                            props["sam2_pixel_polygon"] = [[float(px), float(py)] for px, py in sam_poly_px]

                        features.append({
                            "type": "Feature",
                            "properties": props,
                            "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in pts]]},
                        })

                    if n % 10 == 0:
                        elapsed = max(0.001, time.time() - t0)
                        self.log_detect(
                            f"Processed {n}/{total} tiles | YOLO boxes={yolo_detections} | "
                            f"green rejected={green_rejected} | SAM2 masks={sam_success} | fallback boxes={sam_fallback} | "
                            f"{n/elapsed:.2f} tiles/sec | read={mode} | backend={backend}"
                        )

                if y + tile >= H:
                    break

        finally:
            try:
                if mode == "rasterio" and reader is not None:
                    reader.close()
            except Exception:
                pass

        # CRS sanity correction:
        # Coordinates in the millions are WebMercator meters, not EPSG:4326 degrees.
        try:
            if crs_name == "EPSG:4326" and features:
                coords0 = features[0]["geometry"]["coordinates"][0][0]
                if abs(float(coords0[0])) > 1000 or abs(float(coords0[1])) > 1000:
                    self.log_detect("CRS auto-correction: coordinates are not degrees -> forcing EPSG:3857")
                    crs_name = "EPSG:3857"
        except Exception as exc:
            self.log_detect(f"CRS auto-correction check failed: {exc}")

        fc = {
            "type": "FeatureCollection",
            "name": "mustatil_detections",
            "crs": {"type": "name", "properties": {"name": crs_name}} if crs_name else None,
            "features": features,
        }
        if fc["crs"] is None:
            del fc["crs"]

        write_features_outputs(out_path, fc, crs_name, self.log_detect)
        elapsed = max(0.001, time.time() - t0)

        self.log_detect("=" * 70)
        self.log_detect(f"Done. Wrote {len(features)} detections: {out_path}")
        self.log_detect(f"FINAL: backend={backend}; read_mode={mode}; georef={georef_kind}; YOLO boxes={yolo_detections}; green rejected={green_rejected}; SAM2 masks={sam_success}; fallback boxes={sam_fallback}")
        if georef_kind == "pixel":
            self.log_detect("WARNING: GeoJSON contains PIXEL coordinates, not map coordinates. Use GeoTIFF or worldfile for QGIS overlay.")
        self.log_detect(f"Elapsed: {elapsed:.1f}s; Speed: {n/elapsed:.2f} tiles/sec")

        self.out_geojson.set(str(out_path))
        try:
            self.load_detection_preview()
            self.log_detect("Preview reloaded directly from GeoPackage.")
        except Exception as exc:
            self.log_detect(f"Preview reload failed: {exc}")

        messagebox.showinfo(
            APP_TITLE,
            f"Detection complete.\n"
            f"Backend: {backend}\n"
            f"Read mode: {mode}\n"
            f"Features: {len(features)}\n"
            f"Green/bush rejected: {green_rejected}\n"
            f"SAM2 masks: {sam_success}\n"
            f"Fallback boxes: {sam_fallback}\n"
            f"Speed: {n/elapsed:.2f} tiles/sec\n"
            f"{out_path}"
        )

if __name__ == "__main__":
    app = MustatilGUI()
    app.mainloop()
