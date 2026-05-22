#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble Detect Map GUI
- Select up to 3 YOLO .pt/.onnx models
- Run tiled detection on large images/GeoTIFFs
- Combine detections by overlap / proximity / local density
- Preview detections in different colors with zoom + pan
- Export ensemble detections to GeoJSON
"""
from __future__ import annotations

import json, math, sys, time, threading, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    Image.MAX_IMAGE_PIXELS = None
except Exception as exc:
    raise SystemExit(f"Missing Pillow. Install with: python -m pip install pillow\n{exc}")

try:
    import numpy as np
except Exception as exc:
    raise SystemExit(f"Missing numpy. Install with: python -m pip install numpy\n{exc}")

APP_TITLE = "Ensemble Mustatil Detect Map GUI"

@dataclass
class Detection:
    model_index: int
    model_name: str
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 0.0
    consensus_count: int = 1
    proximity_bonus: float = 0.0
    density_bonus: float = 0.0
    nonoverlap_penalty: float = 0.0
    def bbox(self): return (self.x1, self.y1, self.x2, self.y2)
    def center(self): return ((self.x1+self.x2)/2.0, (self.y1+self.y2)/2.0)

def have_module(name: str) -> bool:
    try:
        __import__(name); return True
    except Exception:
        return False

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0.0, ix2-ix1)*max(0.0, iy2-iy1)
    if inter <= 0: return 0.0
    aa=max(0.0, ax2-ax1)*max(0.0, ay2-ay1); ba=max(0.0, bx2-bx1)*max(0.0, by2-by1)
    return inter / max(1e-9, aa+ba-inter)

def rasterio_available(): return have_module("rasterio")
def is_tiff(path: Path): return path.suffix.lower() in {".tif", ".tiff"}

def open_large_image(path: Path, log_fn):
    if is_tiff(path):
        if not rasterio_available():
            raise RuntimeError("TIFF/GeoTIFF requires rasterio. Install: python -m pip install rasterio")
        import rasterio
        src = rasterio.open(path)
        log_fn("Large image mode: rasterio streaming TIFF/GeoTIFF")
        log_fn(f"Raster size: {src.width} x {src.height}; bands={src.count}; crs={src.crs}")
        return src.width, src.height, "rasterio", src
    im = Image.open(path)
    log_fn("Large image mode: PIL normal image")
    log_fn(f"Image size: {im.width} x {im.height}")
    return im.width, im.height, "pil", im

def read_tile(reader, mode, x, y, tile, W, H):
    w = min(tile, W-x); h = min(tile, H-y)
    if mode == "rasterio":
        from rasterio.windows import Window
        win = Window(x, y, w, h)
        if reader.count >= 3:
            arr = reader.read([1,2,3], window=win, boundless=True, fill_value=0)
            arr = np.moveaxis(arr, 0, -1)
        else:
            arr = reader.read(1, window=win, boundless=True, fill_value=0)
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.dtype != np.uint8:
            arr = arr.astype("float32")
            mn=float(np.nanmin(arr)) if arr.size else 0.0; mx=float(np.nanmax(arr)) if arr.size else 1.0
            if mx > mn: arr = (arr-mn)/(mx-mn)*255.0
            arr = np.clip(arr,0,255).astype("uint8")
        return Image.fromarray(arr, "RGB")
    return reader.crop((x,y,x+w,y+h)).convert("RGB")

def load_preview(path: Path, max_size=2400):
    if is_tiff(path):
        if not rasterio_available(): raise RuntimeError("TIFF preview requires rasterio.")
        import rasterio
        with rasterio.open(path) as src:
            scale = min(max_size/max(1,src.width), max_size/max(1,src.height), 1.0)
            ow=max(1,int(src.width*scale)); oh=max(1,int(src.height*scale))
            if src.count >= 3:
                arr = src.read([1,2,3], out_shape=(3, oh, ow)); arr=np.moveaxis(arr,0,-1)
            else:
                arr = src.read(1, out_shape=(oh,ow)); arr=np.stack([arr,arr,arr],axis=-1)
            if arr.dtype != np.uint8:
                arr=arr.astype("float32"); mn=float(np.nanmin(arr)); mx=float(np.nanmax(arr))
                if mx>mn: arr=(arr-mn)/(mx-mn)*255.0
                arr=np.clip(arr,0,255).astype("uint8")
            return Image.fromarray(arr,"RGB"), src.width, src.height
    im=Image.open(path).convert("RGB"); orig=im.size; im.thumbnail((max_size,max_size)); return im.copy(), orig[0], orig[1]

def get_raster_transform(reader, mode: str, log_fn):
    if mode == "rasterio":
        try:
            tr = reader.transform
            crs_obj = reader.crs
            crs = crs_obj.to_string() if crs_obj else None
            if tr is not None:
                try:
                    is_identity = bool(tr.is_identity)
                except Exception:
                    is_identity = False
                if not is_identity:
                    if log_fn:
                        log_fn("Georeference source: rasterio GeoTIFF transform")
                        log_fn(f"Rasterio transform: {tr}")
                        log_fn(f"Rasterio CRS: {crs}")
                    return "rasterio", tr, crs
            if log_fn:
                log_fn("Rasterio opened file, but transform is identity/invalid.")
        except Exception as exc:
            if log_fn:
                log_fn(f"Rasterio georeference read failed: {exc}")
    if log_fn:
        log_fn("No georeference transform available; export will use pixel coordinates.")
    return "pixel", None, None


def pixel_to_geo(kind, transform, px, py):
    if kind == "rasterio" and transform is not None:
        xg, yg = transform * (px, py); return float(xg), float(yg)
    return float(px), float(py)

def get_onnx_fixed_input_size(model_path: Path, log_fn=None):
    if model_path.suffix.lower() != ".onnx": return None
    try:
        import onnxruntime as ort
        sess=ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        shape=list(sess.get_inputs()[0].shape)
        if len(shape)>=4 and isinstance(shape[2], int) and isinstance(shape[3], int) and shape[2]==shape[3]:
            if log_fn: log_fn(f"{model_path.name}: ONNX fixed input size {shape[3]}x{shape[2]}")
            return int(shape[2])
    except Exception as exc:
        if log_fn: log_fn(f"{model_path.name}: could not read ONNX shape: {exc}")
    return None


def _epsg_int(crs_name):
    """
    Parse EPSG from rasterio CRS string.
    If unknown/pixel coordinates, use -1 so QGIS does not falsely georeference as EPSG:3857.
    """
    try:
        if crs_name is None:
            return -1
        s = str(crs_name).upper().strip()
        if not s or s in {"NONE", "PIXEL"}:
            return -1
        if "EPSG:" in s:
            tail = s.split("EPSG:")[-1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            return int(digits) if digits else -1
        if s.isdigit():
            return int(s)
    except Exception:
        pass
    return -1


def _polygon_wkb(poly_coords):
    import struct
    clean_rings = []
    for ring in poly_coords:
        pts = []
        for p in ring:
            if len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
        if len(pts) >= 3:
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


def write_gpkg_features(gpkg_path: Path, features: list, crs_name=None, log_fn=None):
    """
    QGIS-safe GeoPackage writer.
    First tries Fiona/GDAL, because QGIS reads those GeoPackages reliably.
    Falls back to the internal SQLite writer only if Fiona is unavailable.
    """
    if not features:
        raise RuntimeError("No features to write.")

    gpkg_path = Path(gpkg_path)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    if log_fn:
        log_fn(f"Writing QGIS-safe GeoPackage: {gpkg_path}")
        log_fn(f"Features to write: {len(features)}")
        log_fn(f"CRS input: {crs_name}")
        try:
            log_fn(f"First coordinate: {features[0]['geometry']['coordinates'][0][0]}")
        except Exception:
            pass

    # Preferred: Fiona/GDAL standard GeoPackage
    try:
        import tempfile, time, shutil
        import fiona
        from fiona.crs import CRS

        crs_obj = None
        srs_id = _epsg_int(crs_name)
        if srs_id and srs_id > 0:
            crs_obj = CRS.from_epsg(int(srs_id))
        elif crs_name:
            try:
                crs_obj = CRS.from_string(str(crs_name))
            except Exception:
                crs_obj = None

        tmpdir = Path(tempfile.gettempdir()) / "mustatil_ensemble_fiona_gpkg"
        tmpdir.mkdir(parents=True, exist_ok=True)
        local_tmp = tmpdir / f"{gpkg_path.stem}_{int(time.time())}.gpkg"
        if local_tmp.exists():
            local_tmp.unlink()

        # Use flexible string fields to avoid schema problems.
        schema = {
            "geometry": "Polygon",
            "properties": {
                "feature_type": "str:40",
                "model_index": "int",
                "model_name": "str:120",
                "confidence": "float",
                "ens_score": "float",
                "consensus": "int",
                "model_cnt": "int",
                "member_cnt": "int",
                "models": "str:40",
                "avg_conf": "float",
                "near_cnt": "int",
                "dens_cnt": "int",
                "rank": "int",
                "pixel_bbox": "str:160",
            },
        }

        with fiona.open(
            local_tmp,
            mode="w",
            driver="GPKG",
            layer="detections",
            schema=schema,
            crs=crs_obj,
            encoding="UTF-8",
        ) as dst:
            for feat in features:
                geom = feat.get("geometry")
                if not geom or geom.get("type") != "Polygon":
                    continue
                p = feat.get("properties", {})
                dst.write({
                    "geometry": geom,
                    "properties": {
                        "feature_type": p.get("type") or "",
                        "model_index": int(p.get("model_index") or 0) if p.get("model_index") is not None else None,
                        "model_name": str(p.get("model_name") or ""),
                        "confidence": float(p.get("confidence") or 0.0) if p.get("confidence") is not None else None,
                        "ens_score": float(p.get("ensemble_score") or 0.0) if p.get("ensemble_score") is not None else None,
                        "consensus": int(p.get("consensus_count") or 0) if p.get("consensus_count") is not None else None,
                        "model_cnt": int(p.get("model_count") or 0) if p.get("model_count") is not None else None,
                        "member_cnt": int(p.get("member_count") or 0) if p.get("member_count") is not None else None,
                        "models": str(p.get("models") or ""),
                        "avg_conf": float(p.get("avg_conf") or 0.0) if p.get("avg_conf") is not None else None,
                        "near_cnt": int(p.get("near_count") or 0) if p.get("near_count") is not None else None,
                        "dens_cnt": int(p.get("density_count") or 0) if p.get("density_count") is not None else None,
                        "rank": int(p.get("rank") or 0) if p.get("rank") is not None else None,
                        "pixel_bbox": str(p.get("pixel_bbox") or ""),
                    },
                })

        final_target = gpkg_path
        if gpkg_path.exists():
            try:
                gpkg_path.unlink()
            except Exception:
                final_target = gpkg_path.with_name(gpkg_path.stem + f"_new_{int(time.time())}" + gpkg_path.suffix)
        shutil.copy2(local_tmp, final_target)
        try:
            local_tmp.unlink()
        except Exception:
            pass

        if log_fn:
            log_fn(f"Fiona/GDAL GeoPackage written: {final_target}")
            log_fn("This should load correctly in QGIS.")
        return final_target

    except Exception as fiona_exc:
        if log_fn:
            log_fn(f"Fiona/GDAL GeoPackage writer failed or missing: {fiona_exc}")
            log_fn("Falling back to internal SQLite GeoPackage writer.")

    # Fallback: minimal SQLite GeoPackage
    import sqlite3, json, time, tempfile, shutil

    srs_id = _epsg_int(crs_name)
    tmpdir = Path(tempfile.gettempdir()) / "mustatil_ensemble_gpkg_fallback"
    tmpdir.mkdir(parents=True, exist_ok=True)
    local_tmp = tmpdir / f"{gpkg_path.stem}_{int(time.time())}.gpkg"
    if local_tmp.exists():
        local_tmp.unlink()

    conn = sqlite3.connect(str(local_tmp), timeout=60)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA application_id = 1196437808")
        cur.execute("PRAGMA user_version = 10400")
        cur.execute("""CREATE TABLE gpkg_spatial_ref_sys (
            srs_name TEXT NOT NULL, srs_id INTEGER NOT NULL PRIMARY KEY,
            organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL, description TEXT)""")
        rows = [
            ("Undefined Cartesian SRS", -1, "NONE", -1, "undefined", "undefined cartesian coordinate reference system"),
            ("Undefined Geographic SRS", 0, "NONE", 0, "undefined", "undefined geographic coordinate reference system"),
            ("WGS 84 / Pseudo-Mercator", 3857, "EPSG", 3857, "EPSG:3857", "Web Mercator meters"),
            ("WGS 84 geodetic", 4326, "EPSG", 4326, "EPSG:4326", "longitude/latitude coordinates in decimal degrees on WGS84"),
        ]
        cur.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)", rows)
        if srs_id not in (-1, 0, 4326, 3857):
            cur.execute("INSERT OR IGNORE INTO gpkg_spatial_ref_sys VALUES (?, ?, 'EPSG', ?, ?, ?)",
                        (f"EPSG:{srs_id}", srs_id, srs_id, f"EPSG:{srs_id}", f"EPSG:{srs_id}"))

        cur.execute("""CREATE TABLE gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
            identifier TEXT UNIQUE, description TEXT DEFAULT '',
            last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER)""")
        cur.execute("""CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL, column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL, m TINYINT NOT NULL,
            PRIMARY KEY (table_name, column_name))""")
        cur.execute("""CREATE TABLE detections (
            fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB NOT NULL,
            feature_type TEXT, model_index INTEGER, model_name TEXT, confidence DOUBLE,
            ensemble_score DOUBLE, consensus_count INTEGER, model_count INTEGER,
            member_count INTEGER, models TEXT, avg_conf DOUBLE, near_count INTEGER,
            density_count INTEGER, rank INTEGER, pixel_bbox TEXT)""")
        cur.execute("INSERT INTO gpkg_geometry_columns VALUES ('detections', 'geom', 'POLYGON', ?, 0, 0)", (srs_id,))

        minx = miny = float("inf"); maxx = maxy = float("-inf"); count = 0
        for feat in features:
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])
            wkb = _polygon_wkb(coords)
            if not wkb:
                continue
            for ring in coords:
                for p in ring:
                    if len(p) >= 2:
                        x=float(p[0]); y=float(p[1])
                        minx=min(minx,x); miny=min(miny,y); maxx=max(maxx,x); maxy=max(maxy,y)
            props=feat.get("properties", {})
            cur.execute("""INSERT INTO detections
                (geom, feature_type, model_index, model_name, confidence, ensemble_score, consensus_count,
                model_count, member_count, models, avg_conf, near_count, density_count, rank, pixel_bbox)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sqlite3.Binary(_gpkg_geom_blob(wkb, srs_id)), props.get("type"), props.get("model_index"),
                props.get("model_name"), props.get("confidence"), props.get("ensemble_score"), props.get("consensus_count"),
                props.get("model_count"), props.get("member_count"), props.get("models"), props.get("avg_conf"),
                props.get("near_count"), props.get("density_count"), props.get("rank"),
                json.dumps(props.get("pixel_bbox")) if props.get("pixel_bbox") is not None else None))
            count += 1

        if count == 0:
            raise RuntimeError("No valid polygon features to write.")

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cur.execute("""INSERT INTO gpkg_contents
            (table_name, data_type, identifier, description, last_change, min_x, min_y, max_x, max_y, srs_id)
            VALUES ('detections', 'features', 'detections', 'Ensemble detections', ?, ?, ?, ?, ?, ?)""",
            (now, minx, miny, maxx, maxy, srs_id))
        conn.commit()
    finally:
        conn.close()

    final_target = gpkg_path
    if gpkg_path.exists():
        try:
            gpkg_path.unlink()
        except Exception:
            final_target = gpkg_path.with_name(gpkg_path.stem + f"_new_{int(time.time())}" + gpkg_path.suffix)
    shutil.copy2(local_tmp, final_target)
    try:
        local_tmp.unlink()
    except Exception:
        pass
    if log_fn:
        log_fn(f"Fallback SQLite GeoPackage written: {final_target}")
    return final_target


class EnsembleDetectGUI(tk.Tk):
    def __init__(self):
        super().__init__(); self.title(APP_TITLE); self.geometry("1550x950")
        self.model_paths=[tk.StringVar(), tk.StringVar(), tk.StringVar()]
        self.image_path=tk.StringVar(); self.output_path=tk.StringVar(); self.backend=tk.StringVar(value="Ultralytics Auto")
        self.conf=tk.DoubleVar(value=0.05); self.display_conf=tk.DoubleVar(value=0.05); self.export_conf=tk.DoubleVar(value=0.05); self.tile=tk.IntVar(value=1024); self.overlap=tk.IntVar(value=384); self.shifted_tiles=tk.BooleanVar(value=True); self.save_debug_tiles=tk.BooleanVar(value=False)
        self.w_model1=tk.DoubleVar(value=1.0); self.w_model2=tk.DoubleVar(value=2.0); self.w_model3=tk.DoubleVar(value=3.0)
        self.w_conf=tk.DoubleVar(value=1.0); self.w_proximity=tk.DoubleVar(value=0.5); self.proximity_px=tk.IntVar(value=400)
        self.w_density=tk.DoubleVar(value=0.25); self.density_radius_px=tk.IntVar(value=900); self.w_nonoverlap=tk.DoubleVar(value=-0.5)
        self.iou_match=tk.DoubleVar(value=0.30)
        self.preview_img=None; self.preview_photo=None; self.preview_orig_size=None; self.detections=[]; self.clusters=[]; self._last_georef=("pixel",None,None); self._last_image_path=None
        self.zoom=1.0; self.pan_x=0; self.pan_y=0; self._drag_last=None
        self.show_m1=tk.BooleanVar(value=True); self.show_m2=tk.BooleanVar(value=True); self.show_m3=tk.BooleanVar(value=True)
        self.show_consensus=tk.BooleanVar(value=True); self.show_labels=tk.BooleanVar(value=True)
        # Auto-crop export for training/annotation
        self.crop_size=tk.IntVar(value=1024)
        self.crop_pad=tk.IntVar(value=128)
        self.crop_source=tk.StringVar(value="Visible raw detections")
        self.crop_output_dir=tk.StringVar(value="")
        self.crop_class=tk.IntVar(value=0)  # 0=positive candidate, 1=false_positive candidate
        self._build_ui(); self.after(200, self.dependency_log)

    def _build_ui(self):
        root=ttk.PanedWindow(self, orient=tk.HORIZONTAL); root.pack(fill=tk.BOTH, expand=True)
        left=ttk.Frame(root,padding=8); right=ttk.Frame(root,padding=8); root.add(left, weight=0); root.add(right, weight=1)
        settings=ttk.LabelFrame(left,text="Inputs",padding=8); settings.pack(fill=tk.X)
        colors=["red","green","blue"]
        for i in range(3):
            ttk.Label(settings,text=f"Model {i+1} ({colors[i]})").grid(row=i,column=0,sticky="w",pady=3)
            ttk.Entry(settings,textvariable=self.model_paths[i],width=46).grid(row=i,column=1,sticky="ew",pady=3)
            ttk.Button(settings,text="Browse",command=lambda idx=i:self.pick_model(idx)).grid(row=i,column=2,padx=3)
        r=3
        for label,var,cmd in [("Map image / GeoTIFF",self.image_path,self.pick_image),("Output GeoPackage",self.output_path,self.pick_output)]:
            ttk.Label(settings,text=label).grid(row=r,column=0,sticky="w",pady=3); ttk.Entry(settings,textvariable=var,width=46).grid(row=r,column=1,sticky="ew",pady=3); ttk.Button(settings,text="Browse",command=cmd).grid(row=r,column=2,padx=3); r+=1
        ttk.Label(settings,text="Backend").grid(row=r,column=0,sticky="w",pady=3)
        ttk.Combobox(settings,textvariable=self.backend,values=["Ultralytics Auto","Ultralytics/PT CPU","Ultralytics/ONNX Runtime","ONNX Runtime DirectML GPU"],state="readonly").grid(row=r,column=1,columnspan=2,sticky="ew",pady=3); r+=1
        for label,var in [("Detection conf min",self.conf),("Tile size",self.tile),("Overlap",self.overlap)]:
            ttk.Label(settings,text=label).grid(row=r,column=0,sticky="w",pady=3); ttk.Entry(settings,textvariable=var).grid(row=r,column=1,columnspan=2,sticky="ew",pady=3); r+=1
        settings.columnconfigure(1,weight=1)
        scoring=ttk.LabelFrame(left,text="Priority / Consensus Scoring",padding=8); scoring.pack(fill=tk.X,pady=(8,0))
        rows=[("1 model detects",self.w_model1),("2 models overlap",self.w_model2),("3 models overlap",self.w_model3),("Confidence weight",self.w_conf),("Proximity weight",self.w_proximity),("Proximity radius px",self.proximity_px),("Density weight",self.w_density),("Density radius px",self.density_radius_px),("Non-overlap penalty",self.w_nonoverlap),("Overlap IoU threshold",self.iou_match)]
        for rr,(label,var) in enumerate(rows):
            ttk.Label(scoring,text=label).grid(row=rr,column=0,sticky="w",pady=2); ttk.Entry(scoring,textvariable=var,width=12).grid(row=rr,column=1,sticky="ew",pady=2)
        view=ttk.LabelFrame(left,text="Preview Layers",padding=8); view.pack(fill=tk.X,pady=(8,0))
        for text,var in [("Model 1 red",self.show_m1),("Model 2 green",self.show_m2),("Model 3 blue",self.show_m3),("Consensus yellow/magenta",self.show_consensus),("Labels / scores",self.show_labels)]:
            ttk.Checkbutton(view,text=text,variable=var,command=self.redraw_preview).pack(anchor="w")

        filt=ttk.LabelFrame(left,text="Live Confidence Filter",padding=8); filt.pack(fill=tk.X,pady=(8,0))
        ttk.Label(filt,text="Detection niedrig laufen lassen, danach hier filtern.").pack(anchor="w")
        self.display_conf_label=ttk.Label(filt,text=f"Visible >= {self.display_conf.get():.2f}")
        self.display_conf_label.pack(anchor="w",pady=(2,0))
        ttk.Scale(filt,from_=0.0,to=1.0,orient=tk.HORIZONTAL,variable=self.display_conf,command=self.on_display_conf_change).pack(fill=tk.X,pady=3)
        quick=ttk.Frame(filt); quick.pack(fill=tk.X,pady=(2,0))
        for val in [0.01,0.03,0.05,0.10,0.15,0.20,0.30,0.50]:
            ttk.Button(quick,text=f"{val:.2f}",command=lambda v=val:self.set_display_conf(v)).pack(side=tk.LEFT,expand=True,fill=tk.X,padx=1)
        ttk.Label(filt,text="Export filter").pack(anchor="w",pady=(6,0))
        self.export_conf_label=ttk.Label(filt,text=f"Export >= {self.export_conf.get():.2f}")
        self.export_conf_label.pack(anchor="w",pady=(2,0))
        ttk.Scale(filt,from_=0.0,to=1.0,orient=tk.HORIZONTAL,variable=self.export_conf,command=self.on_export_conf_change).pack(fill=tk.X,pady=3)

        tileopts=ttk.LabelFrame(left,text="Tile Debug / Anti-Miss",padding=8); tileopts.pack(fill=tk.X,pady=(8,0))
        ttk.Checkbutton(tileopts,text="Shifted tile passes: normal + half-stride offsets",variable=self.shifted_tiles).pack(anchor="w")
        ttk.Checkbutton(tileopts,text="Save debug tiles with detections",variable=self.save_debug_tiles).pack(anchor="w")
        ttk.Label(tileopts,text="Hilft bei Treffern, die vorher an Tile-Grenzen verloren gingen.",foreground="#555").pack(anchor="w")
        buttons=ttk.LabelFrame(left,text="Actions",padding=6); buttons.pack(fill=tk.X,pady=6)
        ttk.Button(buttons,text="Load Preview",command=self.load_preview_clicked).grid(row=0,column=0,sticky="ew",padx=2,pady=2)
        ttk.Button(buttons,text="Run Detection",command=self.detect_thread).grid(row=0,column=1,sticky="ew",padx=2,pady=2)
        ttk.Button(buttons,text="Recompute",command=self.recompute_and_redraw).grid(row=0,column=2,sticky="ew",padx=2,pady=2)
        ttk.Button(buttons,text="Reset Zoom",command=self.reset_zoom).grid(row=0,column=3,sticky="ew",padx=2,pady=2)
        ttk.Button(buttons,text="Convert PT to ONNX",command=self.export_onnx_clicked).grid(row=1,column=0,columnspan=4,sticky="ew",padx=2,pady=2)
        for c in range(4): buttons.columnconfigure(c,weight=1)

        export_box=ttk.LabelFrame(left,text="GeoPackage / Export",padding=6); export_box.pack(fill=tk.X,pady=(0,6))
        ttk.Button(export_box,text="GPKG All",command=self.export_all_gpkg_clicked).grid(row=0,column=0,sticky="ew",padx=2,pady=2)
        ttk.Button(export_box,text="GPKG Model 1",command=lambda:self.export_model_gpkg_clicked(1)).grid(row=0,column=1,sticky="ew",padx=2,pady=2)
        ttk.Button(export_box,text="GPKG Model 2",command=lambda:self.export_model_gpkg_clicked(2)).grid(row=0,column=2,sticky="ew",padx=2,pady=2)
        ttk.Button(export_box,text="GPKG Model 3",command=lambda:self.export_model_gpkg_clicked(3)).grid(row=1,column=1,sticky="ew",padx=2,pady=2)
        ttk.Button(export_box,text="GPKG Overlap/Consensus",command=self.export_consensus_gpkg_clicked).grid(row=1,column=2,sticky="ew",padx=2,pady=2)
        ttk.Button(export_box,text="Export 1024 PNG crops for annotation",command=self.export_training_crops_clicked).grid(row=2,column=0,columnspan=3,sticky="ew",padx=2,pady=2)
        for c in range(3): export_box.columnconfigure(c,weight=1)

        crop_box=ttk.LabelFrame(left,text="Training Crop Export",padding=6); crop_box.pack(fill=tk.X,pady=(0,6))
        ttk.Label(crop_box,text="Output folder").grid(row=0,column=0,sticky="w",pady=2)
        ttk.Entry(crop_box,textvariable=self.crop_output_dir).grid(row=0,column=1,sticky="ew",pady=2)
        ttk.Button(crop_box,text="Browse",command=self.pick_crop_output_dir).grid(row=0,column=2,padx=2,pady=2)
        ttk.Label(crop_box,text="Crop size").grid(row=1,column=0,sticky="w",pady=2)
        ttk.Entry(crop_box,textvariable=self.crop_size,width=8).grid(row=1,column=1,sticky="w",pady=2)
        ttk.Label(crop_box,text="Padding px").grid(row=1,column=1,sticky="e",pady=2)
        ttk.Entry(crop_box,textvariable=self.crop_pad,width=8).grid(row=1,column=2,sticky="w",pady=2)
        ttk.Label(crop_box,text="Source").grid(row=2,column=0,sticky="w",pady=2)
        ttk.Combobox(crop_box,textvariable=self.crop_source,state="readonly",values=["Visible raw detections","Export-filter raw detections","Consensus clusters"]).grid(row=2,column=1,columnspan=2,sticky="ew",pady=2)
        ttk.Radiobutton(crop_box,text="Initial label: positive/mustatil",variable=self.crop_class,value=0).grid(row=3,column=0,columnspan=2,sticky="w")
        ttk.Radiobutton(crop_box,text="Initial label: false_positive",variable=self.crop_class,value=1).grid(row=3,column=2,sticky="w")
        ttk.Label(crop_box,text="Erzeugt project/images + project/labels für den Trainer/Annotator.",foreground="#555").grid(row=4,column=0,columnspan=3,sticky="w")
        crop_box.columnconfigure(1,weight=1)

        self.log=tk.Text(left,width=72,height=10); self.log.pack(fill=tk.BOTH,expand=True,pady=(4,0))
        cf=ttk.LabelFrame(right,text="Preview: Mouse wheel zoom, left-drag pan",padding=4); cf.pack(fill=tk.BOTH,expand=True)
        self.canvas=tk.Canvas(cf,bg="#202020",highlightthickness=1,highlightbackground="#555"); self.canvas.pack(fill=tk.BOTH,expand=True)
        self.canvas.bind("<Configure>", lambda e:self.redraw_preview()); self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e:self.zoom_at(e.x,e.y,1.15)); self.canvas.bind("<Button-5>", lambda e:self.zoom_at(e.x,e.y,1/1.15))
        self.canvas.bind("<ButtonPress-1>", self.pan_start); self.canvas.bind("<B1-Motion>", self.pan_move)

    def log_msg(self,s=""):
        self.log.insert(tk.END,str(s)+"\n"); self.log.see(tk.END); self.update_idletasks()
    def dependency_log(self):
        self.log_msg(f"Python: {sys.executable}")
        for mod in ["ultralytics","torch","onnxruntime","rasterio","PIL","numpy"]: self.log_msg(f"{mod}: {'OK' if have_module(mod) else 'missing'}")
        if have_module("onnxruntime"):
            try:
                import onnxruntime as ort; self.log_msg(f"ONNX Runtime providers: {ort.get_available_providers()}")
            except Exception as exc: self.log_msg(f"ONNX provider check failed: {exc}")
    def pick_model(self,idx):
        p=filedialog.askopenfilename(filetypes=[("YOLO models","*.pt *.onnx"),("All","*.*")]);
        if p: self.model_paths[idx].set(p)
    def pick_image(self):
        p=filedialog.askopenfilename(filetypes=[("Images","*.tif *.tiff *.jpg *.jpeg *.png *.bmp *.webp"),("All","*.*")])
        if p:
            self.image_path.set(p)
            if not self.output_path.get():
                ip=Path(p); self.output_path.set(str(ip.with_name(ip.stem+".ensemble.gpkg")))
    def pick_output(self):
        p=filedialog.asksaveasfilename(defaultextension=".gpkg",filetypes=[("GeoJSON","*.geojson"),("JSON","*.json")])
        if p: self.output_path.set(p)
    def load_preview_clicked(self):
        try:
            p=Path(self.image_path.get().strip().strip('"'))
            if not p.exists(): raise RuntimeError("Choose a valid image first.")
            self.preview_img,ow,oh=load_preview(p); self.preview_orig_size=(ow,oh); self.zoom=1.0; self.pan_x=0; self.pan_y=0
            self.log_msg(f"Preview loaded: {self.preview_img.width}x{self.preview_img.height}; original={ow}x{oh}"); self.redraw_preview()
        except Exception as exc:
            self.log_msg(f"Preview ERROR: {exc}"); messagebox.showerror(APP_TITLE,str(exc))
    def reset_zoom(self): self.zoom=1.0; self.pan_x=0; self.pan_y=0; self.redraw_preview()
    def on_mousewheel(self,event): self.zoom_at(event.x,event.y,1.15 if event.delta>0 else 1/1.15)
    def zoom_at(self,cx,cy,factor):
        old=self.zoom; new=max(0.15,min(20.0,old*factor))
        if abs(new-old)<1e-6: return
        self.pan_x = cx-(cx-self.pan_x)*(new/old); self.pan_y = cy-(cy-self.pan_y)*(new/old); self.zoom=new; self.redraw_preview()
    def pan_start(self,event): self._drag_last=(event.x,event.y)
    def pan_move(self,event):
        if self._drag_last is None: return
        lx,ly=self._drag_last; self.pan_x+=event.x-lx; self.pan_y+=event.y-ly; self._drag_last=(event.x,event.y); self.redraw_preview()
    def _preview_transform(self):
        if not self.preview_img: return 1,0,0
        cw=max(1,self.canvas.winfo_width()); ch=max(1,self.canvas.winfo_height())
        base=min(cw/self.preview_img.width, ch/self.preview_img.height, 1.0); scale=base*self.zoom
        return scale, (cw-self.preview_img.width*scale)/2+self.pan_x, (ch-self.preview_img.height*scale)/2+self.pan_y
    def image_to_canvas(self,x,y):
        if not self.preview_img or not self.preview_orig_size: return 0,0
        ow,oh=self.preview_orig_size; px=x/max(1,ow)*self.preview_img.width; py=y/max(1,oh)*self.preview_img.height
        scale,ox,oy=self._preview_transform(); return px*scale+ox, py*scale+oy
    def on_display_conf_change(self, value=None):
        try:
            v=float(self.display_conf.get())
            if hasattr(self, "display_conf_label"):
                self.display_conf_label.config(text=f"Visible >= {v:.2f}")
        except Exception:
            pass
        self.schedule_redraw(20)

    def on_export_conf_change(self, value=None):
        try:
            v=float(self.export_conf.get())
            if hasattr(self, "export_conf_label"):
                self.export_conf_label.config(text=f"Export >= {v:.2f}")
        except Exception:
            pass

    def set_display_conf(self, value):
        try:
            v=float(value)
            self.display_conf.set(v)
            self.export_conf.set(v)
            self.on_display_conf_change()
            self.on_export_conf_change()
        except Exception:
            pass

    def _passes_conf_filter(self, det):
        try:
            return float(det.conf) >= float(self.display_conf.get())
        except Exception:
            return True

    def _passes_export_filter_det(self, det):
        try:
            return float(det.conf) >= float(self.export_conf.get())
        except Exception:
            return True

    def _passes_export_filter_cluster(self, cl):
        try:
            return float(cl.get("avg_conf", 0.0)) >= float(self.export_conf.get())
        except Exception:
            return True

    def _tile_positions(self, W, H, tile, stride):
        offsets=[(0,0)]
        try:
            if bool(self.shifted_tiles.get()):
                half=max(1, stride//2)
                offsets += [(half,0),(0,half),(half,half)]
        except Exception:
            pass
        seen=set()
        positions=[]
        for ox,oy in offsets:
            xs=list(range(ox, W, stride)) if ox < W else []
            ys=list(range(oy, H, stride)) if oy < H else []
            if 0 not in xs: xs=[0]+xs
            if 0 not in ys: ys=[0]+ys
            for y in ys:
                for x in xs:
                    x=max(0,min(W-1,int(x)))
                    y=max(0,min(H-1,int(y)))
                    key=(x,y)
                    if key not in seen:
                        seen.add(key)
                        positions.append(key)
        return positions

    def export_onnx_clicked(self):
        try:
            p=filedialog.askopenfilename(title="Choose YOLO .pt model to export", filetypes=[("PyTorch YOLO model","*.pt"),("All","*.*")])
            if not p:
                return
            out=filedialog.asksaveasfilename(title="Save ONNX model as", defaultextension=".onnx", initialfile=Path(p).with_suffix(".onnx").name, filetypes=[("ONNX model","*.onnx")])
            if not out:
                return
            imgsz=simple_imgsz=1024
            try:
                simple_imgsz=int(self.tile.get())
            except Exception:
                simple_imgsz=1024
            from ultralytics import YOLO
            self.log_msg("="*80)
            self.log_msg(f"Exporting ONNX from: {p}")
            self.log_msg(f"Export imgsz={simple_imgsz}")
            model=YOLO(str(p))
            result=model.export(format="onnx", imgsz=simple_imgsz, simplify=True, opset=12)
            exported=Path(str(result))
            target=Path(out)
            if exported.exists() and exported.resolve() != target.resolve():
                shutil.copy2(exported, target)
                self.log_msg(f"ONNX copied to: {target}")
            elif target.exists():
                self.log_msg(f"ONNX written: {target}")
            else:
                self.log_msg(f"ONNX export result: {exported}")
            messagebox.showinfo(APP_TITLE, "ONNX export complete.")
        except Exception as exc:
            self.log_msg(f"ONNX export ERROR: {exc}")
            messagebox.showerror(APP_TITLE, str(exc))

    def schedule_redraw(self, delay_ms=25):
        try:
            if self._redraw_after_id is not None:
                self.after_cancel(self._redraw_after_id)
        except Exception:
            pass
        self._redraw_after_id = self.after(delay_ms, self.redraw_preview)

    def redraw_preview(self):
        self.canvas.delete("all")
        if self.preview_img is None:
            self.canvas.create_text(20,20,anchor="nw",fill="white",text="Load a preview image first."); return
        scale,ox,oy=self._preview_transform(); w=max(1,int(self.preview_img.width*scale)); h=max(1,int(self.preview_img.height*scale))
        show=self.preview_img.resize((w,h)); self.preview_photo=ImageTk.PhotoImage(show); self.canvas.create_image(ox,oy,anchor="nw",image=self.preview_photo)
        colors={0:"#ff3333",1:"#33ff66",2:"#3399ff"}
        for d in self.detections:
            if not self._passes_conf_filter(d): continue
            if d.model_index==0 and not self.show_m1.get(): continue
            if d.model_index==1 and not self.show_m2.get(): continue
            if d.model_index==2 and not self.show_m3.get(): continue
            x1,y1=self.image_to_canvas(d.x1,d.y1); x2,y2=self.image_to_canvas(d.x2,d.y2); col=colors.get(d.model_index,"white")
            self.canvas.create_rectangle(x1,y1,x2,y2,outline=col,width=max(1,int(2*self.zoom**0.25)))
            if self.show_labels.get() and self.zoom>=0.45: self.canvas.create_text(x1+3,y1+3,anchor="nw",fill=col,text=f"M{d.model_index+1} {d.conf:.2f}")
        if self.show_consensus.get():
            for cl in self.clusters:
                if cl["model_count"] < 2: continue
                try:
                    if float(cl.get("avg_conf",0.0)) < float(self.display_conf.get()): continue
                except Exception:
                    pass
                x1,y1,x2,y2=cl["bbox"]; cx1,cy1=self.image_to_canvas(x1,y1); cx2,cy2=self.image_to_canvas(x2,y2)
                col="#ff00ff" if cl["model_count"]>=3 else "#ffff00"
                self.canvas.create_rectangle(cx1,cy1,cx2,cy2,outline=col,width=3)
                if self.show_labels.get(): self.canvas.create_text(cx1+4,cy2+4,anchor="nw",fill=col,text=f"{cl['model_count']}M score={cl['score']:.2f}")

    def pick_crop_output_dir(self):
        p=filedialog.askdirectory(title="Choose crop training project folder")
        if p: self.crop_output_dir.set(p)

    def _crop_candidates(self):
        src=self.crop_source.get()
        items=[]
        if src == "Consensus clusters":
            for idx,cl in enumerate(self.clusters, start=1):
                if not self._passes_export_filter_cluster(cl):
                    continue
                x1,y1,x2,y2=cl["bbox"]
                items.append(("cluster", idx, [x1,y1,x2,y2], float(cl.get("score",0.0))))
        else:
            for idx,d in enumerate(self.detections, start=1):
                if src == "Visible raw detections":
                    if not self._passes_conf_filter(d):
                        continue
                else:
                    if not self._passes_export_filter_det(d):
                        continue
                items.append(("det", idx, list(d.bbox()), float(d.conf)))
        return items

    def _write_yolo_label_for_crop(self, label_path, bbox_global, crop_left, crop_top, crop_w, crop_h, cls_id):
        x1,y1,x2,y2=map(float,bbox_global)
        lx1=max(0.0,min(float(crop_w),x1-crop_left)); lx2=max(0.0,min(float(crop_w),x2-crop_left))
        ly1=max(0.0,min(float(crop_h),y1-crop_top)); ly2=max(0.0,min(float(crop_h),y2-crop_top))
        if lx2<=lx1 or ly2<=ly1:
            label_path.write_text("",encoding="utf-8"); return
        cx=((lx1+lx2)/2.0)/crop_w; cy=((ly1+ly2)/2.0)/crop_h
        bw=(lx2-lx1)/crop_w; bh=(ly2-ly1)/crop_h
        label_path.write_text(f"{int(cls_id)} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n",encoding="utf-8")

    def export_training_crops_clicked(self):
        try:
            if not self.detections and not self.clusters:
                raise RuntimeError("Run detection first.")
            img_path=Path(self.image_path.get().strip().strip('"'))
            if not img_path.exists():
                raise RuntimeError("Select a valid source image first.")
            out_txt=self.crop_output_dir.get().strip().strip('"')
            if not out_txt:
                default=img_path.with_name(img_path.stem+"_training_crops")
                self.crop_output_dir.set(str(default)); out_txt=str(default)
            out_root=Path(out_txt)
            images_dir=out_root/"images"; labels_dir=out_root/"labels"
            images_dir.mkdir(parents=True,exist_ok=True); labels_dir.mkdir(parents=True,exist_ok=True)
            (out_root/"project.json").write_text(json.dumps({"classes":["mustatil","false_positive"]},indent=2),encoding="utf-8")
            candidates=self._crop_candidates()
            if not candidates:
                raise RuntimeError("No detections/clusters pass the current filter settings.")
            crop_size=max(128,int(self.crop_size.get()))
            pad=max(0,int(self.crop_pad.get()))
            cls_id=int(self.crop_class.get())
            W,H,mode,reader=open_large_image(img_path,self.log_msg)
            meta=[]; saved=0
            try:
                for kind,idx,bbox,score in candidates:
                    x1,y1,x2,y2=map(float,bbox)
                    bw=max(1.0,x2-x1); bh=max(1.0,y2-y1)
                    side=max(float(crop_size), bw+2*pad, bh+2*pad)
                    side=int(min(max(128,round(side)), max(W,H)))
                    cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
                    left=int(round(cx-side/2)); top=int(round(cy-side/2))
                    left=max(0,min(max(0,W-side),left)); top=max(0,min(max(0,H-side),top))
                    read_side=min(side,W-left,H-top)
                    crop=read_tile(reader,mode,left,top,read_side,W,H).convert("RGB")
                    if crop.size != (crop_size,crop_size):
                        # keep final training crops uniform; label is scaled accordingly below
                        sx=crop_size/float(crop.width); sy=crop_size/float(crop.height)
                        scaled_bbox=[(x1-left)*sx,(y1-top)*sy,(x2-left)*sx,(y2-top)*sy]
                        crop=crop.resize((crop_size,crop_size), Image.Resampling.LANCZOS if hasattr(Image,"Resampling") else Image.LANCZOS)
                        label_bbox_global=[scaled_bbox[0],scaled_bbox[1],scaled_bbox[2],scaled_bbox[3]]
                        label_left=0; label_top=0; label_w=crop_size; label_h=crop_size
                    else:
                        label_bbox_global=bbox; label_left=left; label_top=top; label_w=crop.width; label_h=crop.height
                    name=f"{img_path.stem}_{kind}{idx:06d}_x{left}_y{top}_s{side}_conf{score:.3f}".replace(".","p")
                    img_out=images_dir/(name+".png"); lbl_out=labels_dir/(name+".txt")
                    crop.save(img_out)
                    self._write_yolo_label_for_crop(lbl_out,label_bbox_global,label_left,label_top,label_w,label_h,cls_id)
                    meta.append({"image":img_out.name,"label":lbl_out.name,"source_image":str(img_path),"source_type":kind,"source_index":idx,"score_or_conf":score,"crop_left":left,"crop_top":top,"crop_size_requested":side,"final_size":crop_size,"bbox_global":bbox,"initial_class":cls_id})
                    saved+=1
                    if saved%25==0: self.log_msg(f"Training crops exported: {saved}/{len(candidates)}")
            finally:
                if mode=="rasterio":
                    try: reader.close()
                    except Exception: pass
            (out_root/"crop_manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
            self.log_msg(f"Training crop export complete: {saved} PNGs -> {out_root}")
            self.log_msg("Open this folder in the Trainer tab/GUI and correct labels as positive or false_positive.")
            messagebox.showinfo(APP_TITLE,f"Exported {saved} crops to:\n{out_root}")
        except Exception as exc:
            self.log_msg(f"Crop export ERROR: {exc}"); messagebox.showerror(APP_TITLE,str(exc))

    def detect_thread(self): threading.Thread(target=self.run_detection,daemon=True).start()
    def load_model(self,model_path):
        from ultralytics import YOLO
        return YOLO(str(model_path))
    def run_detection(self):
        try:
            self.detections=[]; self.clusters=[]
            selected=[]
            self.active_model_slots=[]
            for slot,var in enumerate(self.model_paths):
                txt=var.get().strip().strip('"')
                if not txt:
                    continue
                p=Path(txt)
                if p.exists():
                    selected.append((slot,p))
                    self.active_model_slots.append(slot)
                else:
                    self.log_msg(f"Model slot {slot+1} ignored, file not found: {p}")
            if not selected: raise RuntimeError("Select at least one valid model.")
            model_files=[p for _,p in selected]
            img_path=Path(self.image_path.get().strip().strip('"'))
            if not img_path.exists(): raise RuntimeError("Select a valid map image.")
            self._last_image_path=img_path
            tile=int(self.tile.get()); overlap=int(self.overlap.get()); stride=max(1,tile-overlap); conf=float(self.conf.get())
            W,H,mode,reader=open_large_image(img_path,self.log_msg); gkind,gtr,crs=get_raster_transform(reader,mode,self.log_msg)
            self._last_georef = (gkind, gtr, crs)
            self.log_msg("="*80); self.log_msg(f"Models selected: {len(model_files)}")
            for slot,m in selected: self.log_msg(f"  GUI Model {slot+1}: {m}")
            self.log_msg(f"Image: {img_path}"); self.log_msg(f"Size: {W}x{H}; tile={tile}; overlap={overlap}; stride={stride}; detect_conf={conf}; visible_filter>={float(self.display_conf.get()):.2f}")
            models=[]; infer_sizes=[]
            for mp in model_files:
                self.log_msg(f"Loading model: {mp.name}"); models.append(self.load_model(mp)); infer_sizes.append(get_onnx_fixed_input_size(mp,self.log_msg) or tile)
            positions=self._tile_positions(W,H,tile,stride)
            total=len(positions); processed=0; t0=time.time()
            self.log_msg(f"Tile positions to scan: {total} | shifted_passes={bool(self.shifted_tiles.get())}")
            debug_dir = img_path.with_suffix("").with_name(img_path.stem + "_debug_detection_tiles")
            debug_saved = 0
            for x,y in positions:
                tile_img=read_tile(reader,mode,x,y,tile,W,H).convert("RGB"); arr=np.array(tile_img)
                tile_had_det=False
                for mi,(model,mp,infer_imgsz) in enumerate(zip(models,model_files,infer_sizes)):
                    try:
                        results=model.predict(arr,conf=conf,imgsz=infer_imgsz,verbose=False)
                        if not results or results[0].boxes is None: continue
                        boxes=results[0].boxes
                        xyxy=boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy,"cpu") else np.asarray(boxes.xyxy)
                        confs=boxes.conf.cpu().numpy() if hasattr(boxes.conf,"cpu") else np.asarray(boxes.conf)
                        clss=boxes.cls.cpu().numpy() if hasattr(boxes.cls,"cpu") else np.asarray(boxes.cls)
                        for bb,cf,cls in zip(xyxy,confs,clss):
                            lx1,ly1,lx2,ly2=map(float,bb[:4]); lx1=max(0,min(tile_img.width-1,lx1)); lx2=max(0,min(tile_img.width-1,lx2)); ly1=max(0,min(tile_img.height-1,ly1)); ly2=max(0,min(tile_img.height-1,ly2))
                            if lx2<=lx1 or ly2<=ly1: continue
                            gui_slot=selected[mi][0]
                            self.detections.append(Detection(gui_slot,mp.name,int(cls),float(cf),x+lx1,y+ly1,x+lx2,y+ly2))
                            tile_had_det=True
                    except Exception as exc: self.log_msg(f"GUI Model {selected[mi][0]+1} failed at tile ({x},{y}): {exc}")
                if tile_had_det and bool(self.save_debug_tiles.get()) and debug_saved < 200:
                    try:
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        tile_img.save(debug_dir / f"tile_x{x}_y{y}_det{debug_saved}.jpg", quality=92)
                        debug_saved += 1
                    except Exception as exc:
                        if debug_saved == 0:
                            self.log_msg(f"Debug tile save failed: {exc}")
                processed+=1
                if processed%10==0:
                    dt=max(1e-6,time.time()-t0); self.log_msg(f"Processed {processed}/{total} tiles | detections={len(self.detections)} | {processed/dt:.2f} tiles/s")
            if mode=="rasterio":
                try: reader.close()
                except Exception: pass
            self.compute_scores()
            if len(set(self.active_model_slots))==1: self.log_msg("Single-model mode: only the selected model was processed; overlap/consensus export is disabled.")
            self.log_msg(f"Detection complete. Raw detections={len(self.detections)} clusters={len(self.clusters)}"); self.redraw_preview()
            # Automatic GeoJSON export disabled. Use GeoPackage buttons instead.
        except Exception as exc:
            self.log_msg(f"ERROR: {exc}"); messagebox.showerror(APP_TITLE,str(exc))
    def recompute_and_redraw(self): self.compute_scores(); self.redraw_preview(); self.log_msg(f"Scores recomputed. Clusters={len(self.clusters)}")
    def compute_scores(self):
        dets=self.detections; iou_thr=float(self.iou_match.get()); prox=float(self.proximity_px.get()); densrad=float(self.density_radius_px.get())
        groups=[]; used=set()
        for i,d in enumerate(dets):
            if i in used: continue
            group=[i]; used.add(i); changed=True
            while changed:
                changed=False
                for j,e in enumerate(dets):
                    if j in used: continue
                    if any(iou_xyxy(e.bbox(), dets[k].bbox()) >= iou_thr for k in group): group.append(j); used.add(j); changed=True
            groups.append(group)
        clusters=[]
        centers=[]
        for group in groups:
            ms=[dets[i] for i in group]; xs1=[d.x1 for d in ms]; ys1=[d.y1 for d in ms]; xs2=[d.x2 for d in ms]; ys2=[d.y2 for d in ms]
            bbox=(min(xs1),min(ys1),max(xs2),max(ys2)); centers.append(((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2,bbox,group))
        for idx,group in enumerate(groups):
            ms=[dets[i] for i in group]; bbox=centers[idx][2]; cx,cy=centers[idx][0],centers[idx][1]
            mcount=len(set(d.model_index for d in ms)); avg=sum(d.conf for d in ms)/max(1,len(ms))
            base=float(self.w_model3.get()) if mcount>=3 else float(self.w_model2.get()) if mcount==2 else float(self.w_model1.get())
            near=sum(1 for k,(ox,oy,_,_) in enumerate(centers) if k!=idx and math.hypot(cx-ox,cy-oy)<=prox)
            dens=sum(1 for d in dets if math.hypot(cx-d.center()[0], cy-d.center()[1])<=densrad)
            penalty=float(self.w_nonoverlap.get()) if mcount==1 else 0.0
            score=base+float(self.w_conf.get())*avg+float(self.w_proximity.get())*min(near,5)+float(self.w_density.get())*min(max(0,dens-len(ms)),10)+penalty
            for d in ms: d.consensus_count=mcount; d.score=score; d.proximity_bonus=near; d.density_bonus=dens; d.nonoverlap_penalty=penalty
            clusters.append({"bbox":bbox,"model_count":mcount,"member_count":len(ms),"avg_conf":avg,"near_count":near,"density_count":dens,"score":score,"members":group,"models":sorted(set(d.model_index+1 for d in ms))})
        clusters.sort(key=lambda c:c["score"], reverse=True); self.clusters=clusters
    def export_geojson_clicked(self):
        if not self.output_path.get().strip(): self.pick_output()
        if not self.output_path.get().strip(): return
        georef=getattr(self,"_last_georef",("pixel",None,None)); self.export_geojson(self.output_path.get().strip(), *georef)
    def export_geojson(self,out_path,gkind="pixel",gtr=None,crs=None):
        feats=[]
        for rank,cl in enumerate(self.clusters,1):
            if not self._passes_export_filter_cluster(cl): continue
            x1,y1,x2,y2=cl["bbox"]; ring_px=[(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]; ring=[pixel_to_geo(gkind,gtr,x,y) for x,y in ring_px]
            feats.append({"type":"Feature","geometry":{"type":"Polygon","coordinates":[ring]},"properties":{"rank":rank,"type":"ensemble_cluster","ensemble_score":cl["score"],"model_count":cl["model_count"],"member_count":cl["member_count"],"models":",".join(map(str,cl["models"])),"avg_conf":cl["avg_conf"],"near_count":cl["near_count"],"density_count":cl["density_count"],"pixel_bbox":[x1,y1,x2,y2]}})
        for d in self.detections:
            if not self._passes_export_filter_det(d): continue
            x1,y1,x2,y2=d.bbox(); ring_px=[(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]; ring=[pixel_to_geo(gkind,gtr,x,y) for x,y in ring_px]
            feats.append({"type":"Feature","geometry":{"type":"Polygon","coordinates":[ring]},"properties":{"type":"raw_detection","model_index":d.model_index+1,"model_name":d.model_name,"confidence":d.conf,"ensemble_score":d.score,"consensus_count":d.consensus_count,"pixel_bbox":[x1,y1,x2,y2]}})
        fc={"type":"FeatureCollection","properties":{"crs":crs,"created_by":APP_TITLE},"features":feats}
        out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(fc,indent=2),encoding="utf-8")
        self.log_msg(f"GeoJSON exported: {out} | features={len(feats)}")

    def _cluster_features(self, georef_kind=None, georef_transform=None):
        georef_kind = georef_kind or self._last_georef[0]
        georef_transform = georef_transform if georef_transform is not None else self._last_georef[1]
        features = []
        for rank, cl in enumerate(self.clusters, start=1):
            if not self._passes_export_filter_cluster(cl):
                continue
            x1, y1, x2, y2 = cl["bbox"]
            ring_px = [(x1,y1), (x2,y1), (x2,y2), (x1,y2), (x1,y1)]
            ring = [pixel_to_geo(georef_kind, georef_transform, x, y) for x, y in ring_px]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "rank": rank,
                    "ensemble_score": cl["score"],
                    "model_count": cl["model_count"],
                    "member_count": cl["member_count"],
                    "models": ",".join(map(str, cl["models"])),
                    "avg_conf": cl["avg_conf"],
                    "near_count": cl["near_count"],
                    "density_count": cl["density_count"],
                    "pixel_bbox": [x1, y1, x2, y2],
                    "type": "ensemble_cluster",
                }
            })
        return features

    def _raw_detection_features(self, model_index=None, georef_kind=None, georef_transform=None):
        georef_kind = georef_kind or self._last_georef[0]
        georef_transform = georef_transform if georef_transform is not None else self._last_georef[1]
        features = []
        for d in self.detections:
            if not self._passes_export_filter_det(d):
                continue
            if model_index is not None and d.model_index != model_index:
                continue
            x1, y1, x2, y2 = d.bbox()
            ring_px = [(x1,y1), (x2,y1), (x2,y2), (x1,y2), (x1,y1)]
            ring = [pixel_to_geo(georef_kind, georef_transform, x, y) for x, y in ring_px]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "type": "raw_detection",
                    "model_index": d.model_index + 1,
                    "model_name": d.model_name,
                    "confidence": d.conf,
                    "ensemble_score": d.score,
                    "consensus_count": d.consensus_count,
                    "pixel_bbox": [x1, y1, x2, y2],
                }
            })
        return features

    def export_model_gpkg_clicked(self, model_number: int):
        if not self.detections:
            messagebox.showerror(APP_TITLE, "No detections available yet.")
            return
        p = filedialog.asksaveasfilename(
            title=f"Save Model {model_number} detections as GeoPackage",
            defaultextension=".gpkg",
            initialfile=f"model_{model_number}_detections.gpkg",
            filetypes=[("GeoPackage", "*.gpkg")]
        )
        if not p:
            return
        try:
            self._ensure_georef_for_export()
            feats = self._raw_detection_features(model_index=model_number-1)
            if not feats:
                raise RuntimeError(f"No detections for model {model_number}.")
            write_gpkg_features(Path(p), feats, self._last_georef[2], self.log_msg)
        except Exception as exc:
            self.log_msg(f"GPKG export error: {exc}")
            messagebox.showerror(APP_TITLE, str(exc))

    def export_consensus_gpkg_clicked(self):
        if not self.clusters:
            messagebox.showerror(APP_TITLE, "No overlap/consensus clusters available yet.")
            return
        p = filedialog.asksaveasfilename(
            title="Save overlap/consensus as GeoPackage",
            defaultextension=".gpkg",
            initialfile="overlap_consensus_detections.gpkg",
            filetypes=[("GeoPackage", "*.gpkg")]
        )
        if not p:
            return
        try:
            self._ensure_georef_for_export()
            if len(set(self.active_model_slots)) < 2:
                raise RuntimeError("Overlap/Consensus needs at least 2 selected models. With one model, use that model's GPKG export.")
            feats = self._cluster_features()
            feats = [f for f in feats if int(f["properties"].get("model_count", 1)) >= 2]
            if not feats:
                raise RuntimeError("No clusters with 2+ overlapping models found.")
            write_gpkg_features(Path(p), feats, self._last_georef[2], self.log_msg)
        except Exception as exc:
            self.log_msg(f"GPKG export error: {exc}")
            messagebox.showerror(APP_TITLE, str(exc))




    def _ensure_georef_for_export(self):
        try:
            if getattr(self, "_last_georef", ("pixel", None, None))[0] != "pixel":
                return
            img_path = getattr(self, "_last_image_path", None)
            if img_path is None:
                txt = self.image_path.get().strip().strip('"')
                img_path = Path(txt) if txt else None
            if img_path and Path(img_path).exists() and is_tiff(Path(img_path)) and rasterio_available():
                import rasterio
                with rasterio.open(img_path) as src:
                    gkind, gtr, crs = get_raster_transform(src, "rasterio", self.log_msg)
                    self._last_georef = (gkind, gtr, crs)
                    self.log_msg(f"Export georef rechecked: kind={gkind}, CRS={crs}")
        except Exception as exc:
            self.log_msg(f"Export georef recheck failed: {exc}")


    def export_all_gpkg_clicked(self):
        if not self.detections and not self.clusters:
            messagebox.showerror(APP_TITLE, "No detections available yet.")
            return
        p = filedialog.asksaveasfilename(
            title="Save all detections as GeoPackage",
            defaultextension=".gpkg",
            initialfile="ensemble_all_detections.gpkg",
            filetypes=[("GeoPackage", "*.gpkg")]
        )
        if not p:
            return
        try:
            self._ensure_georef_for_export()
            feats = []
            try:
                feats.extend(self._cluster_features())
            except Exception:
                pass
            try:
                feats.extend(self._raw_detection_features())
            except Exception:
                pass
            if not feats:
                raise RuntimeError("No features to export.")
            write_gpkg_features(Path(p), feats, self._last_georef[2], self.log_msg)
        except Exception as exc:
            self.log_msg(f"GPKG export error: {exc}")
            messagebox.showerror(APP_TITLE, str(exc))



if __name__ == "__main__":
    EnsembleDetectGUI().mainloop()
