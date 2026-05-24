#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a simple FormLearner JSON from YOLO-labeled crop folders.

Expected structure:
project/
  train/images, train/labels
  val/images, val/labels   optional

Class 0 is treated as positive by default; every other class is negative.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def crop_features(pil_img):
    im = pil_img.convert("RGB").resize((256, 256))
    arr = np.asarray(im).astype("float32") / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green = ((g > r + 0.03) & (g > b + 0.03) & (g > 0.12)).mean()
    gray = (0.299*r + 0.587*g + 0.114*b)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge = float(gx + gy)
    mean = float(gray.mean())
    std = float(gray.std())
    w, h = pil_img.size
    aspect = max(w, h) / max(1, min(w, h))
    area = (w*h)/(256.0*256.0)
    return [float(math.log1p(aspect)), float(area), mean, std, edge, float(green)]

def yolo_boxes(label_path, W, H):
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cls = int(float(p[0]))
        cx, cy, bw, bh = map(float, p[1:])
        x1 = (cx - bw/2) * W
        y1 = (cy - bh/2) * H
        x2 = (cx + bw/2) * W
        y2 = (cy + bh/2) * H
        out.append((cls, x1, y1, x2, y2))
    return out

def sam2_boxes(sidecar, W, H):
    """Read SAM2 sidecar polygons/bboxes and return class-0 pixel boxes.

    The QGIS SAM2 runner writes image.with_suffix('.sam2.json') next to each image.
    FormLearner training must also be able to use those sidecars directly, even if
    the user has not manually pressed the SAM2-to-label conversion button.
    """
    if not sidecar.exists():
        return []
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        print(f"Skipped SAM2 sidecar {sidecar}: {exc}")
        return []
    if isinstance(data, dict):
        items = data.get("polygons") or data.get("segments") or data.get("masks") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    out = []
    for item in items:
        pts = []
        bbox = None
        if isinstance(item, dict):
            pts = item.get("polygon") or item.get("points") or []
            bbox = item.get("bbox")
        elif isinstance(item, list):
            pts = item
        xs, ys = [], []
        if isinstance(pts, list):
            for pt in pts:
                try:
                    if isinstance(pt, dict):
                        xs.append(float(pt.get("x"))); ys.append(float(pt.get("y")))
                    elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        xs.append(float(pt[0])); ys.append(float(pt[1]))
                except Exception:
                    pass
        if xs and ys:
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = map(float, bbox[:4])
            except Exception:
                continue
        else:
            continue
        x1, x2 = sorted([max(0.0, min(float(W), x1)), max(0.0, min(float(W), x2))])
        y1, y2 = sorted([max(0.0, min(float(H), y1)), max(0.0, min(float(H), y2))])
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            out.append((0, x1, y1, x2, y2))
    return out


def candidate_image_label_dirs(project):
    """Return all project layouts used by the QGIS plugin and standalone GUI."""
    pairs = [
        (project / "yolo_datasets" / "train" / "images", project / "yolo_datasets" / "train" / "labels"),
        (project / "yolo_datasets" / "val" / "images", project / "yolo_datasets" / "val" / "labels"),
        (project / "train" / "images", project / "train" / "labels"),
        (project / "val" / "images", project / "val" / "labels"),
        (project / "images", project / "labels"),
        (project / "crops" / "images", project / "crops" / "labels"),
    ]
    seen = set()
    out = []
    for img_dir, lab_dir in pairs:
        key = (str(img_dir.resolve()) if img_dir.exists() else str(img_dir), str(lab_dir))
        if key in seen:
            continue
        seen.add(key)
        if img_dir.exists():
            out.append((img_dir, lab_dir))
    return out


def collect(project):
    X, y = [], []
    used_images = 0
    used_label_files = 0
    used_sam2_sidecars = 0

    for img_dir, lab_dir in candidate_image_label_dirs(project):
        print(f"Scanning images={img_dir} labels={lab_dir}")
        for imgp in img_dir.rglob("*"):
            if imgp.suffix.lower() not in IMG_EXT:
                continue
            try:
                im = Image.open(imgp).convert("RGB")
                W, H = im.size

                lab = lab_dir / (imgp.stem + ".txt")
                labels = yolo_boxes(lab, W, H)
                if labels:
                    used_label_files += 1

                # Direct SAM2 support: if sidecar exists, add its segments as class-0
                # unless a converted YOLO label file already contains positives.
                side = imgp.with_suffix(".sam2.json")
                sam_labels = sam2_boxes(side, W, H)
                if sam_labels:
                    used_sam2_sidecars += 1
                    has_positive_label = any(int(cls) == 0 for cls, *_ in labels)
                    if not has_positive_label:
                        labels = sam_labels + labels

                if not labels:
                    continue

                used_images += 1
                for cls, x1, y1, x2, y2 in labels:
                    x1 = max(0, int(x1)); y1 = max(0, int(y1))
                    x2 = min(W, int(x2)); y2 = min(H, int(y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = im.crop((x1, y1, x2, y2))
                    X.append(crop_features(crop))
                    y.append(1.0 if int(cls) == 0 else 0.0)
            except Exception as exc:
                print(f"Skipped {imgp}: {exc}")

    print(f"Scanned training images with labels/sidecars: {used_images}; label files={used_label_files}; SAM2 sidecars={used_sam2_sidecars}")
    return np.asarray(X, dtype="float64"), np.asarray(y, dtype="float64")

def train(X, y, epochs=1200, lr=0.08, l2=0.001):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xn = (X - mean) / std
    w = np.zeros(Xn.shape[1])
    b = 0.0
    for _ in range(int(epochs)):
        z = Xn @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        w -= lr * ((Xn.T @ (p-y))/len(y) + l2*w)
        b -= lr * float((p-y).mean())
    return {
        "w": w.tolist(),
        "b": float(b),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "features": ["log_aspect","rel_area","gray_mean","gray_std","edge_density","green_ratio"],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=1200)
    args = ap.parse_args()

    project = Path(args.project)
    X, y = collect(project)
    print(f"Samples: {len(y)} positives={int(y.sum())} negatives={int(len(y)-y.sum())}")
    if len(y) < 2 or len(set(y.tolist())) < 2:
        raise RuntimeError("Need at least one positive class 0 and one negative non-class-0 sample.")
    model = train(X, y, args.epochs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"FormLearner written: {out}")

if __name__ == "__main__":
    main()