#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_CLASSES = ["mustatil", "false_positive"]

def yolo_root(project: Path) -> Path:
    return project / "yolo_datasets"

def ensure_yolo_dirs(project: Path):
    root = yolo_root(project)
    dirs = [
        root / "train" / "images",
        root / "train" / "labels",
        root / "val" / "images",
        root / "val" / "labels",
        project / "images",
        project / "labels",
        project / "crops",
        project / "exports",
        project / "weights",
        project / "runs",
        project / "trained_form_models",
        project / "sam2",
        project / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return root

def write_data_yaml(project: Path, classes=None):
    classes = classes or DEFAULT_CLASSES
    root = ensure_yolo_dirs(project)
    names = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: " + str(root).replace("\\", "/") + "\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n" + names + "\n",
        encoding="utf-8"
    )
    print(f"data.yaml written: {data_yaml}")
    return data_yaml

def prepare_dataset_and_yaml(project: Path, classes=None, val_ratio=0.2, images_dir=None, labels_dir=None):
    """Create data.yaml and also populate yolo_datasets from project/images and project/labels.

    This is intentionally copy-based. The original files in images/ and labels/ remain untouched,
    while yolo_datasets/train/... and yolo_datasets/val/... are refreshed/filled for training.
    """
    data_yaml = write_data_yaml(project, classes)
    split_dataset(project, val_ratio=val_ratio, seed=42, copy=True, images_dir=images_dir, labels_dir=labels_dir)
    return data_yaml

def create_project(project: Path, classes=None):
    project.mkdir(parents=True, exist_ok=True)
    root = ensure_yolo_dirs(project)
    data_yaml = write_data_yaml(project, classes or DEFAULT_CLASSES)
    (project / "project.json").write_text(json.dumps({
        "created_by": "Mustatil QGIS Plugin",
        "dataset_root": str(root),
        "data_yaml": str(data_yaml),
        "classes": classes or DEFAULT_CLASSES,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Project created: {project}")
    print(f"YOLO dataset root: {root}")

def split_dataset(project: Path, val_ratio=0.2, seed=42, copy=True, images_dir=None, labels_dir=None):
    project = Path(project)
    root = ensure_yolo_dirs(project)

    # Possible source folders: user-selected training data, then project/images + project/labels OR already train folders
    src_img_dirs = []
    src_lab_dirs = []
    if images_dir:
        src_img_dirs.append(Path(images_dir))
    if labels_dir:
        src_lab_dirs.append(Path(labels_dir))
    src_img_dirs += [project / "images", project / "image", project / "Mustatils" / "images", project / "Mustatils" / "image"]
    src_lab_dirs += [project / "labels", project / "label", project / "Mustatils" / "labels", project / "Mustatils" / "label"]

    images = []
    for d in src_img_dirs:
        if d.exists():
            images.extend([p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT])

    # If no external source images, use all images already in yolo_datasets/train/images
    if not images:
        train_images = root / "train" / "images"
        images = [p for p in train_images.rglob("*") if p.suffix.lower() in IMG_EXT]

    images = sorted(set(images))
    if not images:
        print("No images found to split. Created folders only.")
        return

    random.seed(seed)
    shuffled = images[:]
    random.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * float(val_ratio)))) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:n_val])

    train_img = root / "train" / "images"
    train_lab = root / "train" / "labels"
    val_img = root / "val" / "images"
    val_lab = root / "val" / "labels"

    # Locate label for image by stem in all known label dirs.
    def find_label(img):
        candidates = []
        for lab_dir in src_lab_dirs + [root / "train" / "labels", root / "val" / "labels"]:
            candidates.append(lab_dir / (img.stem + ".txt"))
        for c in candidates:
            if c.exists():
                return c
        return None

    def transfer(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            return
        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))

    for img in images:
        is_val = img in val_set
        dst_img_dir = val_img if is_val else train_img
        dst_lab_dir = val_lab if is_val else train_lab
        dst_img = dst_img_dir / img.name
        transfer(img, dst_img)
        lab = find_label(img)
        if lab:
            transfer(lab, dst_lab_dir / lab.name)
        else:
            # YOLO allows empty label files for negative images.
            (dst_lab_dir / (img.stem + ".txt")).write_text("", encoding="utf-8")

    print(f"Split complete. Images={len(images)} train={len(images)-len(val_set)} val={len(val_set)}")
    print(f"Dataset root: {root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create-project", default="")
    ap.add_argument("--create-yaml", default="")
    ap.add_argument("--split-project", default="")
    ap.add_argument("--prepare-dataset", default="")
    ap.add_argument("--classes", default="mustatil,false_positive")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--images-dir", default="", help="Optional source folder containing training images")
    ap.add_argument("--labels-dir", default="", help="Optional source folder containing YOLO .txt labels")
    args = ap.parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.create_project:
        create_project(Path(args.create_project), classes)
    if args.split_project:
        split_dataset(Path(args.split_project), args.val_ratio, images_dir=args.images_dir or None, labels_dir=args.labels_dir or None)
    if args.create_yaml:
        # Creating/updating YAML now also prepares the dataset from project/images and project/labels.
        prepare_dataset_and_yaml(Path(args.create_yaml), classes, args.val_ratio, images_dir=args.images_dir or None, labels_dir=args.labels_dir or None)
    if args.prepare_dataset:
        prepare_dataset_and_yaml(Path(args.prepare_dataset), classes, args.val_ratio, images_dir=args.images_dir or None, labels_dir=args.labels_dir or None)

if __name__ == "__main__":
    main()