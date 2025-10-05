#!/usr/bin/env python3
"""
yolo_full_metadata.py

Usage:
    python yolo_full_metadata.py --input-dir ./images --output-dir ./out --model yolov8n.pt --conf 0.25

Requirements:
    pip install ultralytics opencv-python-headless tqdm pillow
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
from tqdm import tqdm
from PIL import Image, ExifTags
from ultralytics import YOLO

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

parser = argparse.ArgumentParser(description="Run YOLO on a folder of images, save annotated images and full metadata per image.")
parser.add_argument("--input-dir", "-i", required=True, help="Folder with input images")
parser.add_argument("--output-dir", "-o", required=True, help="Folder to save results")
parser.add_argument("--model", "-m", default="yolov8n.pt", help="YOLO model (default: yolov8n.pt)")
parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold")
parser.add_argument("--ext", "-e", nargs="+", default=[".jpg", ".jpeg", ".png"], help="Extensions to include")
parser.add_argument("--device", "-d", default=None, help="Device: cpu / 0 / cuda:0")
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = ensure_dir(args.output_dir)

images = [p for p in input_dir.iterdir() if p.suffix.lower() in [s.lower() for s in args.ext]]
images.sort()
if not images:
    print("No images found in", input_dir)
    sys.exit(0)

# Load YOLO
device_arg = args.device if args.device else None
model = YOLO(args.model) if device_arg is None else YOLO(args.model, device=device_arg)
model_names = model.names

def draw_boxes_on_image(img, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, (det["x1"], det["y1"], det["x2"], det["y2"]))
        label = f"{det['label']} {det['conf']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def get_file_metadata(path: Path):
    stat = path.stat()
    return {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "file_size_bytes": stat.st_size,
        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }

def get_exif_metadata(path: Path):
    exif_data = {}
    try:
        with Image.open(path) as img:
            exif = img._getexif()
            if exif:
                for tag, val in exif.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    exif_data[str(decoded)] = str(val)
    except Exception:
        pass
    return exif_data

for img_path in tqdm(images, desc="Processing"):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue
    h, w = img_bgr.shape[:2]

    results = model.predict(str(img_path), conf=args.conf, verbose=False)
    res = results[0]

    detections = []
    if res.boxes is not None:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for (b, conf, cls) in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [float(x) for x in b]
            detections.append({
                "label": model_names.get(int(cls), str(cls)),
                "class_id": int(cls),
                "conf": float(conf),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "normalized_bbox": [x1/w, y1/h, x2/w, y2/h]
            })

    # Save annotated image
    annotated = draw_boxes_on_image(img_bgr.copy(), detections)
    out_img = output_dir / img_path.name
    cv2.imwrite(str(out_img), annotated)

    # Metadata
    metadata = {
        "file_info": get_file_metadata(img_path),
        "image_info": {
            "width": w,
            "height": h,
            "channels": img_bgr.shape[2] if len(img_bgr.shape) > 2 else 1,
        },
        "exif_metadata": get_exif_metadata(img_path),
        "detections_count": len(detections),
        "detections": detections,
        "annotated_image": str(out_img),
        "processed_at": datetime.utcnow().isoformat() + "Z"
    }

    # Save per-image JSON
    out_json = out_img.with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

print("✅ Done. Annotated images + per-image JSON with metadata saved in:", output_dir)
