"""
Automatic Face Detection and Privacy Protection using Face Blurring
Clean version – no downloads, uses local DNN model files.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# Paths to your LOCAL MODEL FILES
# -------------------------------------------------------------
DEFAULT_MODELS_DIR = Path("models")

DEFAULT_DNN_PROTO = DEFAULT_MODELS_DIR / "deploy.prototxt"
DEFAULT_DNN_WEIGHTS = DEFAULT_MODELS_DIR / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

# -------------------------------------------------------------
# Data classes
# -------------------------------------------------------------
@dataclass
class DetectionResult:
    x: int
    y: int
    w: int
    h: int
    confidence: Optional[float] = None

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def ensure_results_dir(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory

# -------------------------------------------------------------
# Haar Detector
# -------------------------------------------------------------
def get_haar_cascade():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError("Could not load Haar Cascade.")
    return cascade


def detect_faces_haar(image: np.ndarray) -> List[DetectionResult]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = get_haar_cascade()
    faces = cascade.detectMultiScale(gray, 1.3, 5)

    return [DetectionResult(x, y, w, h) for (x, y, w, h) in faces]


# -------------------------------------------------------------
# DNN Detector
# -------------------------------------------------------------
_dnn_net: Optional[cv2.dnn_Net] = None

def get_dnn_detector(proto_path: Path, weights_path: Path) -> cv2.dnn_Net:
    global _dnn_net
    if _dnn_net is None:
        _dnn_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(weights_path))
    return _dnn_net


def detect_faces_dnn(
    image: np.ndarray,
    proto_path: Path,
    weights_path: Path,
    confidence_threshold: float = 0.5
) -> List[DetectionResult]:

    net = get_dnn_detector(proto_path, weights_path)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")

        x = max(0, start_x)
        y = max(0, start_y)
        width = max(0, end_x - x)
        height = max(0, end_y - y)

        results.append(DetectionResult(x, y, width, height, confidence))

    return results


# -------------------------------------------------------------
# Privacy Effects
# -------------------------------------------------------------
def apply_gaussian_blur(image, detections):
    result = image.copy()
    for det in detections:
        x, y, w, h = det.rect
        roi = result[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        blurred = cv2.GaussianBlur(roi, (31, 31), 0)
        result[y:y+h, x:x+w] = blurred
    return result


def apply_pixelation(image, detections):
    result = image.copy()
    for det in detections:
        x, y, w, h = det.rect
        roi = result[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        small = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        result[y:y+h, x:x+w] = pixelated
    return result


# -------------------------------------------------------------
# Unified detector selector
# -------------------------------------------------------------
def select_detector(name: str, proto, weights) -> Callable:
    if name == "haar":
        return detect_faces_haar
    if name == "dnn":
        return lambda img: detect_faces_dnn(img, proto, weights)
    raise ValueError("Unknown detector: " + name)


# -------------------------------------------------------------
# Image processing
# -------------------------------------------------------------
def process_single_image(image_path, output_dir, mode, detector_name, proto, weights):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Unable to load {image_path}")
        return None

    detector = select_detector(detector_name, proto, weights)
    detections = detector(img)

    if mode == "blur":
        processed = apply_gaussian_blur(img, detections)
    else:
        processed = apply_pixelation(img, detections)

    output_path = output_dir / f"{image_path.stem}_{mode}_{detector_name}{image_path.suffix}"
    cv2.imwrite(str(output_path), processed)

    print(f"[INFO] Saved → {output_path}")
    return output_path


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mode", default="blur", choices=["blur", "pixelate"])
    parser.add_argument("--detector", default="dnn", choices=["haar", "dnn"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_arguments()
    results_dir = ensure_results_dir(args.results_dir)

    if args.image:
        process_single_image(
            args.image,
            results_dir,
            args.mode,
            args.detector,
            DEFAULT_DNN_PROTO,
            DEFAULT_DNN_WEIGHTS,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
