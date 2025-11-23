"""Automatic Face Detection and Privacy Protection using Face Blurring.

This script demonstrates how to detect faces with multiple detectors
and apply privacy preserving transformations such as Gaussian blur or
pixelation. It supports processing individual images, batches of images
in a folder, and live or recorded video streams.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_DNN_PROTO = DEFAULT_MODELS_DIR / "deploy.prototxt"
DEFAULT_DNN_WEIGHTS = DEFAULT_MODELS_DIR / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

DNN_PROTO_URLS: Sequence[str] = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
)

DNN_WEIGHTS_URLS: Sequence[str] = (
    "https://github.com/opencv/opencv/raw/4.x/samples/dnn/face_detector/"
    "res10_300x300_ssd_iter_140000_fp16.caffemodel",
    "https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/"
    "res10_300x300_ssd_iter_140000_fp16.caffemodel",
)

# ------------------------------------------------------------------------------------
# Data containers and utility helpers
# ------------------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """Represents a detected face bounding box."""

    x: int
    y: int
    w: int
    h: int
    confidence: Optional[float] = None

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def ensure_results_dir(directory: Path) -> Path:
    """Create the results directory if it does not exist."""

    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ------------------------------------------------------------------------------------
# Face detection backends
# ------------------------------------------------------------------------------------


def get_haar_cascade() -> cv2.CascadeClassifier:
    """Load the Haar Cascade classifier shipped with OpenCV."""

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(
            "Could not load Haar Cascade classifier. Ensure OpenCV is installed correctly."
        )
    return cascade


def detect_faces_haar(
    image: np.ndarray,
    scale_factor: float = 1.3,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30),
) -> List[DetectionResult]:
    """Detect faces using OpenCV's Haar Cascade classifier."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = get_haar_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return [DetectionResult(x, y, w, h) for (x, y, w, h) in faces]


_dnn_net: Optional[cv2.dnn_Net] = None


def _normalize_urls(urls: Sequence[str]) -> List[str]:
    """Flatten nested URL containers into a simple string list."""

    flat: List[str] = []
    for url in urls:
        if isinstance(url, (list, tuple, set)):
            flat.extend([str(u) for u in url])
        else:
            flat.append(str(url))
    return flat


def _download_file(urls: Sequence[str], destination: Path) -> None:
    """Download a file from one of the given URLs to the destination path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in _normalize_urls(urls):
        try:
            print(f"[INFO] Downloading {destination.name} from {url}")
            urllib.request.urlretrieve(url, destination)
            return
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append((url, exc))
            print(f"[WARN] Failed to download from {url}: {exc}")

    attempted = "\n".join([f"- {url}: {err}" for url, err in errors])
    raise RuntimeError(
        "Could not download required DNN model files.\n"
        "Please download them manually or provide existing paths via --dnn-prototxt/--dnn-weights.\n"
        f"Attempted URLs:\n{attempted}"
    )


def ensure_dnn_model_files(proto_path: Path, weights_path: Path) -> None:
    """Download the OpenCV DNN face detector model files if missing."""

    if not proto_path.exists():
        _download_file(DNN_PROTO_URLS, proto_path)
    if not weights_path.exists():
        _download_file(DNN_WEIGHTS_URLS, weights_path)


def get_dnn_detector(proto_path: Path, weights_path: Path) -> cv2.dnn_Net:
    """Load the OpenCV DNN face detector (SSD with ResNet-10 backbone)."""

    global _dnn_net
    ensure_dnn_model_files(proto_path, weights_path)
    if _dnn_net is None:
        _dnn_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(weights_path))
    return _dnn_net


def detect_faces_dnn(
    image: np.ndarray,
    proto_path: Path,
    weights_path: Path,
    confidence_threshold: float = 0.5,
) -> List[DetectionResult]:
    """Detect faces using the OpenCV DNN face detector."""

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

    results: List[DetectionResult] = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype("int")
        x, y = max(0, start_x), max(0, start_y)
        width, height = max(0, end_x - x), max(0, end_y - y)
        results.append(DetectionResult(x, y, width, height, confidence))
    return results


# ------------------------------------------------------------------------------------
# Privacy-preserving transformations
# ------------------------------------------------------------------------------------


def _sanitize_kernel_size(value: int) -> int:
    """Ensure the kernel size is an odd integer >= 1."""

    value = max(1, value)
    return value + 1 if value % 2 == 0 else value


def apply_gaussian_blur(
    image: np.ndarray,
    detections: Sequence[DetectionResult],
    blur_ratio: float = 0.15,
) -> np.ndarray:
    """Apply Gaussian blur to detected faces."""

    result = image.copy()
    for det in detections:
        x, y, w, h = det.rect
        face_roi = result[y : y + h, x : x + w]
        if face_roi.size == 0:
            continue
        # Kernel is proportional to face size for adaptive blur strength.
        kernel_w = _sanitize_kernel_size(int(w * blur_ratio))
        kernel_h = _sanitize_kernel_size(int(h * blur_ratio))
        blurred_face = cv2.GaussianBlur(face_roi, (kernel_w, kernel_h), 0)
        result[y : y + h, x : x + w] = blurred_face
    return result


def apply_pixelation(
    image: np.ndarray,
    detections: Sequence[DetectionResult],
    downscale_ratio: float = 0.1,
) -> np.ndarray:
    """Apply pixelation effect to detected faces."""

    result = image.copy()
    for det in detections:
        x, y, w, h = det.rect
        face_roi = result[y : y + h, x : x + w]
        if face_roi.size == 0:
            continue
        # Downscale and upscale to create pixelation.
        new_w = max(1, int(w * downscale_ratio))
        new_h = max(1, int(h * downscale_ratio))
        temp = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        result[y : y + h, x : x + w] = pixelated_face
    return result


# ------------------------------------------------------------------------------------
# Processing routines for images and video
# ------------------------------------------------------------------------------------


def select_detector(
    name: str,
    *,
    dnn_prototxt: Path,
    dnn_weights: Path,
    dnn_confidence: float,
) -> Callable[[np.ndarray], List[DetectionResult]]:
    """Return the detection function based on CLI selection."""

    detectors = {
        "haar": detect_faces_haar,
        "dnn": lambda image: detect_faces_dnn(
            image, dnn_prototxt, dnn_weights, dnn_confidence
        ),
    }
    if name not in detectors:
        raise ValueError(f"Unsupported detector '{name}'. Choose from {list(detectors)}")
    return detectors[name]


def annotate_detections(
    image: np.ndarray,
    detections: Sequence[DetectionResult],
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw bounding boxes and optional confidence scores for visualization."""

    annotated = image.copy()
    for det in detections:
        x, y, w, h = det.rect
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        if det.confidence is not None:
            text = f"{det.confidence:.2f}"
            cv2.putText(
                annotated,
                text,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return annotated


def process_single_image(
    image_path: Path,
    output_dir: Path,
    mode: str,
    detector_name: str,
    dnn_prototxt: Path,
    dnn_weights: Path,
    dnn_confidence: float,
    show: bool = False,
) -> Optional[Path]:
    """Process a single image file and save the privacy-preserving result."""

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Could not read image: {image_path}")
        return None

    detector = select_detector(
        detector_name,
        dnn_prototxt=dnn_prototxt,
        dnn_weights=dnn_weights,
        dnn_confidence=dnn_confidence,
    )
    detections = detector(image)

    if mode == "blur":
        processed = apply_gaussian_blur(image, detections)
    elif mode == "pixelate":
        processed = apply_pixelation(image, detections)
    else:
        raise ValueError("mode must be 'blur' or 'pixelate'")

    annotated = annotate_detections(image, detections)

    output_path = output_dir / f"{image_path.stem}_{mode}_{detector_name}{image_path.suffix}"
    cv2.imwrite(str(output_path), processed)

    if show:
        # Display side-by-side comparison using matplotlib.
        rgb_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_original)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(rgb_processed)
        plt.title(f"{mode.capitalize()}ed")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(rgb_annotated)
        plt.title("Detections")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    print(
        f"[INFO] Processed {image_path.name}: {len(detections)} face(s) detected using {detector_name}."
    )
    return output_path


def process_image_folder(
    folder: Path,
    output_dir: Path,
    mode: str,
    detector_name: str,
    dnn_prototxt: Path,
    dnn_weights: Path,
    dnn_confidence: float,
    show: bool = False,
) -> List[Path]:
    """Process all images in a folder."""

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    outputs: List[Path] = []
    for image_file in sorted(folder.iterdir()):
        if image_file.suffix.lower() not in supported_exts:
            continue
        output = process_single_image(
            image_file,
            output_dir,
            mode,
            detector_name,
            dnn_prototxt,
            dnn_weights,
            dnn_confidence,
            show,
        )
        if output:
            outputs.append(output)
    return outputs


def process_video_stream(
    source: str,
    output_dir: Path,
    mode: str,
    detector_name: str,
    dnn_prototxt: Path,
    dnn_weights: Path,
    dnn_confidence: float,
    display: bool = True,
    save_output: bool = False,
) -> Optional[Path]:
    """Process a video file or webcam stream."""

    capture_index: int
    if source.isdigit():
        capture_index = int(source)
        cap = cv2.VideoCapture(capture_index)
        output_path = output_dir / f"webcam_{mode}_{detector_name}.mp4"
    else:
        cap = cv2.VideoCapture(source)
        output_path = output_dir / f"{Path(source).stem}_{mode}_{detector_name}.mp4"

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return None

    detector = select_detector(
        detector_name,
        dnn_prototxt=dnn_prototxt,
        dnn_weights=dnn_weights,
        dnn_confidence=dnn_confidence,
    )

    writer: Optional[cv2.VideoWriter] = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("[INFO] Starting video processing. Press 'q' to quit.")
    frame_count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector(frame)
        if mode == "blur":
            processed = apply_gaussian_blur(frame, detections)
        else:
            processed = apply_pixelation(frame, detections)

        if display:
            annotated = annotate_detections(processed, detections, color=(255, 0, 0))
            cv2.imshow("Privacy Protection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer is not None:
            writer.write(processed)

        frame_count += 1

    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    if frame_count > 0:
        print(
            f"[INFO] Processed {frame_count} frames in {elapsed:.2f}s "
            f"({frame_count / max(elapsed, 1e-6):.2f} FPS)."
        )
    if save_output:
        print(f"[INFO] Saved video to {output_path}")
        return output_path
    return None


# ------------------------------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------------------------------


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic Face Detection and Privacy Protection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, help="Path to a single image to process")
    parser.add_argument(
        "--images",
        type=Path,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Video file path or camera index (e.g., '0' for default webcam)",
    )
    parser.add_argument(
        "--mode",
        default="blur",
        choices=["blur", "pixelate"],
        help="Privacy protection effect to apply",
    )
    parser.add_argument(
        "--detector",
        default="dnn",
        choices=["haar", "dnn"],
        help="Face detection backend",
    )
    parser.add_argument(
        "--dnn-prototxt",
        type=Path,
        default=DEFAULT_DNN_PROTO,
        help="Path to the DNN face detector prototxt file",
    )
    parser.add_argument(
        "--dnn-weights",
        type=Path,
        default=DEFAULT_DNN_WEIGHTS,
        help="Path to the DNN face detector caffemodel weights",
    )
    parser.add_argument(
        "--dnn-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence score for DNN detections (0-1)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display matplotlib previews for images",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV display window during video processing",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save processed video to the results directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where processed outputs are stored",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    results_dir = ensure_results_dir(args.results_dir)

    processed_paths: List[Path] = []

    if args.image:
        if not args.image.exists():
            print(f"[ERROR] Image not found: {args.image}")
        else:
            output = process_single_image(
                args.image,
                results_dir,
                args.mode,
                args.detector,
                args.dnn_prototxt,
                args.dnn_weights,
                args.dnn_confidence,
                show=args.show,
            )
            if output:
                processed_paths.append(output)

    if args.images:
        if not args.images.exists():
            print(f"[ERROR] Image directory not found: {args.images}")
        else:
            outputs = process_image_folder(
                args.images,
                results_dir,
                args.mode,
                args.detector,
                args.dnn_prototxt,
                args.dnn_weights,
                args.dnn_confidence,
                show=args.show,
            )
            processed_paths.extend(outputs)

    if args.video:
        process_video_stream(
            args.video,
            results_dir,
            args.mode,
            args.detector,
            args.dnn_prototxt,
            args.dnn_weights,
            args.dnn_confidence,
            display=not args.no_display,
            save_output=args.save_video,
        )

    if not any([args.image, args.images, args.video]):
        print("[INFO] No inputs provided. Use --image, --images, or --video to get started.")

    if processed_paths:
        print("[INFO] Processed files saved to:")
        for path in processed_paths:
            print(f"  - {path}")

    print(
        "\nSuggestions for comparing performance:\n"
        "  • Measure processing speed (frames-per-second) printed during video runs.\n"
        "  • Count the number of detected faces per frame to evaluate recall.\n"
        "  • Log detection confidences (available with the DNN detector) to compare reliability.\n"
        "  • Inspect qualitative differences between Gaussian blur and pixelation for privacy needs.\n"
        "  • Experiment with detection parameters (scale factor, min neighbors, confidence thresholds) to balance accuracy vs speed.\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
