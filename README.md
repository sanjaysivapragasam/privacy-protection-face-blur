# Privacy Protection Face Blur

This project demonstrates multiple face detectors with privacy-preserving effects such as Gaussian blur and pixelation for images, folders, and video/webcam streams.

## Requirements
- Python 3.8+ (tested with Python 3.12; no TensorFlow dependency)
- [OpenCV](https://pypi.org/project/opencv-python/) with DNN support
- `numpy`, `matplotlib`

Install the dependencies:

```bash
pip install opencv-python numpy matplotlib
```

## Detector options
- **dnn (default)**: OpenCV DNN face detector (ResNet-10 SSD). Works without TensorFlow and is compatible with Python 3.12.
- **haar**: Classic Haar cascade shipped with OpenCV.

### DNN model files
The DNN detector uses two files:
- `models/deploy.prototxt`
- `models/res10_300x300_ssd_iter_140000_fp16.caffemodel`

If these files are missing, the script can automatically download them from the official OpenCV repositories into the `models/` folder (unless you pass `--no-download`). To install them manually, download the files to the same paths:
If these files are missing, the script automatically downloads them from the official OpenCV repositories into the `models/` folder. To install them manually, download the files to the same paths:

```bash
mkdir -p models
curl -L -o models/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -L -o models/res10_300x300_ssd_iter_140000_fp16.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel
# Alternate mirrors if the main link is unavailable:
# curl -L -o models/res10_300x300_ssd_iter_140000_fp16.caffemodel \
#   https://raw.githubusercontent.com/opencv/opencv_3rdparty/55e8c46fcfa66c96b6f4050af3e60c8e2f8b7631/dnn_samples/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel
# curl -L -o models/res10_300x300_ssd_iter_140000_fp16.caffemodel \
#   https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel
  https://github.com/opencv/opencv/raw/4.x/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel
# Alternate mirror if the main link is unavailable:
# curl -L -o models/res10_300x300_ssd_iter_140000_fp16.caffemodel \
#   https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_samples/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel
```

You can also point to custom models via `--dnn-prototxt` and `--dnn-weights`.

If the automatic download fails (for example, due to a moved URL) or you are running offline with `--no-download`, update the paths with your own files or re-run the above commands with a working link. The script now tries multiple URLs (and gracefully handles nested URL lists) before prompting for manual download.
If the automatic download fails (for example, due to a moved URL), update the paths with your own files or re-run the above
commands with a working link. The script now tries multiple URLs (and gracefully handles nested URL lists) before prompting
for manual download.
commands with a working link. The script now tries multiple official OpenCV URLs before prompting for manual download.

## Usage examples
Process a single image with the default DNN detector and blur effect:

```bash
python main.py --image path/to/photo.jpg --mode blur
```

Pixelate faces in all images inside a folder:

```bash
python main.py --images path/to/folder --mode pixelate --detector dnn
```

Run against a webcam or video file and save the output:

```bash
python main.py --video 0 --mode blur --detector dnn --save-video
python main.py --video input.mp4 --detector haar --no-display --save-video
```

### Advanced options
- Adjust DNN confidence threshold: `--dnn-confidence 0.4`
- Use Haar cascade instead: `--detector haar`
- Show Matplotlib previews for images: `--show`
- Disable the live OpenCV window during video processing: `--no-display`
- Force offline/local models only (no auto-download): `--no-download`
- Choose output directory: `--results-dir path/to/results`

## Adding new detectors or packages
1. Install the required package for the detector (e.g., `pip install <package>`).
2. Drop any necessary model files into `models/` (or pass custom paths via new CLI flags).
3. Implement a detection function returning `List[DetectionResult]` in `main.py`.
4. Add the function to `select_detector` and expose any new parameters through argparse.

## Notes
- The script prints FPS metrics during video processing to help compare performance between detectors.
- Detected faces include confidence scores where available (DNN detector only).
