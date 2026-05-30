# YOLOv9 — *E. coli* Detection

Real-time detection of *E. coli* bacteria from a USB microscope feed,
built on YOLOv9 (via Ultralytics) with a fully scaffolded training
pipeline for fine-tuning on custom microscopy data.

The inference code runs as a standalone CLI app today. The modules
under `src/` are intentionally decoupled, so `inference.py` and
`preprocessing.py` can be imported by a FastAPI backend (with a React
frontend on top) with no changes.

---

## Repo layout

```
.
├── config.yaml                # Runtime configuration
├── requirements.txt
├── src/
│   ├── main.py                # CLI entry point — live detection loop
│   ├── capture.py             # cv2.VideoCapture wrapper
│   ├── inference.py           # YOLOv9 detector (loads .pt or .onnx)
│   ├── visualization.py       # Box / HUD overlay rendering
│   ├── preprocessing.py       # CLAHE for microscopy frames
│   ├── config.py              # YAML loader → typed dataclasses
│   └── logging_setup.py       # Console + rotating-file logger
├── training/
│   ├── train.py               # Fine-tune YOLOv9 on your data
│   ├── dataset.yaml.example   # Template — copy to dataset.yaml
│   └── augmentations.py       # Reference Albumentations pipeline
├── models/                    # Drop trained weights here
├── outputs/                   # `--save` writes annotated .mp4 here
└── logs/
```

## Hardware

**Inference (this app):**
- Recommended: Linux / macOS / Windows host with a mid-range NVIDIA
  GPU (RTX 3060 or better) for ≥30 FPS at 640×640.
- Supported: CPU-only fallback. Expect 2–8 FPS depending on CPU.
- USB microscope: any UVC-compatible device that exposes itself as a
  webcam to OpenCV.

**Training:**
- An NVIDIA GPU with ≥8 GB VRAM is strongly recommended. 100 epochs on
  a few thousand microscopy frames typically takes 1–4 hours on an
  RTX 3060 / 4060.

## Installation

```powershell
git clone <this repo>
cd <repo>

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> If you have a CUDA GPU, install the matching PyTorch build **first**
> from <https://pytorch.org/get-started/locally/>, then run
> `pip install -r requirements.txt`. Otherwise pip can pick up a
> CPU-only wheel on some platforms.

## Connecting the USB microscope

Most USB microscopes register as a standard UVC camera and appear
alongside any built-in webcam. You need to find the right device index.

**Windows**
- Plug the microscope in. Open Device Manager → Cameras to confirm it
  is recognised.
- Device indexes follow enumeration order: a built-in webcam is usually
  `0`, the microscope often `1`.

**Linux**
```bash
ls /dev/video*
v4l2-ctl --list-devices   # optional, more detail
```

**macOS**
- System Settings → Privacy & Security → Camera → grant Terminal /
  Python access.

Quick probe to confirm:
```bash
python -c "import cv2; cap=cv2.VideoCapture(1); \
print('opened?', cap.isOpened()); ok,f=cap.read(); \
print('got frame?', ok, getattr(f, 'shape', None)); cap.release()"
```

Set the working index in `config.yaml` under `capture.source`, or pass
`--source 1` at runtime.

## Running inference

Assumes you have trained weights at `models/ecoli_yolov9c.pt` (or
`.onnx`), or you downloaded a generic `yolov9c.pt` from Ultralytics
for smoke-testing.

```powershell
# Live from default source in config.yaml
python -m src.main

# Override the source (USB device index)
python -m src.main --source 1

# Run against a saved video / image file
python -m src.main --source data/sample.mp4

# Save annotated output to outputs/ecoli_<timestamp>.mp4
python -m src.main --save

# Headless run (no preview window) — for SSH or service mode
python -m src.main --no-display --save
```

Press **`q`** in the preview window to quit. Ctrl-C also triggers a
clean shutdown — the capture device and any video writer are released
in a `finally` block, so partial mp4 files are still readable.

The HUD shows the frame number, rolling FPS (averaged over the last
30 frames), and the per-frame detection count. Each bounding box is
labelled with class name and confidence.

## Web UI (FastAPI + React)

A FastAPI backend wraps the inference modules and serves a small React /
Vite frontend with two modes:

- **Upload image** — POST a file to `/api/predict`, view the original
  image with detections overlaid via SVG and a list of class /
  confidence values on the side.
- **Live stream** — `<img src="/api/stream?source=0">` opens an MJPEG
  stream of the camera or a file path, with detections drawn server-side.

### Dev mode (two processes, hot reload)

```powershell
# Terminal 1 — backend
.\.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd frontend
npm install        # first time only
npm run dev        # serves http://localhost:5173 with /api/* proxied to :8000
```

Open <http://localhost:5173>. Vite hot-reloads the UI; uvicorn reloads
the backend on Python edits.

### Production mode (single process)

```powershell
cd frontend
npm install        # first time only
npm run build      # writes the bundle into backend/app/static/
cd ..
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. FastAPI serves the built React app
alongside `/api/*` on the same port — this is the deployment model used
on a Raspberry Pi (one systemd unit, one port, no nginx).

### API reference

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/health` | Returns `{status, model_loaded, device, classes}`. |
| `POST` | `/api/predict` | `multipart/form-data` with `file=<image>` → JSON detections + image size + inference_ms. |
| `GET` | `/api/stream` | `multipart/x-mixed-replace` MJPEG stream. Query: `source` (int or path), `annotate` (default `true`). |

The `source` query param accepts the same values as `capture.source` in
`config.yaml` — an integer device index, a file path, or an
`rtsp://` URL.

## Training on your own E. coli dataset

### 1. Prepare data in YOLO format

One `.txt` per image, one line per box:

```
<class_id> <cx> <cy> <w> <h>          # all coords normalized to [0, 1]
```

Layout:

```
data/ecoli/
├── images/
│   ├── train/   *.jpg | *.png
│   ├── val/
│   └── test/    (optional)
└── labels/
    ├── train/   *.txt
    ├── val/
    └── test/
```

If your annotations are in **COCO JSON**, convert them with the
Ultralytics built-in:

```python
from ultralytics.data.converter import convert_coco
convert_coco("path/to/coco/annotations", use_segments=False, cls91to80=False)
```

This writes YOLO `.txt` labels next to the JSON; move them into the
layout above.

### 2. Configure the dataset

```bash
cp training/dataset.yaml.example training/dataset.yaml
# edit paths inside
```

### 3. Train

```bash
python -m training.train \
    --data training/dataset.yaml \
    --weights yolov9c.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --name ecoli_yolov9c
```

Notes:
- `--weights yolov9c.pt` — Ultralytics auto-downloads recognised names.
  Use `yolov9t.pt` / `yolov9s.pt` for faster, smaller variants
  (recommended for low-power deployment).
- The training script enables microscopy-friendly augmentation by
  default. See the comment block in `training/train.py` for rationale
  (e.g. hue is held at 0 because stain colour is diagnostic; mixup is
  disabled because it blends ground-truth on tiny targets).
- Training output lands in `runs/train/ecoli_yolov9c/`. Best checkpoint
  is `weights/best.pt`.

### 4. Use your trained weights

```bash
cp runs/train/ecoli_yolov9c/weights/best.pt models/ecoli_yolov9c.pt
# update config.yaml model.weights if you used a different filename
python -m src.main
```

### 5. (Optional) Export to ONNX

ONNX removes the PyTorch dependency at deploy time and is usually
faster on CPU.

```bash
yolo export model=models/ecoli_yolov9c.pt format=onnx imgsz=640 simplify=true
```

Then point `model.weights` in `config.yaml` at the new `.onnx` file —
no code changes needed; `YOLOv9Detector` loads both transparently.

## Configuration reference (`config.yaml`)

| Section / key | Type | Notes |
|---|---|---|
| `model.weights` | str | Path to `.pt` or `.onnx` weights. |
| `model.device` | str | `auto` \| `cuda` \| `cuda:0` \| `cpu`. |
| `model.imgsz` | int | Inference size. Multiple of 32. Lower = faster, lower mAP. |
| `model.conf_threshold` | float | Minimum confidence to emit a detection. |
| `model.iou_threshold` | float | NMS IoU threshold. |
| `capture.source` | int / str | Device index, file path, or stream URL. |
| `capture.{width,height,fps}` | int | Requested capture parameters. Camera may ignore unsupported values. |
| `preprocessing.apply_clahe` | bool | Apply CLAHE before inference. |
| `preprocessing.clahe_clip_limit` | float | Higher = stronger enhancement, more noise. |
| `preprocessing.clahe_tile_grid_size` | int | CLAHE tile count per axis. |
| `classes` | list[str] | Display names per class id. |
| `logging.level` | str | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR`. |
| `logging.file` | str | File path, or empty to disable. |
| `output.output_dir` | str | Where `--save` writes annotated mp4s. |

## Library choice — Ultralytics, not WongKinYiu/yolov9

Both repos implement YOLOv9. We use **Ultralytics** because:
- One `pip install ultralytics` covers training, inference, and export
  to ONNX / TensorRT / OpenVINO via a single API.
- `YOLO("foo.pt")` and `YOLO("foo.onnx")` are interchangeable — no
  separate inference backend per export format.
- Actively maintained and ships with Albumentations integration, AMP
  training, EMA, cosine LR, etc., out of the box.

The original `WongKinYiu/yolov9` repo is the right choice only when you
need to modify the architecture itself (custom heads, custom losses).
For fine-tuning + deployment, the Ultralytics path is much shorter.

## Performance tips

- **Smaller variant first.** `yolov9t` or `yolov9s` at 416 input often
  hits 30+ FPS on CPU.
- **Drop input size.** 640 → 320 roughly quadruples FPS.
- **ONNX on CPU.** Export to ONNX and install `onnxruntime` —
  usually faster than PyTorch CPU.
- **Skip CLAHE if it isn't helping.** It costs ~3–8 ms per frame.
  A/B test by toggling `preprocessing.apply_clahe`.
- **Tune confidence threshold up.** Fewer borderline detections =
  less NMS overhead and a cleaner overlay.

## Raspberry Pi deployment notes

The defaults here are tuned for a desktop GPU. For a Pi 4 / 5:

- Train on a workstation; deploy only the exported `.onnx` to the Pi.
- Use `yolov9t` and `imgsz: 416` (or 320). INT8-quantize the ONNX
  with `onnxruntime.quantization` — roughly 2–3× speedup on Pi 5 CPU.
- For lightest install, skip `torch`/`torchvision` and `ultralytics`
  on the Pi entirely — use a pure `onnxruntime` inference path. The
  current `inference.py` uses Ultralytics for ONNX loading; replacing
  it with a pure `onnxruntime` session is a self-contained change in
  one file.
- If you wrap this in FastAPI on the Pi, pin `uvicorn --workers 1` —
  the model fits in RAM once, not twice.
- Use a USB 3.0 port for the microscope; USB 2.0 will throttle 1080p.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Could not open video source 0` | Wrong index, no permission, device in use | Probe other indexes (1, 2, ...); close other apps using the camera; on Linux add your user to the `video` group. |
| `Model weights not found` | `models/` is empty | Fine-tune (see above) and copy `best.pt` in, or point `model.weights` at an existing `.pt`/`.onnx`. |
| Very low FPS on a GPU host | PyTorch installed as CPU-only | Reinstall the matching CUDA build of PyTorch from pytorch.org, then re-run `pip install -r requirements.txt`. |
| All detections at very low confidence | Model under-trained or domain shift | Re-train on more representative data; cautiously lower `conf_threshold` to inspect. |
| Frame reads occasionally fail | USB bandwidth saturation, cable, sleep | Use a USB 3.0 port; shorten the cable; disable USB selective suspend (Windows) / autosuspend (Linux). |
| Preview window freezes but no error | Event-loop starvation | `cv2.imshow` requires the main thread + `waitKey(1)`; don't use `--no-display` if you want a window. |
| Wrong class names on detections | `classes` in `config.yaml` doesn't match the trained model | Edit `config.yaml` `classes` to match the training `names` list, in the same order. |

## Logging

All modules log via Python's standard `logging`. Defaults:
- Console: INFO and above.
- File (`logs/app.log`): rotating, 5 MB × 3 files.

Bump `logging.level` to `DEBUG` in `config.yaml` for per-frame
diagnostics. Tail with:

```bash
tail -f logs/app.log
```

## License / weights

This repo does not ship trained E. coli weights. The Ultralytics
`yolov9c.pt` base, when auto-downloaded for fine-tuning, is subject to
its own licence terms — review them before redistributing derived
checkpoints.
