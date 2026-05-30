"""
src/main.py
===========
CLI entry point for live E. coli detection.

Run as a module so package-relative imports work:

    python -m src.main [--source 0 | path] [--save] [--config config.yaml] [--no-display]

The main loop:

1.  read a frame from the USB microscope / file,
2.  optionally apply CLAHE,
3.  run YOLOv9 inference,
4.  draw bounding boxes + HUD (frame#, FPS, detection count),
5.  optionally write the annotated frame to ``outputs/ecoli_<ts>.mp4``,
6.  display in a window unless ``--no-display``.

Press ``q`` in the preview window — or send SIGINT — to quit cleanly.
"""
import argparse
import logging
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Union

import cv2

from src.capture import VideoSource, CaptureError
from src.config import load_config
from src.inference import YOLOv9Detector, InferenceError
from src.logging_setup import setup_logging
from src.preprocessing import apply_clahe
from src.visualization import draw_detections, draw_hud


# ----------------------------------------------------------------------
# CLI argument parsing
# ----------------------------------------------------------------------

def _parse_source(value: str) -> Union[int, str]:
    """Convert the ``--source`` string to int (device index) or keep as path/URL."""
    try:
        return int(value)
    except ValueError:
        return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time E. coli detection from a USB microscope feed.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Override capture source: integer device index (0, 1, ...) for a "
            "USB camera, or a path to a video/image file, or an rtsp:// URL. "
            "Falls back to capture.source in config.yaml."
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated stream to outputs/ecoli_<timestamp>.mp4.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the OpenCV preview window — useful headless / over SSH.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Signal handling — graceful shutdown
# ----------------------------------------------------------------------

_shutdown = False


def _install_signal_handlers() -> None:
    """Translate SIGINT/SIGTERM into a clean shutdown flag.

    We avoid raising KeyboardInterrupt out of the C extension on Windows
    by setting a flag the main loop polls. The cv2.VideoWriter and the
    capture device are then released in the finally block.
    """
    def handler(signum, _frame):
        global _shutdown
        logging.getLogger(__name__).info(
            "Received signal %s — initiating graceful shutdown.", signum,
        )
        _shutdown = True

    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

# Abort the live loop if reads fail this many times in a row — most
# likely the USB device has been physically disconnected or the file
# stream ended.
_MAX_CONSECUTIVE_READ_FAILURES = 30


def main() -> None:
    args = _parse_args()

    # Load config + logging first so subsequent errors are captured.
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        # No logging configured yet — write directly to stderr.
        print(f"[FATAL] Could not load config: {exc}", file=sys.stderr)
        sys.exit(1)

    setup_logging(cfg.logging.level, cfg.logging.file or None)
    log = logging.getLogger(__name__)
    _install_signal_handlers()

    source = _parse_source(args.source) if args.source is not None else cfg.capture.source
    log.info("Source: %r", source)

    log.info("Loading YOLOv9 detector: weights=%s device=%s",
             cfg.model.weights, cfg.model.device)
    try:
        detector = YOLOv9Detector(
            weights_path=cfg.model.weights,
            device=cfg.model.device,
            imgsz=cfg.model.imgsz,
            conf_threshold=cfg.model.conf_threshold,
            iou_threshold=cfg.model.iou_threshold,
            class_names=cfg.classes,
        )
    except InferenceError as exc:
        log.error("Failed to load detector: %s", exc)
        sys.exit(1)

    fps_window: deque = deque(maxlen=30)
    writer: cv2.VideoWriter = None
    output_path: Path = None
    consecutive_read_failures = 0
    no_detection_streak = 0
    frame_number = 0

    try:
        with VideoSource(
            source,
            cfg.capture.width,
            cfg.capture.height,
            cfg.capture.fps,
        ) as cap:
            while not _shutdown:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_read_failures += 1
                    log.warning(
                        "Frame read failed (consecutive=%d).",
                        consecutive_read_failures,
                    )
                    if consecutive_read_failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                        log.error(
                            "Capture appears dead after %d failures; aborting.",
                            consecutive_read_failures,
                        )
                        break
                    # Small backoff so we don't spin on a disconnected device.
                    time.sleep(0.05)
                    continue
                consecutive_read_failures = 0
                frame_number += 1

                # CLAHE happens BEFORE inference so the model sees the
                # enhanced image. We then annotate the SAME enhanced
                # frame so what the user sees matches what the model saw.
                if cfg.preprocessing.apply_clahe:
                    proc = apply_clahe(
                        frame,
                        clip_limit=cfg.preprocessing.clahe_clip_limit,
                        tile_grid_size=cfg.preprocessing.clahe_tile_grid_size,
                    )
                else:
                    proc = frame

                try:
                    detections = detector.predict(proc)
                except InferenceError as exc:
                    # Per-frame inference failures are recoverable — log
                    # and skip rather than tearing down the loop.
                    log.error("Inference failed on frame %d: %s", frame_number, exc)
                    continue

                if not detections:
                    no_detection_streak += 1
                    # Log infrequently so an empty field of view doesn't
                    # flood the log file.
                    if no_detection_streak % 150 == 0:
                        log.info(
                            "No detections in last %d frames.",
                            no_detection_streak,
                        )
                else:
                    if no_detection_streak >= 150:
                        log.info(
                            "Detections resumed after %d empty frames.",
                            no_detection_streak,
                        )
                    no_detection_streak = 0

                draw_detections(proc, detections)

                dt = time.perf_counter() - t0
                if dt > 0:
                    fps_window.append(1.0 / dt)
                fps = sum(fps_window) / len(fps_window) if fps_window else 0.0
                draw_hud(
                    proc,
                    fps=fps,
                    detection_count=len(detections),
                    frame_number=frame_number,
                )

                if args.save or cfg.output.save_video:
                    if writer is None:
                        out_dir = Path(cfg.output.output_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        output_path = out_dir / f"ecoli_{int(time.time())}.mp4"
                        h, w = proc.shape[:2]
                        # mp4v is broadly available without extra codecs.
                        # avc1 is smaller but needs an H.264 encoder
                        # installed alongside OpenCV — not guaranteed.
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        # Use a sane minimum FPS for files in case the
                        # measured FPS is still 0 on the first frame.
                        record_fps = max(fps, float(cfg.capture.fps or 15))
                        writer = cv2.VideoWriter(
                            str(output_path), fourcc, record_fps, (w, h),
                        )
                        if not writer.isOpened():
                            log.error("Could not open VideoWriter at %s", output_path)
                            writer = None
                        else:
                            log.info("Recording annotated stream to %s", output_path)
                    if writer is not None:
                        writer.write(proc)

                if not args.no_display:
                    cv2.imshow("E. coli Detection — q to quit", proc)
                    # waitKey(1) is mandatory for imshow to repaint.
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        log.info("Quit key pressed.")
                        break

    except CaptureError as exc:
        log.error("Capture error: %s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
            log.info("Saved output: %s", output_path)
        cv2.destroyAllWindows()
        log.info("Shutdown complete. Processed %d frames.", frame_number)


if __name__ == "__main__":
    main()
