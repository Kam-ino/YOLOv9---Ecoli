"""
training/train.py
=================
Fine-tune YOLOv9 on a custom E. coli microscopy dataset.

Quick start:
    python -m training.train \\
        --data training/dataset.yaml \\
        --weights yolov9c.pt \\
        --epochs 100 --batch 16 --imgsz 640 --device 0

``--data`` points to an Ultralytics-format dataset yaml. If your
annotations are in COCO JSON, convert first:

    from ultralytics.data.converter import convert_coco
    convert_coco("path/to/coco/annotations", use_segments=False)

The augmentation defaults below are tuned for stained-light microscopy
— see the inline rationale block. Override individual values with the
``--<key>`` flags if you need to experiment.
"""
import argparse
import logging
import sys
from pathlib import Path


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv9 on a custom E. coli microscopy dataset.",
    )
    p.add_argument("--data", required=True,
                   help="Path to the YOLO-format dataset yaml (see "
                        "training/dataset.yaml.example).")
    p.add_argument("--weights", default="yolov9c.pt",
                   help="Pretrained weights to fine-tune from. Ultralytics "
                        "auto-downloads recognised names (yolov9t.pt, "
                        "yolov9s.pt, yolov9c.pt, yolov9e.pt).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size. Reduce if you OOM on GPU.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Training image size. Must be a multiple of 32.")
    p.add_argument("--device", default=None,
                   help="CUDA device id ('0'), 'cpu', or comma list for "
                        "multi-GPU. None lets Ultralytics auto-select.")
    p.add_argument("--name", default="ecoli_yolov9c",
                   help="Run name under <project>/.")
    p.add_argument("--project", default="runs/train",
                   help="Parent directory for training runs.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last.pt of the same run name.")
    p.add_argument("--patience", type=int, default=30,
                   help="Early stopping patience in epochs without "
                        "validation improvement.")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader worker count. Reduce on Windows / "
                        "low-RAM machines.")
    p.add_argument("--lr0", type=float, default=0.001,
                   help="Initial learning rate (AdamW).")
    p.add_argument("--optimizer", default="AdamW",
                   choices=("SGD", "Adam", "AdamW", "auto"),
                   help="Optimizer. AdamW is a safer default on small "
                        "microscopy datasets.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        log.error("Dataset yaml not found: %s", data_path)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    log.info("Loading base weights: %s", args.weights)
    model = YOLO(args.weights)

    log.info(
        "Starting fine-tuning: data=%s epochs=%d batch=%d imgsz=%d device=%s",
        args.data, args.epochs, args.batch, args.imgsz, args.device,
    )

    # ------------------------------------------------------------------
    # Augmentation rationale — microscopy-specific
    # ------------------------------------------------------------------
    # degrees=180     orientation under a microscope is arbitrary
    # flipud/fliplr   bacteria are not chiral at this scale
    # hsv_h=0.0       stain hue is diagnostic — do NOT shift it
    # hsv_s=0.2       mild saturation jitter for camera white-balance drift
    # hsv_v=0.4       large value jitter — illumination varies between samples
    # scale=0.3       moderate zoom; aggressive scale alters apparent cell size
    # mosaic=0.5      moderate; full mosaic can synthesise unrealistic FOVs
    # mixup=0.0       avoid — blends ground-truth boxes on tiny targets
    # copy_paste=0.0  avoid — same reason
    # erasing=0.2     light random erasing helps occlusion robustness
    # ------------------------------------------------------------------

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        project=args.project,
        resume=args.resume,
        patience=args.patience,
        workers=args.workers,

        # Microscopy-tuned augmentation
        hsv_h=0.0,
        hsv_s=0.2,
        hsv_v=0.4,
        degrees=180.0,
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.2,

        # Optimizer — conservative defaults for small datasets
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=True,

        # Reporting / checkpoints
        plots=True,
        save=True,
        save_period=10,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    log.info("Training complete.")
    log.info("Best weights: %s", best)
    log.info("To deploy:")
    log.info("    cp %s models/ecoli_yolov9c.pt", best)
    log.info("    # then update config.yaml model.weights if needed")
    log.info("To export to ONNX (lighter inference, no PyTorch at deploy):")
    log.info("    yolo export model=%s format=onnx imgsz=%d simplify=true",
             best, args.imgsz)


if __name__ == "__main__":
    main()
