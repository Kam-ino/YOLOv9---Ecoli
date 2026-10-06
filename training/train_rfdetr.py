"""
training/train_rfdetr.py
========================
Fine-tune RF-DETR on the E. coli dataset — the RF-DETR twin of ``train.py``.

Same CLI as ``training/train.py`` so ``backend/app/training.py`` can spawn
either one with identical arguments:

    python -m training.train_rfdetr \\
        --data training/dataset.yaml \\
        --weights medium --epochs 200 --batch 4 --imgsz 640 --device 0

``--weights`` is either a variant name (``nano|small|medium|large`` →
COCO-pretrained from Roboflow's CDN) or a ``.pth`` checkpoint to continue
from (then ``--variant`` says which architecture it is).

Output lands in ``<project>/<name>/``: ``checkpoint_best_ema.pth`` (what the
backend auto-activates into ``models/best_rfdetr.pth``), ``last.ckpt`` for
``--resume``, ``training_config.json`` and TensorBoard logs.

Recipe
------
* lr 1e-4 / lr_encoder 1.5e-4 / wd 1e-4 — the library defaults.
* Effective batch 16 via gradient accumulation (``--grad-accum 0`` = auto).
* EMA weights, early stopping on validation mAP50-95 of the EMA model.
* bf16 autocast on CUDA.
* ``--num-queries`` is the per-image detection cap. RF-DETR has no NMS, so
  this must exceed the densest slide (1,043 labelled cells) — hence 1200,
  not the library's 300.
* Augmentation is RF-DETR's own (scale jitter + horizontal flip +
  multi-scale). None of train.py's mosaic / rotation / HSV / erasing — a
  known confound, documented in experiments/rfdetr_vs_yolov9/NOTES.md.
"""
import argparse
import logging
import sys
from pathlib import Path

from training.dataset_view import make_view, names_of_yaml, splits_of_yaml


log = logging.getLogger(__name__)

VARIANTS = ("nano", "small", "medium", "large")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune RF-DETR on a custom E. coli microscopy dataset.",
    )
    p.add_argument("--data", required=True,
                   help="Path to the YOLO-format dataset yaml (same file train.py uses).")
    p.add_argument("--weights", default="medium",
                   help="Variant name (nano|small|medium|large = COCO-pretrained) "
                        "or a .pth checkpoint to continue from.")
    p.add_argument("--variant", default=None, choices=VARIANTS,
                   help="Architecture of a .pth given via --weights (default: medium).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4,
                   help="Batch size. 8 GB GPUs: 2-4 at 640 px with 1200 queries.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Square input resolution. Must be a multiple of 32.")
    p.add_argument("--device", default=None,
                   help="'0' / 'cuda:0' / 'cpu'. None = CUDA if available.")
    p.add_argument("--name", default="ecoli_rfdetr",
                   help="Run name under <project>/.")
    p.add_argument("--project", default="runs/train",
                   help="Parent directory for training runs.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last.ckpt of the same run name.")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience in epochs (0 = off).")
    p.add_argument("--save-period", type=int, default=-1,
                   help="Accepted for CLI parity with train.py; RF-DETR keeps "
                        "its own periodic checkpoints.")
    p.add_argument("--workers", type=int, default=2,
                   help="DataLoader worker count. Reduce on Windows / low-RAM machines.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-queries", type=int, default=1200,
                   help="Decoder queries = max detections per image.")
    p.add_argument("--grad-accum", type=int, default=0,
                   help="Gradient accumulation steps; 0 = 16 // batch.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="~30-40%% less VRAM for ~20%% more time.")
    return p.parse_args()


def resolve_device(spec) -> str:
    """Map train.py-style device specs onto a torch device string."""
    spec = (str(spec).strip().lower() if spec is not None else "auto")
    if spec in ("", "auto"):
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if spec.isdigit():
        return f"cuda:{spec}"
    if spec == "cuda":
        return "cuda:0"
    return spec


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
    if args.imgsz % 32 != 0:
        log.error("--imgsz must be a multiple of 32 (got %d).", args.imgsz)
        sys.exit(1)

    try:
        import rfdetr
    except ImportError:
        log.error("rfdetr not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    # --weights: variant name → COCO-pretrained; otherwise a checkpoint file.
    pretrain: Path | None = None
    variant = args.variant
    if args.weights.lower() in VARIANTS:
        variant = variant or args.weights.lower()
    else:
        pretrain = Path(args.weights)
        if not pretrain.is_file():
            log.error("Weights not found: %s (expected a variant name or a .pth file).", pretrain)
            sys.exit(1)
        variant = variant or "medium"
    model_cls = getattr(rfdetr, f"RFDETR{variant.capitalize()}")

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # RF-DETR wants <dataset>/train, <dataset>/valid and data.yaml at the
    # root; the repo keeps images/<split>. Build a hard-linked view per run.
    splits = splits_of_yaml(data_path)
    names = names_of_yaml(data_path)
    if not splits.get("train") or not splits.get("valid"):
        log.error("%s needs non-empty train and val splits (got %s).",
                  data_path, {k: len(v) for k, v in splits.items()})
        sys.exit(1)
    dataset_dir = run_dir / "dataset"
    make_view({"train": splits["train"], "valid": splits["valid"]}, dataset_dir, names)
    class_names = [names[k] for k in sorted(names)]

    device = resolve_device(args.device)
    grad_accum = args.grad_accum or max(1, 16 // args.batch)

    model_kwargs = dict(
        resolution=args.imgsz,
        num_queries=args.num_queries,
        num_select=args.num_queries,
        device=device,
    )
    if pretrain is not None:
        model_kwargs["pretrain_weights"] = str(pretrain)
    if args.gradient_checkpointing:
        model_kwargs["gradient_checkpointing"] = True

    log.info(
        "Starting RF-DETR fine-tune: variant=%s weights=%s data=%s "
        "(train=%d val=%d) epochs=%d batch=%d x accum=%d imgsz=%d queries=%d device=%s",
        variant, pretrain or "COCO", data_path, len(splits["train"]), len(splits["valid"]),
        args.epochs, args.batch, grad_accum, args.imgsz, args.num_queries, device,
    )
    model = model_cls(**model_kwargs)

    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(run_dir),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=grad_accum,
        lr=args.lr,
        # resolution is a model-constructor field (set above), not a train() kwarg
        amp_dtype="bf16" if device.startswith("cuda") else None,
        use_ema=True,
        early_stopping=args.patience > 0,
        early_stopping_patience=max(args.patience, 1),
        early_stopping_use_ema=True,
        skip_best_epochs=min(5, max(args.epochs - 1, 0)),
        eval_max_dets=args.num_queries,
        seed=args.seed,
        num_workers=args.workers,
        tensorboard=True,
        run_test=False,
        class_names=class_names,
        resume=str(run_dir / "last.ckpt") if args.resume else None,
        progress_bar="tqdm",
        notes={
            "recipe": "training/train_rfdetr.py",
            "data": str(data_path),
            "variant": variant,
            "args": vars(args),
        },
    )

    best = run_dir / "checkpoint_best_ema.pth"
    log.info("Training complete.")
    log.info("Best weights: %s", best)
    log.info("To deploy: copy it to models/best_rfdetr.pth (the backend does "
             "this automatically after a Train-tab run).")


if __name__ == "__main__":
    # The __main__ guard matters on Windows: Lightning's DataLoader workers
    # re-import this module via spawn.
    main()
