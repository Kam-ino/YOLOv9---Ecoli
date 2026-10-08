"""
scripts/compare_bench.py
========================
Speed, size and compute of both detectors under identical conditions.

    python scripts/compare_bench.py [--fold 0] [--imgsz 640] [--passes 5] [--warmup 20] [--fp32]

Same GPU, batch size 1, FP16 for both unless ``--fp32`` (YOLO ``half=True``;
RF-DETR ``model.inference(dtype="float16")``), cuDNN autotune on, a warm-up,
then ``--passes`` passes over all pooled images. Each timing wraps
pre-processing + forward + post-processing (NMS / query decode) + rescale to
image coordinates — i.e. ``Detector.predict(frame)`` — between two
``torch.cuda.synchronize()`` calls. Parameters are ``sum(numel)``; GFLOPs
come from ``torch.utils.flop_counter`` on a ``1×3×imgsz×imgsz`` forward (plus
Ultralytics' own thop figure for YOLO as a cross-check).

Weights: the fold's trained models under runs/compare/<algo>/fold<k>/.
Output: runs/compare/bench.json and a markdown row per model.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import RFDETRDetector, YOLOv9Detector  # noqa: E402

OUT_DIR = ROOT / "runs" / "compare"
EXP_DIR = ROOT / "experiments" / "rfdetr_vs_yolov9"
BEST = {"yolov9": "weights/best.pt", "rfdetr": "checkpoint_best_ema.pth"}


def find_module(obj, depth: int = 4):
    """First torch.nn.Module reachable from ``obj`` (wrappers nest it differently)."""
    import torch
    if isinstance(obj, torch.nn.Module):
        return obj
    if depth == 0:
        return None
    for v in vars(obj).values() if hasattr(obj, "__dict__") else []:
        m = find_module(v, depth - 1) if not isinstance(v, (str, int, float, list, dict, tuple, type(None))) else None
        if m is not None:
            return m
    return None


def count_flops(module, imgsz: int, device: str):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    module.eval()
    dtype = next(module.parameters()).dtype
    x = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=dtype)
    # Some graphs (RF-DETR's deformable attention) assert on grad state under
    # no_grad; count with autograd on and drop the result.
    with FlopCounterMode(display=False) as fc:
        module(x)
    return fc.get_total_flops() / 1e9


def bench(det, frames, passes: int, warmup: int) -> dict:
    import torch
    cuda = det.device.startswith("cuda")
    for f in frames[:warmup]:
        det.predict(f)
    ms = []
    for _ in range(passes):
        for f in frames:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            det.predict(f)
            if cuda:
                torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
    return {"n": len(ms), "mean_ms": statistics.mean(ms), "median_ms": statistics.median(ms),
            "std_ms": statistics.pstdev(ms), "p95_ms": float(np.percentile(ms, 95)),
            "fps": 1000 / statistics.mean(ms)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--suffix", default="", help="Run-name suffix, e.g. _x8.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-det", type=int, default=1200)
    ap.add_argument("--variant", default="medium")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ecoli")
    a = ap.parse_args()

    import torch
    torch.backends.cudnn.benchmark = True
    gt = json.loads((OUT_DIR / "gt_all.json").read_text(encoding="utf-8"))
    plan = json.loads((EXP_DIR / "folds.json").read_text(encoding="utf-8"))
    meta = {m["file_name"]: m for m in plan["images"]}
    names = [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])]
    frames = [cv2.imread(str(a.data / "images" / meta[im["file_name"]]["origin_split"] / im["file_name"]))
              for im in gt["images"]]
    assert all(f is not None for f in frames)

    out = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
           "torch": torch.__version__, "dtype": "fp32" if a.fp32 else "fp16", "imgsz": a.imgsz,
           "batch_size": 1, "n_images": len(frames), "passes": a.passes, "warmup": a.warmup,
           "post_processing": {"yolov9": f"NMS iou=0.7, max_det={a.max_det}",
                               "rfdetr": f"top-{a.max_det} query decode, no NMS"},
           "models": {}}
    rows = []
    for algo in ("yolov9", "rfdetr"):
        weights = OUT_DIR / algo / f"fold{a.fold}{a.suffix}" / BEST[algo]
        if not weights.is_file():
            print(f"skip {algo}: no weights at {weights}")
            continue
        common = dict(conf_threshold=0.25, iou_threshold=0.7, class_names=names, device="auto", warmup=True)
        if algo == "yolov9":
            det = YOLOv9Detector(str(weights), imgsz=a.imgsz, max_det=a.max_det, half=not a.fp32, **common)
        else:
            det = RFDETRDetector(str(weights), variant=a.variant, resolution=a.imgsz, num_queries=a.max_det, **common)
            if not a.fp32:
                try:
                    det.model.inference(compile=False, dtype="float16", inplace=True)
                except Exception as exc:
                    out["models"].setdefault(algo, {})["fp16_note"] = f"inference(dtype=float16) failed: {exc!r}; ran fp32"
        module = find_module(det.model)
        info = out["models"].setdefault(algo, {})
        info.update({"weights": str(weights), "params_M": sum(p.numel() for p in module.parameters()) / 1e6
                     if module is not None else None})
        try:
            info["gflops_flop_counter"] = count_flops(module, a.imgsz, det.device)
        except Exception as exc:
            info["gflops_flop_counter"] = None
            info["gflops_error"] = repr(exc)
        if algo == "yolov9":
            try:
                info["gflops_ultralytics"] = det.model.info(verbose=False)[3]
            except Exception as exc:
                info["gflops_ultralytics_error"] = repr(exc)
        info.update(bench(det, frames, a.passes, a.warmup))
        g = info.get("gflops_flop_counter") or info.get("gflops_ultralytics")
        rows.append(f"| {algo} | {info['params_M']:.1f} | {g if g is None else f'{g:.1f}'} | "
                    f"{info['mean_ms']:.1f} ± {info['std_ms']:.1f} | {info['median_ms']:.1f} | {info['fps']:.1f} |")
        print(f"{algo}: {info['mean_ms']:.1f} ms mean ({info['fps']:.1f} FPS), {info['params_M']:.1f} M params, "
              f"GFLOPs {g}", flush=True)
        del det
        torch.cuda.empty_cache()

    (OUT_DIR / "bench.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n| model | params (M) | GFLOPs @{a.imgsz} | latency ms (mean +/- sd) | median | FPS |")
    print("|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
