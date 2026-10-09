"""
scripts/slim_checkpoint.py
==========================
Shrink an RF-DETR checkpoint for distribution: keep only the model weights
(as fp16) plus the metadata rfdetr needs to rebuild the architecture; drop
the Lightning trainer state, optimizer and scheduler. 146 MB -> 73 MB, which
fits GitHub's 100 MB per-file limit without LFS. Loading is unchanged
(torch casts fp16 tensors back into the fp32 model).

    python scripts/slim_checkpoint.py models/best_rfdetr.pth            # in place, original -> *.fp32.pth
    python scripts/slim_checkpoint.py in.pth --out slim.pth
"""
import argparse
import shutil
from pathlib import Path

import torch

KEEP = ("args", "model_config", "model_name", "rfdetr_version", "epoch", "class_names", "query_expansion")


def slim(src: Path, dst: Path) -> dict:
    torch.serialization.add_safe_globals([argparse.Namespace])
    ck = torch.load(str(src), map_location="cpu", weights_only=True)
    model = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}
    out = {"model": model, **{k: ck[k] for k in KEEP if k in ck}, "slim": {"source": src.name, "dtype": "fp16"}}
    torch.save(out, str(dst))
    return {"params_M": sum(v.numel() for v in model.values()) / 1e6,
            "before_MB": src.stat().st_size / 1e6, "after_MB": dst.stat().st_size / 1e6}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="Default: replace src, keeping src as <stem>.fp32.pth")
    a = ap.parse_args()
    if a.out is None:
        backup = a.src.with_name(a.src.stem + ".fp32" + a.src.suffix)
        if not backup.exists():
            shutil.copy2(a.src, backup)
        tmp = a.src.with_suffix(".tmp")
        info = slim(backup, tmp)
        tmp.replace(a.src)
        print(f"{a.src}: {info['before_MB']:.0f} MB -> {info['after_MB']:.0f} MB ({info['params_M']:.1f} M params); original kept as {backup.name}")
    else:
        info = slim(a.src, a.out)
        print(f"{a.out}: {info['before_MB']:.0f} MB -> {info['after_MB']:.0f} MB")


if __name__ == "__main__":
    main()
