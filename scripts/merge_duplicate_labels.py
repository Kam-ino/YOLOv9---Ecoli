"""
scripts/merge_duplicate_labels.py
=================================
Collapse duplicate copies of the same image into one, merging their labels.

The dataset tab auto-renames on filename collision ("x.png" -> "x - 1.png"),
so re-saving an image with a second batch of boxes creates a *new copy*
carrying only that batch. Every copy then teaches the model that the boxes
drawn on its siblings are background, and copies that land in different
splits leak train into val.

This script:
  1. groups images that are pixel-identical up to rotation / flip,
  2. maps every copy's boxes into one canonical frame and unions them,
     dropping re-drawn duplicates of the same object,
  3. re-splits by *group* so no image appears in both train and val,
  4. writes the result to a new dataset root (the source is never modified).

Usage:
    python scripts/merge_duplicate_labels.py --src data/ecoli --dst data/ecoli_merged
    python scripts/merge_duplicate_labels.py --selftest
"""
import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# The 8 dihedral transforms: (number of CCW quarter turns, then left-right flip).
DIHEDRAL = [(k, f) for k in range(4) for f in (False, True)]
# Mean abs gray difference (0-255, at 128px) below which two images count as
# the same frame. Identical PNGs score 0; different fields of view score 15+.
SAME_IMAGE_MAD = 4.0


def transform_image(img: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(img, k)
    return np.fliplr(out) if flip else out


def transform_boxes(boxes: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Apply the same transform to normalized YOLO rows (cls, x, y, w, h)."""
    b = boxes.copy()
    for _ in range(k):  # one CCW quarter turn: (x, y) -> (y, 1 - x), w <-> h
        b[:, [1, 2, 3, 4]] = np.c_[b[:, 2], 1.0 - b[:, 1], b[:, 4], b[:, 3]]
    if flip:
        b[:, 1] = 1.0 - b[:, 1]
    return b


def _thumb(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(np.ascontiguousarray(gray), (128, 128),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def find_transform(img: np.ndarray, ref_thumb: np.ndarray):
    """Return ((k, flip), mad) for the transform that best maps img onto ref."""
    best = None
    for k, flip in DIHEDRAL:
        mad = float(np.abs(_thumb(transform_image(img, k, flip)) - ref_thumb).mean())
        if best is None or mad < best[1]:
            best = ((k, flip), mad)
    return best


def _xyxy(b: np.ndarray) -> np.ndarray:
    return np.c_[b[:, 1] - b[:, 3] / 2, b[:, 2] - b[:, 4] / 2,
                 b[:, 1] + b[:, 3] / 2, b[:, 2] + b[:, 4] / 2]


def dedupe(boxes: np.ndarray) -> Tuple[np.ndarray, int]:
    """Drop re-drawn copies of the same object; return (kept, n_class_conflicts).

    Two boxes are the same object when they overlap heavily (IoU > 0.45), when
    one's centre sits inside the other and their areas are comparable
    (hand-drawn boxes on a 15 px cell rarely reach a high IoU twice), or when
    they share a class and one lies mostly inside the other — the same clump
    boxed tight in one copy and loose in another. A cluster box enclosing
    single-cell boxes is kept: different class, very different area.
    """
    if len(boxes) == 0:
        return boxes, 0
    # Smallest first so the tighter box of a duplicate pair survives, and a
    # sloppy box thrown around several already-kept clumps is the one dropped.
    boxes = boxes[np.argsort(boxes[:, 3] * boxes[:, 4], kind="stable")]
    xy = _xyxy(boxes)
    area = boxes[:, 3] * boxes[:, 4]
    keep: List[int] = []
    conflicts = 0
    for i in range(len(boxes)):
        dup = False
        for j in keep:
            ix = min(xy[i, 2], xy[j, 2]) - max(xy[i, 0], xy[j, 0])
            iy = min(xy[i, 3], xy[j, 3]) - max(xy[i, 1], xy[j, 1])
            if ix <= 0 or iy <= 0:
                continue
            inter = ix * iy
            iou = inter / (area[i] + area[j] - inter)
            centre_in = (xy[j, 0] <= boxes[i, 1] <= xy[j, 2]
                         and xy[j, 1] <= boxes[i, 2] <= xy[j, 3])
            similar = 0.5 <= area[i] / area[j] <= 2.0
            nested = boxes[i, 0] == boxes[j, 0] and inter / min(area[i], area[j]) > 0.6
            if iou > 0.45 or (centre_in and similar) or nested:
                dup = True
                if boxes[i, 0] != boxes[j, 0]:
                    conflicts += 1
                break
        if not dup:
            keep.append(i)
    return boxes[keep], conflicts


def load_labels(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    return np.loadtxt(path, ndmin=2, dtype=np.float64)[:, :5]


def group_images(files: List[Path]) -> List[List[Tuple[Path, Tuple[int, bool]]]]:
    """Greedy clustering: each group is [(path, transform-onto-first-member)]."""
    groups: List[dict] = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  ! unreadable, skipped: {f}", file=sys.stderr)
            continue
        placed = False
        for g in groups:
            # Cheap reject: a rotated copy keeps the same set of side lengths.
            if sorted(img.shape[:2]) != g["dims"]:
                continue
            t, mad = find_transform(img, g["thumb"])
            if mad < SAME_IMAGE_MAD:
                g["members"].append((f, t))
                placed = True
                break
        if not placed:
            groups.append({"dims": sorted(img.shape[:2]), "thumb": _thumb(img),
                           "members": [(f, (0, False))]})
    return [g["members"] for g in groups]


def domain_of(name: str) -> str:
    # Two visually unrelated sources share this dataset; stratify the split
    # so val covers both.
    return "microscope" if "microscope" in name.lower() else "stained"


def merge(src: Path, dst: Path, val_fraction: float, seed: int) -> None:
    files = sorted(p for s in ("train", "val", "test")
                   for p in (src / "images" / s).glob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        sys.exit(f"No images under {src}/images")
    if dst.exists():
        sys.exit(f"{dst} already exists — remove it or pick another --dst")

    print(f"Grouping {len(files)} images ...")
    groups = group_images(files)

    merged: List[dict] = []
    total_in = total_out = total_conflicts = 0
    for members in groups:
        # Canonical frame = first member; every transform already maps onto it.
        rows = []
        for path, (k, flip) in members:
            lab = load_labels(src / "labels" / path.parent.name / (path.stem + ".txt"))
            rows.append(transform_boxes(lab, k, flip))
        allb = np.vstack(rows)
        kept, conflicts = dedupe(allb)
        total_in += len(allb)
        total_out += len(kept)
        total_conflicts += conflicts
        merged.append({"image": members[0][0], "boxes": kept, "copies": len(members)})

    # Split by group, stratified by domain.
    rng = random.Random(seed)
    split_of: Dict[int, str] = {}
    for dom in ("microscope", "stained"):
        idx = [i for i, m in enumerate(merged) if domain_of(m["image"].name) == dom]
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * val_fraction)) if idx else 0
        for n, i in enumerate(idx):
            split_of[i] = "val" if n < n_val else "train"

    for s in ("train", "val", "test"):
        (dst / "images" / s).mkdir(parents=True)
        (dst / "labels" / s).mkdir(parents=True)
    for i, m in enumerate(merged):
        s = split_of[i]
        shutil.copy2(m["image"], dst / "images" / s / m["image"].name)
        with open(dst / "labels" / s / (m["image"].stem + ".txt"), "w", encoding="utf-8") as fh:
            for c, x, y, w, h in m["boxes"]:
                fh.write(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    n_val = sum(1 for s in split_of.values() if s == "val")
    print(f"{len(files)} files -> {len(merged)} unique images "
          f"({len(merged) - n_val} train / {n_val} val)")
    print(f"{total_in} boxes -> {total_out} after removing {total_in - total_out} "
          f"re-drawn duplicates ({total_conflicts} had conflicting classes)")
    print(f"Wrote {dst}")


def selftest() -> None:
    """A box drawn on a marked patch must still cover it after every transform."""
    img = np.zeros((60, 100), dtype=np.uint8)
    img[10:20, 70:90] = 255                                  # patch: x 70-90, y 10-20
    box = np.array([[0, 0.80, 0.25, 0.20, 10 / 60]])
    for k, flip in DIHEDRAL:
        t_img = transform_image(img, k, flip)
        _, x, y, w, h = transform_boxes(box, k, flip)[0]
        H, W = t_img.shape
        x1, x2 = round((x - w / 2) * W), round((x + w / 2) * W)
        y1, y2 = round((y - h / 2) * H), round((y + h / 2) * H)
        assert t_img[y1:y2, x1:x2].all() and t_img.sum() == t_img[y1:y2, x1:x2].sum(), (k, flip)
    # Same cell boxed twice collapses to one; a cluster around it survives.
    b = np.array([[0, .50, .50, .02, .02], [0, .505, .50, .022, .02], [1, .50, .50, .20, .20]])
    kept, conflicts = dedupe(b)
    assert len(kept) == 2 and conflicts == 0, kept
    # Same clump boxed tight and loose collapses to the tight box; the cell inside stays.
    b = np.array([[1, .50, .50, .10, .10], [1, .52, .51, .20, .18], [0, .50, .50, .02, .02]])
    kept, _ = dedupe(b)
    assert sorted(kept[:, 3].tolist()) == [.02, .10], kept
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=Path("data/ecoli"))
    ap.add_argument("--dst", type=Path, default=Path("data/ecoli_merged"))
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else merge(a.src, a.dst, a.val_fraction, a.seed)
