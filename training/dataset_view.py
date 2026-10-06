"""
training/dataset_view.py
========================
Build a dataset *view* — ``<dst>/{train,valid,test}/{images,labels}`` plus
``data.yaml`` — out of hard links to the real files under ``data/ecoli``.

Why a view: RF-DETR wants ``train/`` and ``valid/`` folders with a
``data.yaml`` at the dataset root, Ultralytics wants whatever the yaml says,
and the k-fold scripts need per-fold subsets. Hard links give all three
without copying a byte or touching the originals. Same-volume only; falls
back to a copy otherwise.

``expand8_train=True`` additionally writes the 8 dihedral variants of every
*training* image (4 rotations × {original, mirrored}) with transformed
labels — offline orientation augmentation that both detectors see
identically. Validation and test splits are never expanded: evaluation stays
on real frames, and expanding a held-out image would leak its near-twins into
training.

Used by ``training/train.py``, ``training/train_rfdetr.py`` and
``scripts/compare_folds.py``.
"""
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import yaml

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# RF-DETR / Roboflow name the validation split "valid"; Ultralytics calls it "val".
SPLIT_ALIAS = {"val": "valid", "valid": "valid", "train": "train", "test": "test"}
D4_SUFFIX = "__d4-"      # <stem>__d4-<k>.png, k = 0..7 (0 = the original)


def label_of(img: Path) -> Path:
    """Label path for either YOLO layout.

    ``<root>/images/<split>/x.png`` -> ``<root>/labels/<split>/x.txt`` (data/ecoli)
    ``<root>/<split>/images/x.png`` -> ``<root>/<split>/labels/x.txt`` (views, Roboflow)
    """
    if img.parent.name == "images":
        return img.parent.parent / "labels" / (img.stem + ".txt")
    return img.parent.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:          # ponytail: cross-volume → copy; fine for a handful of images
        shutil.copy2(src, dst)


def images_in(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS) if folder.is_dir() else []


def splits_of_yaml(yaml_path: Union[str, Path]) -> Dict[str, List[Path]]:
    """Resolve an Ultralytics ``data.yaml`` into ``{train|valid|test: [image paths]}``."""
    yaml_path = Path(yaml_path)
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    base = Path(raw.get("path", "."))
    if not base.is_absolute():
        # Ultralytics resolves relative to the cwd (the app runs from the
        # repo root); fall back to the yaml's own folder.
        base = base if base.is_dir() else (yaml_path.parent / base)
    out: Dict[str, List[Path]] = {}
    for key in ("train", "val", "test"):
        if key not in raw:
            continue
        entry = raw[key]
        dirs = entry if isinstance(entry, list) else [entry]
        imgs: List[Path] = []
        for d in dirs:
            d = Path(d)
            imgs += images_in(d if d.is_absolute() else base / d)
        out[SPLIT_ALIAS[key]] = imgs
    return out


def names_of_yaml(yaml_path: Union[str, Path]) -> Dict[int, str]:
    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    names = raw.get("names", [])
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# Dihedral (rotate + mirror) expansion
# ---------------------------------------------------------------------------

# Each entry: (image transform, point transform on normalised (x, y), swaps w/h).
# Rotations are clockwise, as cv2.ROTATE_90_CLOCKWISE; mirror = horizontal flip.
def _d4_ops() -> List[Tuple[Callable, Callable, bool]]:
    import cv2
    rot = {
        0: (lambda im: im, lambda x, y: (x, y), False),
        1: (lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), lambda x, y: (1 - y, x), True),
        2: (lambda im: cv2.rotate(im, cv2.ROTATE_180), lambda x, y: (1 - x, 1 - y), False),
        3: (lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE), lambda x, y: (y, 1 - x), True),
    }
    ops = []
    for mirror in (False, True):
        for k in range(4):
            img_t, pt_t, swap = rot[k]
            if mirror:
                def img_fn(im, _t=img_t):
                    return _t(cv2.flip(im, 1))

                def pt_fn(x, y, _t=pt_t):
                    return _t(1 - x, y)
            else:
                img_fn, pt_fn = img_t, pt_t
            ops.append((img_fn, pt_fn, swap))
    return ops


def transform_labels(rows: List[List[float]], pt: Callable, swap: bool) -> List[List[float]]:
    """YOLO rows ``[cls, cx, cy, w, h]`` (normalised) under a D4 point map."""
    out = []
    for cls, cx, cy, w, h in rows:
        nx, ny = pt(cx, cy)
        nw, nh = (h, w) if swap else (w, h)
        out.append([cls, min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0), nw, nh])
    return out


def read_labels(path: Path) -> List[List[float]]:
    if not path.exists():
        return []
    return [[float(v) for v in ln.split()] for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_labels(path: Path, rows: List[List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n" for r in rows),
                    encoding="utf-8")


def expand8(img: Path, images_dir: Path, labels_dir: Path) -> List[Path]:
    """Write the 8 D4 variants of ``img`` (+ labels) as ``<stem>__d4-<k>``; k=0 is a hard link."""
    import cv2
    rows = read_labels(label_of(img))
    frame = cv2.imread(str(img))
    assert frame is not None, img
    written = []
    for k, (img_fn, pt_fn, swap) in enumerate(_d4_ops()):
        dst = images_dir / f"{img.stem}{D4_SUFFIX}{k}{img.suffix}"
        if k == 0:
            link(img, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(dst), img_fn(frame), [cv2.IMWRITE_PNG_COMPRESSION, 3])
            assert ok, dst
        write_labels(labels_dir / f"{dst.stem}.txt", transform_labels(rows, pt_fn, swap))
        written.append(dst)
    return written


def make_view(splits: Dict[str, Iterable[Path]], dst: Path, names: Dict[int, str],
              expand8_train: bool = False) -> Path:
    """Materialise ``splits`` (keys train/valid/test) under ``dst``; return ``dst/data.yaml``.

    Images without a label file get an empty ``.txt`` (= background image)
    so both loaders see a complete pair. With ``expand8_train`` the train
    split holds 8 orientation variants per source image.
    """
    dst = Path(dst)
    present = []
    missing: List[Path] = []
    for split, images in splits.items():
        split = SPLIT_ALIAS[split]
        images = list(images)
        if not images:
            continue
        present.append(split)
        images_dir, labels_dir = dst / split / "images", dst / split / "labels"
        for img in images:
            lab = label_of(img)
            if not lab.exists():
                missing.append(img)
            if split == "train" and expand8_train:
                expand8(img, images_dir, labels_dir)
                continue
            link(img, images_dir / img.name)
            target = labels_dir / (img.stem + ".txt")
            if lab.exists():
                link(lab, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")
    if missing:
        # Legitimate for background images, but a whole split without labels
        # means the layout was misread — say so loudly rather than train on nothing.
        import logging
        logging.getLogger(__name__).warning(
            "dataset_view: %d of %d images have no label file (treated as background), e.g. %s",
            len(missing), sum(len(list(v)) for v in splits.values()), missing[0])
    data = {"path": dst.resolve().as_posix()}
    for split in ("train", "valid", "test"):
        if split in present:
            data["val" if split == "valid" else split] = f"{split}/images"
    data["nc"] = len(names)
    data["names"] = {int(k): names[k] for k in sorted(names)}
    if expand8_train:
        data["expand8_train"] = True
    out = dst / "data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def _selftest() -> None:
    import tempfile

    import cv2
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ds"
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        # 80x40 image with one bright box; label = that box in YOLO format.
        frame = np.zeros((40, 80, 3), np.uint8)
        x1, y1, x2, y2 = 10, 4, 30, 16                     # px, inclusive-exclusive
        frame[y1:y2, x1:x2] = 255
        cv2.imwrite(str(root / "images" / "train" / "a.png"), frame)
        cv2.imwrite(str(root / "images" / "val" / "a.png"), frame)
        row = f"0 {(x1 + x2) / 2 / 80:.6f} {(y1 + y2) / 2 / 40:.6f} {(x2 - x1) / 80:.6f} {(y2 - y1) / 40:.6f}\n"
        (root / "labels" / "train" / "a.txt").write_text(row)
        (root / "labels" / "val" / "a.txt").write_text(row)
        cv2.imwrite(str(root / "images" / "train" / "b.png"), np.zeros((8, 8, 3), np.uint8))   # label-less
        y = root / "data.yaml"
        y.write_text(yaml.safe_dump({"path": root.as_posix(), "train": "images/train",
                                     "val": "images/val", "names": {0: "ecoli"}}))
        splits = splits_of_yaml(y)
        assert [p.name for p in splits["train"]] == ["a.png", "b.png"], splits
        assert [p.name for p in splits["valid"]] == ["a.png"]

        out = make_view({"train": [splits["train"][0]], "valid": splits["valid"]}, Path(td) / "v8",
                        names_of_yaml(y), expand8_train=True)
        d = yaml.safe_load(out.read_text())
        assert d["train"] == "train/images" and d["val"] == "valid/images" and d["expand8_train"] is True
        imgs = sorted((Path(td) / "v8" / "train" / "images").iterdir())
        assert len(imgs) == 8 and len(list((Path(td) / "v8" / "valid" / "images").iterdir())) == 1
        for p in imgs:
            im = cv2.imread(str(p))
            h, w = im.shape[:2]
            ys, xs = np.where(im[:, :, 0] > 127)
            px = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)          # box recovered from pixels
            cls, cx, cy, bw, bh = read_labels(Path(td) / "v8" / "train" / "labels" / (p.stem + ".txt"))[0]
            lb = (round((cx - bw / 2) * w), round((cy - bh / 2) * h), round((cx + bw / 2) * w), round((cy + bh / 2) * h))
            assert lb == px, (p.name, lb, px)
        shapes = {cv2.imread(str(p)).shape[:2] for p in imgs}
        assert shapes == {(40, 80), (80, 40)}, shapes

        out = make_view({"train": splits["train"], "valid": splits["valid"]}, Path(td) / "v1", names_of_yaml(y))
        assert (Path(td) / "v1" / "train" / "labels" / "b.txt").read_text() == ""
        assert (Path(td) / "v1" / "train" / "labels" / "a.txt").read_text() == row
        assert "expand8_train" not in yaml.safe_load(out.read_text())

        # A view built from a view (fold dir → run dataset) must carry labels too.
        s2 = splits_of_yaml(out)
        assert label_of(s2["valid"][0]).exists(), label_of(s2["valid"][0])
        out2 = make_view(s2, Path(td) / "v2", names_of_yaml(out), expand8_train=True)
        assert (Path(td) / "v2" / "valid" / "labels" / "a.txt").read_text() == row
        rows2 = read_labels(Path(td) / "v2" / "train" / "labels" / f"a{D4_SUFFIX}3.txt")
        assert len(rows2) == 1 and rows2[0][0] == 0.0, rows2
        assert yaml.safe_load(out2.read_text())["names"] == {0: "ecoli"}
    print("dataset_view selftest ok (8 D4 variants, labels match pixels, view-of-view keeps labels)")


if __name__ == "__main__":
    _selftest()
