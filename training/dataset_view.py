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

Used by ``training/train_rfdetr.py`` and ``scripts/compare_folds.py``.
"""
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Union

import yaml

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# RF-DETR / Roboflow name the validation split "valid"; Ultralytics calls it "val".
SPLIT_ALIAS = {"val": "valid", "valid": "valid", "train": "train", "test": "test"}


def label_of(img: Path) -> Path:
    """``<root>/images/<split>/x.png`` -> ``<root>/labels/<split>/x.txt``."""
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


def make_view(splits: Dict[str, Iterable[Path]], dst: Path, names: Dict[int, str]) -> Path:
    """Materialise ``splits`` (keys train/valid/test) under ``dst``; return ``dst/data.yaml``.

    Images without a label file get an empty ``.txt`` (= background image)
    so both loaders see a complete pair.
    """
    dst = Path(dst)
    present = []
    for split, images in splits.items():
        split = SPLIT_ALIAS[split]
        images = list(images)
        if not images:
            continue
        present.append(split)
        for img in images:
            link(img, dst / split / "images" / img.name)
            lab = label_of(img)
            target = dst / split / "labels" / (img.stem + ".txt")
            if lab.exists():
                link(lab, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")
    data = {"path": dst.resolve().as_posix()}
    for split in ("train", "valid", "test"):
        if split in present:
            data["val" if split == "valid" else split] = f"{split}/images"
    data["nc"] = len(names)
    data["names"] = {int(k): names[k] for k in sorted(names)}
    out = dst / "data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def _selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ds"
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            (root / "images" / split / "a.png").write_bytes(b"x")
            (root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (root / "images" / "train" / "b.png").write_bytes(b"y")      # no label
        y = root / "data.yaml"
        y.write_text(yaml.safe_dump({"path": root.as_posix(), "train": "images/train",
                                     "val": "images/val", "names": {0: "ecoli"}}))
        splits = splits_of_yaml(y)
        assert [p.name for p in splits["train"]] == ["a.png", "b.png"], splits
        assert [p.name for p in splits["valid"]] == ["a.png"]
        out = make_view(splits, Path(td) / "view", names_of_yaml(y))
        d = yaml.safe_load(out.read_text())
        assert d["train"] == "train/images" and d["val"] == "valid/images" and d["names"] == {0: "ecoli"}
        assert (Path(td) / "view" / "train" / "labels" / "b.txt").read_text() == ""
        assert (Path(td) / "view" / "valid" / "images" / "a.png").read_bytes() == b"x"
    print("dataset_view selftest ok")


if __name__ == "__main__":
    _selftest()
