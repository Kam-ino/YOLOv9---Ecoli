"""
backend/app/dataset.py
======================
On-disk dataset store in YOLO format.

Saves uploaded / snapshotted images and their bounding-box annotations
directly into the layout consumed by ``training/train.py``::

    data/ecoli/
      images/{train,val,test}/<filename>
      labels/{train,val,test}/<stem>.txt   # one box per line:
                                           #   <class_id> <cx> <cy> <w> <h>
"""
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from .schemas import (
    DatasetEntry, DatasetSplitStats, DatasetStats, LabelBox,
)


log = logging.getLogger(__name__)


VALID_SPLITS = ("train", "val", "test")

# Filenames are quoted into URLs and path-joined onto disk, so restrict
# hard to a safe character set. The Label UI generates timestamps that
# satisfy this; user-uploaded names are sanitised by the route. Spaces
# are permitted so the " - N" collision suffix produces readable names.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9 ._-]+$")


class DatasetStore:
    """File-system backed dataset in YOLO format."""

    def __init__(self, root: Path):
        self.root = root

    # -- path helpers --------------------------------------------------------

    def _resolve(self, filename: str, split: str) -> Tuple[Path, Path]:
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {split!r}")
        if not _FILENAME_RE.match(filename):
            raise ValueError(
                f"Invalid filename: {filename!r} "
                f"(allowed: A-Z a-z 0-9 . _ - and space)"
            )
        image_path = self.root / "images" / split / filename
        label_path = self.root / "labels" / split / (Path(filename).stem + ".txt")
        return image_path, label_path

    def _exists_in_any_split(self, filename: str) -> bool:
        return any(
            (self.root / "images" / s / filename).exists() for s in VALID_SPLITS
        )

    def unique_filename(self, filename: str) -> str:
        """Return ``filename`` unchanged if free, else append ' - N' before
        the extension. Scans across all splits so the same name can't
        appear in train/val/test simultaneously and confuse the user.

        ``ecoli.png`` → first free of ``ecoli - 1.png`` → ``ecoli - 2.png`` → …

        We iterate from 1 (re-using gaps left by deletes) rather than
        max+1, which keeps numbers small in normal use.
        """
        if not self._exists_in_any_split(filename):
            return filename
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        n = 1
        # No upper bound — at 1 collision/ms this loop is fine even for
        # millions of entries. In practice it terminates almost immediately.
        while True:
            candidate = f"{stem} - {n}{suffix}"
            if not self._exists_in_any_split(candidate):
                return candidate
            n += 1

    # -- write ---------------------------------------------------------------

    def save(
        self,
        image_bytes: bytes,
        boxes: List[LabelBox],
        split: str,
        filename: str,
    ) -> DatasetEntry:
        # Validate first, then auto-rename on collision so we never
        # silently overwrite a previously-labelled image.
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {split!r}")
        if not _FILENAME_RE.match(filename):
            raise ValueError(
                f"Invalid filename: {filename!r} "
                f"(allowed: A-Z a-z 0-9 . _ - and space)"
            )
        final_name = self.unique_filename(filename)
        if final_name != filename:
            log.info(
                "Filename collision on %s — saving as %s instead.",
                filename, final_name,
            )

        image_path, label_path = self._resolve(final_name, split)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        image_path.write_bytes(image_bytes)

        lines = [
            f"{b.class_id} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}"
            for b in boxes
        ]
        # Trailing newline matches Ultralytics' own writer; harmless if empty.
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

        log.info(
            "Saved %s with %d box(es) to split=%s", final_name, len(boxes), split,
        )
        return DatasetEntry(
            filename=final_name,
            split=split,
            image_url=f"/api/dataset/image/{split}/{final_name}",
            num_boxes=len(boxes),
            created_at=image_path.stat().st_mtime,
        )

    def delete(self, filename: str, split: str) -> None:
        image_path, label_path = self._resolve(filename, split)
        image_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)
        log.info("Deleted %s from split=%s", filename, split)

    # -- read ----------------------------------------------------------------

    def get_image_path(self, filename: str, split: str) -> Path:
        image_path, _ = self._resolve(filename, split)
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        return image_path

    def list_entries(self, split: Optional[str] = None) -> List[DatasetEntry]:
        splits = [split] if split else list(VALID_SPLITS)
        entries: List[DatasetEntry] = []
        for s in splits:
            if s not in VALID_SPLITS:
                continue
            img_dir = self.root / "images" / s
            if not img_dir.is_dir():
                continue
            for img in img_dir.iterdir():
                if not img.is_file():
                    continue
                entries.append(
                    DatasetEntry(
                        filename=img.name,
                        split=s,
                        image_url=f"/api/dataset/image/{s}/{img.name}",
                        num_boxes=self._count_boxes(img.name, s),
                        created_at=img.stat().st_mtime,
                    )
                )
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    def _count_boxes(self, image_filename: str, split: str) -> int:
        label_path = self.root / "labels" / split / (
            Path(image_filename).stem + ".txt"
        )
        if not label_path.exists():
            return 0
        return sum(
            1
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def read_labels(self, filename: str, split: str) -> List[LabelBox]:
        """Parse the YOLO-format label file for ``filename`` in ``split``.

        Returns an empty list if the file is missing. Malformed lines
        are skipped with a warning rather than aborting the whole read
        — we want to recover gracefully from hand-edited label files.
        """
        _, label_path = self._resolve(filename, split)
        if not label_path.exists():
            return []
        boxes: List[LabelBox] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                log.warning("Skipping malformed label line in %s: %r", label_path, line)
                continue
            try:
                boxes.append(LabelBox(
                    class_id=int(parts[0]),
                    cx=float(parts[1]),
                    cy=float(parts[2]),
                    w=float(parts[3]),
                    h=float(parts[4]),
                ))
            except (ValueError, Exception) as exc:
                log.warning("Skipping unparseable label in %s (%s): %r",
                            label_path, exc, line)
        return boxes

    def find_entry(self, filename: str) -> Optional[Tuple[DatasetEntry, List[LabelBox]]]:
        """Locate ``filename`` across all splits, returning entry + boxes."""
        for split in VALID_SPLITS:
            try:
                image_path = self.get_image_path(filename, split)
            except (ValueError, FileNotFoundError):
                continue
            boxes = self.read_labels(filename, split)
            entry = DatasetEntry(
                filename=filename,
                split=split,
                image_url=f"/api/dataset/image/{split}/{filename}",
                num_boxes=len(boxes),
                created_at=image_path.stat().st_mtime,
            )
            return entry, boxes
        return None

    def stats(self) -> DatasetStats:
        per_split = {}
        total_images = total_boxes = 0
        for s in VALID_SPLITS:
            img_dir = self.root / "images" / s
            if not img_dir.is_dir():
                per_split[s] = DatasetSplitStats(images=0, boxes=0)
                continue
            images = [f for f in img_dir.iterdir() if f.is_file()]
            boxes = sum(self._count_boxes(f.name, s) for f in images)
            per_split[s] = DatasetSplitStats(images=len(images), boxes=boxes)
            total_images += len(images)
            total_boxes += boxes
        return DatasetStats(
            classes=load_label_classes(),
            splits=per_split,
            totals=DatasetSplitStats(images=total_images, boxes=total_boxes),
        )


# ---------------------------------------------------------------------------
# Label vocabulary
# ---------------------------------------------------------------------------

# Where to read class names from. If this file doesn't exist (the user
# hasn't created training/dataset.yaml yet), fall back to a single
# "ecoli" class — which matches the shipped dataset.yaml.example and
# config.yaml defaults.
_LABEL_VOCAB_PATH = Path("training/dataset.yaml")
_FALLBACK_LABEL_CLASSES: List[str] = ["ecoli"]

# Class names end up in YOLO label files, dataset.yaml, and URLs (e.g.
# the entries list in the sidebar). Restrict to a safe character set so
# we never have to think about escaping.
_CLASS_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def load_label_classes() -> List[str]:
    """Read the labeling vocabulary, with a sensible fallback."""
    if not _LABEL_VOCAB_PATH.exists():
        return list(_FALLBACK_LABEL_CLASSES)
    try:
        with _LABEL_VOCAB_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        names = data.get("names", {})
        if isinstance(names, dict):
            # Ultralytics format: {0: "ecoli", 1: "..."}
            return [str(names[i]) for i in sorted(names)]
        if isinstance(names, list):
            return [str(n) for n in names]
    except Exception as exc:
        log.warning("Could not parse %s: %s", _LABEL_VOCAB_PATH, exc)
    return list(_FALLBACK_LABEL_CLASSES)


def add_label_class(name: str) -> List[str]:
    """Append ``name`` to the labeling vocabulary in ``training/dataset.yaml``.

    Bootstraps the file if it doesn't exist yet (so the user doesn't
    have to start from a manual template just to add a second class).

    Raises:
        ValueError: name is empty, malformed, or already present.

    Returns the new full class list.
    """
    name = name.strip()
    if not name:
        raise ValueError("Class name cannot be empty.")
    if not _CLASS_NAME_RE.match(name):
        raise ValueError(
            f"Invalid class name {name!r}: only letters, digits, '.', "
            f"'_' and '-' allowed (no spaces or slashes)."
        )
    if len(name) > 50:
        raise ValueError("Class name too long (max 50 chars).")

    current = load_label_classes()
    if name in current:
        raise ValueError(f"Class {name!r} already exists.")
    new_list = current + [name]

    # Preserve any existing keys (path, train, val, test, download, ...)
    # by round-tripping through yaml; only replace `names`. Comments are
    # lost in the round trip — acceptable trade-off for not pulling in
    # ruamel.yaml just for this.
    if _LABEL_VOCAB_PATH.exists():
        with _LABEL_VOCAB_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    else:
        # Mirror what _ensure_dataset_yaml writes on first training run
        # so the trainer can still consume the file.
        dataset_root = (Path.cwd() / "data" / "ecoli").resolve()
        data = {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
        }

    data["names"] = {i: n for i, n in enumerate(new_list)}

    _LABEL_VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LABEL_VOCAB_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)

    log.info("Added label class %r. Vocabulary: %s", name, new_list)
    return new_list


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: Optional[DatasetStore] = None


def init_store(root: Path) -> DatasetStore:
    global _store
    root.mkdir(parents=True, exist_ok=True)
    _store = DatasetStore(root)
    log.info("DatasetStore initialised at %s", root.resolve())
    return _store


def get_store() -> DatasetStore:
    if _store is None:
        raise RuntimeError("DatasetStore not initialised.")
    return _store
