"""
scripts/merge_expanded_into_data.py
===================================
Fold the expanded, pseudo-labelled dataset (``data/ecoli_x8``) back into the
app's dataset (``data/ecoli``), and freeze the human-only labels first.

    python scripts/merge_expanded_into_data.py

1. ``data/ecoli_human/``: the 69 original images (hard links) with their
   current human-only labels (copies) + data.yaml. This is the ground truth
   the comparison scripts keep using (scripts/compare_folds.py --data).
2. ``data/ecoli/``: adds the 7 rotated/mirrored variants of every image
   (``<stem>__d4-1..7``) with their labels, and replaces each original's
   label file with the ``__d4-0`` version (human + pseudo boxes). Label files
   are written as new files (temp + rename), so hard-linked copies in
   runs/ keep their old content.

Idempotent: re-running overwrites the same files.
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.dataset_view import IMG_EXTS, D4_SUFFIX, link  # noqa: E402


def replace_file(src: Path, dst: Path) -> None:
    """Copy src over dst via a temp file + rename, giving dst a fresh inode."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".tmp")
    os.close(fd)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ecoli")
    ap.add_argument("--expanded", type=Path, default=ROOT / "data" / "ecoli_x8")
    ap.add_argument("--human", type=Path, default=ROOT / "data" / "ecoli_human")
    ap.add_argument("--names", type=Path, default=ROOT / "training" / "dataset.yaml")
    a = ap.parse_args()

    names = yaml.safe_load(a.names.read_text(encoding="utf-8"))["names"]
    originals = {s: sorted(p for p in (a.data / "images" / s).glob("*")
                           if p.suffix.lower() in IMG_EXTS and D4_SUFFIX not in p.stem)
                 for s in ("train", "val", "test")}

    # 1. Freeze human-only labels.
    n_frozen = 0
    for split, imgs in originals.items():
        for img in imgs:
            link(img, a.human / "images" / split / img.name)
            lab = a.data / "labels" / split / f"{img.stem}.txt"
            if lab.exists():
                replace_file(lab, a.human / "labels" / split / lab.name)
                n_frozen += 1
    (a.human / "data.yaml").write_text(yaml.safe_dump({
        "path": a.human.resolve().as_posix(), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": names, "note": "human-only labels frozen by scripts/merge_expanded_into_data.py"},
        sort_keys=False), encoding="utf-8")

    # 2. Merge variants + pseudo-labelled originals into data/ecoli.
    n_new, n_relabelled = 0, 0
    for split, imgs in originals.items():
        for img in imgs:
            for k in range(8):
                src_img = a.expanded / "images" / split / f"{img.stem}{D4_SUFFIX}{k}{img.suffix}"
                src_lab = a.expanded / "labels" / split / f"{img.stem}{D4_SUFFIX}{k}.txt"
                assert src_img.exists() and src_lab.exists(), src_img
                if k == 0:
                    replace_file(src_lab, a.data / "labels" / split / f"{img.stem}.txt")
                    n_relabelled += 1
                else:
                    dst_img = a.data / "images" / split / src_img.name
                    if not dst_img.exists():
                        shutil.copyfile(src_img, dst_img)
                        n_new += 1
                    replace_file(src_lab, a.data / "labels" / split / src_lab.name)
    total = sum(1 for s in ("train", "val", "test") for p in (a.data / "images" / s).glob("*") if p.suffix.lower() in IMG_EXTS)
    print(f"froze {n_frozen} human label files in {a.human}; added {n_new} variant images and "
          f"relabelled {n_relabelled} originals in {a.data}; {a.data} now holds {total} images")


if __name__ == "__main__":
    main()
