"""
scripts/compare_folds.py
========================
Build the k-fold cross-validation plan shared by the YOLOv9 / RF-DETR
comparison (experiments/rfdetr_vs_yolov9).

Pools every labelled image (train + val + test) and assigns each one to
exactly one held-out fold, like ``scripts/crossval.py``'s ``make_folds``
(seeded shuffle, round-robin within image domain) with two additions:

* near-duplicate images (mean |grey diff| at 128 px below DUP_THRESHOLD,
  the thesis's "different view" boundary) travel together, so a frame never
  trains a fold whose twin is held out;
* each fold gets an *inner* validation set (``--inner-val`` images drawn
  from the other folds, domain-balanced) for early stopping and threshold
  selection, so the held-out fold is never used for tuning.

Outputs:
  experiments/rfdetr_vs_yolov9/folds.json         the plan + dup groups + dataset stats
  experiments/rfdetr_vs_yolov9/label_hashes.txt   sha256 of every label file (GT snapshot)
  runs/compare/folds/fold<k>/{train,valid,test}/… hard-linked views + data.yaml
  runs/compare/gt_all.json                        COCO GT over all images (image_id = 1-based sorted index)
  runs/compare/gt_valid_<k>.json                  COCO GT of each fold's inner val

Usage (from the repo root):
    python scripts/compare_folds.py [--folds 5] [--inner-val 6] [--seed 0]
"""
import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.dataset_view import IMG_EXTS, label_of, make_view, names_of_yaml  # noqa: E402

# scripts/merge_duplicate_labels.py calibrates this metric as "<4 = same
# frame, 15+ = different view". Distinct frames of the same microscope setup
# score 5-10 here (uniform background), so only the "same frame" band merges;
# anything under BORDERLINE is listed in folds.json for the notes.
DUP_THRESHOLD = 4.0
BORDERLINE = 8.0
EXP_DIR = ROOT / "experiments" / "rfdetr_vs_yolov9"
OUT_DIR = ROOT / "runs" / "compare"


def domain_of(name: str) -> str:
    """Same rule as the thesis scripts: microscope captures vs stained slides."""
    return "microscope" if "microscope" in name.lower() else "stained"


def load_labels(path: Path) -> np.ndarray:
    """YOLO txt → (n, 5) float array [cls, cx, cy, w, h]; empty file → (0, 5)."""
    if not path.exists():
        return np.zeros((0, 5), dtype=float)
    rows = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return np.array(rows, dtype=float).reshape(-1, 5) if rows else np.zeros((0, 5), dtype=float)


def dup_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def thumb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)


def dup_groups(images, threshold=DUP_THRESHOLD):
    """Union-find over pairs scoring below ``threshold``. Returns (groups, pairs)."""
    thumbs = {p: thumb(p) for p in images}
    parent = {p: p for p in images}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    pairs = []            # everything under BORDERLINE, merged only if under threshold
    for i, a in enumerate(images):
        for b in images[i + 1:]:
            s = dup_score(thumbs[a], thumbs[b])
            if s < max(threshold, BORDERLINE):
                pairs.append((a, b, s))
            if s < threshold:
                parent[find(a)] = find(b)
    groups = defaultdict(list)
    for p in images:
        groups[find(p)].append(p)
    return [sorted(g) for g in groups.values()], pairs


def make_folds_grouped(groups, k: int, seed: int):
    """Round-robin within each domain over dup *groups* (crossval.make_folds, grouped)."""
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for dom in ("microscope", "stained"):
        gs = sorted((g for g in groups if domain_of(g[0].name) == dom), key=lambda g: g[0].name)
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            folds[i % k].extend(g)
    return [sorted(f) for f in folds]


def pick_inner_val(groups, exclude, n: int, seed: int):
    """``n`` images (whole dup groups, alternating domains) from groups not in ``exclude``."""
    rng = random.Random(seed)
    pool = [g for g in groups if not (set(g) & exclude)]
    by_dom = {d: [g for g in pool if domain_of(g[0].name) == d] for d in ("microscope", "stained")}
    for gs in by_dom.values():
        rng.shuffle(gs)
    chosen, turn = [], 0
    while len(chosen) < n and any(by_dom.values()):
        dom = ("microscope", "stained")[turn % 2]
        turn += 1
        if by_dom[dom]:
            chosen.extend(by_dom[dom].pop())
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ecoli")
    ap.add_argument("--names", type=Path, default=ROOT / "training" / "dataset.yaml")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-val", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-from", type=Path, default=None,
                    help="Expanded dataset (scripts/make_pseudo_dataset.py output, images/<split>/<stem>__d4-k.png). "
                         "Training splits then use all 8 variants + its labels; inner val and held-out images stay "
                         "original with human labels.")
    a = ap.parse_args()

    names = names_of_yaml(a.names)
    images = sorted(
        (p for s in ("train", "val", "test") for p in (a.data / "images" / s).glob("*")
         if p.suffix.lower() in IMG_EXTS),
        key=lambda p: p.name,
    )
    assert len({p.name for p in images}) == len(images), "duplicate file names across splits"

    groups, pairs = dup_groups(images)
    folds = make_folds_grouped(groups, a.folds, a.seed)
    plan = []
    for k, held in enumerate(folds):
        inner = pick_inner_val(groups, set(held), a.inner_val, a.seed * 100 + k)
        train = sorted(p for p in images if p not in held and p not in inner)
        plan.append({"test": held, "valid": inner, "train": train})

    # --- views ------------------------------------------------------------
    def expanded_variants(p: Path):
        """The 8 orientation files of ``p`` inside --train-from (same split folder)."""
        folder = a.train_from / "images" / p.parent.name
        vs = sorted(folder.glob(f"{p.stem}__d4-*{p.suffix}"))
        assert len(vs) == 8, (p.name, len(vs), folder)
        return vs

    import shutil
    for k, sp in enumerate(plan):
        view = {"test": sp["test"], "valid": sp["valid"], "train": sp["train"]}
        if a.train_from:
            view["train"] = [v for p in sp["train"] for v in expanded_variants(p)]
        view_dir = OUT_DIR / "folds" / f"fold{k}"
        shutil.rmtree(view_dir, ignore_errors=True)       # views are links/derived files; rebuild clean
        out_yaml = make_view(view, view_dir, names)
        if a.train_from:
            import yaml
            d = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
            d["expanded"] = True
            d["train_from"] = a.train_from.resolve().as_posix()
            out_yaml.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")

    # --- COCO ground truth over the pooled set ------------------------------
    fold_of = {p: k for k, f in enumerate(folds) for p in f}
    inner_of = defaultdict(list)
    for k, sp in enumerate(plan):
        for p in sp["valid"]:
            inner_of[p].append(k)
    coco_images, coco_anns, meta = [], [], []
    # pycocotools stores the matched GT id in dtMatches and reads 0 as "no
    # match", so both image and annotation ids are 1-based.
    ann_id, total_lines = 1, 0
    for img_id, p in enumerate(images, start=1):
        h, w = cv2.imread(str(p)).shape[:2]
        lab = load_labels(label_of(p))
        total_lines += len(lab)
        for cls, cx, cy, bw, bh in lab:
            x, y, pw, ph = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
            assert -1 <= x <= w and -1 <= y <= h and pw > 0 and ph > 0, (p.name, cls, cx, cy, bw, bh)
            coco_anns.append({"id": ann_id, "image_id": img_id, "category_id": int(cls),
                              "bbox": [round(x, 2), round(y, 2), round(pw, 2), round(ph, 2)],
                              "area": round(pw * ph, 2), "iscrowd": 0})
            ann_id += 1
        entry = {"id": img_id, "file_name": p.name, "width": w, "height": h,
                 "fold": fold_of[p], "domain": domain_of(p.name), "origin_split": p.parent.name}
        coco_images.append(entry)
        meta.append({**entry, "inner_val_of": inner_of.get(p, []), "n_boxes": int(len(lab)),
                     "label_sha256": hashlib.sha256(label_of(p).read_bytes()).hexdigest()
                     if label_of(p).exists() else None})
    assert len(coco_anns) == total_lines
    categories = [{"id": int(i), "name": n, "supercategory": "bacteria"} for i, n in sorted(names.items())]
    gt = {"images": coco_images, "annotations": coco_anns, "categories": categories}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gt_all.json").write_text(json.dumps(gt), encoding="utf-8")
    for k, sp in enumerate(plan):
        ids = {images.index(p) + 1 for p in sp["valid"]}
        sub = {"images": [i for i in coco_images if i["id"] in ids],
               "annotations": [x for x in coco_anns if x["image_id"] in ids],
               "categories": categories}
        (OUT_DIR / f"gt_valid_{k}.json").write_text(json.dumps(sub), encoding="utf-8")

    # --- plan + provenance ----------------------------------------------------
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    assert sorted((p for f in folds for p in f), key=lambda p: p.name) == images, "image missing from folds"
    for k, sp in enumerate(plan):
        assert not (set(sp["test"]) & set(sp["valid"])) and not (set(sp["test"]) & set(sp["train"]))
        assert not (set(sp["valid"]) & set(sp["train"]))
        assert len(sp["test"]) + len(sp["valid"]) + len(sp["train"]) == len(images)
    out = {
        "seed": a.seed, "folds": a.folds, "inner_val": a.inner_val, "dup_threshold": DUP_THRESHOLD,
        "train_from": a.train_from.resolve().as_posix() if a.train_from else None,
        "n_images": len(images), "n_boxes": len(coco_anns), "categories": names,
        "fold_sizes": [{"test": len(sp["test"]), "valid": len(sp["valid"]), "train": len(sp["train"])}
                       for sp in plan],
        "dup_groups": [[p.name for p in g] for g in groups if len(g) > 1],
        "dup_pairs": [{"a": x.name, "b": y.name, "score": round(s, 2)} for x, y, s in pairs],
        "images": meta,
    }
    (EXP_DIR / "folds.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (EXP_DIR / "label_hashes.txt").write_text(
        "".join(f"{m['label_sha256']}  labels/{m['origin_split']}/{Path(m['file_name']).stem}.txt\n"
                for m in meta), encoding="utf-8")

    # --- summary ----------------------------------------------------------------
    boxes = np.array([m["n_boxes"] for m in meta])
    sides = np.array([min(x["bbox"][2], x["bbox"][3]) for x in coco_anns])
    print(f"{len(images)} images, {len(coco_anns)} boxes; GT/image max {boxes.max()} "
          f"p95 {np.percentile(boxes, 95):.0f} median {np.median(boxes):.0f}; "
          f"box short side px median {np.median(sides):.1f} p10 {np.percentile(sides, 10):.1f} "
          f"p90 {np.percentile(sides, 90):.1f}; {sum(s.item() < 32 for s in sides) / len(sides):.1%} of boxes < 32 px")
    print(f"dup groups (<{DUP_THRESHOLD}): {out['dup_groups']}")
    for k, sp in enumerate(plan):
        dom = lambda ps: sum(domain_of(p.name) == "microscope" for p in ps)
        nb = lambda ps: sum(len(load_labels(label_of(p))) for p in ps)
        print(f"fold {k}: test {len(sp['test'])} imgs ({dom(sp['test'])} micro, {nb(sp['test'])} boxes) | "
              f"inner val {len(sp['valid'])} ({dom(sp['valid'])} micro, {nb(sp['valid'])} boxes) | "
              f"train {len(sp['train'])} ({nb(sp['train'])} boxes)")
    print(f"wrote {EXP_DIR / 'folds.json'}, {OUT_DIR / 'gt_all.json'}, views under {OUT_DIR / 'folds'}")


if __name__ == "__main__":
    main()
