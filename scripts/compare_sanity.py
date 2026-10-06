"""
scripts/compare_sanity.py
=========================
Draw ground-truth boxes *from the COCO json* (not the source labels) on a
few random images of each split of one fold, so a conversion bug shows up
as a visibly wrong box. Eyeball runs/compare/sanity/<split>/ afterwards.

    python scripts/compare_sanity.py [--fold 0] [--n 10] [--seed 0]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "runs" / "compare"
EXP_DIR = ROOT / "experiments" / "rfdetr_vs_yolov9"

COLORS = {0: (0, 255, 0), 1: (0, 128, 255)}


def group(items: list, key: str = "image_id") -> dict:
    out: dict = {}
    for it in items:
        out.setdefault(it[key], []).append(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    gt = json.loads((OUT_DIR / "gt_all.json").read_text(encoding="utf-8"))
    plan = json.loads((EXP_DIR / "folds.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in gt["categories"]}
    anns = group(gt["annotations"])
    meta = {m["file_name"]: m for m in plan["images"]}
    rng = random.Random(a.seed)
    for split in ("train", "valid", "test"):
        folder = OUT_DIR / "folds" / f"fold{a.fold}" / split / "images"
        files = sorted(p.name for p in folder.iterdir())
        out = OUT_DIR / "sanity" / split
        out.mkdir(parents=True, exist_ok=True)
        for name in rng.sample(files, min(a.n, len(files))):
            img = cv2.imread(str(folder / name))
            im = next((i for i in gt["images"] if i["file_name"] == name), None)
            if im is not None:                      # original image: boxes from the COCO json
                assert img.shape[1] == im["width"] and img.shape[0] == im["height"], (name, img.shape, im)
                boxes = [(x["bbox"], x["category_id"]) for x in anns.get(im["id"], [])]
                tag = f"id={im['id']} dom={meta[name]['domain']}"
            else:                                   # expanded/pseudo-labelled variant: boxes from its txt
                h, w = img.shape[:2]
                boxes = []
                for ln in (folder.parent / "labels" / (Path(name).stem + ".txt")).read_text(encoding="utf-8").splitlines():
                    c, cx, cy, bw, bh = (float(v) for v in ln.split())
                    boxes.append(([(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h], int(c)))
                tag = "expanded (human + pseudo labels)"
            for (bx, by, bw, bh), cid in boxes:
                cv2.rectangle(img, (int(bx), int(by)), (int(bx + bw), int(by + bh)), COLORS.get(cid, (255, 0, 255)), 1)
            label = f"{split} fold{a.fold} {tag} {len(boxes)} boxes " \
                    f"{' '.join(f'{names[k]}={c}' for k, c in COLORS.items() if k in names)}"
            cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imwrite(str(out / name), img)
        print(f"{split}: {min(a.n, len(files))} images -> {out}")


if __name__ == "__main__":
    main()
