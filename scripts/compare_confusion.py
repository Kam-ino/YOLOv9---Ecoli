"""
scripts/compare_confusion.py
============================
Confusion matrices for both detectors under the one shared protocol:
cross-validated held-out predictions, human labels, original orientation,
each fold's inner-val threshold. Matching is class-agnostic (IoU >= 0.5,
highest score first) so a cell that is found but called the wrong class
lands in an off-diagonal cell instead of counting as miss + false alarm.
Rows = ground truth (+ "background" = false positives), columns = prediction
(+ "background" = missed cells), like Ultralytics' confusion_matrix.png.

    python scripts/compare_confusion.py --preds yolov9 rfdetr
    -> experiments/rfdetr_vs_yolov9/figures/confusion_matrices.png (+ .json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.compare_eval import OUT_DIR, EXP_DIR, group, iou_matrix  # noqa: E402

LABEL = {"yolov9": "YOLOv9-c", "rfdetr": "RF-DETR"}


def confusion(gt_by_img, dt_by_img, taus, fold_of, n_cls):
    m = np.zeros((n_cls + 1, n_cls + 1), dtype=int)      # [gt, pred]; index n_cls = background
    for img_id, fold in fold_of.items():
        gts = gt_by_img.get(img_id, [])
        dets = sorted((d for d in dt_by_img.get(img_id, []) if d["score"] >= taus[fold]), key=lambda d: -d["score"])
        used = np.zeros(len(gts), dtype=bool)
        if gts and dets:
            ious = iou_matrix(np.array([d["bbox"] for d in dets], float), np.array([g["bbox"] for g in gts], float))
        for i, d in enumerate(dets):
            if gts:
                cand = ious[i].copy()
                cand[used] = 0
                j = int(cand.argmax())
                if cand[j] >= 0.5:
                    used[j] = True
                    m[gts[j]["category_id"], d["category_id"]] += 1
                    continue
            m[n_cls, d["category_id"]] += 1                 # false positive
        for j, g in enumerate(gts):
            if not used[j]:
                m[g["category_id"], n_cls] += 1               # missed
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", nargs="+", default=["yolov9", "rfdetr"])
    a = ap.parse_args()
    gt = json.loads((OUT_DIR / "gt_all.json").read_text(encoding="utf-8"))
    names = [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])] + ["background"]
    fold_of = {im["id"]: im["fold"] for im in gt["images"]}
    gt_by = group(gt["annotations"])
    mats = {}
    for tag in a.preds:
        taus = {int(k): v for k, v in json.loads((OUT_DIR / f"metrics_{tag}.json").read_text(encoding="utf-8"))
                ["prf_at_tau_star"]["tau_per_fold"].items()}
        dt_by = group(json.loads((OUT_DIR / f"preds_{tag}.json").read_text(encoding="utf-8")))
        mats[tag] = confusion(gt_by, dt_by, taus, fold_of, len(names) - 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(mats), figsize=(5.2 * len(mats), 4.6))
    for ax, (tag, m) in zip(np.atleast_1d(axes), mats.items()):
        row_tot = m.sum(1, keepdims=True)
        norm = np.where(row_tot > 0, m / np.maximum(row_tot, 1), 0)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if i == m.shape[0] - 1 and j == m.shape[1] - 1:
                    continue
                ax.text(j, i, f"{m[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(names)), names, rotation=20)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("ground truth")
        tp = sum(m[i, i] for i in range(len(names) - 1))
        ax.set_title(f"{LABEL.get(tag, tag)} — held-out, human labels, τ* per fold\n"
                     f"recall {tp / max(m[:-1].sum(), 1):.2f} · precision {tp / max(m[:, :-1].sum(), 1):.2f}", fontsize=10)
    fig.tight_layout()
    out = EXP_DIR / "figures" / "confusion_matrices.png"
    fig.savefig(out, dpi=150)
    (out.with_suffix(".json")).write_text(json.dumps({t: {"labels": names, "matrix": m.tolist()} for t, m in mats.items()}, indent=1),
                                         encoding="utf-8")
    for tag, m in mats.items():
        print(LABEL.get(tag, tag), "rows=GT, cols=pred, labels", names)
        print(m)
    print("wrote", out)


if __name__ == "__main__":
    main()
