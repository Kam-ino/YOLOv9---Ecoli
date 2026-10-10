"""
scripts/model_report_figures.py
===============================
Ultralytics-style report figures for either detector, computed by ONE
script so the two models' plots are comparable: training curves, confusion
matrix (counts and normalised), and F1 / precision / recall vs confidence.

    python scripts/model_report_figures.py --algo yolov9 --weights models/best_yolov9c.pt --run runs/train/deploy_yolov9_r2
    python scripts/model_report_figures.py --algo rfdetr --weights models/best_rfdetr.pth --run runs/train/deploy_rfdetr_r2

Evaluation split: ``--val-dir`` (default: the val split of the dataset the
model was trained with). Conventions follow Ultralytics' validator so the
YOLO figures match its own: whole-image inference, confusion matrix at
conf >= 0.25 and IoU >= 0.45 (rows = predicted, columns = true, normalised
by column), curves swept over confidence with matches at IoU >= 0.5.

Output: <out>/<algo>/{results,confusion_matrix,confusion_matrix_normalized,
F1_curve,P_curve,R_curve}.png
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.compare_eval import iou_matrix  # noqa: E402
from src.inference import RFDETRDetector, YOLOv9Detector  # noqa: E402
from training.dataset_view import IMG_EXTS, label_of, read_labels  # noqa: E402

LABEL = {"yolov9": "YOLOv9-c", "rfdetr": "RF-DETR"}


def collect(det, val_dir: Path, tiled: bool):
    """Per image: GT (n,5 xyxy+cls) and detections (m,6 xyxy+score+cls)."""
    data = []
    for img in sorted(p for p in val_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        frame = cv2.imread(str(img))
        h, w = frame.shape[:2]
        rows = read_labels(label_of(img))
        gt = np.array([[(r[1] - r[3] / 2) * w, (r[2] - r[4] / 2) * h, (r[1] + r[3] / 2) * w, (r[2] + r[4] / 2) * h, r[0]]
                       for r in rows], float).reshape(-1, 5)
        dets = det.predict(frame, tiled=tiled)
        dt = np.array([[*d.bbox, d.confidence, d.class_id] for d in dets], float).reshape(-1, 6)
        data.append((gt, dt))
    return data


def xyxy_to_xywh(b):
    return np.column_stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]) if len(b) else np.zeros((0, 4))


def confusion(data, nc, conf=0.25, iou_thr=0.45):
    """Ultralytics ConfusionMatrix: m[pred, true]; last index = background."""
    m = np.zeros((nc + 1, nc + 1), int)
    for gt, dt in data:
        dt = dt[dt[:, 4] >= conf]
        gc = gt[:, 4].astype(int) if len(gt) else np.zeros(0, int)
        if len(gt) == 0:
            for c in dt[:, 5].astype(int):
                m[c, nc] += 1
            continue
        if len(dt) == 0:
            for c in gc:
                m[nc, c] += 1
            continue
        iou = iou_matrix(xyxy_to_xywh(gt[:, :4]), xyxy_to_xywh(dt[:, :4]))
        x = np.argwhere(iou > iou_thr)
        if len(x):
            matches = np.concatenate([x, iou[x[:, 0], x[:, 1]][:, None]], 1)
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))
        m0, m1 = matches[:, 0].astype(int), matches[:, 1].astype(int)
        dc = dt[:, 5].astype(int)
        for i, g in enumerate(gc):
            j = m0 == i
            if j.any():
                m[dc[m1[j][0]], g] += 1
            else:
                m[nc, g] += 1
        for j, c in enumerate(dc):
            if not (m1 == j).any():
                m[c, nc] += 1
    return m


def pr_curves(data, nc, iou_thr=0.5, n_pts=1000):
    """Per class P/R/F1 over a confidence grid, Ultralytics-style (IoU >= 0.5 greedy by score)."""
    x = np.linspace(0, 1, n_pts)
    tp_conf = {c: [] for c in range(nc)}       # (conf, is_tp) per detection
    n_gt = np.zeros(nc, int)
    for gt, dt in data:
        for c in range(nc):
            g = gt[gt[:, 4] == c]
            d = dt[dt[:, 5] == c]
            d = d[d[:, 4].argsort()[::-1]]
            n_gt[c] += len(g)
            used = np.zeros(len(g), bool)
            iou = iou_matrix(xyxy_to_xywh(d[:, :4]), xyxy_to_xywh(g[:, :4])) if len(d) and len(g) else None
            for i in range(len(d)):
                hit = False
                if iou is not None:
                    cand = iou[i].copy()
                    cand[used] = 0
                    j = int(cand.argmax())
                    if cand[j] >= iou_thr:
                        used[j] = True
                        hit = True
                tp_conf[c].append((d[i, 4], hit))
    P, R = np.zeros((nc, n_pts)), np.zeros((nc, n_pts))
    for c in range(nc):
        arr = np.array(tp_conf[c], float).reshape(-1, 2)
        for k, t in enumerate(x):
            sel = arr[arr[:, 0] >= t]
            tp = sel[:, 1].sum() if len(sel) else 0.0
            P[c, k] = tp / len(sel) if len(sel) else 1.0
            R[c, k] = tp / n_gt[c] if n_gt[c] else 0.0
    F1 = 2 * P * R / (P + R + 1e-16)
    return x, P, R, F1


def plot_curve(x, Y, names, ylabel, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, n in enumerate(names):
        ax.plot(x, Y[i], lw=1, label=n)
    y = Y.mean(0)
    k = int(y.argmax())
    lab = f"all classes {y.max():.2f} at {x[k]:.3f}" if ylabel != "Recall" else f"all classes {y[0]:.2f} at 0.000"
    if ylabel == "Precision":
        k = int(np.argmax(y >= 0.999)) if (y >= 0.999).any() else int(y.argmax())
        lab = f"all classes {y[k]:.2f} at {x[k]:.3f}"
    ax.plot(x, y, lw=3, color="blue", label=lab)
    ax.set_xlabel("Confidence")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(m, names, path, title, normalize):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = names + ["background"]
    arr = m.astype(float)
    if normalize:
        arr = arr / np.maximum(arr.sum(0, keepdims=True), 1)
    arr_show = arr.copy()
    arr_show[-1, -1] = np.nan
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(np.nan_to_num(arr_show), cmap="Blues", vmin=0, vmax=np.nanmax(arr_show) if np.nanmax(arr_show) > 0 else 1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j == len(labels) - 1 or arr[i, j] == 0:
                continue
            ax.text(j, i, f"{arr[i, j]:.2f}" if normalize else f"{int(arr[i, j])}", ha="center", va="center",
                    color="white" if arr_show[i, j] > 0.6 * np.nanmax(arr_show) else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training(run: Path, algo: str, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader((run / ("results.csv" if algo == "yolov9" else "metrics.csv")).open(encoding="utf-8")))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    if algo == "yolov9":
        keys = ["train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/precision(B)", "metrics/recall(B)",
                "val/box_loss", "val/cls_loss", "val/dfl_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
        ep = [float(r["epoch"]) for r in rows]
        series = {k: (ep, [float(r[k]) for r in rows]) for k in keys}
    else:
        keys = ["train/loss", "train/loss_ce", "train/loss_bbox", "train/loss_giou", "train/lr",
                "val/precision", "val/recall", "val/F1", "val/ema_mAP_50", "val/ema_mAP_50_95"]
        series = {}
        for k in keys:
            pts = [(float(r["epoch"]), float(r[k])) for r in rows if r.get(k) not in (None, "")]
            series[k] = ([p[0] for p in pts], [p[1] for p in pts])
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, k in zip(axes.flat, keys):
        xs, ys = series[k]
        ax.plot(xs, ys, marker=".", lw=1, label="results")
        if len(ys) > 5:
            w = max(3, len(ys) // 10)
            sm = np.convolve(ys, np.ones(w) / w, mode="valid")
            ax.plot(xs[w - 1:], sm, ":", lw=2, label="smooth")
        ax.set_title(k, fontsize=11)
    axes.flat[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--algo", required=True, choices=("yolov9", "rfdetr"))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--run", type=Path, required=True, help="Training run dir (results.csv / metrics.csv).")
    ap.add_argument("--val-dir", type=Path, default=ROOT / "data" / "ecoli_x8_r2d" / "images" / "val")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "rfdetr_vs_yolov9" / "figures" / "latest")
    ap.add_argument("--tiled", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    names = ["ecoli", "ecoli_cluster"]
    out = a.out / a.algo
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{LABEL[a.algo]} (latest, {a.run.name}) — val split {a.val_dir.parent.parent.name}"

    if a.algo == "yolov9":
        det = YOLOv9Detector(a.weights, imgsz=a.imgsz, conf_threshold=0.001, iou_threshold=0.7, class_names=names,
                             max_det=1200, warmup=False)
    else:
        det = RFDETRDetector(a.weights, resolution=a.imgsz, num_queries=1200, conf_threshold=0.001, class_names=names,
                             warmup=False)
    data = collect(det, a.val_dir, a.tiled)
    m = confusion(data, len(names))
    plot_confusion(m, names, out / "confusion_matrix.png", f"Confusion Matrix — {tag}", normalize=False)
    plot_confusion(m, names, out / "confusion_matrix_normalized.png", f"Confusion Matrix Normalized — {tag}", normalize=True)
    x, P, R, F1 = pr_curves(data, len(names))
    plot_curve(x, F1, names, "F1", out / "F1_curve.png", f"F1-Confidence Curve — {tag}")
    plot_curve(x, P, names, "Precision", out / "P_curve.png", f"Precision-Confidence Curve — {tag}")
    plot_curve(x, R, names, "Recall", out / "R_curve.png", f"Recall-Confidence Curve — {tag}")
    plot_training(a.run, a.algo, out / "results.png", f"Training curves — {tag}")
    f1 = F1.mean(0)
    print(f"{LABEL[a.algo]}: {len(data)} val images; confusion (pred x true):\n{m}\n"
          f"best F1 {f1.max():.3f} at conf {x[int(f1.argmax())]:.3f}; figures in {out}")


if __name__ == "__main__":
    main()
