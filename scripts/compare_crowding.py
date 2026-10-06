"""
scripts/compare_crowding.py
===========================
The paper's core question: does the NMS-free detector keep finding bacteria
as they crowd together?

    python scripts/compare_crowding.py --preds yolov9 rfdetr [--nms-sweep]

Per GT box: ``crowd`` = max IoU with any other GT box in the same image
(0 = isolated), plus its image's GT count, boxes per megapixel and the
nearest-neighbour centre distance over the box diagonal.

Bins: isolated (crowd = 0) · touching (0 < crowd ≤ 0.2) · overlapping
(0.2–0.5) · heavy (> 0.5); image bins by GT-count quartile.

Per bin and model: recall at τ*_k (compare_eval's greedy matcher and the
per-fold inner-val thresholds), a paired bootstrap over images for the
recall difference, stratified mAP per image bin, and the flip side of
NMS-free decoding: the duplicate rate (same-class predictions at τ* with
IoU > 0.7 to a higher-scoring kept prediction) and false positives per
image bin.

``--nms-sweep`` evaluates every ``preds_yolov9_nms<iou>.json`` produced by
``compare_predict.py --nms-iou …`` (threshold re-chosen on inner val each
time) → heavy-bin recall vs NMS IoU. If RF-DETR's edge shrinks as the IoU
rises, the finding is "NMS threshold", not "NMS-free".

Outputs runs/compare/crowding_<tag>.json, crowding_summary.json and PNGs
under experiments/rfdetr_vs_yolov9/figures/.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compare_eval import (  # noqa: E402
    OUT_DIR, EXP_DIR, ap_of, best_tau, evaluate, group, iou_matrix, match_image, prf,
)

FIG_DIR = EXP_DIR / "figures"
CROWD_BINS = [("isolated", 0.0, 0.0), ("touching", 0.0, 0.2), ("overlapping", 0.2, 0.5), ("heavy", 0.5, 1.01)]
DUP_IOU = 0.7


# ---------------------------------------------------------------------------
# Crowding descriptors
# ---------------------------------------------------------------------------

def crowd_bin(c: float) -> str:
    if c <= 0:
        return "isolated"
    for name, lo, hi in CROWD_BINS[1:]:
        if lo < c <= hi:
            return name
    return "heavy"


def describe_gt(gt: dict) -> dict:
    """ann id → {crowd, bin, n_gt, density, nn_dist}."""
    by_img = group(gt["annotations"])
    size = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    out = {}
    for img_id, anns in by_img.items():
        b = np.array([a["bbox"] for a in anns], dtype=float)
        n = len(anns)
        ious = iou_matrix(b, b) if n > 1 else np.zeros((1, 1))
        np.fill_diagonal(ious, 0)
        cx, cy = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
        diag = np.hypot(b[:, 2], b[:, 3])
        d = np.hypot(cx[:, None] - cx[None], cy[:, None] - cy[None])
        np.fill_diagonal(d, np.inf)
        w, h = size[img_id]
        for i, a in enumerate(anns):
            c = float(ious[i].max()) if n > 1 else 0.0
            out[a["id"]] = {"image_id": img_id, "crowd": c, "bin": crowd_bin(c), "n_gt": n,
                            "density": n / (w * h / 1e6),
                            "nn_dist": float(d[i].min() / diag[i]) if n > 1 else float("inf")}
    return out


def image_bins(gt: dict) -> tuple:
    """Quartile bins of GT count per image → (bin label per image id, edges)."""
    counts = {im["id"]: 0 for im in gt["images"]}
    for a in gt["annotations"]:
        counts[a["image_id"]] += 1
    vals = np.array(list(counts.values()))
    q = np.percentile(vals, [25, 50, 75])
    labels = {}
    for i, n in counts.items():
        labels[i] = "Q1" if n <= q[0] else "Q2" if n <= q[1] else "Q3" if n <= q[2] else "Q4"
    return labels, [float(x) for x in q], counts


# ---------------------------------------------------------------------------
# Per-tag analysis
# ---------------------------------------------------------------------------

def load_tau(tag: str, gt: dict, plan: dict) -> dict:
    """τ*_k per fold: from metrics_<tag>.json if present, else re-chosen on inner val."""
    mp = OUT_DIR / f"metrics_{tag}.json"
    if mp.exists():
        return {int(k): v for k, v in json.loads(mp.read_text(encoding="utf-8"))["prf_at_tau_star"]["tau_per_fold"].items()}
    taus = {}
    for k in range(plan["folds"]):
        vgt = json.loads((OUT_DIR / f"gt_valid_{k}.json").read_text(encoding="utf-8"))
        vdt = json.loads((OUT_DIR / f"preds_{tag}_valid_{k}.json").read_text(encoding="utf-8"))
        taus[k] = best_tau(group(vgt["annotations"]), group(vdt), [im["id"] for im in vgt["images"]])
    return taus


def analyse(tag: str, gt: dict, plan: dict, desc: dict, img_bin: dict) -> dict:
    preds = json.loads((OUT_DIR / f"preds_{tag}.json").read_text(encoding="utf-8"))
    taus = load_tau(tag, gt, plan)
    fold_of = {im["id"]: im["fold"] for im in gt["images"]}
    gt_by_img, dt_by_img = group(gt["annotations"]), group(preds)
    img_ids = sorted(fold_of)

    bins = [b[0] for b in CROWD_BINS]
    # per-image counts: matched / total GT per crowding bin, FP, duplicates, kept
    per_img = {i: {"matched": dict.fromkeys(bins, 0), "total": dict.fromkeys(bins, 0),
                   "fp": 0, "kept": 0, "dup": 0, "tp": 0, "fn": 0} for i in img_ids}
    for i in img_ids:
        tau = taus[fold_of[i]]
        gts, dets = gt_by_img.get(i, []), dt_by_img.get(i, [])
        tp, fp, fn, matched = match_image(gts, dets, tau)
        rec = per_img[i]
        rec.update(fp=fp, tp=tp, fn=fn)
        for g in gts:
            b = desc[g["id"]]["bin"]
            rec["total"][b] += 1
            if g["id"] in matched:
                rec["matched"][b] += 1
        kept = sorted((d for d in dets if d["score"] >= tau), key=lambda d: -d["score"])
        rec["kept"] = len(kept)
        if len(kept) > 1:
            kb = np.array([d["bbox"] for d in kept], dtype=float)
            kc = np.array([d["category_id"] for d in kept])
            ious = iou_matrix(kb, kb)
            same = kc[:, None] == kc[None]
            upper = np.triu(np.ones_like(ious, dtype=bool), 1)       # j > i: i is higher-scoring
            dup = ((ious > DUP_IOU) & same & upper).any(axis=0)       # j duplicates some earlier i
            rec["dup"] = int(dup.sum())

    def recall_by_bin(ids):
        return {b: (sum(per_img[i]["matched"][b] for i in ids) / max(1, sum(per_img[i]["total"][b] for i in ids)))
                for b in bins}

    out = {"tag": tag, "tau_per_fold": taus, "recall_by_crowd_bin": recall_by_bin(img_ids),
           "n_gt_by_crowd_bin": {b: sum(per_img[i]["total"][b] for i in img_ids) for b in bins},
           "image_bins": {}, "duplicate_rate": (sum(r["dup"] for r in per_img.values())
                                                / max(1, sum(r["kept"] for r in per_img.values()))),
           "per_image": per_img}
    for q in ("Q1", "Q2", "Q3", "Q4"):
        ids = [i for i in img_ids if img_bin[i] == q]
        if not ids:
            continue
        tp = sum(per_img[i]["tp"] for i in ids); fp = sum(per_img[i]["fp"] for i in ids)
        fn = sum(per_img[i]["fn"] for i in ids)
        E = evaluate(gt, [p for p in preds if p["image_id"] in set(ids)], img_ids=ids)
        out["image_bins"][q] = {"n_images": len(ids), "mAP": ap_of(E, 0), "mAP50": ap_of(E, 0, 0),
                                **prf(tp, fp, fn), "fp_per_image": fp / len(ids),
                                "dup_per_image": sum(per_img[i]["dup"] for i in ids) / len(ids)}
    return out


def bootstrap_recall(res_a: dict, res_b: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap over images of recall-by-bin for a and b (and their difference)."""
    ids = sorted(res_a["per_image"])
    bins = [b[0] for b in CROWD_BINS]
    ma = np.array([[res_a["per_image"][i]["matched"][b] for b in bins] for i in ids], float)
    mb = np.array([[res_b["per_image"][i]["matched"][b] for b in bins] for i in ids], float)
    tot = np.array([[res_a["per_image"][i]["total"][b] for b in bins] for i in ids], float)
    rng = np.random.default_rng(seed)
    ra, rb = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        t = tot[idx].sum(0)
        ra.append(ma[idx].sum(0) / np.maximum(t, 1))
        rb.append(mb[idx].sum(0) / np.maximum(t, 1))
    ra, rb = np.array(ra), np.array(rb)
    d = rb - ra
    out = {}
    for j, b in enumerate(bins):
        out[b] = {"a_ci95": [float(np.percentile(ra[:, j], 2.5)), float(np.percentile(ra[:, j], 97.5))],
                  "b_ci95": [float(np.percentile(rb[:, j], 2.5)), float(np.percentile(rb[:, j], 97.5))],
                  "delta": float(d[:, j].mean()),
                  "delta_ci95": [float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))]}
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_recall_bins(results: dict, boot: dict | None, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = [b[0] for b in CROWD_BINS]
    tags = list(results)
    x = np.arange(len(bins))
    w = 0.8 / len(tags)
    fig, ax = plt.subplots(figsize=(7, 4))
    for t, tag in enumerate(tags):
        r = [results[tag]["recall_by_crowd_bin"][b] for b in bins]
        err = None
        if boot and len(tags) == 2:
            key = "a_ci95" if t == 0 else "b_ci95"
            lo = [r[j] - boot[b][key][0] for j, b in enumerate(bins)]
            hi = [boot[b][key][1] - r[j] for j, b in enumerate(bins)]
            err = [np.maximum(lo, 0), np.maximum(hi, 0)]
        ax.bar(x + (t - (len(tags) - 1) / 2) * w, r, w, yerr=err, capsize=3, label=tag)
    n = results[tags[0]]["n_gt_by_crowd_bin"]
    ax.set_xticks(x, [f"{b}\n(n={n[b]})" for b in bins])
    ax.set_ylabel("recall at τ* (IoU ≥ 0.5)")
    ax.set_ylim(0, 1)
    ax.set_title("Recall by GT crowding (max IoU with another GT box)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_image_bins(results: dict, edges, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qs = ["Q1", "Q2", "Q3", "Q4"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, lab in zip(axes, ("R", "mAP"), ("recall at τ*", "mAP50-95")):
        for tag, r in results.items():
            ax.plot(qs, [r["image_bins"].get(q, {}).get(key, np.nan) for q in qs], marker="o", label=tag)
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"images by GT count quartile (edges {', '.join(f'{e:.0f}' for e in edges)})")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_nms_sweep(sweep: list, ref: dict | None, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ious = [s["nms_iou"] for s in sweep]
    for b in ("heavy", "overlapping", "isolated"):
        ax.plot(ious, [s["recall_by_crowd_bin"][b] for s in sweep], marker="o", label=f"YOLOv9 · {b}")
    if ref:
        for b, ls in (("heavy", "--"), ("overlapping", ":")):
            ax.axhline(ref["recall_by_crowd_bin"][b], ls=ls, color="k", label=f"RF-DETR · {b}")
    ax.set_xlabel("YOLOv9 NMS IoU threshold")
    ax.set_ylabel("recall at τ*")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_qualitative(gt: dict, tags: list, results: dict, counts: dict, plan: dict, path: Path, n: int = 6) -> None:
    """GT | model A | model B boxes on the n densest held-out images."""
    meta = {m["file_name"]: m for m in plan["images"]}
    by_id = {im["id"]: im for im in gt["images"]}
    dense = sorted(counts, key=lambda i: -counts[i])[:n]
    gt_by_img = group(gt["annotations"])
    preds = {t: group(json.loads((OUT_DIR / f"preds_{t}.json").read_text(encoding="utf-8"))) for t in tags}
    cols = ["GT"] + tags
    rows = []
    for i in dense:
        im = by_id[i]
        src = ROOT / "data" / "ecoli" / "images" / meta[im["file_name"]]["origin_split"] / im["file_name"]
        base = cv2.imread(str(src))
        scale = 640 / base.shape[1]
        panels = []
        for c in cols:
            img = cv2.resize(base, None, fx=scale, fy=scale)
            if c == "GT":
                boxes, color = [g["bbox"] for g in gt_by_img.get(i, [])], (0, 255, 0)
            else:
                tau = results[c]["tau_per_fold"][im["fold"]]
                boxes, color = [d["bbox"] for d in preds[c].get(i, []) if d["score"] >= tau], (0, 128, 255)
            for x, y, w, h in boxes:
                cv2.rectangle(img, (int(x * scale), int(y * scale)),
                              (int((x + w) * scale), int((y + h) * scale)), color, 1)
            cv2.putText(img, f"{c}: {len(boxes)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"{c}: {len(boxes)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
            panels.append(img)
        h = min(p.shape[0] for p in panels)
        rows.append(np.concatenate([p[:h] for p in panels], axis=1))
    w = min(r.shape[1] for r in rows)
    cv2.imwrite(str(path), np.concatenate([r[:, :w] for r in rows], axis=0))


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", nargs="+", required=True, help="Tags, e.g. yolov9 rfdetr.")
    ap.add_argument("--nms-sweep", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    gt = json.loads((OUT_DIR / "gt_all.json").read_text(encoding="utf-8"))
    plan = json.loads((EXP_DIR / "folds.json").read_text(encoding="utf-8"))
    desc = describe_gt(gt)
    img_bin, edges, counts = image_bins(gt)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for tag in a.preds:
        r = analyse(tag, gt, plan, desc, img_bin)
        results[tag] = r
        (OUT_DIR / f"crowding_{tag}.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
        print(f"{tag}: recall by crowd bin {json.dumps({k: round(v, 3) for k, v in r['recall_by_crowd_bin'].items()})}"
              f"  duplicate rate {r['duplicate_rate']:.3%}")

    summary = {"crowd_bins": CROWD_BINS, "n_gt_by_bin": results[a.preds[0]]["n_gt_by_crowd_bin"],
               "image_bin_edges": edges, "gt_descriptors_summary": {
                   "crowd_mean": float(np.mean([d["crowd"] for d in desc.values()])),
                   "share_crowd_gt0": float(np.mean([d["crowd"] > 0 for d in desc.values()])),
                   "share_heavy": float(np.mean([d["bin"] == "heavy" for d in desc.values()]))},
               "tags": {t: {k: v for k, v in r.items() if k != "per_image"} for t, r in results.items()}}

    boot = None
    if len(a.preds) == 2:
        ta, tb = a.preds
        boot = bootstrap_recall(results[ta], results[tb], a.n_boot)
        summary["recall_bootstrap"] = {"direction": f"{tb} minus {ta}", **boot}
        for b, v in boot.items():
            print(f"  delta recall[{b}] = {v['delta']:+.3f}  95% CI {v['delta_ci95']}")

    if a.nms_sweep:
        sweep = []
        for p in sorted(OUT_DIR.glob("preds_yolov9_nms*.json")):
            if "_valid_" in p.name or p.name.endswith(".meta.json"):
                continue
            tag = p.stem[len("preds_"):]
            iou = float(tag.split("nms")[1])
            r = analyse(tag, gt, plan, desc, img_bin)
            m = json.loads((OUT_DIR / f"preds_{tag}.meta.json").read_text(encoding="utf-8")) if (OUT_DIR / f"preds_{tag}.meta.json").exists() else {}
            sweep.append({"tag": tag, "nms_iou": iou, "recall_by_crowd_bin": r["recall_by_crowd_bin"],
                          "tau_per_fold": r["tau_per_fold"], "duplicate_rate": r["duplicate_rate"],
                          "image_bins": {q: {k: v["image_bins"][q][k] for k in ("mAP", "mAP50", "P", "R", "F1")}
                                         for q, v in [(q, r) for q in r["image_bins"]]},
                          "weights_md5": {k: v.get("md5") for k, v in m.get("folds", {}).items()}})
        if "yolov9" in results:
            sweep.append({"tag": "yolov9", "nms_iou": 0.7, "recall_by_crowd_bin": results["yolov9"]["recall_by_crowd_bin"],
                          "tau_per_fold": results["yolov9"]["tau_per_fold"],
                          "duplicate_rate": results["yolov9"]["duplicate_rate"], "image_bins": {}})
        sweep.sort(key=lambda s: s["nms_iou"])
        summary["nms_sweep"] = sweep
        if sweep:
            fig_nms_sweep(sweep, results.get("rfdetr"), FIG_DIR / "heavy_recall_vs_nms_iou.png")
            print("NMS sweep (heavy-bin recall):", {s["nms_iou"]: round(s["recall_by_crowd_bin"]["heavy"], 3) for s in sweep})

    fig_recall_bins(results, boot, FIG_DIR / "recall_by_crowding_bin.png")
    fig_image_bins(results, edges, FIG_DIR / "recall_by_gt_count_quartile.png")
    fig_qualitative(gt, a.preds, results, counts, plan, FIG_DIR / "qualitative_dense.png")
    (OUT_DIR / "crowding_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'crowding_summary.json'} and figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
