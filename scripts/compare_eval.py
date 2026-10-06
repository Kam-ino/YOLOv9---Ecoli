"""
scripts/compare_eval.py
=======================
ONE evaluator for both detectors, so the paper never compares numbers from
two frameworks' validators.

    python scripts/compare_eval.py --preds yolov9 rfdetr      # tags from compare_predict.py
    python scripts/compare_eval.py --selftest                  # GT as predictions → mAP 1.0

Per tag (runs/compare/metrics_<tag>.json):
  * pycocotools bbox metrics on the pooled held-out predictions (every image
    predicted once by a model that never saw it): mAP50-95, mAP50, AP by
    COCO size and by GT-area tercile, AR, per-class AP;
  * the same per fold → mean ± std across folds;
  * precision / recall / F1 at τ*_k, the F1-maximising threshold chosen on
    fold k's *inner val*, applied to fold k's held-out images (plus fixed
    τ = 0.25 and 0.5 for reference). Matching is greedy, class-aware,
    IoU ≥ 0.5, highest score first.

With exactly two tags also runs/compare/bootstrap_<a>_vs_<b>.json: a paired
bootstrap over the pooled images (1,000 resamples, duplicates counted with
multiplicity) → 95 % CI on ΔmAP50-95 and ΔmAP50, and a Wilcoxon test on
per-image F1 at τ*. Bootstrapping re-uses pycocotools' per-image match
tables (``evalImgs``), so a resample costs milliseconds, not a full re-eval.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs" / "compare"
EXP_DIR = ROOT / "experiments" / "rfdetr_vs_yolov9"
MAX_DETS = 1200
MATCH_IOU = 0.5
FIXED_TAUS = (0.25, 0.5)


# ---------------------------------------------------------------------------
# pycocotools helpers
# ---------------------------------------------------------------------------

def load_coco(gt: dict) -> COCO:
    coco = COCO()
    coco.dataset = json.loads(json.dumps(gt))      # pycocotools mutates; give it a copy
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def evaluate(gt: dict, results: list, img_ids=None, area_rng=None) -> COCOeval:
    """Run COCOeval.evaluate() (+ accumulate) silently. ``results`` may be empty."""
    coco_gt = load_coco(gt)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results) if results else load_coco({**gt, "annotations": []})
        E = COCOeval(coco_gt, coco_dt, "bbox")
        E.params.maxDets = [1, 10, MAX_DETS]
        if img_ids is not None:
            E.params.imgIds = sorted(img_ids)
        if area_rng is not None:
            E.params.areaRng, E.params.areaRngLbl = area_rng
        E.evaluate()
        E.accumulate()
    return E


def ap_of(E: COCOeval, area: int = 0, iou_idx=None) -> float:
    """Mean precision over recall thresholds and classes for one area range."""
    p = E.eval["precision"][:, :, :, area, -1]        # [T, R, K] at the largest maxDet
    if iou_idx is not None:
        p = p[iou_idx]
    p = p[p > -1]
    return float(p.mean()) if p.size else float("nan")


def ar_of(E: COCOeval, area: int = 0) -> float:
    r = E.eval["recall"][:, :, area, -1]
    r = r[r > -1]
    return float(r.mean()) if r.size else float("nan")


def summarize(E: COCOeval, cat_names) -> dict:
    out = {
        "mAP": ap_of(E, 0), "mAP50": ap_of(E, 0, 0), "mAP75": ap_of(E, 0, 5),
        "AP_small": ap_of(E, 1), "AP_medium": ap_of(E, 2), "AP_large": ap_of(E, 3),
        "AR": ar_of(E, 0), "per_class": {},
    }
    for k, cid in enumerate(E.params.catIds):
        p = E.eval["precision"][:, :, k, 0, -1]
        valid = p > -1
        out["per_class"][cat_names[cid]] = {
            "mAP": float(p[valid].mean()) if valid.any() else float("nan"),
            "mAP50": float(p[0][p[0] > -1].mean()) if (p[0] > -1).any() else float("nan"),
        }
    return out


def accumulate_subset(E: COCOeval, img_idx, max_det: int = MAX_DETS) -> np.ndarray:
    """pycocotools' accumulate() restricted to image *indices* (repeats allowed).

    Returns precision[T, R, K] for area 'all' at ``max_det``. Mirrors
    COCOeval.accumulate exactly (checked by ``--selftest``), with the
    precision envelope vectorised.
    """
    p = E.params
    T, R, K, A, I = len(p.iouThrs), len(p.recThrs), len(p.catIds), len(p.areaRng), len(p.imgIds)
    precision = -np.ones((T, R, K))
    a = 0                                               # area 'all'
    for k in range(K):
        Es = [E.evalImgs[k * A * I + a * I + i] for i in img_idx]
        Es = [e for e in Es if e is not None]
        if not Es:
            continue
        dt_scores = np.concatenate([e["dtScores"][:max_det] for e in Es])
        order = np.argsort(-dt_scores, kind="mergesort")
        dtm = np.concatenate([e["dtMatches"][:, :max_det] for e in Es], axis=1)[:, order]
        dt_ig = np.concatenate([e["dtIgnore"][:, :max_det] for e in Es], axis=1)[:, order]
        gt_ig = np.concatenate([e["gtIgnore"] for e in Es])
        npig = int(np.count_nonzero(gt_ig == 0))
        if npig == 0:
            continue
        tps = np.logical_and(dtm, np.logical_not(dt_ig))
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dt_ig))
        tp_sum = np.cumsum(tps, axis=1).astype(float)
        fp_sum = np.cumsum(fps, axis=1).astype(float)
        for t in range(T):
            tp, fp = tp_sum[t], fp_sum[t]
            rc = tp / npig
            pr = tp / (fp + tp + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1]        # precision envelope
            inds = np.searchsorted(rc, p.recThrs, side="left")
            q = np.zeros(R)
            ok = inds < len(pr)
            q[ok] = pr[inds[ok]]
            precision[t, :, k] = q
    return precision


def map_from_precision(prec: np.ndarray) -> tuple:
    """(mAP50-95, mAP50) from a precision[T, R, K] array."""
    v = prec[prec > -1]
    v50 = prec[0][prec[0] > -1]
    return (float(v.mean()) if v.size else float("nan"),
            float(v50.mean()) if v50.size else float("nan"))


# ---------------------------------------------------------------------------
# Greedy matching → precision / recall / F1
# ---------------------------------------------------------------------------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,4) xywh, b: (m,4) xywh → (n, m) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def match_image(gts: list, dets: list, tau: float, iou_thr: float = MATCH_IOU):
    """Greedy class-aware matching at score ≥ tau. Returns (tp, fp, fn, matched_gt_ids)."""
    dets = sorted((d for d in dets if d["score"] >= tau), key=lambda d: -d["score"])
    if not gts:
        return 0, len(dets), 0, set()
    gb = np.array([g["bbox"] for g in gts], dtype=float)
    gc = np.array([g["category_id"] for g in gts])
    used = np.zeros(len(gts), dtype=bool)
    tp = 0
    matched = set()
    if dets:
        db = np.array([d["bbox"] for d in dets], dtype=float)
        ious = iou_matrix(db, gb)
        for i, d in enumerate(dets):
            cand = ious[i].copy()
            cand[(gc != d["category_id"]) | used] = 0
            j = int(cand.argmax())
            if cand[j] >= iou_thr:
                used[j] = True
                tp += 1
                matched.add(gts[j]["id"])
    fp = len(dets) - tp
    fn = int((~used).sum())
    return tp, fp, fn, matched


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"P": p, "R": r, "F1": f, "TP": tp, "FP": fp, "FN": fn}


def group(items: list, key="image_id") -> dict:
    out = {}
    for it in items:
        out.setdefault(it[key], []).append(it)
    return out


def best_tau(gt_by_img: dict, dt_by_img: dict, img_ids) -> float:
    """F1-maximising threshold over a 0.01 grid (ties → higher threshold)."""
    best = (-1.0, 0.25)
    for tau in np.round(np.arange(0.01, 0.96, 0.01), 2):
        tp = fp = fn = 0
        for i in img_ids:
            a, b, c, _ = match_image(gt_by_img.get(i, []), dt_by_img.get(i, []), tau)
            tp, fp, fn = tp + a, fp + b, fn + c
        f = prf(tp, fp, fn)["F1"]
        if f >= best[0]:
            best = (f, float(tau))
    return best[1]


# ---------------------------------------------------------------------------
# Per-tag evaluation
# ---------------------------------------------------------------------------

def tercile_area_rng(gt: dict):
    areas = np.array([a["area"] for a in gt["annotations"]])
    e1, e2 = np.percentile(areas, [100 / 3, 200 / 3])
    return ([[0, 1e10], [0, e1], [e1, e2], [e2, 1e10]], ["all", "tercile_s", "tercile_m", "tercile_l"]), (e1, e2)


def evaluate_tag(tag: str, gt: dict, plan: dict) -> dict:
    preds = json.loads((OUT_DIR / f"preds_{tag}.json").read_text(encoding="utf-8"))
    cat_names = {c["id"]: c["name"] for c in gt["categories"]}
    fold_of = {img["id"]: img["fold"] for img in gt["images"]}
    n_folds = plan["folds"]

    E = evaluate(gt, preds)
    pooled = summarize(E, cat_names)
    rng, edges = tercile_area_rng(gt)
    Et = evaluate(gt, preds, area_rng=rng)
    pooled["size_terciles"] = {"edges_px2": [float(edges[0]), float(edges[1])],
                               "AP_tercile_s": ap_of(Et, 1), "AP_tercile_m": ap_of(Et, 2), "AP_tercile_l": ap_of(Et, 3)}

    gt_by_img = group(gt["annotations"])
    dt_by_img = group(preds)
    per_fold, taus = [], {}
    pooled_tp = pooled_fp = pooled_fn = 0
    per_image_f1 = {}
    for k in range(n_folds):
        ids = [i for i, f in fold_of.items() if f == k]
        vgt = json.loads((OUT_DIR / f"gt_valid_{k}.json").read_text(encoding="utf-8"))
        vdt = json.loads((OUT_DIR / f"preds_{tag}_valid_{k}.json").read_text(encoding="utf-8"))
        tau = best_tau(group(vgt["annotations"]), group(vdt), [im["id"] for im in vgt["images"]])
        taus[k] = tau
        tp = fp = fn = 0
        for i in ids:
            a, b, c, _ = match_image(gt_by_img.get(i, []), dt_by_img.get(i, []), tau)
            tp, fp, fn = tp + a, fp + b, fn + c
            per_image_f1[i] = prf(a, b, c)["F1"]
        pooled_tp, pooled_fp, pooled_fn = pooled_tp + tp, pooled_fp + fp, pooled_fn + fn
        Ek = evaluate(gt, [p for p in preds if p["image_id"] in set(ids)], img_ids=ids)
        per_fold.append({"fold": k, "n_images": len(ids), "tau": tau,
                         "mAP": ap_of(Ek, 0), "mAP50": ap_of(Ek, 0, 0), **prf(tp, fp, fn)})

    fixed = {}
    for tau in FIXED_TAUS:
        tp = fp = fn = 0
        for i in fold_of:
            a, b, c, _ = match_image(gt_by_img.get(i, []), dt_by_img.get(i, []), tau)
            tp, fp, fn = tp + a, fp + b, fn + c
        fixed[str(tau)] = prf(tp, fp, fn)

    def mean_std(key):
        v = np.array([f[key] for f in per_fold], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

    return {
        "tag": tag, "n_images": len(fold_of), "n_predictions": len(preds),
        "pooled": pooled,
        "prf_at_tau_star": {"tau_per_fold": taus, **prf(pooled_tp, pooled_fp, pooled_fn)},
        "prf_fixed": fixed,
        "per_fold": per_fold,
        "fold_mean_std": {k: mean_std(k) for k in ("mAP", "mAP50", "P", "R", "F1")},
        "per_image_f1": per_image_f1,
    }


# ---------------------------------------------------------------------------
# Paired bootstrap between two tags
# ---------------------------------------------------------------------------

def bootstrap(gt: dict, tag_a: str, tag_b: str, metrics_a: dict, metrics_b: dict,
              n_boot: int = 1000, seed: int = 0) -> dict:
    Ea = evaluate(gt, json.loads((OUT_DIR / f"preds_{tag_a}.json").read_text(encoding="utf-8")))
    Eb = evaluate(gt, json.loads((OUT_DIR / f"preds_{tag_b}.json").read_text(encoding="utf-8")))
    assert Ea.params.imgIds == Eb.params.imgIds
    n = len(Ea.params.imgIds)
    # Self-check: the subset accumulator reproduces pycocotools on the full set.
    for E in (Ea, Eb):
        full = accumulate_subset(E, range(n))
        ref = E.eval["precision"][:, :, :, 0, -1]
        assert np.allclose(full, ref, atol=1e-9), "accumulate_subset drifted from pycocotools"

    rng = np.random.default_rng(seed)
    deltas, deltas50 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ma, ma50 = map_from_precision(accumulate_subset(Ea, idx))
        mb, mb50 = map_from_precision(accumulate_subset(Eb, idx))
        deltas.append(mb - ma)
        deltas50.append(mb50 - ma50)
    deltas, deltas50 = np.array(deltas), np.array(deltas50)

    fa, fb = metrics_a["per_image_f1"], metrics_b["per_image_f1"]
    ids = sorted(set(fa) & set(fb))
    xa, xb = np.array([fa[i] for i in ids]), np.array([fb[i] for i in ids])
    try:
        from scipy.stats import wilcoxon
        diff = xb - xa
        wil = {"p_value": float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
               "n": len(ids), "mean_delta_F1": float(diff.mean())}
    except Exception as exc:
        wil = {"error": repr(exc)}

    def ci(d):
        return {"delta": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "p_delta_le_0": float((d <= 0).mean())}

    return {"a": tag_a, "b": tag_b, "direction": f"{tag_b} minus {tag_a}", "n_boot": n_boot, "seed": seed,
            "n_images": n, "point": {"mAP": [metrics_a["pooled"]["mAP"], metrics_b["pooled"]["mAP"]],
                                     "mAP50": [metrics_a["pooled"]["mAP50"], metrics_b["pooled"]["mAP50"]]},
            "delta_mAP": ci(deltas), "delta_mAP50": ci(deltas50), "wilcoxon_per_image_F1": wil}


# ---------------------------------------------------------------------------

def md_row(m: dict) -> str:
    p, t, f = m["pooled"], m["prf_at_tau_star"], m["fold_mean_std"]
    return (f"| {m['tag']} | {p['mAP50']:.3f} | {p['mAP']:.3f} | {p['AP_small']:.3f} | "
            f"{t['P']:.3f} | {t['R']:.3f} | {t['F1']:.3f} | "
            f"{f['mAP50']['mean']:.3f} ± {f['mAP50']['std']:.3f} | {f['mAP']['mean']:.3f} ± {f['mAP']['std']:.3f} |")


def selftest(gt: dict, plan: dict) -> None:
    """Feed the GT back as predictions → every metric must be 1.0."""
    preds = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"],
              "score": 1.0, "fold": 0} for a in gt["annotations"]]
    (OUT_DIR / "preds_selftest.json").write_text(json.dumps(preds), encoding="utf-8")
    for k in range(plan["folds"]):
        vgt = json.loads((OUT_DIR / f"gt_valid_{k}.json").read_text(encoding="utf-8"))
        vp = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
              for a in vgt["annotations"]]
        (OUT_DIR / f"preds_selftest_valid_{k}.json").write_text(json.dumps(vp), encoding="utf-8")
    m = evaluate_tag("selftest", gt, plan)
    assert abs(m["pooled"]["mAP"] - 1.0) < 1e-6 and abs(m["pooled"]["mAP50"] - 1.0) < 1e-6, m["pooled"]
    assert m["prf_at_tau_star"]["F1"] == 1.0 and all(f["mAP"] > 0.999 for f in m["per_fold"])
    E = evaluate(gt, preds)
    full = accumulate_subset(E, range(len(E.params.imgIds)))
    assert np.allclose(full, E.eval["precision"][:, :, :, 0, -1], atol=1e-9)
    b = bootstrap(gt, "selftest", "selftest", m, m, n_boot=20)
    assert b["delta_mAP"]["delta"] == 0.0
    for f in OUT_DIR.glob("preds_selftest*"):
        f.unlink()
    print("compare_eval selftest ok (mAP=1.0, accumulate_subset == pycocotools, bootstrap delta =0)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", nargs="*", default=[], help="Tags from compare_predict.py.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    gt = json.loads((OUT_DIR / "gt_all.json").read_text(encoding="utf-8"))
    plan = json.loads((EXP_DIR / "folds.json").read_text(encoding="utf-8"))
    if a.selftest:
        selftest(gt, plan)
        return

    metrics = {}
    print("| model | mAP50 | mAP50-95 | AP_small | P@tau* | R@tau* | F1@tau* | fold mAP50 | fold mAP50-95 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tag in a.preds:
        m = evaluate_tag(tag, gt, plan)
        metrics[tag] = m
        (OUT_DIR / f"metrics_{tag}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
        print(md_row(m), flush=True)

    if len(a.preds) == 2 and not a.no_bootstrap:
        ta, tb = a.preds
        b = bootstrap(gt, ta, tb, metrics[ta], metrics[tb], n_boot=a.n_boot)
        (OUT_DIR / f"bootstrap_{ta}_vs_{tb}.json").write_text(json.dumps(b, indent=1), encoding="utf-8")
        print(f"\ndelta mAP50-95 ({tb} - {ta}) = {b['delta_mAP']['delta']:+.4f}  95% CI {b['delta_mAP']['ci95']}  "
              f"P(Δ≤0)={b['delta_mAP']['p_delta_le_0']:.3f}")
        print(f"delta mAP50     ({tb} - {ta}) = {b['delta_mAP50']['delta']:+.4f}  95% CI {b['delta_mAP50']['ci95']}")
        print(f"Wilcoxon per-image F1: {b['wilcoxon_per_image_F1']}")


if __name__ == "__main__":
    main()
