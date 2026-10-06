"""
scripts/compare_report.py
=========================
Assemble experiments/rfdetr_vs_yolov9/RESULTS.md from the JSON the other
compare_* scripts leave in runs/compare/. Pure formatting — no numbers are
computed here.

    python scripts/compare_report.py [--preds yolov9 rfdetr]
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "runs" / "compare"
EXP_DIR = ROOT / "experiments" / "rfdetr_vs_yolov9"
LABEL = {"yolov9": "YOLOv9-c", "rfdetr": "RF-DETR"}


def load(name: str):
    p = OUT_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def pm(d: dict, nd=3):
    return f"{d['mean']:.{nd}f} ± {d['std']:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", nargs="+", default=["yolov9", "rfdetr"])
    a = ap.parse_args()

    plan = json.loads((EXP_DIR / "folds.json").read_text(encoding="utf-8"))
    metrics = {t: load(f"metrics_{t}.json") for t in a.preds}
    metas = {t: load(f"preds_{t}.meta.json") for t in a.preds}
    boot = load(f"bootstrap_{a.preds[0]}_vs_{a.preds[1]}.json") if len(a.preds) == 2 else None
    crowd = load("crowding_summary.json")
    bench = load("bench.json")
    missing = [t for t, m in metrics.items() if m is None]
    if missing:
        sys.exit(f"missing metrics for {missing}: run compare_eval.py first")

    L = []
    L.append(f"# RF-DETR vs YOLOv9-c on E. coli — results\n")
    L.append(f"Generated {date.today()} by `scripts/compare_report.py`. Design, provenance and caveats: `NOTES.md`.\n")
    L.append(f"{plan['folds']}-fold cross-validation over {plan['n_images']} pooled images "
             f"({plan['n_boxes']} boxes), seed {plan['seed']}; every image scored once by a model that never saw it; "
             f"thresholds and early stopping chosen on a {plan['inner_val']}-image inner val per fold. "
             f"One evaluator (pycocotools) for both models.\n")

    # ---- main table -----------------------------------------------------------
    L.append("## Detection quality (pooled held-out, 69 images)\n")
    L.append("| model | mAP50 | mAP50-95 | AP_small (<32²px) | P@τ* | R@τ* | F1@τ* | fold mAP50 (mean ± sd) | fold mAP50-95 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for t, m in metrics.items():
        p, s, fm = m["pooled"], m["prf_at_tau_star"], m["fold_mean_std"]
        L.append(f"| {LABEL.get(t, t)} | {f(p['mAP50'])} | {f(p['mAP'])} | {f(p['AP_small'])} | "
                 f"{f(s['P'])} | {f(s['R'])} | {f(s['F1'])} | {pm(fm['mAP50'])} | {pm(fm['mAP'])} |")
    L.append("")
    L.append("τ* = F1-maximising confidence on each fold's inner val, applied to that fold's held-out images "
             "(per-fold values: " + "; ".join(
                 f"{LABEL.get(t, t)} " + ", ".join(f"{k}:{v:.2f}" for k, v in m["prf_at_tau_star"]["tau_per_fold"].items())
                 for t, m in metrics.items()) + ").\n")

    if boot:
        d, d50, w = boot["delta_mAP"], boot["delta_mAP50"], boot["wilcoxon_per_image_F1"]
        L.append("### Paired bootstrap (1,000 image resamples)\n")
        L.append(f"Δ = {LABEL.get(boot['b'], boot['b'])} − {LABEL.get(boot['a'], boot['a'])}\n")
        L.append("| metric | Δ | 95 % CI | P(Δ ≤ 0) |")
        L.append("|---|---|---|---|")
        L.append(f"| mAP50-95 | {d['delta']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | {d['p_delta_le_0']:.3f} |")
        L.append(f"| mAP50 | {d50['delta']:+.4f} | [{d50['ci95'][0]:+.4f}, {d50['ci95'][1]:+.4f}] | {d50['p_delta_le_0']:.3f} |")
        if "p_value" in w:
            L.append(f"\nWilcoxon signed-rank on per-image F1@τ* (n = {w['n']}): mean Δ = {w['mean_delta_F1']:+.3f}, p = {w['p_value']:.4f}.\n")

    # ---- secondary tables -------------------------------------------------------
    L.append("### Fixed thresholds, classes, box sizes\n")
    L.append("| model | P/R/F1 @0.25 | P/R/F1 @0.50 | AP ecoli | AP ecoli_cluster | AP tercile S/M/L (GT area) |")
    L.append("|---|---|---|---|---|---|")
    for t, m in metrics.items():
        fx, pc, st = m["prf_fixed"], m["pooled"]["per_class"], m["pooled"]["size_terciles"]
        pr = lambda d: f"{d['P']:.2f}/{d['R']:.2f}/{d['F1']:.2f}"
        L.append(f"| {LABEL.get(t, t)} | {pr(fx['0.25'])} | {pr(fx['0.5'])} | "
                 f"{f(pc.get('ecoli', {}).get('mAP'))} | {f(pc.get('ecoli_cluster', {}).get('mAP'))} | "
                 f"{f(st['AP_tercile_s'])} / {f(st['AP_tercile_m'])} / {f(st['AP_tercile_l'])} |")
    st = next(iter(metrics.values()))["pooled"]["size_terciles"]["edges_px2"]
    L.append(f"\nGT-area tercile edges: {st[0]:.0f} and {st[1]:.0f} px².\n")

    L.append("### Per fold\n")
    L.append("| fold | n | " + " | ".join(f"{LABEL.get(t, t)} mAP50 / mAP50-95 / F1@τ*" for t in metrics) + " |")
    L.append("|---|---|" + "---|" * len(metrics))
    for k in range(plan["folds"]):
        cells = []
        for m in metrics.values():
            r = m["per_fold"][k]
            cells.append(f"{r['mAP50']:.3f} / {r['mAP']:.3f} / {r['F1']:.3f}")
        L.append(f"| {k} | {metrics[a.preds[0]]['per_fold'][k]['n_images']} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- speed ---------------------------------------------------------------------
    if bench:
        L.append(f"## Speed and size ({bench['gpu']}, batch 1, {bench['dtype']}, {bench['imgsz']} px, "
                 f"{bench['passes']}×{bench['n_images']} images, post-processing included)\n")
        L.append("| model | params (M) | GFLOPs | latency ms (mean ± sd) | median ms | FPS |")
        L.append("|---|---|---|---|---|---|")
        for t, b in bench["models"].items():
            g = b.get("gflops_flop_counter") or b.get("gflops_ultralytics")
            L.append(f"| {LABEL.get(t, t)} | {f(b.get('params_M'), 1)} | {f(g, 1)} | "
                     f"{b['mean_ms']:.1f} ± {b['std_ms']:.1f} | {b['median_ms']:.1f} | {b['fps']:.1f} |")
        notes = [f"{LABEL.get(t, t)}: {b['fp16_note']}" for t, b in bench["models"].items() if "fp16_note" in b]
        L.append(("\n" + "; ".join(notes) + "\n") if notes else "")

    # ---- crowding ------------------------------------------------------------------
    if crowd:
        L.append("## Crowding analysis\n")
        n = crowd["n_gt_by_bin"]
        L.append("Crowding of a GT box = max IoU with any other GT box in its image. "
                 f"{crowd['gt_descriptors_summary']['share_crowd_gt0']:.0%} of boxes touch another box; "
                 f"{crowd['gt_descriptors_summary']['share_heavy']:.1%} overlap heavily (IoU > 0.5).\n")
        L.append("| recall @τ* | " + " | ".join(f"{b} (n={n[b]})" for b in n) + " | duplicate rate |")
        L.append("|---|" + "---|" * (len(n) + 1))
        for t, r in crowd["tags"].items():
            L.append(f"| {LABEL.get(t, t)} | " + " | ".join(f"{r['recall_by_crowd_bin'][b]:.3f}" for b in n)
                     + f" | {r['duplicate_rate']:.2%} |")
        rb = crowd.get("recall_bootstrap")
        if rb:
            L.append(f"| Δ ({rb['direction']}) 95 % CI | " + " | ".join(
                f"{rb[b]['delta']:+.3f} [{rb[b]['delta_ci95'][0]:+.3f}, {rb[b]['delta_ci95'][1]:+.3f}]" for b in n) + " | |")
        L.append("\nDuplicate rate = kept predictions at τ* overlapping a higher-scoring same-class prediction with IoU > 0.7 "
                 "(the cost of skipping NMS).\n")
        L.append("| images by GT count | " + " | ".join(f"{LABEL.get(t, t)} R / mAP50-95 / FP per img" for t in crowd["tags"]) + " |")
        L.append("|---|" + "---|" * len(crowd["tags"]))
        for q in ("Q1", "Q2", "Q3", "Q4"):
            cells = [f"{r['image_bins'][q]['R']:.3f} / {r['image_bins'][q]['mAP']:.3f} / {r['image_bins'][q]['fp_per_image']:.1f}"
                     if q in r["image_bins"] else "–" for r in crowd["tags"].values()]
            L.append(f"| {q} | " + " | ".join(cells) + " |")
        e = crowd["image_bin_edges"]
        L.append(f"\nQuartile edges on GT boxes per image: {e[0]:.0f}, {e[1]:.0f}, {e[2]:.0f}.\n")
        if crowd.get("nms_sweep"):
            L.append("### YOLOv9 NMS IoU sweep (threshold re-chosen on inner val each time)\n")
            L.append("| NMS IoU | recall heavy | recall overlapping | recall isolated | duplicate rate |")
            L.append("|---|---|---|---|---|")
            for s in crowd["nms_sweep"]:
                r = s["recall_by_crowd_bin"]
                L.append(f"| {s['nms_iou']:.2f} | {r['heavy']:.3f} | {r['overlapping']:.3f} | {r['isolated']:.3f} | {s['duplicate_rate']:.2%} |")
            L.append("")
        L.append("Figures: `figures/recall_by_crowding_bin.png`, `figures/recall_by_gt_count_quartile.png`, "
                 "`figures/heavy_recall_vs_nms_iou.png`, `figures/qualitative_dense.png`.\n")

    # ---- provenance --------------------------------------------------------------------
    L.append("## Provenance\n")
    for t, m in metas.items():
        if m:
            L.append(f"- {LABEL.get(t, t)}: versions {m.get('versions')}; weights md5 per fold: "
                     + ", ".join(f"{k}:{v['md5'][:8]}" for k, v in m["folds"].items())
                     + f"; conf {m['args']['conf']}, max_det {m['args']['max_det']}, nms_iou {m['args']['nms_iou']}, tiled {m['args']['tiled']}.")
    L.append(f"- Fold plan `folds.json` (seed {plan['seed']}); label snapshot `label_hashes.txt`.")
    L.append("- Thesis reference (not comparable: 9-image val split, pre-relabel GT, Ultralytics validator): "
             "YOLOv9-c mAP50 0.499 / mAP50-95 0.280.\n")

    (EXP_DIR / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {EXP_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
