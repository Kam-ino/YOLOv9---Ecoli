# RF-DETR vs YOLOv9-c on E. coli — results

Generated 2026-10-08 by `scripts/compare_report.py`. Design, provenance and caveats: `NOTES.md`.

5-fold cross-validation over 69 pooled images (5937 boxes), seed 0; every image scored once by a model that never saw it; thresholds and early stopping chosen on a 6-image inner val per fold. One evaluator (pycocotools) for both models.

## Detection quality (pooled held-out, 69 images)

| model | mAP50 | mAP50-95 | AP_small (<32²px) | P@τ* | R@τ* | F1@τ* | fold mAP50 (mean ± sd) | fold mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| YOLOv9-c | 0.362 | 0.156 | 0.096 | 0.347 | 0.470 | 0.400 | 0.400 ± 0.116 | 0.179 ± 0.080 |
| RF-DETR | 0.473 | 0.210 | 0.121 | 0.400 | 0.560 | 0.467 | 0.511 ± 0.052 | 0.229 ± 0.046 |

τ* = F1-maximising confidence on each fold's inner val, applied to that fold's held-out images (per-fold values: YOLOv9-c 0:0.16, 1:0.32, 2:0.14, 3:0.34, 4:0.21; RF-DETR 0:0.25, 1:0.35, 2:0.18, 3:0.47, 4:0.28).

### Paired bootstrap (1,000 image resamples)

Δ = RF-DETR − YOLOv9-c

| metric | Δ | 95 % CI | P(Δ ≤ 0) |
|---|---|---|---|
| mAP50-95 | +0.0521 | [+0.0202, +0.0836] | 0.000 |
| mAP50 | +0.1093 | [+0.0681, +0.1528] | 0.000 |

Wilcoxon signed-rank on per-image F1@τ* (n = 69): mean Δ = +0.095, p = 0.0000.

### Fixed thresholds, classes, box sizes

| model | P/R/F1 @0.25 | P/R/F1 @0.50 | AP ecoli | AP ecoli_cluster | AP tercile S/M/L (GT area) |
|---|---|---|---|---|---|
| YOLOv9-c | 0.37/0.47/0.41 | 0.58/0.26/0.36 | 0.171 | 0.142 | 0.253 / 0.069 / 0.126 |
| RF-DETR | 0.39/0.61/0.47 | 0.61/0.41/0.49 | 0.220 | 0.199 | 0.303 / 0.097 / 0.179 |

GT-area tercile edges: 529 and 1047 px².

### Per fold

| fold | n | YOLOv9-c mAP50 / mAP50-95 / F1@τ* | RF-DETR mAP50 / mAP50-95 / F1@τ* |
|---|---|---|---|
| 0 | 15 | 0.417 / 0.183 / 0.463 | 0.475 / 0.187 / 0.509 |
| 1 | 14 | 0.237 / 0.075 / 0.261 | 0.437 / 0.178 / 0.450 |
| 2 | 14 | 0.369 / 0.131 / 0.320 | 0.546 / 0.259 / 0.368 |
| 3 | 13 | 0.418 / 0.224 / 0.448 | 0.553 / 0.285 / 0.541 |
| 4 | 13 | 0.560 / 0.282 / 0.558 | 0.546 / 0.239 / 0.546 |

## Speed and size (NVIDIA GeForce RTX 5060 Laptop GPU, batch 1, fp16, 640 px, 5×69 images, post-processing included)

| model | params (M) | GFLOPs | latency ms median | p95 | mean | FPS (1/median) |
|---|---|---|---|---|---|---|
| YOLOv9-c | 25.3 | 102.3 | 25.1 | 38.9 | 279.0 | 39.8 |
| RF-DETR | 36.5 | – | 46.5 | 54.0 | 47.5 | 21.5 |

Median is the headline: YOLOv9's mean is dominated by a few dense slides where Ultralytics' NMS hits its 2 s time limit (tens of thousands of candidate boxes at conf 0.001–0.25); RF-DETR has no such tail.

RF-DETR: inference(dtype=float16) failed: ValueError('inference(inplace=True) requires compile=False. Compiled models can retain references to the original parameter storage, so setting model.model=None may not free the weight tensors and inplace=True would not reliably reduce memory usage.'); ran fp32

## Crowding analysis

Crowding of a GT box = max IoU with any other GT box in its image. 72% of boxes touch another box; 12.3% overlap heavily (IoU > 0.5).

| recall @τ* | isolated (n=1676) | touching (n=3159) | overlapping (n=369) | heavy (n=733) | duplicate rate |
|---|---|---|---|---|---|
| YOLOv9-c | 0.547 | 0.448 | 0.201 | 0.527 | 0.10% |
| RF-DETR | 0.709 | 0.525 | 0.285 | 0.508 | 3.78% |
| Δ (rfdetr minus yolov9) 95 % CI | +0.162 [+0.107, +0.215] | +0.073 [+0.016, +0.122] | +0.074 [-0.033, +0.134] | -0.047 [-0.500, +0.023] | |

Duplicate rate = kept predictions at τ* overlapping a higher-scoring same-class prediction with IoU > 0.7 (the cost of skipping NMS).

| images by GT count | YOLOv9-c R / mAP50-95 / FP per img | RF-DETR R / mAP50-95 / FP per img |
|---|---|---|
| Q1 | 0.420 / 0.141 / 17.7 | 0.668 / 0.190 / 20.8 |
| Q2 | 0.511 / 0.119 / 39.0 | 0.623 / 0.182 / 32.4 |
| Q3 | 0.595 / 0.196 / 63.6 | 0.683 / 0.284 / 43.1 |
| Q4 | 0.447 / 0.157 / 191.6 | 0.521 / 0.194 / 197.2 |

Quartile edges on GT boxes per image: 23, 34, 98.

### YOLOv9 NMS IoU sweep (threshold re-chosen on inner val each time)

| NMS IoU | recall heavy | recall overlapping | recall isolated | duplicate rate |
|---|---|---|---|---|
| 0.50 | 0.487 | 0.214 | 0.564 | 0.00% |
| 0.60 | 0.501 | 0.214 | 0.560 | 0.00% |
| 0.70 | 0.527 | 0.201 | 0.547 | 0.10% |
| 0.80 | 0.573 | 0.171 | 0.514 | 18.81% |
| 0.90 | 0.756 | 0.146 | 0.461 | 38.83% |

Figures: `figures/recall_by_crowding_bin.png`, `figures/recall_by_gt_count_quartile.png`, `figures/heavy_recall_vs_nms_iou.png`, `figures/qualitative_dense.png`.

## Provenance

- YOLOv9-c: versions {'torch': '2.11.0+cu128', 'ultralytics': '8.4.48'}; weights md5 per fold: 0:78919e8f, 1:e15b1d10, 2:b00d3457, 3:9d022a0f, 4:aae8f182; conf 0.001, max_det 1200, nms_iou 0.7, tiled False.
- RF-DETR: versions {'torch': '2.11.0+cu128', 'rfdetr': '1.11.2'}; weights md5 per fold: 0:89fa26f5, 1:90c8e1f8, 2:2835c1d7, 3:e6f08090, 4:e2c8492f; conf 0.001, max_det 1200, nms_iou 0.7, tiled False.
- Fold plan `folds.json` (seed 0); label snapshot `label_hashes.txt`.
- Thesis reference (not comparable: 9-image val split, pre-relabel GT, Ultralytics validator): YOLOv9-c mAP50 0.499 / mAP50-95 0.280.
