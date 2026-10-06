# RF-DETR vs YOLOv9-c on E. coli — experiment notes

Working notes for the second paper. Results tables are generated into `RESULTS.md`
by `scripts/compare_report.py`; this file holds the design, provenance, decisions
and caveats. Dates are absolute.

## Question

Does an NMS-free transformer detector (RF-DETR, DINOv2 backbone + DETR decoder)
handle dense / overlapping bacteria better than the thesis YOLOv9-c, under one
shared evaluation on data neither model was tuned on?

## Decisions (2026-10-06, with Dominic)

1. **5-fold cross-validation for both models** on the pooled 69 images. The thesis
   `test/` split holds 2 untracked images (one a borderline near-duplicate of a
   training frame) and every thesis number comes from the 9-image `val` split that
   also drove checkpoint selection. CV with an inner validation set per fold is
   the only design that gives held-out numbers with any statistical weight.
2. **Ground truth = current working-tree labels** (1,629 val boxes), not the
   committed HEAD (487). The relabel completed the densest slide
   (`ecoli - 1.png`, 123 → 1,043 boxes). YOLO's numbers will therefore not match
   the thesis table; `label_hashes.txt` pins the exact label files used.
3. Both detectors live in the app (`src/inference.py`) and are trained through
   the app's own recipes (`training/train.py`, `training/train_rfdetr.py`), so
   what the paper measures is what the app ships.

## Dataset (pooled, 2026-10-06 snapshot)

| | value |
|---|---|
| images | 69 (train 58 / val 9 / test 2 in the thesis layout), all PNG |
| boxes | 5,585 — `ecoli` + `ecoli_cluster` |
| boxes per image | median 34, p95 253, max **1,043** (`ecoli - 1.png`, 2000×1693) |
| box short side | median 25.9 px, p10 15.5, p90 62.0; **70 % of boxes are COCO-"small" (< 32 px)** |
| image sizes | mostly 640×480 (microscope captures); stained slides up to 2000×1693 |
| domains | filename rule from the thesis: `microscope` if the name contains "microscope", else `stained` |

Consequences: `AP_small` is nearly the whole dataset, so `RESULTS.md` also reports
AP by GT-area tercile. The 1,043-box image exceeds both libraries' default
per-image caps (Ultralytics `max_det=300`, RF-DETR `num_queries=300`); the
comparison uses 1,200 for both.

### Near-duplicate scan

`scripts/compare_folds.py` scores every image pair by mean |grey difference| at
128 px (the metric of `scripts/merge_duplicate_labels.py`, calibrated there as
"< 4 = same frame, 15+ = different view"). No pair scores below 5.6; distinct
frames of the same microscope setup score 5–10 because the background is uniform,
so merging at 15 would have fused 41 microscope frames into one group. Decision:
merge only below 4 (none), list the 13 pairs below 8 in `folds.json` as
borderline. The one pair flagged by the thesis tooling
(`image_2026-09-26_183914758.png` vs `MicroscopeCapture(5).png`, score 6.7) was
checked visually: different fields of view.

## Fold plan (`folds.json`, seed 0)

Assignment follows `scripts/crossval.py::make_folds` (seeded shuffle, round-robin
within domain). Per fold: held-out = the fold; inner val = 6 images drawn
domain-balanced from the other four folds (used for early stopping and the
F1-maximising confidence threshold τ*); train = the rest.

| fold | held-out images (microscope) / boxes | inner val images / boxes | train images / boxes |
|---|---|---|---|
| 0 | 15 (12) / 978 | 6 / 1,721 | 48 / 2,886 |
| 1 | 14 (11) / 1,953 | 6 / 683 | 49 / 2,949 |
| 2 | 14 (11) / 951 | 6 / 651 | 49 / 3,983 |
| 3 | 13 (11) / 728 | 6 / 1,046 | 50 / 3,811 |
| 4 | 13 (11) / 975 | 6 / 1,271 | 50 / 3,339 |

Views under `runs/compare/folds/fold<k>/{train,valid,test}` are hard links to
`data/ecoli` (no copies, originals untouched). GT for evaluation:
`runs/compare/gt_all.json` (COCO, `image_id` = 1-based index in sorted-filename
order, annotation ids 1-based — pycocotools reads a matched id of 0 as "no
match" — `category_id` = YOLO class index). Sanity renders of the COCO GT:
`runs/compare/sanity/` (checked 2026-10-06).

## Models and recipes

| | YOLOv9-c | RF-DETR Medium |
|---|---|---|
| library | Ultralytics 8.4.48 | rfdetr 1.11.x (PyPI, 2026-10-04) |
| pretraining | COCO (`models/yolov9c.pt`) | COCO (`rf-detr-medium.pth`) |
| input | 640 letterbox (rect) | 640×640 square stretch |
| params | 25.3 M | ~33.7 M (DINOv2-S backbone; Small/Medium/Large differ in resolution + decoder depth, not params) |
| recipe | `training/train.py` = thesis exp_F: AdamW lr0 1e-3 cos, wd 5e-4, warmup 3, batch 8, 230 ep, patience 30, seed 0, AMP; aug hsv_s .2 hsv_v .4, rot 180, translate .1, scale .3, flips .5, mosaic .5 (close 10), erasing .2, randaugment | `training/train_rfdetr.py`: lr 1e-4 / lr_enc 1.5e-4, wd 1e-4, effective batch 16 (grad accum), 200 ep, early stop patience 20 on inner-val mAP50-95 (EMA), bf16; aug = scale jitter + hflip + multi-scale only |
| per-image cap | `max_det 1200` (thesis val used 300) | `num_queries = num_select = 1200` |
| selection | best.pt by inner-val fitness (mAP50-95) | `checkpoint_best_ema.pth` by inner-val mAP50-95 |

### Known confounds (state in the paper)

- Augmentation: YOLO's mosaic / rotation / HSV / erasing vs RF-DETR's scale
  jitter + flip. Both are each library's tuned default; neither was tuned here.
- Optimiser, schedule, EMA and epoch budget differ (library defaults).
- Letterbox-rect (YOLO) vs square stretch (RF-DETR) at the same long side.
- YOLO's NMS IoU (0.7, thesis val) is a tunable knob; the NMS sweep in
  `compare_crowding.py` shows how much of any gap it explains.
- RF-DETR's 1,200-query cap vs YOLO's post-NMS `max_det` 1200.
- Whole-image inference at 640 for the paper; the app additionally tiles large
  slides (`src/inference.py`, `tiled=True`) — reported as an extra row if run.
- Both COCO-pretrained; neither saw microscopy before fine-tuning.

## Evaluation (one pipeline for both)

`scripts/compare_predict.py` (conf 0.001, cap 1,200, whole image) →
`scripts/compare_eval.py` (pycocotools; τ* on inner val; per fold mean ± sd;
paired bootstrap over images with multiplicity-aware re-accumulation of
pycocotools' per-image match tables; Wilcoxon on per-image F1) →
`scripts/compare_crowding.py` (crowding bins, NMS sweep, duplicate rate,
figures) → `scripts/compare_bench.py` (bs 1, FP16, post-processing included,
params, GFLOPs) → `scripts/compare_report.py`.

Self-checks: `compare_eval.py --selftest` (GT as predictions → mAP 1.0, subset
accumulator == pycocotools, bootstrap Δ = 0), fold assertions in
`compare_folds.py`, `tests/test_detector_base.py` for the shared tiling path.

## Provenance

- Thesis repo at start: HEAD `42c1e00` ("ui", 2026-09-22) with uncommitted edits
  to `training/train.py`, `scripts/crossval.py` and 3 val label files, plus
  untracked `data/ecoli/images|labels/test/` and 3 new train images. The
  comparison uses the working tree as found; sha256 of every label file is in
  `label_hashes.txt`. Re-run `scripts/compare_folds.py` and diff that file
  before any evaluation if labels may have changed.
- Thesis weights `models/best_yolov9c.pt` (md5 `7b32598fb05e12a9200554441cfa5bd5`,
  = run `exp_F_mergedv2_640_long` epoch 165/230): **not used** in the CV (it was
  trained on train+val, which overlaps every fold); kept as the app's default
  YOLO model and quoted as a reference row only.
- Thesis headline (val, 9 images, 487 boxes, old labels, Ultralytics validator):
  P .564 R .472 mAP50 .499 mAP50-95 .280. Thesis 5-fold CV
  (`runs/crossval/baseline/summary.json`, leaky selection on the held-out fold):
  mAP50 .467 ± .064.
- Environment: thesis `.venv` (Python 3.11.9, torch 2.11.0+cu128, Ultralytics
  8.4.48) + `rfdetr[train]` + tensorboard; `logs/freeze_before_rfdetr.txt` is the
  pre-install snapshot; `pip freeze` after install → `requirements.lock.txt`.
- Hardware: NVIDIA RTX 5060 Laptop GPU, 8 GB, Windows 11.

## Gotchas

- Ultralytics rebases a *relative* `project` under its `runs_dir` setting
  (`runs/detect/runs/compare/...`), which is why the thesis runs sit under
  `runs/detect/runs/train/`. `scripts/compare_train.py` passes an absolute
  `--project`; the backend's auto-activate globs by run name and is unaffected.
- Windows consoles default to cp1252: the compare scripts print ASCII only.
- `rfdetr[train]` pins `numpy < 2.4` (downgraded 2.4.4 → 2.3.5 on install)
  and pulls `roboflow`, `peft`, `accelerate`, `pyarrow`, `av`.

## Log

- 2026-10-06: plan agreed; app integration (algorithm switch) built; folds,
  GT, sanity renders done; YOLOv9 smoke train on fold 0 OK (2 epochs,
  `max_det 1200` accepted); RF-DETR smoke pending `rfdetr` install.
