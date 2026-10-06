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

### Query expansion (RF-DETR)

The published COCO checkpoints carry 300 object queries packed as
`nn.Embedding(300 × 13 groups)`; rfdetr can shrink that on load but not grow
it, and the densest slide has 1,043 cells. `training/rfdetr_queries.py`
derives `models/rf-detr-medium-q1200.pth` by tiling each group's 300 learned
queries 4× (copies get Gaussian noise of 2 % of the tensor's std so they
separate during fine-tuning) and updating the stored `num_queries`;
`train_rfdetr.py` does this automatically when `--num-queries` exceeds the
checkpoint. The 1,200 slots are therefore initialised from the 300 learned
ones, not at random — note this in the paper as a method detail. Smoke runs
(2 epochs, batch 2, 640 px, 1,200 queries, fold 0 and fold 2): ~1.5 min,
Lightning `max_mem` 4.6 GB, total GPU peak 6.6–7.2 GB with ~1.2–1.6 GB held
by the running app backend → full training uses `--gradient-checkpointing`.

### Offline ×8 orientation expansion (`--expand8`)

Requested 2026-10-07: every *training* image is written in its 8 dihedral
variants (4 rotations × {original, mirrored}) with transformed labels
(`training/dataset_view.py`, self-test checks each rotated label against the
rotated pixels). Applied identically to both models, so both see all
orientations and the augmentation confound shrinks; inner-val and held-out
images are never expanded (evaluation stays on real frames, and expanding a
held-out image would leak its twins into training). Epoch and patience
budgets are divided by 8 so the optimiser-step budget is unchanged
(`scripts/compare_train.py --expand8` → YOLO 29 ep / patience 4, RF-DETR
25 ep / patience 3; outputs `fold<k>_x8`). YOLO's online rotation/flip
augmentation stays on top, so for YOLO the expansion is largely redundant;
for RF-DETR (hflip only) it is the new signal.

### Pseudo-labelled expanded dataset (`data/ecoli_x8/`, requested 2026-10-07)

`scripts/make_pseudo_dataset.py` writes all 69 images × 8 orientations (552)
with the rotated human labels **plus** teacher detections (thesis
`models/best_yolov9c.pt`, tiled, conf ≥ 0.25) that overlap no existing box
(IoU < 0.3 — the app's "Suggest missing" rule). `pseudo.json` lists every
added box with its confidence. Motivation: the val relabel (487 → 1,629
boxes) showed the human labels are incomplete; the teacher fills gaps.
`scripts/compare_folds.py --train-from data/ecoli_x8` builds fold views
whose *training* split is the expanded, pseudo-labelled copy of the fold's
training images; inner-val and held-out images stay original with human
labels, and `runs/compare/gt_all.json` is unchanged.

Confounds this adds: only YOLOv9 can act as teacher (COCO RF-DETR knows no
bacteria), so both students inherit YOLO's detections and errors — a bias
towards YOLOv9 that the paper must state. The teacher trained on
train + val, so its pseudo-labels on a fold's training images are not
independent of that fold's held-out images; evaluation is still against
human labels only.

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
- The COCO checkpoint `rf-detr-medium.pth` (386 MB, md5
  `7223f764a87b863f02eb8d52bf0ce2ee`, from
  `https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth`)
  is fetched by rfdetr into `~/.roboflow/models/` (override with `RF_HOME`)
  without resume; on a slow link, download it with `curl -C -` to that path
  instead — rfdetr accepts a pre-placed file whose md5 matches.

## Log

- 2026-10-06: plan agreed; app integration (algorithm switch) built, tested
  in the browser and committed (`058069a`); comparison pipeline committed
  (`e3cf08f`); folds, GT (1-based ids), sanity renders and
  `compare_eval.py --selftest` done; YOLOv9 smoke train on fold 0 OK
  (2 epochs, `max_det 1200` accepted, output landed under
  `runs/detect/runs/compare/...` → absolute `--project` in compare_train.py).
  RF-DETR smoke train waits on the 386 MB COCO checkpoint download
  (`~/.roboflow/models/rf-detr-medium.pth`, ~100 kB/s link).
- 2026-10-07: checkpoint fetched (md5 ok); `training/rfdetr_queries.py`
  added (1,200-query checkpoint); RF-DETR runs through the API (COCO base).
  **Bug found and fixed**: `dataset_view.label_of` only knew the
  `images/<split>` layout, so views built from fold views (train_rfdetr.py,
  `--expand8`) had empty labels — the first RF-DETR smokes trained on nothing
  (zero losses, val mAP −1). After the fix, 2-epoch smokes on fold 2 with ×8
  expansion: RF-DETR val mAP50 0.50 / mAP50-95 0.22, YOLOv9 mAP50 0.39.
  RF-DETR ×8 at batch 2 + gradient checkpointing: 4.5 min/epoch, 5.6 GB
  Lightning `max_mem`, GPU full (app backend holding 2 GB) → batch-1 timing
  test pending. `--expand8` and `make_pseudo_dataset.py` added.
- 2026-10-07 02:08–04:15: first full run on the pseudo-labelled ×8 folds
  (YOLO 29 ep / patience 4, RF-DETR 25 ep / patience 3, batch 1 + grad
  checkpointing after batch 2 spilled VRAM). YOLO finished all folds in
  2.7–9.2 min each; fold 1 "best" epoch was 2 and training stopped at
  epoch 6 (inside Ultralytics' 3-epoch warm-up) → fold mAP50 0.17 vs 0.34–0.48
  elsewhere. Verdict: patience 3–4 on a 6-image inner val is noise-driven.
  **Protocol change**: no early stopping; full scaled budget (YOLO 29 ep,
  RF-DETR 25 ep) with best-checkpoint selection on inner val for both. The
  early-stopped YOLO predictions are kept as `yolov9_es` (sensitivity row);
  RF-DETR's early-stopped folds 0–1 (`fold0_x8`, `fold1_x8`) are incomplete
  and unused. Full-budget runs: `fold<k>_x8_full`, started 04:20.
