"""
training/augmentations.py
=========================
Reference Albumentations pipeline for microscopy data.

This file is intentionally documentation-oriented. Ultralytics will
automatically apply Albumentations transformations during training when
the package is installed — the defaults are reasonable but generic.

The pipeline below is what an *explicit*, microscopy-aware augmentation
schedule looks like. To wire it in directly you would subclass
:class:`ultralytics.data.dataset.YOLODataset` and override
``build_transforms``. For most projects, tuning the hyperparameters in
``training/train.py`` (which delegate to Ultralytics' built-in augmenter)
is sufficient and matches this reference closely.
"""
from typing import Optional

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:  # albumentations is optional
    A = None  # type: ignore[assignment]
    ALBUMENTATIONS_AVAILABLE = False


def build_microscopy_train_transform(imgsz: int = 640) -> Optional["A.Compose"]:
    """Augmentations appropriate for stained-light microscopy.

    Notable choices:
      * CLAHE matches our *inference*-time preprocessing (keeps train /
        inference distributions consistent).
      * GaussianBlur covers focus-plane drift between samples.
      * GaussNoise covers low-light sensor noise from cheap USB scopes.
      * Rotation up to 180° — microscope orientation is arbitrary.
      * Vertical *and* horizontal flips — bacteria are not chiral.
      * NO hue shift / channel shuffle — stain colour is diagnostic and
        must be preserved.
      * NO MixUp / CutMix — they blend ground-truth boxes on tiny
        targets and hurt small-object detection.

    Returns ``None`` if albumentations is not installed.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    return A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=0),
            A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.Rotate(limit=180, border_mode=0, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def build_val_transform(imgsz: int = 640) -> Optional["A.Compose"]:
    """Validation transforms — resize + letterbox only, no augmentation."""
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=0),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )
