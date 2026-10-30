"""Multi-source dataset loader for MicroSeg 5-class segmentation.

Supported data sources:
  1. Cityscapes  — provides Walkable (1), Non-walkable (2), Boundary (3)
  2. Kaggle FPV Crosswalk — provides Crosswalk (4) only
  3. Local pilot — provides all 5 classes (future)

All sources are normalized to:
  - 96×96×3 float32 images in [0.0, 1.0]
  - 96×96 int32 class-ID masks (values 0-4 or 255=ignore)
  - 96×96 float32 sample-weight maps (1.0 = valid, 0.0 = ignore)

Combining sources:
  tf.data.Dataset.sample_from_datasets([cityscapes_ds, crosswalk_ds], weights)
  This interleaves batches stochastically, ensuring the Crosswalk class
  appears regularly even though most Cityscapes pixels are non-crosswalk.
"""
import os
from pathlib import Path

import tensorflow as tf

from microseg_dataset_config import (
    CITYSCAPES_ROOT, KAGGLE_CROSSWALK_ROOT, LOCAL_PILOT_ROOT,
    validate_cityscapes, validate_crosswalk,
)
from microseg_taxonomy import IGNORE_LABEL, cityscapes_lookup

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
BATCH_SIZE = int(os.environ.get("MICROSEG_BATCH_SIZE", "8"))
AUTOTUNE = tf.data.AUTOTUNE

# Sampling weight: fraction of batches drawn from each source.
# Crosswalk is upweighted to compensate for the very small number of
# Crosswalk pixels relative to Walkable/Non-walkable pixels.
CITYSCAPES_SAMPLE_WEIGHT = 0.65
CROSSWALK_SAMPLE_WEIGHT = 0.35


# ---------------------------------------------------------------------------
# Shared preprocessing helpers
# ---------------------------------------------------------------------------

def _load_and_resize_image(path: tf.Tensor) -> tf.Tensor:
    """Decode JPEG or PNG image and bilinear-resize to 96×96."""
    raw = tf.io.read_file(path)
    # decode_image handles both JPEG and PNG automatically.
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(image, tf.float32) / 255.0


def _load_and_resize_mask_png(path: tf.Tensor) -> tf.Tensor:
    """Decode single-channel PNG mask and nearest-neighbor resize to 96×96."""
    raw = tf.io.read_file(path)
    mask = tf.io.decode_png(raw, channels=1)
    mask = tf.image.resize(mask, [IMAGE_HEIGHT, IMAGE_WIDTH],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.cast(tf.squeeze(mask, axis=-1), tf.int32)


def _make_weights(mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Zero-weight the ignore pixels (255) without changing their label storage."""
    valid = tf.not_equal(mask, IGNORE_LABEL)
    weights = tf.cast(valid, tf.float32)
    safe_mask = tf.where(valid, mask, tf.zeros_like(mask))
    return safe_mask, weights


# ---------------------------------------------------------------------------
# Source 1: Cityscapes
# ---------------------------------------------------------------------------

def _cityscapes_remap_tf(mask: tf.Tensor) -> tf.Tensor:
    """Apply the Cityscapes trainId → MicroSeg LUT inside the tf.data pipeline."""
    lut = tf.constant(cityscapes_lookup(), dtype=tf.int32)
    valid = (mask >= 0) & (mask < 256)
    remapped = tf.gather(lut, tf.clip_by_value(mask, 0, 255))
    return tf.where(valid, remapped, tf.fill(tf.shape(mask), IGNORE_LABEL))


def _collect_cityscapes_pairs(split: str) -> tuple[list[str], list[str]]:
    """Scan Cityscapes directory structure and return paired image/mask paths."""
    image_root = CITYSCAPES_ROOT / "leftImg8bit" / split
    mask_root = CITYSCAPES_ROOT / "gtFine" / split

    image_paths, mask_paths, missing = [], [], []
    for image_path in sorted(image_root.glob("*/*_leftImg8bit.png")):
        city = image_path.parent.name
        mask_name = image_path.name.replace("_leftImg8bit.png",
                                            "_gtFine_labelTrainIds.png")
        mask_path = mask_root / city / mask_name
        if mask_path.exists():
            image_paths.append(str(image_path))
            mask_paths.append(str(mask_path))
        else:
            missing.append(str(mask_path))

    split_label = split.upper()
    print(f"[Cityscapes {split_label}] images={len(image_paths)}, missing={len(missing)}")
    if not image_paths:
        raise RuntimeError(f"No Cityscapes pairs found for split '{split}'.")
    return image_paths, mask_paths


def _load_cityscapes_pair(image_path: tf.Tensor, mask_path: tf.Tensor):
    """Load one Cityscapes image+mask, remap to MicroSeg IDs, return triple."""
    image = _load_and_resize_image(image_path)
    mask = _load_and_resize_mask_png(mask_path)
    mask = _cityscapes_remap_tf(mask)
    safe_mask, weights = _make_weights(mask)
    return image, safe_mask, weights


def make_cityscapes_dataset(split: str, training: bool = False,
                            batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Return a batched tf.data.Dataset from Cityscapes for MicroSeg training."""
    validate_cityscapes()
    image_paths, mask_paths = _collect_cityscapes_pairs(split)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if training:
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    ds = ds.map(_load_cityscapes_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Source 2: Kaggle FPV Crosswalk
# ---------------------------------------------------------------------------

_CROSSWALK_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_CROSSWALK_MASK_DIRECTORY_NAMES = {
    "annotation", "annotations", "gt", "ground_truth", "groundtruth",
    "label", "labels", "mask", "masks",
}
_CROSSWALK_MASK_SUFFIXES = (
    "_mask", "-mask", ".mask", "_label", "-label", "_labels",
    "_segmentation", "_gt",
)
_CROSSWALK_ROLE_DIRECTORY_NAMES = (
    "ground_truth", "groundtruth", "annotations", "annotation", "images",
    "image", "masks", "mask", "labels", "label", "imgs", "img", "gt",
)


def _is_crosswalk_mask(path: Path) -> bool:
    """Identify masks by directory or filename without assuming one layout."""
    for part in path.parts[:-1]:
        directory = part.lower().replace(" ", "_").replace("-", "_")
        if any(
            directory == role
            or directory.endswith("_" + role)
            or directory.startswith(role + "_")
            for role in _CROSSWALK_MASK_DIRECTORY_NAMES
        ):
            return True
    stem = path.stem.lower()
    return (
        any(stem.endswith(suffix) for suffix in _CROSSWALK_MASK_SUFFIXES)
        or stem.startswith(("mask_", "mask-", "gt_", "gt-"))
    )


def _crosswalk_pair_key(path: Path, root: Path) -> tuple[str, ...]:
    """Normalize mirrored image/mask paths to a common relative key."""
    parts = []
    for part in path.relative_to(root).parts[:-1]:
        normalized = part.lower().replace(" ", "_").replace("-", "_")
        for role in _CROSSWALK_ROLE_DIRECTORY_NAMES:
            if normalized == role:
                normalized = ""
                break
            if normalized.endswith("_" + role):
                normalized = normalized[:-(len(role) + 1)]
                break
            if normalized.startswith(role + "_"):
                normalized = normalized[len(role) + 1:]
                break
        if normalized:
            parts.append(normalized)
    parts.append(path.stem.lower())

    stem = parts[-1]
    for suffix in _CROSSWALK_MASK_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    for prefix in ("mask_", "mask-", "gt_", "gt-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    parts[-1] = stem
    return tuple(parts)


def _collect_crosswalk_pairs() -> tuple[list[str], list[str]]:
    """Pair FPV images and binary masks across common mirrored folder layouts.

    Supports same-stem files in separate image/mask directories, mirrored
    nested source/condition folders, and sibling ``*_mask`` files. Image and
    mask extensions may differ; masks are decoded as single-channel images.
    """
    root = KAGGLE_CROSSWALK_ROOT
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _CROSSWALK_IMAGE_EXTENSIONS
    )
    mask_paths_by_key: dict[tuple[str, ...], list[Path]] = {}
    image_candidates = []

    for path in files:
        if _is_crosswalk_mask(path):
            key = _crosswalk_pair_key(path, root)
            mask_paths_by_key.setdefault(key, []).append(path)
        else:
            image_candidates.append(path)

    image_paths, paired_masks, missing = [], [], []
    ambiguous = []
    for image_path in image_candidates:
        key = _crosswalk_pair_key(image_path, root)
        matches = mask_paths_by_key.get(key, [])
        if len(matches) > 1:
            ambiguous.append((image_path, matches))
            continue
        if matches:
            image_paths.append(str(image_path))
            paired_masks.append(str(matches[0]))
        else:
            missing.append(str(image_path))

    if ambiguous:
        details = "\n".join(
            f"  {image}: " + ", ".join(str(path) for path in masks)
            for image, masks in ambiguous[:10]
        )
        raise RuntimeError(
            "Ambiguous FPV Crosswalk image/mask matches; make the relative "
            "image and mask paths unique:\n" + details
        )

    print(
        f"[Kaggle Crosswalk] paired={len(image_paths)}, "
        f"images_without_masks={len(missing)}"
    )
    if missing:
        examples = "\n".join(f"  {path}" for path in missing[:10])
        raise RuntimeError(
            f"Found {len(missing)} FPV Crosswalk images without masks. "
            "The dataset is expected to contain paired images and masks; "
            "check the extracted folder layout and naming:\n" + examples
        )
    if not image_paths:
        raise RuntimeError(
            "No FPV Crosswalk image/mask pairs found under "
            f"KAGGLE_CROSSWALK_ROOT={root}. Expected matching relative paths "
            "under image and mask directories, or same-stem *_mask files."
        )
    return image_paths, paired_masks


def _load_crosswalk_pair(image_path: tf.Tensor, mask_path: tf.Tensor):
    """Load one Kaggle crosswalk image and binary mask as a MicroSeg triple.

    Mask is a binary image:
      pixel >  127  →  Crosswalk (ID 4)
      pixel <= 127  →  Ignore (255)  — NOT Background, to prevent contamination
    """
    image = _load_and_resize_image(image_path)

    raw_mask = tf.io.read_file(mask_path)
    # Accept PNG/JPEG masks while keeping their binary values single-channel.
    mask_raw = tf.io.decode_image(raw_mask, channels=1, expand_animations=False)
    mask_raw = tf.image.resize(mask_raw, [IMAGE_HEIGHT, IMAGE_WIDTH],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask_raw = tf.cast(tf.squeeze(mask_raw, axis=-1), tf.int32)

    # Threshold: JPEG artifacts may shift values away from exactly 0/255.
    crosswalk = tf.fill(tf.shape(mask_raw), 4)
    ignore = tf.fill(tf.shape(mask_raw), IGNORE_LABEL)
    mask = tf.where(mask_raw > 127, crosswalk, ignore)

    safe_mask, weights = _make_weights(mask)
    return image, safe_mask, weights


def make_crosswalk_dataset(training: bool = False,
                           batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Return a batched tf.data.Dataset from the Kaggle FPV Crosswalk dataset."""
    validate_crosswalk()
    image_paths, mask_paths = _collect_crosswalk_pairs()

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if training:
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    ds = ds.map(_load_crosswalk_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Source 3: Local pilot (PNG masks, IDs 0-4 and 255)
# ---------------------------------------------------------------------------

def _collect_local_pairs() -> tuple[list[str], list[str]]:
    """Scan local pilot dataset: images/ paired with masks/ by matching stem."""
    images_dir = LOCAL_PILOT_ROOT / "images"
    masks_dir = LOCAL_PILOT_ROOT / "masks"

    image_paths, mask_paths, missing = [], [], []
    for image_path in sorted(images_dir.glob("*.png")):
        mask_path = masks_dir / image_path.name
        if mask_path.exists():
            image_paths.append(str(image_path))
            mask_paths.append(str(mask_path))
        else:
            missing.append(str(mask_path))

    print(f"[Local Pilot] images={len(image_paths)}, missing={len(missing)}")
    return image_paths, mask_paths


def _load_local_pair(image_path: tf.Tensor, mask_path: tf.Tensor):
    """Load one local pilot image+mask. Mask already contains MicroSeg IDs."""
    image = _load_and_resize_image(image_path)
    mask = _load_and_resize_mask_png(mask_path)
    safe_mask, weights = _make_weights(mask)
    return image, safe_mask, weights


def make_local_dataset(training: bool = False,
                       batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Return a batched tf.data.Dataset from the local PathClear pilot."""
    image_paths, mask_paths = _collect_local_pairs()
    if not image_paths:
        return None  # Local data not yet available — caller handles None.

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if training:
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    ds = ds.map(_load_local_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------

def make_dataset(split: str, training: bool = False,
                 batch_size: int = BATCH_SIZE,
                 use_crosswalk: bool = True) -> tf.data.Dataset:
    """Build a combined Cityscapes + Kaggle dataset for MicroSeg training.

    Args:
        split:         'train' or 'val'. Kaggle dataset has no official split;
                       for 'val' only Cityscapes validation images are used.
        training:      Enable shuffle and augmentation.
        batch_size:    Batch size (default from MICROSEG_BATCH_SIZE env var).
        use_crosswalk: Include the Kaggle FPV Crosswalk source (train only).

    Returns:
        A batched, prefetched tf.data.Dataset of (image, mask, weights).
    """
    cityscapes_ds = make_cityscapes_dataset(split, training=training,
                                            batch_size=batch_size)

    if split == "val" or not use_crosswalk:
        # Validation: Cityscapes only (Kaggle has no official val split).
        return cityscapes_ds

    crosswalk_ds = make_crosswalk_dataset(training=training, batch_size=batch_size)

    # Interleave: CITYSCAPES_SAMPLE_WEIGHT fraction from Cityscapes,
    # CROSSWALK_SAMPLE_WEIGHT fraction from Kaggle crosswalk.
    combined = tf.data.Dataset.sample_from_datasets(
        [cityscapes_ds, crosswalk_ds],
        weights=[CITYSCAPES_SAMPLE_WEIGHT, CROSSWALK_SAMPLE_WEIGHT],
        rerandomize_each_iteration=training,
    )
    return combined
