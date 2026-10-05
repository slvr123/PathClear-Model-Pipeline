"""Cityscapes 96 x 96 pipeline for the four-class MicroSeg model.

Mirrors the Fast-SCNN loader geometry: bilinear RGB resize, nearest-neighbor
mask resize, and a zero sample weight on ignored pixels. The differences are the
96 x 96 resolution and the four-class remapping. Kaggle FPVCrosswalk2025 pairs
can provide Crosswalk-only supervision; unmarked mask pixels receive zero
sample weight and do not become Background labels.
"""
import os
from functools import partial
from pathlib import Path

import tensorflow as tf

from dataset_config import KAGGLE_CROSSWALK_ROOT
from dataset_config import CITYSCAPES_ROOT
from microseg.taxonomy import (
    IGNORE_LABEL,
    NUM_CLASSES,
    cityscapes_label_ids_lookup,
    cityscapes_lookup,
)

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96

# Weak negative supervision for unmarked Crosswalk pixels. Non-walkable is the
# safest of the four classes for an unmarked first-person road scene: it is the
# dominant surface class and it is never Crosswalk.
NEGATIVE_CLASS_ID = 1

BATCH_SIZE = int(os.environ.get("MICROSEG_BATCH_SIZE", "8"))
if BATCH_SIZE < 1:
    raise ValueError("MICROSEG_BATCH_SIZE must be at least 1.")

AUTOTUNE = tf.data.AUTOTUNE
CROSSWALK_CLASS_ID = 3
CROSSWALK_SAMPLE_WEIGHT = float(os.environ.get("MICROSEG_CROSSWALK_WEIGHT", "0.35"))
if not 0.0 < CROSSWALK_SAMPLE_WEIGHT < 1.0:
    raise ValueError("MICROSEG_CROSSWALK_WEIGHT must be between 0 and 1.")

_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
_MASK_DIRECTORY_NAMES = {
    "annotation", "annotations", "gt", "ground_truth", "groundtruth",
    "label", "labels", "mask", "masks",
}
_IMAGE_DIRECTORY_NAMES = {"image", "images", "img", "imgs"}
_ROLE_DIRECTORY_NAMES = _MASK_DIRECTORY_NAMES | _IMAGE_DIRECTORY_NAMES
_UNLABELED_IMAGE_DIRECTORIES = {"fine_tuning_seed"}
_MASK_SUFFIXES = (
    "_mask", "-mask", ".mask", "_label", "-label", "_labels",
    "_segmentation", "_gt",
)


def _normalized_directory(part):
    """Remove a leading/trailing image or mask role from a path component."""
    normalized = part.lower().replace(" ", "_").replace("-", "_")
    for role in sorted(_ROLE_DIRECTORY_NAMES, key=len, reverse=True):
        if normalized == role:
            return ""
        if normalized.endswith("_" + role):
            return normalized[:-(len(role) + 1)]
        if normalized.startswith(role + "_"):
            return normalized[len(role) + 1:]
    return normalized


def _crosswalk_pair_key(path, root):
    parts = [
        normalized
        for part in path.relative_to(root).parts[:-1]
        if (normalized := _normalized_directory(part))
    ]
    stem = path.stem.lower()
    for suffix in _MASK_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    for prefix in ("mask_", "mask-", "gt_", "gt-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    parts.append(stem)
    return tuple(parts)


def _is_crosswalk_mask(path):
    for part in path.parts[:-1]:
        normalized = part.lower().replace(" ", "_").replace("-", "_")
        if any(
            normalized == role
            or normalized.endswith("_" + role)
            or normalized.startswith(role + "_")
            for role in _MASK_DIRECTORY_NAMES
        ):
            return True
    stem = path.stem.lower()
    return (
        any(stem.endswith(suffix) for suffix in _MASK_SUFFIXES)
        or stem.startswith(("mask_", "mask-", "gt_", "gt-"))
    )


def _is_unlabeled_auxiliary_image(path, root):
    """Exclude the Kaggle seed-image collection, which has no masks."""
    return any(
        _normalized_directory(part) in _UNLABELED_IMAGE_DIRECTORIES
        for part in path.relative_to(root).parts[:-1]
    )


def collect_crosswalk_pairs(root=None):
    """Find deterministic FPVCrosswalk2025 image/mask pairs under ``root``.

    Supports mirrored image/mask directories and sibling mask-suffixed files.
    Ambiguous and unpaired image files raise instead of being silently omitted.
    """
    root = Path(KAGGLE_CROSSWALK_ROOT if root is None else root)
    if not root.is_dir():
        raise FileNotFoundError(f"Kaggle Crosswalk dataset root does not exist: {root}")

    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )
    masks_by_key = {}
    image_candidates = []
    ignored_auxiliary_images = 0
    for path in files:
        if _is_crosswalk_mask(path):
            masks_by_key.setdefault(_crosswalk_pair_key(path, root), []).append(path)
        elif _is_unlabeled_auxiliary_image(path, root):
            ignored_auxiliary_images += 1
        else:
            image_candidates.append(path)

    pairs = []
    unpaired = []
    ambiguous = []
    for image_path in image_candidates:
        matches = masks_by_key.get(_crosswalk_pair_key(image_path, root), [])
        if len(matches) > 1:
            ambiguous.append((image_path, matches))
        elif matches:
            pairs.append((str(image_path), str(matches[0])))
        else:
            unpaired.append(image_path)

    if ambiguous:
        detail = "\n".join(
            f"  {image}: " + ", ".join(str(mask) for mask in masks)
            for image, masks in ambiguous[:10]
        )
        raise RuntimeError(
            "Ambiguous Kaggle Crosswalk image/mask matches:\n" + detail
        )
    if unpaired:
        detail = "\n".join(f"  {path}" for path in unpaired[:10])
        raise RuntimeError(
            f"Found {len(unpaired)} Kaggle Crosswalk image(s) without masks:\n"
            + detail
        )
    if not pairs:
        raise RuntimeError(f"No Kaggle Crosswalk image/mask pairs found under {root}.")

    print(
        f"[Kaggle Crosswalk] paired={len(pairs)} "
        f"ignored_unlabeled_seed_images={ignored_auxiliary_images} root={root}"
    )
    return [image for image, _ in pairs], [mask for _, mask in pairs]


def load_crosswalk_pair(
    image_path,
    mask_path,
    negative_weight=0.0,
    negative_class_id=NEGATIVE_CLASS_ID,
):
    """Load a Kaggle pair; supervise positive Crosswalk pixels only.

    Source mask values above 127 map to MicroSeg Crosswalk ID 3. By default
    every other pixel becomes Ignore (255) with zero sample weight, which
    leaves the model free to paint Crosswalk anywhere in the frame without
    penalty, because ignored pixels never enter the loss.

    When ``negative_weight`` is greater than zero the unmarked pixels become
    weak negative evidence instead: they carry ``negative_class_id`` with
    that reduced weight, so the unmarked part of a Crosswalk frame pushes the
    Crosswalk logit down. The weight stays below 1 because the unmarked area
    of a first-person frame also contains sidewalk, buildings and sky that
    this class does not describe correctly. It is pressure against Crosswalk
    flooding, not a claim about those surfaces.
    """
    raw_image = tf.io.read_file(image_path)
    image = tf.io.decode_image(raw_image, channels=3, expand_animations=False)
    image = tf.image.resize(
        image, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.BILINEAR
    )
    image = tf.cast(image, tf.float32) / 255.0

    raw_mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_image(raw_mask, channels=1, expand_animations=False)
    mask = tf.image.resize(
        mask, [IMAGE_HEIGHT, IMAGE_WIDTH],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    mask = tf.cast(tf.squeeze(mask, axis=-1), tf.int32)
    positive = mask > 127
    crosswalk = tf.fill(tf.shape(mask), CROSSWALK_CLASS_ID)
    if negative_weight > 0:
        negative = tf.fill(tf.shape(mask), negative_class_id)
        targets_raw = tf.where(positive, crosswalk, negative)
        weights_raw = tf.where(
            positive,
            tf.constant(1.0, tf.float32),
            tf.constant(negative_weight, tf.float32),
        )
    else:
        ignored = tf.fill(tf.shape(mask), IGNORE_LABEL)
        targets_raw = tf.where(positive, crosswalk, ignored)
        weights_raw = None
    targets, weights = prepare_targets(targets_raw, weights_raw)
    return image, targets, weights


def make_crosswalk_dataset(
    training=False, batch_size=None, root=None, pairs=None, negative_weight=0.0
):
    """Build the Kaggle Crosswalk dataset at MicroSeg input resolution."""
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not 0.0 <= negative_weight <= 1.0:
        raise ValueError("negative_weight must be between 0 and 1.")
    if pairs is None:
        image_paths, mask_paths = collect_crosswalk_pairs(root)
    else:
        if not pairs:
            raise ValueError("Crosswalk pairs must not be empty.")
        image_paths, mask_paths = zip(*pairs)
        image_paths, mask_paths = list(image_paths), list(mask_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if training:
        dataset = dataset.shuffle(len(image_paths), reshuffle_each_iteration=True)
    dataset = dataset.map(
        partial(load_crosswalk_pair, negative_weight=negative_weight),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def collect_cityscapes_pairs(split):
    """Collect Cityscapes pairs, accepting original labelIds or trainIds masks.

    Prefer precomputed trainIds masks when both formats exist. Do not modify
    source masks; the MicroSeg loader remaps either format in memory.
    """
    image_root = CITYSCAPES_ROOT / "leftImg8bit" / split
    mask_root = CITYSCAPES_ROOT / "gtFine" / split
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(
            f"Cityscapes split '{split}' is incomplete. Expected image and mask "
            f"directories:\n  {image_root}\n  {mask_root}"
        )

    image_paths = sorted(image_root.glob("*/*_leftImg8bit.png"))
    pairs = []
    missing = []
    for image_path in image_paths:
        stem = image_path.name.removesuffix("_leftImg8bit.png")
        train_ids = mask_root / image_path.parent.name / f"{stem}_gtFine_labelTrainIds.png"
        label_ids = mask_root / image_path.parent.name / f"{stem}_gtFine_labelIds.png"
        if train_ids.is_file():
            pairs.append((str(image_path), str(train_ids)))
        elif label_ids.is_file():
            pairs.append((str(image_path), str(label_ids)))
        else:
            missing.append(str(image_path))

    if missing:
        details = "\n".join(f"  {path}" for path in missing[:10])
        raise RuntimeError(
            f"Found {len(missing)} Cityscapes image(s) without a labelIds or "
            f"labelTrainIds mask in split '{split}':\n{details}"
        )
    if not pairs:
        raise RuntimeError(
            f"No Cityscapes image/mask pairs found for split '{split}' under "
            f"{CITYSCAPES_ROOT}."
        )

    train_id_count = sum(path.endswith("_labelTrainIds.png") for _, path in pairs)
    label_id_count = len(pairs) - train_id_count
    print(
        f"[Cityscapes {split}] pairs={len(pairs)} "
        f"labelTrainIds={train_id_count} labelIds={label_id_count}"
    )
    return [image for image, _ in pairs], [mask for _, mask in pairs]


def attach_cityscapes_ignore_masks(cityscapes_pairs, ignore_root):
    """Attach available reviewed masks mirrored under ``ignore_root/{split}/{city}``.

    Each optional mask is named ``{cityscapes_stem}_ignore.png``. Missing masks
    mean no reviewed exclusions for that image; they do not imply the image
    was reviewed.
    """
    ignore_root = Path(ignore_root)
    try:
        ignore_root.resolve().relative_to(CITYSCAPES_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Cityscapes ignore-mask root must be inside {CITYSCAPES_ROOT}: "
            f"{ignore_root}"
        ) from error
    image_root = CITYSCAPES_ROOT / "leftImg8bit"
    pairs_with_ignore = []
    for image_path, mask_path in cityscapes_pairs:
        image_path = Path(image_path)
        try:
            relative_image = image_path.relative_to(image_root)
        except ValueError as error:
            raise ValueError(
                f"Cityscapes image must be under {image_root}: {image_path}"
            ) from error
        stem = image_path.stem.removesuffix("_leftImg8bit")
        ignore_path = (
            ignore_root / relative_image.parent / f"{stem}_ignore.png"
        )
        pairs_with_ignore.append((
            str(image_path),
            mask_path,
            str(ignore_path) if ignore_path.is_file() else "",
        ))
    return pairs_with_ignore


def remap_cityscapes_tensor(mask):
    """Apply the shared Cityscapes -> MicroSeg lookup as a TensorFlow op."""
    mask = tf.cast(mask, tf.int32)
    valid = (mask >= 0) & (mask < 256)
    mapped = tf.gather(
        tf.constant(cityscapes_lookup(), dtype=tf.int32),
        tf.clip_by_value(mask, 0, 255),
    )
    return tf.where(valid, mapped, IGNORE_LABEL)


def remap_cityscapes_label_ids_tensor(mask):
    """Map original Cityscapes labelIds to MicroSeg IDs inside TensorFlow."""
    mask = tf.cast(mask, tf.int32)
    valid = (mask >= 0) & (mask < 256)
    mapped = tf.gather(
        tf.constant(cityscapes_label_ids_lookup(), dtype=tf.int32),
        tf.clip_by_value(mask, 0, 255),
    )
    return tf.where(valid, mapped, IGNORE_LABEL)


def prepare_targets(mask, weights=None):
    """Return safe class IDs and sample weights; always keep both together.

    ``weights`` optionally scales per-pixel supervision. Values above zero
    mark a pixel as supervised and scale its contribution, which lets a
    source contribute weak evidence without claiming full confidence.
    ``None`` keeps full weight on every supervised pixel.
    """
    mask = tf.convert_to_tensor(mask)
    if not mask.dtype.is_integer:
        raise TypeError("MicroSeg masks must contain integer IDs.")
    mask = tf.cast(mask, tf.int32)
    valid = (mask >= 0) & (mask < NUM_CLASSES)
    tf.debugging.assert_equal(
        tf.reduce_all(valid | (mask == IGNORE_LABEL)), True,
        message="MicroSeg masks must contain only 0..3 or 255",
    )
    supervised = tf.cast(valid, tf.float32)
    if weights is None:
        scaled = supervised
    else:
        scaled = supervised * tf.cast(weights, tf.float32)
    return tf.where(valid, mask, 0), scaled


def load_pair(image_path, mask_path, label_format="auto", ignore_mask_path=None):
    """Load Cityscapes labels and optionally apply a reviewed binary ignore mask."""
    raw_image = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
    raw_mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=1)
    tf.debugging.assert_equal(
        tf.shape(raw_image)[:2],
        tf.shape(raw_mask)[:2],
        message="Cityscapes image and semantic mask dimensions must match.",
    )

    if ignore_mask_path is None:
        raw_ignore_mask = None
    else:
        ignore_mask_path = tf.convert_to_tensor(ignore_mask_path, dtype=tf.string)

        def read_ignore_mask():
            ignore = tf.io.decode_png(
                tf.io.read_file(ignore_mask_path), channels=1
            )
            tf.debugging.assert_equal(
                tf.shape(raw_image)[:2],
                tf.shape(ignore)[:2],
                message="Cityscapes image and ignore mask dimensions must match.",
            )
            binary_values = (
                tf.equal(ignore, 0)
                | tf.equal(ignore, 1)
                | tf.equal(ignore, 255)
            )
            tf.debugging.assert_equal(
                tf.reduce_all(binary_values),
                True,
                message="Cityscapes ignore masks must contain only 0, 1, or 255.",
            )
            return ignore

        raw_ignore_mask = tf.cond(
            tf.strings.length(ignore_mask_path) > 0,
            read_ignore_mask,
            lambda: tf.zeros_like(raw_mask),
        )

    image = tf.image.resize(
        raw_image, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.BILINEAR
    )
    image = tf.cast(image, tf.float32) / 255.0

    if label_format == "auto":
        is_label_ids = tf.strings.regex_full_match(
            mask_path, r".*_labelIds\.png"
        )
        mask = tf.cond(
            is_label_ids,
            lambda: remap_cityscapes_label_ids_tensor(raw_mask),
            lambda: remap_cityscapes_tensor(raw_mask),
        )
    elif label_format == "labelIds":
        mask = remap_cityscapes_label_ids_tensor(raw_mask)
    elif label_format == "trainIds":
        mask = remap_cityscapes_tensor(raw_mask)
    else:
        raise ValueError(
            "label_format must be 'auto', 'labelIds', or 'trainIds', "
            f"got {label_format!r}."
        )
    # Remap at native resolution, then nearest-neighbor resize so no invalid
    # class IDs are introduced between categories.
    mask = tf.image.resize(
        mask, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    mask = tf.squeeze(mask, axis=-1)
    if raw_ignore_mask is not None:
        ignore_mask = tf.image.resize(
            raw_ignore_mask,
            [IMAGE_HEIGHT, IMAGE_WIDTH],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        ignore_mask = tf.squeeze(ignore_mask, axis=-1)
        mask = tf.where(ignore_mask > 0, IGNORE_LABEL, mask)
    targets, weights = prepare_targets(mask)
    return image, targets, weights


def make_dataset(
    split,
    training=False,
    batch_size=None,
    cityscapes_pairs=None,
    cityscapes_ignore_root=None,
):
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if cityscapes_pairs is None:
        image_paths, mask_paths = collect_cityscapes_pairs(split)
    else:
        if not cityscapes_pairs:
            raise ValueError("Cityscapes pairs must not be empty.")
        cityscapes_pairs = list(cityscapes_pairs)
    if cityscapes_ignore_root is not None:
        if cityscapes_pairs is None:
            cityscapes_pairs = list(zip(image_paths, mask_paths))
        elif any(len(pair) != 2 for pair in cityscapes_pairs):
            raise ValueError(
                "Pass two-path Cityscapes pairs with cityscapes_ignore_root, "
                "or pass explicit three-path pairs, not both."
            )
        cityscapes_pairs = attach_cityscapes_ignore_masks(
            cityscapes_pairs, cityscapes_ignore_root
        )
    if cityscapes_pairs is None:
        cityscapes_pairs = list(zip(image_paths, mask_paths))

    normalized_pairs = []
    for pair in cityscapes_pairs:
        if len(pair) == 2:
            normalized_pairs.append((*pair, ""))
        elif len(pair) == 3:
            normalized_pairs.append(pair)
        else:
            raise ValueError(
                "Cityscapes pairs must contain image/mask paths and an optional "
                f"ignore-mask path, got {pair!r}."
            )
    image_paths, mask_paths, ignore_paths = zip(*normalized_pairs)
    dataset = tf.data.Dataset.from_tensor_slices(
        (list(image_paths), list(mask_paths), list(ignore_paths))
    )
    if training:
        dataset = dataset.shuffle(
            len(normalized_pairs), reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        lambda image, mask, ignore: load_pair(
            image, mask, ignore_mask_path=ignore
        ),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def make_combined_dataset(
    split,
    training=False,
    batch_size=None,
    include_crosswalk=True,
    crosswalk_weight=CROSSWALK_SAMPLE_WEIGHT,
    crosswalk_root=None,
    crosswalk_pairs=None,
    crosswalk_real_pairs=None,
    crosswalk_synthetic_pairs=None,
    crosswalk_real_weight=0.5,
    cityscapes_pairs=None,
    cityscapes_ignore_root=None,
    crosswalk_negative_weight=0.0,
):
    """Interleave Cityscapes and Kaggle batches for MicroSeg training.

    Kaggle is included only for the training split. When real and synthetic
    pair lists are supplied separately, their streams can be sampled at a
    controlled ratio; training repeats each stream to avoid exhausting the
    smaller source. Cityscapes validation remains the return value for
    validation so Kaggle training data cannot leak into that metric.
    """
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    cityscapes = make_dataset(
        split,
        training=training,
        batch_size=batch_size,
        cityscapes_pairs=cityscapes_pairs,
        cityscapes_ignore_root=cityscapes_ignore_root,
    )
    if split != "train" or not include_crosswalk:
        return cityscapes
    if not 0.0 < crosswalk_weight < 1.0:
        raise ValueError("crosswalk_weight must be between 0 and 1.")

    if crosswalk_real_pairs is not None or crosswalk_synthetic_pairs is not None:
        if crosswalk_pairs is not None:
            raise ValueError(
                "Pass either crosswalk_pairs or source-specific Crosswalk pairs, not both."
            )
        if not crosswalk_real_pairs or not crosswalk_synthetic_pairs:
            raise ValueError(
                "Both real and synthetic Crosswalk pairs are required for "
                "source-balanced sampling."
            )
        if not 0.0 < crosswalk_real_weight < 1.0:
            raise ValueError("crosswalk_real_weight must be between 0 and 1.")
        real_crosswalk = make_crosswalk_dataset(
            training=training,
            batch_size=batch_size,
            root=crosswalk_root,
            pairs=crosswalk_real_pairs,
            negative_weight=crosswalk_negative_weight,
        )
        synthetic_crosswalk = make_crosswalk_dataset(
            training=training,
            batch_size=batch_size,
            root=crosswalk_root,
            pairs=crosswalk_synthetic_pairs,
            negative_weight=crosswalk_negative_weight,
        )
        if training:
            real_crosswalk = real_crosswalk.repeat()
            synthetic_crosswalk = synthetic_crosswalk.repeat()
        crosswalk = tf.data.Dataset.sample_from_datasets(
            [real_crosswalk, synthetic_crosswalk],
            weights=[crosswalk_real_weight, 1.0 - crosswalk_real_weight],
            rerandomize_each_iteration=training,
        )
    else:
        crosswalk = make_crosswalk_dataset(
            training=training,
            batch_size=batch_size,
            root=crosswalk_root,
            pairs=crosswalk_pairs,
            negative_weight=crosswalk_negative_weight,
        )
    return tf.data.Dataset.sample_from_datasets(
        [cityscapes, crosswalk],
        weights=[1.0 - crosswalk_weight, crosswalk_weight],
        rerandomize_each_iteration=training,
    )


if __name__ == "__main__":
    print("\n========== MICROSEG DATASET TEST ==========")
    print(f"\nResolution: {IMAGE_HEIGHT} x {IMAGE_WIDTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print("\nCityscapes trainId -> MicroSeg ID:")
    lookup = cityscapes_lookup()
    for index in range(19):
        print(f"  {index:3d} -> {int(lookup[index])}")
