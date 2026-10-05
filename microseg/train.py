"""Train the four-class MicroSeg model with a reproducible Crosswalk split."""
import argparse
import json
import math
import random
import re
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from dataset_config import CITYSCAPES_ROOT, KAGGLE_CROSSWALK_ROOT
from microseg import dataset as microseg_dataset
from microseg.dataset import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    attach_cityscapes_ignore_masks,
    collect_crosswalk_pairs,
    make_combined_dataset,
    make_crosswalk_dataset,
    make_dataset,
)
from microseg.model import NUM_CLASSES, build_microseg
from microseg.taxonomy import CLASS_NAMES
from segmentation_metrics import make_metrics

_MANIFEST_NAME = "crosswalk_split.json"
_STATE_NAME = "state.json"
_CLIP_FRAME = re.compile(r"^(?P<video>.+\.mp4)_\d+\s+\(Custom\)$", re.IGNORECASE)


def create_crosswalk_split_manifest(root, seed=42, validation_fraction=0.2):
    """Create a deterministic split that keeps frames from each source video together."""
    root = Path(root).resolve()
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    image_paths, mask_paths = collect_crosswalk_pairs(root)
    records = []
    real_groups_by_condition = {}
    for image_path, mask_path in zip(image_paths, mask_paths):
        image = Path(image_path)
        relative = image.relative_to(root)
        parts = relative.parts
        source = parts[0].casefold()
        if source == "real-world dataset" and len(parts) >= 4:
            condition = parts[1].casefold()
            match = _CLIP_FRAME.fullmatch(image.stem)
            if match is None:
                raise ValueError(
                    "Cannot identify source video for real Crosswalk image: "
                    f"{image}"
                )
            group = match.group("video").casefold()
            real_groups_by_condition.setdefault(condition, set()).add(group)
        elif source == "synthetic dataset":
            condition = parts[1].casefold() if len(parts) >= 4 else "unknown"
            group = None
        else:
            raise ValueError(
                f"Crosswalk pair is outside Real-world/Synthetic dataset folders: "
                f"{image}"
            )
        records.append({
            "image": relative.as_posix(),
            "mask": Path(mask_path).resolve().relative_to(root).as_posix(),
            "source": "real" if group is not None else "synthetic",
            "condition": condition,
            "group": group,
        })

    validation_groups = set()
    for condition, groups in sorted(real_groups_by_condition.items()):
        if len(groups) < 2:
            raise ValueError(
                f"Condition {condition!r} needs at least two source-video groups "
                "for a leakage-safe train/validation split."
            )
        ordered = sorted(groups)
        random.Random(f"{seed}:{condition}").shuffle(ordered)
        validation_count = min(
            len(groups) - 1, max(1, round(len(groups) * validation_fraction))
        )
        validation_groups.update(ordered[:validation_count])

    for record in records:
        record["split"] = (
            "validation"
            if record["source"] == "real" and record["group"] in validation_groups
            else "train"
        )

    train_groups = {
        item["group"] for item in records
        if item["source"] == "real" and item["split"] == "train"
    }
    validation_groups_found = {
        item["group"] for item in records
        if item["source"] == "real" and item["split"] == "validation"
    }
    overlap = train_groups & validation_groups_found
    if overlap:
        raise AssertionError(f"Source-video leakage across splits: {sorted(overlap)}")

    return {
        "schema_version": 1,
        "dataset_root": str(root),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "items": records,
    }


def _pairs_from_manifest(manifest, root, split, source=None):
    pairs = []
    for item in manifest["items"]:
        if item["split"] != split or (source and item["source"] != source):
            continue
        image_relative, mask_relative = Path(item["image"]), Path(item["mask"])
        if (
            image_relative.is_absolute()
            or mask_relative.is_absolute()
            or ".." in image_relative.parts
            or ".." in mask_relative.parts
        ):
            raise ValueError(f"Crosswalk paths must be relative to {root}: {item!r}")
        image = root / image_relative
        mask = root / mask_relative
        pairs.append((str(image), str(mask)))
    return pairs


def _read_cityscapes_manifest(path, root):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported Cityscapes pair manifest: {path}")
    if Path(manifest.get("dataset_root", "")).resolve() != root.resolve():
        raise ValueError(
            f"Manifest dataset root {manifest.get('dataset_root')} does not match {root}."
        )
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"Malformed Cityscapes pair manifest: {path}")

    result = {}
    for split in ("train", "val"):
        entries = splits.get(split)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Cityscapes manifest has no {split!r} pairs: {path}")
        pairs = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Malformed Cityscapes pair in {path}: {entry!r}")
            image, mask = entry.get("image"), entry.get("mask")
            if not isinstance(image, str) or not isinstance(mask, str):
                raise ValueError(f"Malformed Cityscapes pair in {path}: {entry!r}")
            paths = [image, mask]
            if "ignore_mask" in entry:
                if not isinstance(entry["ignore_mask"], str):
                    raise ValueError(
                        f"Malformed Cityscapes ignore-mask path in {path}: {entry!r}"
                    )
                paths.append(entry["ignore_mask"])
            relatives = [Path(value) for value in paths]
            if any(
                relative.is_absolute() or ".." in relative.parts
                for relative in relatives
            ):
                raise ValueError(
                    f"Cityscapes paths must be relative to {root}: {entry!r}"
                )
            pairs.append(tuple(str(root / relative) for relative in relatives))
        result[split] = pairs
    return result


def _read_manifest(path, root):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported Crosswalk split manifest: {path}")
    if Path(manifest["dataset_root"]).resolve() != root.resolve():
        raise ValueError(
            f"Manifest dataset root {manifest['dataset_root']} does not match {root}."
        )
    required_splits = {"train", "validation"}
    if not isinstance(manifest.get("items"), list) or any(
        item.get("split") not in required_splits
        or item.get("source") not in {"real", "synthetic"}
        or not isinstance(item.get("image"), str)
        or not isinstance(item.get("mask"), str)
        for item in manifest["items"]
    ):
        raise ValueError(f"Malformed Crosswalk split manifest: {path}")
    return manifest


def _load_or_create_manifest(
    output, root, seed, validation_fraction, resume, source_manifest=None
):
    path = output / _MANIFEST_NAME
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"Resume requires {path}")
        return _read_manifest(path, root)
    if path.exists():
        raise FileExistsError(f"{path} already exists; use --resume or another run directory.")
    if source_manifest is not None:
        manifest = _read_manifest(source_manifest, root)
    else:
        manifest = create_crosswalk_split_manifest(root, seed, validation_fraction)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _positive_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _sampling_weight(value):
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be between 0 and 1") from error
    if not 0.0 < number < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


class VerifyFinite(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            if value is not None and not np.isfinite(value):
                raise FloatingPointError(f"Non-finite {name} at epoch {epoch + 1}.")


class SmokeChecks(keras.callbacks.Callback):
    """Verify the smoke run updates weights and validation leaves them intact."""

    def on_train_begin(self, logs=None):
        self.initial_weights = [weight.copy() for weight in self.model.get_weights()]

    def on_test_begin(self, logs=None):
        self.before_validation = [weight.copy() for weight in self.model.get_weights()]

    def on_test_end(self, logs=None):
        for before, after in zip(self.before_validation, self.model.get_weights()):
            np.testing.assert_array_equal(before, after)
        print("Validation left all model weights unchanged.", flush=True)

    def on_train_end(self, logs=None):
        changes = [
            np.max(np.abs(before - after))
            for before, after in zip(self.initial_weights, self.model.get_weights())
        ]
        change = max(changes, default=0.0)
        if not np.isfinite(change) or change == 0:
            raise AssertionError("Training did not produce a finite model-weight update.")
        print(f"Verified training weight change: {change:.8f}", flush=True)


class CrosswalkValidation(keras.callbacks.Callback):
    """Evaluate held-out real Crosswalk clips separately from Cityscapes."""

    def __init__(self, dataset, smoke_test=False):
        super().__init__()
        self.dataset = dataset.take(1) if smoke_test else dataset

    def on_epoch_end(self, epoch, logs=None):
        before = [weight.copy() for weight in self.model.get_weights()]
        values = self.model.evaluate(self.dataset, verbose=0, return_dict=True)
        for before_weight, after_weight in zip(before, self.model.get_weights()):
            np.testing.assert_array_equal(before_weight, after_weight)
        if logs is not None:
            logs.update({f"val_crosswalk_{key}": float(value) for key, value in values.items()})
        print(
            "Held-out real Crosswalk metrics:",
            {key: float(value) for key, value in values.items()},
            flush=True,
        )


class CrosswalkFullFrameMetrics(keras.callbacks.Callback):
    """Score Crosswalk across the whole frame so over-prediction is visible.

    The Kaggle label contract marks only the crossing and ignores every other
    pixel, so the weighted validation metrics can only ever see the marked
    quad. A model that paints Crosswalk across the whole image is not
    penalised, because the flood lands on pixels with zero sample weight: it
    scores perfect recall and a near-perfect pixel accuracy while being
    completely wrong. This callback recomputes the Crosswalk confusion matrix
    over every pixel, so a prediction on an unmarked pixel counts as a false
    positive. Precision and the predicted-pixel fraction are therefore honest
    flooding measures, and the full-frame IoU is safe to monitor.
    """

    def __init__(self, dataset, class_id=3):
        super().__init__()
        self.dataset = dataset
        self.class_id = class_id

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        true_positive = false_positive = false_negative = true_negative = 0
        for images, targets, _ in self.dataset:
            predicted = np.argmax(
                np.asarray(self.model(images, training=False)), axis=-1
            )
            truth = np.asarray(targets)
            predicted_crosswalk = predicted == self.class_id
            true_crosswalk = truth == self.class_id
            true_positive += int(np.count_nonzero(predicted_crosswalk & true_crosswalk))
            false_positive += int(np.count_nonzero(predicted_crosswalk & ~true_crosswalk))
            false_negative += int(np.count_nonzero(~predicted_crosswalk & true_crosswalk))
            true_negative += int(np.count_nonzero(~predicted_crosswalk & ~true_crosswalk))

        total = true_positive + false_positive + false_negative + true_negative
        if total == 0:
            raise ValueError("Crosswalk full-frame evaluation saw no pixels.")
        predicted_pixels = true_positive + false_positive
        marked_pixels = true_positive + false_negative
        union = true_positive + false_positive + false_negative
        precision = true_positive / predicted_pixels if predicted_pixels else 0.0
        recall = true_positive / marked_pixels if marked_pixels else 0.0
        iou = true_positive / union if union else 0.0
        logs["val_crosswalk_fullframe_precision"] = float(precision)
        logs["val_crosswalk_fullframe_recall"] = float(recall)
        logs["val_crosswalk_fullframe_iou"] = float(iou)
        logs["val_crosswalk_predicted_fraction"] = float(predicted_pixels / total)
        logs["val_crosswalk_true_fraction"] = float(marked_pixels / total)
        print(
            "Crosswalk full-frame: "
            f"IoU={iou:.4f} precision={precision:.4f} recall={recall:.4f} "
            f"predicted={predicted_pixels / total:.2%} of frame "
            f"(marked={marked_pixels / total:.2%})",
            flush=True,
        )


COMBINED_SCORE = "val_combined_score"


class CombinedScore(keras.callbacks.Callback):
    """Blend both validation sources into one checkpoint/LR metric.

    The Cityscapes split only supervises Walkable, Non-walkable and Boundary,
    and the held-out real Crosswalk split only supervises Crosswalk. Selecting
    a checkpoint on one source alone can therefore promote a model that has
    collapsed the other half of the taxonomy, which is exactly how a strong
    Cityscapes score was paired with a decaying Crosswalk score.
    """

    def __init__(
        self,
        cityscapes_key="val_mean_iou",
        crosswalk_key="val_crosswalk_fullframe_iou",
        cityscapes_weight=0.5,
    ):
        super().__init__()
        if not 0.0 <= cityscapes_weight <= 1.0:
            raise ValueError("cityscapes_weight must be between 0 and 1.")
        self.cityscapes_key = cityscapes_key
        self.crosswalk_key = crosswalk_key
        self.cityscapes_weight = cityscapes_weight

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        missing = [
            key for key in (self.cityscapes_key, self.crosswalk_key) if key not in logs
        ]
        if missing:
            raise ValueError(f"Combined score is missing validation metrics: {missing}")
        score = (
            self.cityscapes_weight * float(logs[self.cityscapes_key])
            + (1.0 - self.cityscapes_weight) * float(logs[self.crosswalk_key])
        )
        if not math.isfinite(score):
            raise FloatingPointError(
                f"Non-finite combined score at epoch {epoch}: {score}"
            )
        logs[COMBINED_SCORE] = score


class ValidationDriftGuard(keras.callbacks.Callback):
    """Fail fast when validation decisions freeze or Crosswalk collapses.

    IoU, Dice and pixel accuracy are derived from an argmax confusion matrix,
    so they only move when a predicted class changes. If they stay exactly
    equal while the validation loss keeps improving, the model is still
    refining logits without changing a single decision: a class-collapse
    signature that a best-checkpoint rule would otherwise keep selecting
    without comment. A Crosswalk mean IoU of exactly zero means the model
    predicted no Crosswalk pixel at all on the held-out real split.
    """

    def __init__(self, patience=3):
        super().__init__()
        if patience < 0:
            raise ValueError("patience must not be negative.")
        self.patience = patience
        self.previous = None
        self.frozen_epochs = 0
        self.empty_crosswalk_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or not self.patience:
            return
        current = (
            float(logs["val_mean_iou"]),
            float(logs["val_pixel_accuracy"]),
            float(logs["val_dice"]),
        )
        if self.previous is not None and current == self.previous:
            self.frozen_epochs += 1
        else:
            self.frozen_epochs = 0
        self.previous = current

        if float(logs["val_crosswalk_fullframe_iou"]) == 0.0:
            self.empty_crosswalk_epochs += 1
        else:
            self.empty_crosswalk_epochs = 0

        if self.frozen_epochs >= self.patience:
            raise RuntimeError(
                f"Cityscapes validation decisions did not change for "
                f"{self.frozen_epochs} consecutive epochs while the model kept "
                f"training. mIoU/pixel accuracy/Dice are frozen at {current}, so "
                f"the run is refining logits without changing predictions. "
                f"Stop and inspect the training signal before spending more epochs."
            )
        if self.empty_crosswalk_epochs >= self.patience:
            raise RuntimeError(
                f"Held-out real Crosswalk mean IoU stayed exactly 0.0 for "
                f"{self.empty_crosswalk_epochs} consecutive epochs: the model "
                f"predicts no Crosswalk pixels at all. Stop and rebalance the "
                f"Crosswalk sampling weight or fix the Crosswalk label contract."
            )


class SaveLatestState(keras.callbacks.Callback):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def on_epoch_end(self, epoch, logs=None):
        pending = self.output / "last.pending.keras"
        self.model.save(pending)
        pending.replace(self.output / "last.keras")
        state = {
            "completed_epochs": epoch + 1,
            "optimizer_iterations": int(self.model.optimizer.iterations.numpy()),
        }
        temporary = self.output / "state.pending.json"
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output / _STATE_NAME)


def _unit_interval(value):
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=_positive_int, default=50)
    parser.add_argument("--batch-size", type=_positive_int, default=microseg_dataset.BATCH_SIZE)
    parser.add_argument("--steps-per-epoch", type=_positive_int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--crosswalk-weight",
        type=_sampling_weight,
        default=microseg_dataset.CROSSWALK_SAMPLE_WEIGHT,
        help="Probability of sampling Crosswalk batches during training",
    )
    parser.add_argument(
        "--crosswalk-real-weight",
        type=_sampling_weight,
        default=0.5,
        help="Fraction of sampled Crosswalk batches drawn from real training images",
    )
    parser.add_argument(
        "--cityscapes-score-weight",
        type=float,
        default=0.5,
        help=(
            "Weight of the Cityscapes validation mIoU inside the combined "
            "checkpoint score; the remainder is the held-out Crosswalk mIoU"
        ),
    )
    parser.add_argument(
        "--crosswalk-negative-weight",
        type=_unit_interval,
        default=0.25,
        help=(
            "Sample weight for unmarked pixels in Crosswalk training frames, "
            "which become weak negative evidence against a Crosswalk flood. "
            "0 disables negative supervision. Evaluation is never affected."
        ),
    )
    parser.add_argument(
        "--frozen-metric-patience",
        type=int,
        default=3,
        help=(
            "Consecutive epochs with bit-identical validation decisions, or a "
            "Crosswalk mIoU of exactly 0, that abort the run. Use 0 to disable."
        ),
    )
    parser.add_argument("--crosswalk-root", type=Path, default=KAGGLE_CROSSWALK_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path.home() / "microseg_runs" / "run"
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Reuse an existing split manifest instead of rescanning the dataset tree",
    )
    parser.add_argument(
        "--cityscapes-manifest",
        type=Path,
        help="Reuse Cityscapes train/validation pair paths from a JSON manifest",
    )
    parser.add_argument(
        "--cityscapes-ignore-root",
        type=Path,
        help=(
            "Reviewed masks mirrored as {root}/{split}/{city}/{image-stem}_ignore.png; "
            "every Cityscapes pair must have one"
        ),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.crosswalk_root.resolve()
    output = args.output_dir.resolve()
    if args.resume and args.split_manifest:
        raise ValueError("--split-manifest cannot be used together with --resume.")
    if args.resume and args.cityscapes_ignore_root:
        raise ValueError(
            "--cityscapes-ignore-root is stored in the run manifest; "
            "do not pass it with --resume."
        )
    if args.cityscapes_manifest and args.cityscapes_ignore_root:
        raise ValueError(
            "Pass Cityscapes ignore masks through the manifest or "
            "--cityscapes-ignore-root, not both."
        )
    if args.cityscapes_ignore_root and not args.cityscapes_ignore_root.is_dir():
        raise FileNotFoundError(
            f"Cityscapes ignore-mask root does not exist: "
            f"{args.cityscapes_ignore_root}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"Crosswalk dataset root does not exist: {root}")
    if args.split_manifest is not None and not args.split_manifest.is_file():
        raise FileNotFoundError(f"Crosswalk split manifest does not exist: {args.split_manifest}")
    if args.cityscapes_manifest is not None and not args.cityscapes_manifest.is_file():
        raise FileNotFoundError(
            f"Cityscapes pair manifest does not exist: {args.cityscapes_manifest}"
        )
    if not CITYSCAPES_ROOT.is_dir():
        raise FileNotFoundError(f"Cityscapes dataset root does not exist: {CITYSCAPES_ROOT}")
    if args.resume:
        for name in (
            _MANIFEST_NAME, "cityscapes_pairs.json", _STATE_NAME,
            "last.keras", "history.csv",
        ):
            if not (output / name).is_file():
                raise FileNotFoundError(f"Resume requires {output / name}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; use --resume or choose another directory.")

    keras.utils.set_random_seed(args.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    print(
        f"Devices: {tf.config.list_physical_devices('GPU') or 'CPU'}; "
        f"input={IMAGE_HEIGHT}x{IMAGE_WIDTH}; batch={args.batch_size}",
        flush=True,
    )

    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(
        output,
        root,
        args.seed,
        args.validation_fraction,
        args.resume,
        source_manifest=args.split_manifest,
    )
    crosswalk_train_pairs = _pairs_from_manifest(manifest, root, "train")
    crosswalk_train_real_pairs = _pairs_from_manifest(
        manifest, root, "train", source="real"
    )
    crosswalk_train_synthetic_pairs = _pairs_from_manifest(
        manifest, root, "train", source="synthetic"
    )
    crosswalk_validation_pairs = _pairs_from_manifest(
        manifest, root, "validation", source="real"
    )
    crosswalk_train_real_count = sum(
        item["source"] == "real" and item["split"] == "train"
        for item in manifest["items"]
    )
    crosswalk_train_synthetic_count = sum(
        item["source"] == "synthetic" and item["split"] == "train"
        for item in manifest["items"]
    )
    if not crosswalk_train_pairs or not crosswalk_validation_pairs:
        raise ValueError("Crosswalk split must contain training and held-out real examples.")
    if not crosswalk_train_real_pairs or not crosswalk_train_synthetic_pairs:
        raise ValueError(
            "Crosswalk training split must contain both real and synthetic examples."
        )
    cityscapes_manifest_path = (
        output / "cityscapes_pairs.json" if args.resume else args.cityscapes_manifest
    )
    if cityscapes_manifest_path is None:
        cityscape_pairs = {}
        for split in ("train", "val"):
            images, masks = microseg_dataset.collect_cityscapes_pairs(split)
            cityscape_pairs[split] = list(zip(images, masks))
            if args.cityscapes_ignore_root is not None:
                cityscape_pairs[split] = attach_cityscapes_ignore_masks(
                    cityscape_pairs[split], args.cityscapes_ignore_root
                )
    else:
        cityscape_pairs = _read_cityscapes_manifest(
            cityscapes_manifest_path, CITYSCAPES_ROOT
        )
    if not args.resume:
        serialized_splits = {}
        for split, pairs in cityscape_pairs.items():
            entries = []
            for pair in pairs:
                image, mask = pair[:2]
                entry = {
                    "image": Path(image).relative_to(CITYSCAPES_ROOT).as_posix(),
                    "mask": Path(mask).relative_to(CITYSCAPES_ROOT).as_posix(),
                }
                if len(pair) == 3 and pair[2]:
                    entry["ignore_mask"] = (
                        Path(pair[2]).relative_to(CITYSCAPES_ROOT).as_posix()
                    )
                entries.append(entry)
            serialized_splits[split] = entries
        saved_cityscapes_manifest = {
            "schema_version": 1,
            "dataset_root": str(CITYSCAPES_ROOT.resolve()),
            "splits": serialized_splits,
        }
        (output / "cityscapes_pairs.json").write_text(
            json.dumps(saved_cityscapes_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    cityscape_train_pairs = cityscape_pairs["train"]
    cityscape_validation_pairs = cityscape_pairs["val"]
    cityscapes_ignore_mask_pairs = sum(
        len(pair) == 3 and bool(pair[2])
        for split_pairs in cityscape_pairs.values()
        for pair in split_pairs
    )
    cityscape_train_images = [pair[0] for pair in cityscape_train_pairs]
    cityscape_validation_images = [pair[0] for pair in cityscape_validation_pairs]

    cityscape_train = make_dataset(
        "train",
        training=True,
        batch_size=args.batch_size,
        cityscapes_pairs=cityscape_train_pairs,
    )
    crosswalk_train = make_crosswalk_dataset(
        training=True,
        batch_size=args.batch_size,
        root=root,
        pairs=crosswalk_train_pairs,
        negative_weight=args.crosswalk_negative_weight,
    )
    cityscape_validation = make_dataset(
        "val",
        training=False,
        batch_size=args.batch_size,
        cityscapes_pairs=cityscape_validation_pairs,
    )
    crosswalk_validation = make_crosswalk_dataset(
        training=False,
        batch_size=args.batch_size,
        root=root,
        pairs=crosswalk_validation_pairs,
    )
    held_out_crosswalk_pixels = 0
    held_out_valid_pixels = 0
    crosswalk_preflight = (
        crosswalk_validation.take(1) if args.smoke_test else crosswalk_validation
    )
    for _, targets, weights in crosswalk_preflight:
        held_out_crosswalk_pixels += int(
            tf.reduce_sum(
                tf.cast(tf.equal(targets, 3) & (weights > 0), tf.int64)
            ).numpy()
        )
        held_out_valid_pixels += int(tf.reduce_sum(tf.cast(weights, tf.int64)).numpy())
    if held_out_crosswalk_pixels == 0 or held_out_valid_pixels == 0:
        raise ValueError(
            "Held-out real Crosswalk split has no supervised Crosswalk pixels "
            "after 96x96 resizing."
        )

    if args.smoke_test:
        args.epochs = 1
        train_data = cityscape_train.take(1).concatenate(crosswalk_train.take(1))
        validation_data = cityscape_validation.take(1)
        steps_per_epoch = 2
        crosswalk_eval = crosswalk_validation.take(1)
        validation_steps = 1
    else:
        train_data = make_combined_dataset(
            "train",
            training=True,
            batch_size=args.batch_size,
            crosswalk_weight=args.crosswalk_weight,
            crosswalk_real_weight=args.crosswalk_real_weight,
            crosswalk_root=root,
            crosswalk_real_pairs=crosswalk_train_real_pairs,
            crosswalk_synthetic_pairs=crosswalk_train_synthetic_pairs,
            cityscapes_pairs=cityscape_train_pairs,
            crosswalk_negative_weight=args.crosswalk_negative_weight,
        )
        validation_data = cityscape_validation
        cityscape_batches = math.ceil(
            len(cityscape_train_images) / args.batch_size
        )
        synthetic_crosswalk_batches = math.ceil(
            len(crosswalk_train_synthetic_pairs) / args.batch_size
        )
        steps_per_epoch = args.steps_per_epoch or max(
            math.ceil(cityscape_batches / (1.0 - args.crosswalk_weight)),
            cityscape_batches
            + math.ceil(
                synthetic_crosswalk_batches / (1.0 - args.crosswalk_real_weight)
            ),
        )
        validation_steps = math.ceil(
            len(cityscape_validation_images) / args.batch_size
        )
        crosswalk_eval = crosswalk_validation

    print(
        f"Crosswalk split: train={len(crosswalk_train_pairs)} "
        f"held-out real validation={len(crosswalk_validation_pairs)}; "
        f"Cityscapes train={len(cityscape_train_images)} "
        f"validation={len(cityscape_validation_images)}; "
        f"steps/epoch={steps_per_epoch}, validation_steps={validation_steps}",
        flush=True,
    )

    state = {}
    if args.resume:
        state = json.loads((output / _STATE_NAME).read_text(encoding="utf-8"))
        model = keras.models.load_model(output / "last.keras")
        if int(model.optimizer.iterations.numpy()) != state["optimizer_iterations"]:
            raise ValueError("Checkpoint and resume state disagree on optimizer steps.")
        initial_epoch = state["completed_epochs"]
    else:
        model = build_microseg()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction="mean_with_sample_weight"
            ),
            weighted_metrics=make_metrics(NUM_CLASSES),
            jit_compile=False,
        )
        initial_epoch = 0

    config_path = output / "run_config.json"
    config = {
        "seed": args.seed,
        "validation_fraction": manifest["validation_fraction"],
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
        "cityscapes_root": str(CITYSCAPES_ROOT),
        "cityscapes_manifest": "cityscapes_pairs.json",
        "cityscapes_ignore_mask_pairs": cityscapes_ignore_mask_pairs,
        "cityscapes_ignore_mask_root": (
            str(args.cityscapes_ignore_root)
            if args.cityscapes_ignore_root is not None
            else None
        ),
        "crosswalk_root": str(root),
        "crosswalk_sampling_weight": (
            0.5 if args.smoke_test else args.crosswalk_weight
        ),
        "checkpoint_metric": COMBINED_SCORE,
        "cityscapes_score_weight": args.cityscapes_score_weight,
        "crosswalk_negative_weight": args.crosswalk_negative_weight,
        "frozen_metric_patience": args.frozen_metric_patience,
        "crosswalk_train_real_pairs": crosswalk_train_real_count,
        "crosswalk_train_synthetic_pairs": crosswalk_train_synthetic_count,
        "crosswalk_validation_real_pairs": len(crosswalk_validation_pairs),
        "crosswalk_real_sampling_weight": (
            None if args.smoke_test else args.crosswalk_real_weight
        ),
        "input_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "class_names": list(CLASS_NAMES),
        "normalization": {
            "type": "group_normalization",
            "groups": 8,
        },
        "held_out_crosswalk_pixels_after_resize": held_out_crosswalk_pixels,
        "held_out_valid_pixels_after_resize": held_out_valid_pixels,
        "resume_from_epoch": initial_epoch,
        "smoke_test": args.smoke_test,
    }
    if not args.resume:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor=COMBINED_SCORE, mode="max", factor=0.5, patience=3,
        min_lr=1e-6, verbose=1,
    )
    callbacks = [
        keras.callbacks.TerminateOnNaN(),
        # CrosswalkValidation publishes val_crosswalk_* first, then the
        # full-frame pass, which cannot be gamed by ignore-as-supervision.
        # CombinedScore derives val_combined_score, and the guard runs before
        # anything is saved.
        CrosswalkValidation(crosswalk_eval, smoke_test=args.smoke_test),
        CrosswalkFullFrameMetrics(crosswalk_eval),
        CombinedScore(cityscapes_weight=args.cityscapes_score_weight),
        ValidationDriftGuard(
            patience=0 if args.smoke_test else args.frozen_metric_patience
        ),
        VerifyFinite(),
        scheduler,
        keras.callbacks.ModelCheckpoint(
            str(output / "best.keras"),
            monitor=COMBINED_SCORE,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output / "history.csv"), append=args.resume),
        SaveLatestState(output),
    ]
    if args.smoke_test:
        callbacks.append(SmokeChecks())
    history = model.fit(
        train_data,
        validation_data=validation_data,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        initial_epoch=initial_epoch,
        epochs=initial_epoch + args.epochs,
        callbacks=callbacks,
        verbose=2,
        shuffle=False,
    )
    print(
        "Final epoch:",
        {key: float(values[-1]) for key, values in history.history.items()},
        flush=True,
    )
    print(f"Saved run manifest, checkpoints, and history in {output}", flush=True)


if __name__ == "__main__":
    main()
