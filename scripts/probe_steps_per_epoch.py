"""Empirically probe how --steps-per-epoch affects the batches actually seen.

Rebuilds the real training pipeline and iterates it, recording the sequence of
image-file identities per step. Reports whether step counts below/above a
threshold see new samples or replay the same prefix.

Run under WSL:
    bash run_wsl.sh scripts/probe_steps_per_epoch.py --steps 50 100 200 400
"""
import argparse
import collections
import json
from pathlib import Path
from unittest import mock

import numpy as np
import tensorflow as tf

from dataset_config import CITYSCAPES_ROOT, DEFAULT_KAGGLE_CROSSWALK_ROOT
from microseg.dataset import IMAGE_HEIGHT, IMAGE_WIDTH, make_combined_dataset
from microseg.train import (
    _load_or_create_manifest,
    _pairs_from_manifest,
    parse_args,
)


def build(args, workdir):
    root = Path(DEFAULT_KAGGLE_CROSSWALK_ROOT)
    manifest = _load_or_create_manifest(
        workdir,
        root,
        args.seed,
        args.validation_fraction,
        False,
        source_manifest=args.split_manifest,
    )
    real = _pairs_from_manifest(manifest, root, "train", source="real")
    synthetic = _pairs_from_manifest(manifest, root, "train", source="synthetic")
    city_train = _pairs_from_manifest(manifest, root, "train")
    # make_combined_dataset only needs the Cityscapes pair list here.
    return make_combined_dataset(
        "train",
        training=True,
        batch_size=args.batch_size,
        crosswalk_weight=args.crosswalk_weight,
        crosswalk_root=root,
        crosswalk_real_pairs=real,
        crosswalk_synthetic_pairs=synthetic,
        crosswalk_real_weight=args.crosswalk_real_weight,
        cityscapes_pairs=city_train,
        crosswalk_negative_weight=args.crosswalk_negative_weight,
        crosswalk_horizon_fraction=args.crosswalk_horizon_fraction,
        augment=args.augment,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--cityscapes-manifest", type=Path, default=None)
    known, rest = parser.parse_known_args()
    args = parse_args(rest)
    args.cityscapes_manifest = known.cityscapes_manifest

    workdir = Path("/home/nacsi/microseg_runs/probe_steps")
    workdir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(args.seed)
    sequences = {}
    for count in known.steps:
        # A fresh directory per step count: _load_or_create_manifest refuses to
        # overwrite, so reuse would abort on the second iteration.
        step_dir = workdir / f"steps{count}"
        step_dir.mkdir(parents=True, exist_ok=True)
        dataset = build(args, step_dir)
        # Record the per-step *label mix* as well as the image mean, so the
        # Cityscapes/Crosswalk balance can be measured, not guessed at.
        seen = []
        for images, targets, _ in dataset.take(count):
            labels = np.asarray(targets)
            # Crosswalk is class 3; Cityscapes pairs never contain it.
            crosswalk_pixels = float(np.count_nonzero(labels == 3)) / labels.size
            seen.append(
                (
                    round(float(tf.reduce_mean(images)), 6),
                    round(crosswalk_pixels, 6),
                )
            )
        sequences[count] = seen
        print(f"steps={count}: collected {len(seen)} batches", flush=True)

    print("\n--- Crosswalk share of batches actually drawn ---")
    for count in sorted(known.steps):
        rows = sequences[count]
        with_crosswalk = sum(1 for _, share in rows if share > 0.0)
        total_crosswalk_pixels = sum(share for _, share in rows)
        print(
            f"steps={count:>4}: batches_containing_crosswalk="
            f"{with_crosswalk:>4}/{len(rows)} "
            f"({with_crosswalk / len(rows):.1%})  "
            f"mean_crosswalk_pixel_share={total_crosswalk_pixels / len(rows):.4f}"
        )

    print("\n--- are distinct step counts seeing distinct batches? ---")
    for count in sorted(known.steps):
        rows = sequences[count]
        print(f"steps={count:>4}: distinct_batch_signatures={len(set(rows)):>4}")

    print("\n--- Cityscapes stream exhaustion check ---")
    # make_dataset() does not repeat(), so the Cityscapes branch is finite:
    # ceil(2975 / 8) = 372 batches. Once exhausted, sample_from_datasets drops
    # it and every remaining batch must come from Crosswalk.
    city_batches_needed = known.steps[-1]
    print(
        "A Cityscapes-only epoch at batch 8 needs about "
        f"{city_batches_needed} batches to cover the training split."
    )

    # Direct measurement: label each batch with which *branch* produced it by
    # using synthetic stand-ins that differ only by constant image value, so a
    # branch can be identified regardless of its Crosswalk pixel count.
    print("\n--- direct branch measurement (synthetic stand-ins) ---")
    for count in sorted(known.steps):
        step_dir = workdir / f"branch{count}"
        step_dir.mkdir(parents=True, exist_ok=True)
        manifest = _load_or_create_manifest(
            step_dir,
            Path(DEFAULT_KAGGLE_CROSSWALK_ROOT),
            args.seed,
            args.validation_fraction,
            False,
            source_manifest=args.split_manifest,
        )
        root = Path(DEFAULT_KAGGLE_CROSSWALK_ROOT)
        real = _pairs_from_manifest(manifest, root, "train", source="real")
        synthetic = _pairs_from_manifest(manifest, root, "train", source="synthetic")
        city_train = _pairs_from_manifest(manifest, root, "train")
        city = tf.data.Dataset.from_tensors(
            (
                tf.fill((args.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3), 0.2),
                tf.ones(
                    (args.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=tf.int32
                ),
                tf.ones((args.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH)),
            )
        ).take(372)
        cross = tf.data.Dataset.from_tensors(
            (
                tf.fill((args.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3), 0.8),
                tf.ones(
                    (args.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=tf.int32
                ),
                tf.ones((args.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH)),
            )
        )
        with mock.patch(
            "microseg.dataset.make_dataset", return_value=city
        ), mock.patch(
            "microseg.dataset.make_crosswalk_dataset",
            side_effect=lambda **kwargs: cross.repeat(),
        ):
            combined = make_combined_dataset(
                "train",
                training=True,
                batch_size=args.batch_size,
                crosswalk_weight=args.crosswalk_weight,
                crosswalk_root=root,
                crosswalk_real_pairs=real,
                crosswalk_synthetic_pairs=synthetic,
                crosswalk_real_weight=args.crosswalk_real_weight,
                cityscapes_pairs=city_train,
                crosswalk_negative_weight=args.crosswalk_negative_weight,
                crosswalk_horizon_fraction=args.crosswalk_horizon_fraction,
                augment=args.augment,
                seed=args.seed,
            )
            values = [
                float(tf.reduce_mean(images))
                for images, _, _ in combined.take(count)
            ]
        share = sum(1 for value in values if value > 0.5) / len(values)
        print(
            f"steps={count:>4}: realised_crosswalk_share={share:.1%} "
            f"(requested {args.crosswalk_weight:.0%})"
        )

if __name__ == "__main__":
    main()