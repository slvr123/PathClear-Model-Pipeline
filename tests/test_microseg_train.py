"""Tests for MicroSeg's reproducible, leakage-safe Crosswalk split."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

from microseg.train import (
    COMBINED_SCORE,
    CombinedScore,
    CrosswalkFullFrameMetrics,
    ValidationDriftGuard,
    _read_cityscapes_manifest,
    create_crosswalk_split_manifest,
    parse_args,
)


class CrosswalkSplitTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def write_pair(self, relative_image):
        image_path = self.root / relative_image
        mask_path = (
            image_path.parent.parent / "masks" /
            f"{image_path.stem}_mask.jpg"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        Image.fromarray(mask).save(mask_path)

    def make_dataset(self):
        for video in ("clip_a", "clip_b", "clip_c"):
            for frame in (10, 20):
                self.write_pair(
                    Path("Real-world dataset") / "sunny" / "images" /
                    f"20260101_120000_{video}.mp4_{frame:09d} (Custom).jpg"
                )
        self.write_pair(
            Path("Synthetic dataset") / "sunny" / "images" / "img_sunny_1.jpg"
        )

    def test_split_keeps_video_frames_together_and_synthetic_in_training(self):
        self.make_dataset()
        manifest = create_crosswalk_split_manifest(self.root, seed=42)

        real_splits = {}
        for item in manifest["items"]:
            if item["source"] == "real":
                real_splits.setdefault(item["group"], set()).add(item["split"])
            else:
                self.assertEqual(item["split"], "train")

        self.assertEqual(len(real_splits), 3)
        self.assertTrue(all(len(splits) == 1 for splits in real_splits.values()))
        self.assertEqual(
            sum(item["split"] == "validation" for item in manifest["items"]), 2
        )
        self.assertEqual(
            sum(item["source"] == "synthetic" for item in manifest["items"]), 1
        )

    def test_split_is_deterministic_for_same_seed(self):
        self.make_dataset()

        first = create_crosswalk_split_manifest(self.root, seed=91)
        second = create_crosswalk_split_manifest(self.root, seed=91)

        self.assertEqual(first, second)

    def test_cli_accepts_an_explicit_crosswalk_sampling_weight(self):
        args = parse_args(["--crosswalk-weight", "0.65"])

        self.assertEqual(args.crosswalk_weight, 0.65)

    def test_cli_negative_supervision_defaults_and_overrides(self):
        self.assertEqual(parse_args([]).crosswalk_negative_weight, 0.25)
        self.assertEqual(
            parse_args(["--crosswalk-negative-weight", "0"]).crosswalk_negative_weight,
            0.0,
        )

    def test_cli_rejects_an_out_of_range_negative_weight(self):
        with self.assertRaises(SystemExit):
            parse_args(["--crosswalk-negative-weight", "1.5"])

    def test_cityscapes_manifest_resolves_relative_pair_paths(self):
        manifest_path = self.root / "cityscapes.json"
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "dataset_root": str(self.root),
            "splits": {
                split: [{
                    "image": f"leftImg8bit/{split}/city/image.png",
                    "mask": f"gtFine/{split}/city/mask.png",
                }]
                for split in ("train", "val")
            },
        }), encoding="utf-8")

        pairs = _read_cityscapes_manifest(manifest_path, self.root)

        self.assertEqual(
            pairs["train"],
            [(
                str(self.root / "leftImg8bit" / "train" / "city" / "image.png"),
                str(self.root / "gtFine" / "train" / "city" / "mask.png"),
            )],
        )

    def test_cityscapes_manifest_resolves_optional_ignore_mask_path(self):
        manifest_path = self.root / "cityscapes.json"
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "dataset_root": str(self.root),
            "splits": {
                split: [{
                    "image": f"leftImg8bit/{split}/city/image.png",
                    "mask": f"gtFine/{split}/city/mask.png",
                    "ignore_mask": f"ignore_masks/{split}/city/image_ignore.png",
                }]
                for split in ("train", "val")
            },
        }), encoding="utf-8")

        pairs = _read_cityscapes_manifest(manifest_path, self.root)

        self.assertEqual(
            pairs["val"],
            [(
                str(self.root / "leftImg8bit" / "val" / "city" / "image.png"),
                str(self.root / "gtFine" / "val" / "city" / "mask.png"),
                str(
                    self.root / "ignore_masks" / "val" / "city"
                    / "image_ignore.png"
                ),
            )],
        )

    def test_cityscapes_manifest_rejects_ignore_mask_parent_traversal(self):
        manifest_path = self.root / "cityscapes.json"
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "dataset_root": str(self.root),
            "splits": {
                "train": [{
                    "image": "image.png",
                    "mask": "mask.png",
                    "ignore_mask": "../outside.png",
                }],
                "val": [{"image": "image.png", "mask": "mask.png"}],
            },
        }), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "relative to"):
            _read_cityscapes_manifest(manifest_path, self.root)

    def test_cityscapes_manifest_rejects_parent_traversal(self):
        manifest_path = self.root / "cityscapes.json"
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "dataset_root": str(self.root),
            "splits": {
                "train": [{"image": "../outside.png", "mask": "mask.png"}],
                "val": [{"image": "image.png", "mask": "mask.png"}],
            },
        }), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "relative to"):
            _read_cityscapes_manifest(manifest_path, self.root)


class CombinedScoreTests(unittest.TestCase):
    @staticmethod
    def logs(cityscapes=0.4, crosswalk=0.2):
        return {
            "val_mean_iou": cityscapes,
            "val_pixel_accuracy": 0.8,
            "val_dice": 0.5,
            "val_crosswalk_fullframe_iou": crosswalk,
        }

    def test_default_crosswalk_key_is_the_non_gameable_metric(self):
        # The ignored-pixel Crosswalk score cannot detect flooding, so the
        # combined score must never monitor it.
        self.assertEqual(
            CombinedScore().crosswalk_key, "val_crosswalk_fullframe_iou"
        )

    def test_blends_both_validation_sources_equally_by_default(self):
        logs = self.logs()
        CombinedScore().on_epoch_end(0, logs)

        self.assertAlmostEqual(logs[COMBINED_SCORE], 0.3, places=9)

    def test_honours_an_explicit_cityscapes_weight(self):
        logs = self.logs()
        CombinedScore(cityscapes_weight=0.75).on_epoch_end(0, logs)

        self.assertAlmostEqual(logs[COMBINED_SCORE], 0.75 * 0.4 + 0.25 * 0.2, places=9)

    def test_penalises_a_collapsed_crosswalk_half(self):
        healthy = self.logs(cityscapes=0.30, crosswalk=0.30)
        collapsed = self.logs(cityscapes=0.40, crosswalk=0.0)
        CombinedScore().on_epoch_end(0, healthy)
        CombinedScore().on_epoch_end(0, collapsed)

        self.assertAlmostEqual(healthy[COMBINED_SCORE], 0.30, places=9)
        self.assertAlmostEqual(collapsed[COMBINED_SCORE], 0.20, places=9)
        # A better Cityscapes score must not rescue a collapsed Crosswalk half.
        self.assertGreater(collapsed["val_mean_iou"], healthy["val_mean_iou"])
        self.assertGreater(healthy[COMBINED_SCORE], collapsed[COMBINED_SCORE])

    def test_requires_the_crosswalk_metric(self):
        logs = {"val_mean_iou": 0.4}

        with self.assertRaisesRegex(ValueError, "missing validation metrics"):
            CombinedScore().on_epoch_end(0, logs)

    def test_rejects_a_non_finite_score(self):
        logs = self.logs(cityscapes=float("nan"))

        with self.assertRaises(FloatingPointError):
            CombinedScore().on_epoch_end(0, logs)

    def test_rejects_an_out_of_range_weight(self):
        with self.assertRaises(ValueError):
            CombinedScore(cityscapes_weight=1.5)


class ValidationDriftGuardTests(unittest.TestCase):
    @staticmethod
    def logs(miou=0.28, accuracy=0.84, dice=0.30, crosswalk=0.0):
        return {
            "val_mean_iou": miou,
            "val_pixel_accuracy": accuracy,
            "val_dice": dice,
            "val_crosswalk_fullframe_iou": crosswalk,
        }

    def test_aborts_when_validation_decisions_stop_changing(self):
        # Reproduces the pilot_cityscapes_kpv symptom: identical discrete
        # metrics every epoch while the loss keeps improving.
        guard = ValidationDriftGuard(patience=2)
        frozen = self.logs(crosswalk=0.31)

        guard.on_epoch_end(0, dict(frozen))
        guard.on_epoch_end(1, dict(frozen))
        with self.assertRaisesRegex(RuntimeError, "did not change"):
            guard.on_epoch_end(2, dict(frozen))

    def test_aborts_when_crosswalk_stays_exactly_zero(self):
        guard = ValidationDriftGuard(patience=3)

        guard.on_epoch_end(0, self.logs(miou=0.28))
        guard.on_epoch_end(1, self.logs(miou=0.29))
        with self.assertRaisesRegex(RuntimeError, "no Crosswalk pixels"):
            guard.on_epoch_end(2, self.logs(miou=0.30))

    def test_allows_metrics_that_keep_improving(self):
        guard = ValidationDriftGuard(patience=2)

        for epoch, miou in enumerate((0.20, 0.25, 0.31, 0.40)):
            guard.on_epoch_end(epoch, self.logs(miou=miou, crosswalk=0.30))

        self.assertEqual(guard.frozen_epochs, 0)
        self.assertEqual(guard.empty_crosswalk_epochs, 0)

    def test_crosswalk_recovery_resets_the_collapse_counter(self):
        guard = ValidationDriftGuard(patience=3)

        guard.on_epoch_end(0, self.logs(crosswalk=0.0))
        guard.on_epoch_end(1, self.logs(crosswalk=0.0))
        guard.on_epoch_end(2, self.logs(crosswalk=0.3))

        self.assertEqual(guard.empty_crosswalk_epochs, 0)

    def test_can_be_disabled(self):
        guard = ValidationDriftGuard(patience=0)

        for epoch in range(5):
            guard.on_epoch_end(epoch, self.logs())

        with self.assertRaises(ValueError):
            ValidationDriftGuard(patience=-1)


class _ScriptedModel:
    """Minimal stand-in for a Keras model that replays scripted predictions."""

    def __init__(self, predictions):
        self._predictions = list(predictions)
        self.calls = 0

    def __call__(self, images, training=False):
        prediction = self._predictions[self.calls]
        self.calls += 1
        logits = np.full(prediction.shape + (4,), -10.0, dtype=np.float32)
        for class_id in range(4):
            logits[..., class_id] = np.where(
                prediction == class_id, 10.0, -10.0
            )
        return logits


class CrosswalkFullFrameMetricsTests(unittest.TestCase):
    @staticmethod
    def batch(targets):
        images = np.zeros((1, 4, 4, 3), dtype=np.float32)
        # Every sample weight is zero, exactly like a Kaggle frame where only
        # the marked crossing is supervised.
        weights = np.zeros((1, 4, 4), dtype=np.float32)
        return [(images, targets, weights)]

    @staticmethod
    def marked_quad():
        # Two marked pixels out of sixteen: 12.5% of the frame.
        targets = np.zeros((1, 4, 4), dtype=np.int32)
        targets[0, 0, :2] = 3
        return targets

    def test_perfect_prediction_scores_one(self):
        targets = self.marked_quad()
        callback = CrosswalkFullFrameMetrics(self.batch(targets))
        callback.set_model(_ScriptedModel([targets]))
        logs = {}

        callback.on_epoch_end(0, logs)

        self.assertAlmostEqual(logs["val_crosswalk_fullframe_iou"], 1.0, places=9)
        self.assertAlmostEqual(logs["val_crosswalk_fullframe_precision"], 1.0, places=9)
        self.assertAlmostEqual(logs["val_crosswalk_fullframe_recall"], 1.0, places=9)
        self.assertAlmostEqual(logs["val_crosswalk_predicted_fraction"], 0.125, places=9)
        self.assertAlmostEqual(logs["val_crosswalk_true_fraction"], 0.125, places=9)

    def test_flooding_the_frame_is_penalised(self):
        # The exact failure the pilot visualizer exposed: Crosswalk painted
        # across the whole image while the marked quad is a quarter of it.
        targets = self.marked_quad()
        flooded = np.full((1, 4, 4), 3, dtype=np.int32)
        callback = CrosswalkFullFrameMetrics(self.batch(targets))
        callback.set_model(_ScriptedModel([flooded]))
        logs = {}

        callback.on_epoch_end(0, logs)

        # Perfect recall, but the flood drags IoU and precision down hard.
        self.assertAlmostEqual(logs["val_crosswalk_fullframe_recall"], 1.0, places=9)
        self.assertAlmostEqual(logs["val_crosswalk_predicted_fraction"], 1.0, places=9)
        self.assertLess(logs["val_crosswalk_fullframe_precision"], 0.25)
        self.assertLess(logs["val_crosswalk_fullframe_iou"], 0.25)
    def test_ignored_pixels_are_still_false_positives(self):
        targets = np.zeros((1, 4, 4), dtype=np.int32)
        targets[0, 0, 0] = 3
        flooded = np.full((1, 4, 4), 3, dtype=np.int32)
        callback = CrosswalkFullFrameMetrics(self.batch(targets))
        callback.set_model(_ScriptedModel([flooded]))
        logs = {}

        callback.on_epoch_end(0, logs)

        self.assertAlmostEqual(
            logs["val_crosswalk_fullframe_precision"], 1 / 16, places=9
        )

    def test_predicting_no_crosswalk_scores_zero(self):
        targets = self.marked_quad()
        absent = np.zeros((1, 4, 4), dtype=np.int32)
        callback = CrosswalkFullFrameMetrics(self.batch(targets))
        callback.set_model(_ScriptedModel([absent]))
        logs = {}

        callback.on_epoch_end(0, logs)

        self.assertEqual(logs["val_crosswalk_fullframe_iou"], 0.0)
        self.assertEqual(logs["val_crosswalk_predicted_fraction"], 0.0)

    def test_empty_dataset_is_rejected(self):
        callback = CrosswalkFullFrameMetrics([])

        with self.assertRaisesRegex(ValueError, "saw no pixels"):
            callback.on_epoch_end(0, {})


if __name__ == "__main__":
    unittest.main()
