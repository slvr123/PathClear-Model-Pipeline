"""Synthetic correctness tests; no real images, training, or checkpoint writes."""
import json
from pathlib import Path
import tempfile
import unittest

import keras
import numpy as np
from PIL import Image
import tensorflow as tf

from pathclear_pipeline import make_loss, prepare_targets, remap_cityscapes_tensor
from pathclear_taxonomy import remap_cityscapes_train_ids, colorize
from segmentation_metrics import make_metrics
from scripts.validate_pathclear_pilot import validate_manifest


class MappingTests(unittest.TestCase):
    def test_every_cityscapes_class_and_void(self):
        original = np.array(list(range(19)) + [255], dtype=np.uint8)
        before = original.copy()
        # Independent expectation covers car, motorcycle, rider, wall, fence,
        # pole, sign, vegetation, building, terrain, sky and source void.
        expected = [0, 1, 255, 4, 4, 5, 5, 5, 6, 255, 255, 2, 2, 3, 3, 3, 3, 3, 3, 255]
        np.testing.assert_array_equal(remap_cityscapes_train_ids(original), expected)
        np.testing.assert_array_equal(remap_cityscapes_tensor(original).numpy(), expected)
        np.testing.assert_array_equal(original, before)
        mapped = remap_cityscapes_train_ids(np.arange(256))
        self.assertEqual(set(np.unique(mapped)), {0, 1, 2, 3, 4, 5, 6, 255})

    def test_unknown_ids_are_ignored(self):
        original = np.array([-1, 19, 254, 256, 1000])
        np.testing.assert_array_equal(remap_cityscapes_train_ids(original), [255] * 5)
        np.testing.assert_array_equal(remap_cityscapes_tensor(original), [255] * 5)

    def test_local_labels_preserve_new_classes(self):
        safe, weights = prepare_targets(tf.constant([7, 8, 9, 255]))
        np.testing.assert_array_equal(safe, [7, 8, 9, 0])
        np.testing.assert_array_equal(weights, [1, 1, 1, 0])
        for bad in (-1, 10, 254):
            with self.assertRaises(tf.errors.InvalidArgumentError):
                prepare_targets(tf.constant([bad]))
        with self.assertRaises(TypeError):
            prepare_targets(tf.constant([1.5]))

    def test_color_mask_keeps_ignore_black(self):
        rgb = colorize(np.array([[0, 7, 8, 9, 255]], dtype=np.uint8))
        self.assertEqual(rgb.shape, (1, 5, 3))
        np.testing.assert_array_equal(rgb[0, -1], [0, 0, 0])
        self.assertEqual(len(np.unique(rgb.reshape(-1, 3), axis=0)), 5)


class LossAndMetricTests(unittest.TestCase):
    def test_ignore_does_not_change_loss_or_gradient(self):
        loss = make_loss()
        logits = tf.Variable(np.arange(20, dtype=np.float32).reshape(2, 10) / 10)
        safe, weights = prepare_targets(tf.constant([2, 255]))
        with tf.GradientTape() as tape:
            value = loss(safe, logits, sample_weight=weights)
        gradient = tape.gradient(value, logits).numpy()
        reference = loss(tf.constant([2]), logits[:1], sample_weight=tf.ones(1))
        self.assertAlmostEqual(float(value), float(reference), places=6)
        np.testing.assert_array_equal(gradient[1], np.zeros(10))
        changed = tf.concat([logits[:1], tf.ones((1, 10)) * 1234], axis=0)
        self.assertAlmostEqual(float(value), float(loss(safe, changed, sample_weight=weights)), places=6)
        restored = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(loss))
        self.assertAlmostEqual(float(restored(safe, logits, sample_weight=weights)), float(value), places=6)

    def test_all_ignored_has_zero_loss_gradient_and_metrics(self):
        safe, weights = prepare_targets(tf.constant([255, 255]))
        logits = tf.Variable(tf.zeros((2, 10)))
        with tf.GradientTape() as tape:
            value = make_loss()(safe, logits, sample_weight=weights)
        self.assertEqual(float(value), 0)
        np.testing.assert_array_equal(tape.gradient(value, logits), np.zeros((2, 10)))
        for metric in make_metrics(10):
            metric.update_state(safe, tf.convert_to_tensor(logits), sample_weight=weights)
            self.assertEqual(float(metric.result()), 0)

    def test_exact_confusion_counts_ignore_and_serialization(self):
        safe, weights = prepare_targets(tf.constant([0, 1, 1, 255]))
        logits = tf.one_hot([0, 0, 1, 9], 10) * 10
        for metric, expected in zip(make_metrics(10), [2 / 3, 1 / 2, 2 / 3]):
            metric.update_state(safe[:2], logits[:2], sample_weight=weights[:2])
            metric.update_state(safe[2:], logits[2:], sample_weight=weights[2:])
            self.assertAlmostEqual(float(metric.result()), expected, places=6)
            restored = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(metric))
            restored.update_state(safe, logits, sample_weight=weights)
            self.assertAlmostEqual(float(restored.result()), expected, places=6)

    def test_false_positive_missing_class_is_counted(self):
        # Class 9 has zero truth support but predictions still count as errors.
        safe, weights = prepare_targets(tf.constant([0, 0]))
        logits = tf.one_hot([0, 9], 10) * 10
        for metric, expected in zip(make_metrics(10), [0.5, 0.25, 1 / 3]):
            metric.update_state(safe, logits, sample_weight=weights)
            self.assertAlmostEqual(float(metric.result()), expected, places=6)


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.samples = []

    def sample(self, index=0, split="train", route="r1", session="s1"):
        Image.fromarray(np.full((2, 5, 3), index, dtype=np.uint8)).save(self.root / f"image{index}.png")
        Image.fromarray(np.arange(10, dtype=np.uint8).reshape(2, 5)).save(self.root / f"mask{index}.png")
        row = {"sample_id": f"sample{index}", "image": f"image{index}.png", "mask": f"mask{index}.png",
               "route_id": route, "session_id": session, "split": split, "camera": "synthetic-test",
               "source": "synthetic test fixture, not field data", "reviewed": True, "reviewer": "test"}
        self.samples.append(row)
        return row

    def validate(self):
        self.manifest.write_text(json.dumps({"schema_version": 1, "samples": self.samples}), encoding="utf-8")
        return validate_manifest(self.manifest)

    def test_empty_template_reports_missing_data(self):
        report = self.validate()
        self.assertFalse(report["coverage_and_review_checks_pass"])
        self.assertEqual(len(report["missing_classes"]["train"]), 10)

    def test_three_independent_splits_with_all_classes(self):
        for i, split in enumerate(("train", "val", "test")):
            self.sample(i, split, f"route{i}", f"session{i}")
        report = self.validate()
        self.assertTrue(report["coverage_and_review_checks_pass"])
        self.assertFalse(report["training_authorized"])
        self.assertEqual(report["frames_by_split"]["train"]["Stairs"], 1)

    def test_route_leakage_is_rejected(self):
        self.sample()
        self.sample(1, "val", "r1", "s2")
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            self.validate()

    def test_session_leakage_is_rejected(self):
        self.sample()
        self.sample(1, "val", "r2", "s1")
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            self.validate()

    def test_missing_mask_is_pending(self):
        row = self.sample()
        row.update(mask=None, reviewed=False)
        self.assertEqual(self.validate()["labeled"], 0)

    def test_bad_mask_ids_rejected(self):
        self.sample()
        Image.fromarray(np.full((2, 5), 19, dtype=np.uint8)).save(self.root / "mask0.png")
        with self.assertRaisesRegex(ValueError, "Invalid mask IDs"):
            self.validate()

    def test_rgb_mask_rejected(self):
        self.sample()
        Image.fromarray(np.zeros((2, 5, 3), dtype=np.uint8)).save(self.root / "mask0.png")
        with self.assertRaisesRegex(ValueError, "8-bit ID PNG"):
            self.validate()

    def test_duplicate_image_rejected(self):
        self.sample()
        second = self.sample(1, "val", "r2", "s2")
        second["image"] = "image0.png"
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
