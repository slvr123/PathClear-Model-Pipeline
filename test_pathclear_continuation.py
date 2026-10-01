"""Synthetic training checks for optimizer resume and best-checkpoint retention."""
from pathlib import Path
import tempfile
import unittest

import keras
import numpy as np
import tensorflow as tf

from continue_pathclear_cityscapes import EpochCheckpointer, assert_optimizer_state, read_json
from pathclear_pipeline import sha256
from pathclear_stages import cityscapes_model


def tiny_stage():
    keras.utils.set_random_seed(19)
    inputs = keras.Input((2, 2, 3))
    features = keras.layers.Conv2D(4, 1, trainable=False, name="features")(inputs)
    outputs = keras.layers.Conv2D(10, 1, kernel_initializer="zeros", name="classifier_logits")(features)
    return cityscapes_model(keras.Model(inputs, outputs))


class ContinuationTests(unittest.TestCase):
    def test_resumed_adam_update_matches_uninterrupted_update(self):
        model = tiny_stage()
        image, labels, weights = tf.ones((1, 2, 2, 3)), tf.zeros((1, 2, 2), tf.int32), tf.ones((1, 2, 2))
        model.train_on_batch(image, labels, sample_weight=weights)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.keras"
            model.save(path)
            restored = keras.models.load_model(path)
            assert_optimizer_state(restored, {"optimizer_iterations": 1})
            with self.assertRaisesRegex(ValueError, "optimizer steps"):
                assert_optimizer_state(restored, {"optimizer_iterations": 0})
            for a, b in zip(model.optimizer.variables, restored.optimizer.variables):
                np.testing.assert_array_equal(a.numpy(), b.numpy())
            model.train_on_batch(image, labels, sample_weight=weights)
            restored.train_on_batch(image, labels, sample_weight=weights)
            self.assertEqual(int(restored.optimizer.iterations), 2)
            for a, b in zip(model.get_weights(), restored.get_weights()):
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
            for a, b in zip(model.optimizer.variables, restored.optimizer.variables):
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6, atol=1e-7)

    def test_keeps_previous_best_then_replaces_only_on_improvement(self):
        model = tiny_stage()
        image, labels, weights = tf.ones((1, 2, 2, 3)), tf.zeros((1, 2, 2), tf.int32), tf.ones((1, 2, 2))
        model.train_on_batch(image, labels, sample_weight=weights)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            model.save(output / "best.keras")
            previous_best_hash = sha256(output / "best.keras")
            state = {"best_val_mean_iou": 0.5, "best_epoch": 1, "completed_epochs": 1,
                     "optimizer_iterations": 1}
            check = EpochCheckpointer(output, 1, state, {})
            check.set_model(model)
            check.on_train_begin()
            for epoch, validation_label in ((1, 6), (2, 0)):
                check.on_epoch_begin(epoch)
                model.train_on_batch(image, labels, sample_weight=weights)
                model.reset_metrics()
                values = model.test_on_batch(image, tf.fill((1, 2, 2), validation_label),
                                             sample_weight=weights, return_dict=True)
                check.on_epoch_end(epoch, {"val_" + k: float(v) for k, v in values.items()})
                saved = read_json(output / "state.json")
                self.assertEqual(saved["completed_epochs"], epoch + 1)
                self.assertEqual(saved["optimizer_iterations"], epoch + 1)
                self.assertEqual(saved["last_sha256"], sha256(output / "last.keras"))
                if epoch == 1:
                    self.assertEqual(saved["best_epoch"], 1)
                    self.assertEqual(sha256(output / "best.keras"), previous_best_hash)
                else:
                    self.assertEqual(saved["best_epoch"], 3)
                    self.assertGreater(saved["best_val_mean_iou"], 0.5)
                    self.assertNotEqual(sha256(output / "best.keras"), previous_best_hash)
            loaded = keras.models.load_model(output / "last.keras")
            assert_optimizer_state(loaded, saved)
            for a, b in zip(model.get_weights(), loaded.get_weights()):
                np.testing.assert_array_equal(a, b)
            self.assertFalse(list(output.glob("*.pending.*")))


if __name__ == "__main__":
    unittest.main()
