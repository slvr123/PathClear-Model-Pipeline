"""Check masked-class training, persistence, and re-enabling local outputs."""
from pathlib import Path
import tempfile
import unittest

import keras
import numpy as np
import tensorflow as tf

from pathclear_pipeline import make_loss
from pathclear_stages import ActiveClasses, cityscapes_model, enable_local_classes, GATE_NAME
from segmentation_metrics import make_metrics


class StageTests(unittest.TestCase):
    def test_inactive_logits_do_not_compete_or_receive_gradients(self):
        raw = tf.Variable([[1., 2., 3., 4., 5., 6., 7., 100., 200., 300.]])
        gate = ActiveClasses()
        with tf.GradientTape() as tape:
            value = make_loss()(tf.constant([2]), gate(raw), sample_weight=tf.ones(1))
        gradient = tape.gradient(value, raw).numpy()
        expected = make_loss()(tf.constant([2]), raw[:, :7], sample_weight=tf.ones(1))
        self.assertAlmostEqual(float(value), float(expected), places=6)
        np.testing.assert_array_equal(gradient[:, 7:], [[0, 0, 0]])
        self.assertEqual(int(tf.argmax(gate(raw), axis=-1)[0]), 6)
        for metric in make_metrics(10):
            metric.update_state(tf.constant([6]), gate(raw), sample_weight=tf.ones(1))
            self.assertAlmostEqual(float(metric.result()), 1.0, places=6)

    def test_ignored_pixels_have_zero_loss_and_gradient_with_gate(self):
        raw = tf.Variable(tf.zeros((2, 10)))
        with tf.GradientTape() as tape:
            value = make_loss()(tf.constant([0, 0]), ActiveClasses()(raw), sample_weight=tf.zeros(2))
        self.assertEqual(float(value), 0)
        np.testing.assert_array_equal(tape.gradient(value, raw), np.zeros((2, 10)))

    def test_training_save_reload_and_local_reactivation(self):
        inputs = keras.Input((2, 2, 3))
        features = keras.layers.Conv2D(4, 1, trainable=False, name="features")(inputs)
        logits = keras.layers.Conv2D(10, 1, name="classifier_logits")(features)
        stage = cityscapes_model(keras.Model(inputs, logits))
        image = tf.ones((1, 2, 2, 3))
        labels = tf.zeros((1, 2, 2), dtype=tf.int32)
        weights = tf.ones((1, 2, 2))
        before = stage.get_layer("classifier_logits").get_weights()
        frozen = stage.get_layer("features").get_weights()
        stage.train_on_batch(image, labels, sample_weight=weights)
        after = stage.get_layer("classifier_logits").get_weights()
        self.assertTrue(any(np.any(a[..., :7] != b[..., :7]) for a, b in zip(before, after)))
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a[..., 7:], b[..., 7:])
        for a, b in zip(frozen, stage.get_layer("features").get_weights()):
            np.testing.assert_array_equal(a, b)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage.keras"
            stage.save(path)
            loaded = keras.models.load_model(path)
            self.assertEqual(int(loaded.optimizer.iterations), 1)
            self.assertEqual(tuple(loaded.get_layer(GATE_NAME).active_ids), tuple(range(7)))
            np.testing.assert_allclose(loaded(image), stage(image), rtol=0, atol=0)
            local = enable_local_classes(loaded)
            self.assertEqual(local.output_shape[-1], 10)
            self.assertFalse(any(layer.name == GATE_NAME for layer in local.layers))
            self.assertEqual(int(local.optimizer.iterations), 0)
            for a, b in zip(loaded.get_weights(), local.get_weights()):
                np.testing.assert_array_equal(a, b)
            before_local = local.get_layer("classifier_logits").get_weights()
            local.train_on_batch(image, tf.fill((1, 2, 2), 9), sample_weight=weights)
            after_local = local.get_layer("classifier_logits").get_weights()
            for class_id in (7, 8, 9):
                self.assertTrue(any(np.any(a[..., class_id] != b[..., class_id])
                                    for a, b in zip(before_local, after_local)))


if __name__ == "__main__":
    unittest.main()
