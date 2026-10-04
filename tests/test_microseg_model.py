"""Shape, finite-forward, and parameter-budget checks for the MicroSeg model."""
import unittest

import keras
import numpy as np

from microseg.model import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NUM_CLASSES,
    build_microseg,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_microseg()

    def test_input_and_output_shapes(self):
        self.assertEqual(self.model.input_shape, (None, INPUT_HEIGHT, INPUT_WIDTH, 3))
        self.assertEqual(
            self.model.output_shape, (None, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES)
        )
        # The classifier emits exactly four logits per pixel.
        self.assertEqual(int(self.model.get_layer("classifier_logits").filters), NUM_CLASSES)

    def test_forward_pass_is_finite_and_raw_logits(self):
        images = keras.ops.convert_to_tensor(
            np.random.RandomState(0).rand(2, INPUT_HEIGHT, INPUT_WIDTH, 3).astype("float32")
        )
        logits = self.model(images, training=False)
        self.assertEqual(tuple(logits.shape), (2, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES))
        self.assertTrue(bool(keras.ops.all(keras.ops.isfinite(logits)).numpy()))
        # Every argmax is a valid class id.
        predicted = np.argmax(np.asarray(logits), axis=-1)
        self.assertLess(int(predicted.max()), NUM_CLASSES)

    def test_output_is_raw_logits_without_softmax(self):
        # Raw logits: no softmax is applied inside the model; it belongs in the
        # loss or post-processing.
        self.assertIsInstance(self.model.layers[-1], keras.layers.UpSampling2D)
        self.assertFalse(
            any(isinstance(layer, keras.layers.Softmax) for layer in self.model.layers)
        )

    def test_parameter_budget(self):
        total = int(self.model.count_params())
        self.assertGreater(total, 0)
        self.assertLess(total, 120_000)

    def test_uses_batch_independent_group_normalization(self):
        group_norms = [
            layer for layer in self.model.layers
            if isinstance(layer, keras.layers.GroupNormalization)
        ]
        self.assertTrue(group_norms)
        self.assertTrue(
            all(
                isinstance(layer, keras.layers.GroupNormalization)
                and layer.groups == 8
                for layer in group_norms
            )
        )
        self.assertFalse(
            any(
                isinstance(layer, keras.layers.BatchNormalization)
                for layer in self.model.layers
            )
        )

    def test_downsample_factor_matches_grid(self):
        low_res = self.model.get_layer("classifier_logits")
        self.assertEqual(
            tuple(low_res.output.shape[1:3]),
            (INPUT_HEIGHT // 8, INPUT_WIDTH // 8),
        )

    def test_rejects_indivisible_input(self):
        with self.assertRaises(ValueError):
            build_microseg(input_shape=(100, 100, 3))

    def test_rejects_wrong_channels(self):
        with self.assertRaises(ValueError):
            build_microseg(input_shape=(96, 96, 1))


if __name__ == "__main__":
    unittest.main()
