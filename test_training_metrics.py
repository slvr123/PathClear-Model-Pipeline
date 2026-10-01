"""Small exact-count checks for void masking and metric serialization."""
import unittest
import warnings
import keras
import numpy as np
from segmentation_metrics import make_metrics


class MetricTests(unittest.TestCase):
    def test_weighted_sparse_logits(self):
        # Valid confusion matrix: [[1, 0], [1, 1]]. Fourth pixel is void.
        labels = np.array([0, 1, 1, 0])
        logits = np.eye(19, dtype=np.float32)[[0, 0, 1, 18]] * 10
        weights = np.array([1, 1, 1, 0], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            for metric, expected in zip(make_metrics(), [2/3, 1/2, 2/3]):
                metric.update_state(labels[:2], logits[:2], sample_weight=weights[:2])
                metric.update_state(labels[2:], logits[2:], sample_weight=weights[2:])
                self.assertAlmostEqual(float(metric.result()), expected, places=6)
                restored = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(metric))
                restored.update_state(labels, logits, sample_weight=weights)
                self.assertAlmostEqual(float(restored.result()), expected, places=6)
                metric.reset_state()
                metric.update_state(labels, logits, sample_weight=np.zeros_like(weights))
                self.assertEqual(float(metric.result()), 0)

    def test_false_positive_class_counts_in_macro(self):
        for metric, expected in zip(make_metrics(), [0, 0, 0]):
            metric.update_state(np.array([0]), np.eye(19)[[18]], sample_weight=np.array([1.]))
            self.assertEqual(float(metric.result()), expected)


if __name__ == "__main__":
    unittest.main()
