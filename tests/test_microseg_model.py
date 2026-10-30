"""Tests for microseg_model.py — model structure, shapes, and parameter count.

Run with: bash run_wsl.sh tests/test_microseg_model.py
Or:        pytest tests/test_microseg_model.py

No GPU required. No dataset required.
"""
import numpy as np
import pytest


class TestBuildMicroseg:
    """Structural tests that require only TensorFlow (CPU)."""

    def setup_method(self):
        from microseg_model import build_microseg
        self.build_microseg = build_microseg

    def test_default_output_shape(self):
        model = self.build_microseg()
        # Batch dim is None; spatial dims and class dim must match.
        assert model.output_shape == (None, 96, 96, 5)

    def test_input_shape(self):
        model = self.build_microseg()
        assert model.input_shape == (None, 96, 96, 3)

    def test_custom_num_classes(self):
        model = self.build_microseg(num_classes=3)
        assert model.output_shape == (None, 96, 96, 3)

    def test_parameter_count_under_200k(self):
        model = self.build_microseg()
        total = model.count_params()
        assert total < 200_000, (
            f"Model has {total:,} parameters. "
            "Expected under 200,000 for ESP32-CAM deployment."
        )

    def test_parameter_count_sanity_lower_bound(self):
        # Too few params → model is likely degenerate.
        model = self.build_microseg()
        assert model.count_params() > 10_000

    def test_model_name(self):
        model = self.build_microseg()
        assert model.name == "MicroSeg"

    def test_classifier_logits_layer_exists(self):
        model = self.build_microseg()
        names = [layer.name for layer in model.layers]
        assert "classifier_logits" in names

    def test_upsample_layer_exists(self):
        model = self.build_microseg()
        names = [layer.name for layer in model.layers]
        assert "upsample_x8" in names


class TestForwardPass:
    """Forward-pass tests using random synthetic inputs."""

    def setup_method(self):
        import numpy as np
        from microseg_model import build_microseg
        self.model = build_microseg()
        self.rng = np.random.default_rng(0)

    def test_forward_pass_single_image(self):
        import numpy as np
        x = self.rng.uniform(0.0, 1.0, (1, 96, 96, 3)).astype(np.float32)
        y = self.model(x, training=False)
        assert y.shape == (1, 96, 96, 5)

    def test_forward_pass_batch(self):
        import numpy as np
        x = self.rng.uniform(0.0, 1.0, (4, 96, 96, 3)).astype(np.float32)
        y = self.model(x, training=False)
        assert y.shape == (4, 96, 96, 5)

    def test_output_is_finite(self):
        import numpy as np
        x = self.rng.uniform(0.0, 1.0, (2, 96, 96, 3)).astype(np.float32)
        y = self.model(x, training=False).numpy()
        assert np.all(np.isfinite(y)), "Model output contains NaN or Inf."

    def test_output_is_logits_not_probabilities(self):
        """Logits are not constrained to [0, 1] or sum to 1."""
        import numpy as np
        x = self.rng.uniform(0.0, 1.0, (2, 96, 96, 3)).astype(np.float32)
        y = self.model(x, training=False).numpy()
        # At least some logit values should be outside [0, 1]
        outside_01 = np.any((y < 0) | (y > 1))
        assert outside_01, (
            "All logits are in [0,1] — model may be applying softmax internally. "
            "Check that no activation is applied to the classifier_logits layer."
        )

    def test_training_mode_forward_pass(self):
        import numpy as np
        x = self.rng.uniform(0.0, 1.0, (2, 96, 96, 3)).astype(np.float32)
        y = self.model(x, training=True)
        assert y.shape == (2, 96, 96, 5)

    def test_all_pixels_predicted(self):
        """Output spatial resolution must match input — all pixels get a label."""
        import numpy as np
        x = self.rng.uniform(0.0, 1.0, (1, 96, 96, 3)).astype(np.float32)
        y = self.model(x, training=False).numpy()
        assert y.shape[1] == 96 and y.shape[2] == 96


class TestModelSerializability:
    """Verify the model can be saved and reloaded cleanly."""

    def test_save_and_reload(self, tmp_path):
        import numpy as np
        from microseg_model import build_microseg
        model = build_microseg()

        save_path = tmp_path / "microseg_test.keras"
        model.save(str(save_path))

        import keras
        loaded = keras.models.load_model(str(save_path))
        assert loaded.output_shape == (None, 96, 96, 5)

        rng = np.random.default_rng(1)
        x = rng.uniform(0.0, 1.0, (1, 96, 96, 3)).astype(np.float32)
        orig_out = model(x, training=False).numpy()
        load_out = loaded(x, training=False).numpy()
        np.testing.assert_allclose(orig_out, load_out, atol=1e-5)
