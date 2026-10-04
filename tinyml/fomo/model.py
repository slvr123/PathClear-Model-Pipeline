"""Local FOMO-inspired architecture, not Edge Impulse's training implementation."""
import json
from pathlib import Path

from config import CLASS_NAMES, GRID_CUT_POINTS, SETTINGS, contract


def build_model(backbone_weights="imagenet", grid_size=SETTINGS.grid_size):
    contract(grid_size)  # Validate before allocating or downloading weights.
    import tensorflow as tf
    # Public Keras ImageNet weights initialize visual features; FOMO-specific training is local.
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(SETTINGS.image_size, SETTINGS.image_size, 3),
        alpha=SETTINGS.alpha, include_top=False,
        weights="imagenet" if backbone_weights == "imagenet" else None)
    if backbone_weights not in (None, "random", "imagenet"):
        backbone.load_weights(backbone_weights)
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9
    trunk = tf.keras.Model(backbone.input, backbone.get_layer(GRID_CUT_POINTS[grid_size]).output, name="mobilenet_trunk")
    inputs = tf.keras.Input((SETTINGS.image_size, SETTINGS.image_size, 3), name="rgb_0_to_1")
    # Keep the public RGB [0,1] contract; embed MobileNet's [-1,1] transform in the model.
    features = trunk(tf.keras.layers.Rescaling(2.0, offset=-1.0, name="mobilenet_scaling")(inputs))
    features = tf.keras.layers.Conv2D(SETTINGS.head_channels, 1, activation="relu", name="grid_head")(features)
    probabilities = tf.keras.layers.Conv2D(len(CLASS_NAMES) + 1, 1, activation="softmax",
                                           name="grid_probabilities")(features)
    model = tf.keras.Model(inputs, probabilities, name="pathclear_local_fomo_inspired")
    model.compile(optimizer=tf.keras.optimizers.Adam(SETTINGS.learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), jit_compile=False)
    return model


def load_model(path):
    import tensorflow as tf
    path = Path(path)
    metadata_path = path.parent / "report.json" if path.suffix == ".tflite" else path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    grid_size = metadata["contract"].get("grid_size")
    if metadata["contract"] != contract(grid_size):
        raise ValueError("Model contract differs from current config; do not silently reinterpret it.")
    model = TFLiteModel(path) if path.suffix == ".tflite" else tf.keras.models.load_model(path, compile=False)
    if model.input_shape[1:] != (SETTINGS.image_size, SETTINGS.image_size, 3) or model.output_shape[1:] != (grid_size, grid_size, len(CLASS_NAMES)+1):
        raise ValueError("Unexpected model tensor shapes")
    return model, metadata


class TFLiteModel:
    """PC adapter to evaluate exported INT8 predictions as an independent detector."""
    def __init__(self, path):
        import numpy as np
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        if self.input["dtype"] != np.int8 or self.output["dtype"] != np.int8:
            raise ValueError("This adapter expects full INT8 I/O")
        self.input_shape = (None,) + tuple(self.input["shape"][1:])
        self.output_shape = (None,) + tuple(self.output["shape"][1:])

    def __call__(self, batch, training=False):
        import numpy as np
        if training:
            raise ValueError("TFLite inference cannot train")
        scale, zero = self.input["quantization"]
        quantized = np.clip(np.rint(np.asarray(batch) / scale + zero), -128, 127).astype(np.int8)
        self.interpreter.set_tensor(self.input["index"], quantized)
        self.interpreter.invoke()
        scale, zero = self.output["quantization"]
        probabilities = (self.interpreter.get_tensor(self.output["index"]).astype(np.float32) - zero) * scale
        # Correct small rounding errors in quantized softmax sums, preserving class order.
        return probabilities / np.maximum(probabilities.sum(axis=-1, keepdims=True), 1e-8)
