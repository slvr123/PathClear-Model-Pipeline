"""Explicit output selection for Cityscapes adaptation and later local training."""
import keras
from keras import ops

from pathclear_pipeline import compile_pathclear
from pathclear_taxonomy import NUM_CLASSES, CITYSCAPES_SUPPORTED_IDS

GATE_NAME = "pathclear_active_classes"


@keras.saving.register_keras_serializable(package="PathClear")
class ActiveClasses(keras.layers.Layer):
    """Keep ten output slots, with no gradient or competition from inactive IDs.

    Float32 logits use a finite sentinel rather than infinity so cross-entropy
    and all-ignored batches remain finite. This layer is saved in the model.
    """
    def __init__(self, active_ids=CITYSCAPES_SUPPORTED_IDS, **kwargs):
        super().__init__(**kwargs)
        self.active_ids = tuple(active_ids)
        if not self.active_ids or len(set(self.active_ids)) != len(self.active_ids):
            raise ValueError("Expected distinct active class IDs.")
        if any(i not in range(NUM_CLASSES) for i in self.active_ids):
            raise ValueError("Active class IDs must be in 0..9.")

    def call(self, logits):
        active = ops.convert_to_tensor([i in self.active_ids for i in range(NUM_CLASSES)], dtype="bool")
        return ops.where(active, ops.cast(logits, "float32"), -1e9)

    def get_config(self):
        return {**super().get_config(), "active_ids": list(self.active_ids)}


def cityscapes_model(raw_model):
    """Attach a serialized gate without changing the ten-class learned head."""
    if raw_model.output_shape[-1] != NUM_CLASSES:
        raise ValueError("Expected the transferred ten-class model.")
    if any(layer.name == GATE_NAME for layer in raw_model.layers):
        raise ValueError("Model already has an active-class gate.")
    model = keras.Model(raw_model.inputs, ActiveClasses(name=GATE_NAME)(raw_model.output),
                        name="PathClear_Cityscapes_Stage")
    compile_pathclear(model)
    return model


def enable_local_classes(stage_model):
    """Remove the Cityscapes gate and create a fresh optimizer for all ten IDs.

    Learned weights and the feature-layer freeze policy are preserved.
    Calling this helper does not train or save anything.
    """
    gate = stage_model.get_layer(GATE_NAME)
    if not isinstance(gate, ActiveClasses) or tuple(gate.active_ids) != CITYSCAPES_SUPPORTED_IDS:
        raise ValueError("Expected the seven-active-class Cityscapes stage.")
    if stage_model.layers[-1] is not gate:
        raise ValueError("Gate must be the final layer to enable local outputs safely.")
    model = keras.Model(stage_model.inputs, gate.input, name="PathClear_Local_Stage")
    compile_pathclear(model)
    return model
