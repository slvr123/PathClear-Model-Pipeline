"""Epoch-level segmentation metrics using weighted confusion matrices."""

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="PathClear")
class MacroDice(keras.metrics.MeanIoU):
    """Hard multiclass Dice, averaged over classes with nonzero support.

    Inherits sparse-label/logit conversion and sample weighting from MeanIoU.
    A class contributes when its true count + predicted count is positive.
    Counts accumulate across batches before computing the macro average.
    """

    def result(self):
        cm = self.total_cm
        denominator = ops.sum(cm, axis=0) + ops.sum(cm, axis=1)
        dice = ops.divide_no_nan(2 * ops.diagonal(cm), denominator)
        present = ops.cast(denominator > 0, self.dtype)
        return ops.divide_no_nan(ops.sum(dice * present), ops.sum(present))


def make_metrics(num_classes=19):
    # float64 accepts float sample weights and retains exact large pixel counts.
    return [
        keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy", dtype="float64"),
        keras.metrics.MeanIoU(
            num_classes=num_classes, sparse_y_pred=False,
            name="mean_iou", dtype="float64",
        ),
        MacroDice(
            num_classes=num_classes, sparse_y_pred=False,
            name="dice", dtype="float64",
        ),
    ]
