"""Compact four-class MicroSeg dense-segmentation network.

A lightweight encoder/fusion/head network inspired by MobileNetV2 and
Fast-SCNN building blocks, sized to investigate INT8 deployment on an
ESP32-CAM. It is a separate experiment from the Fast-SCNN PathClear pipeline
and reuses only Fast-SCNN's low-level convolution blocks.

Logits are returned raw; softmax belongs in the loss or post-processing.
"""
from keras import layers, Model

from fast_scnn import conv_bn_relu, depthwise_separable_conv, inverted_residual

INPUT_HEIGHT = 96
INPUT_WIDTH = 96
NUM_CLASSES = 4

# Three stride-2 stages, matching the 12 x 12 low-resolution grid at 96 x 96.
DOWNSAMPLE_FACTOR = 8
SHORTCUT_CHANNELS = 32
CONTEXT_CHANNELS = 40
EXPANSION_FACTOR = 4
NUM_RESIDUAL_BLOCKS = 4
NORMALIZATION = layers.GroupNormalization
NORMALIZATION_KWARGS = {"groups": 8, "axis": -1}


def learning_to_downsample(x):
    """Encoder: a standard convolution followed by two depthwise-separable
    downsampling blocks, ending at 1/8 resolution and SHORTCUT_CHANNELS."""

    x = conv_bn_relu(
        x,
        filters=16,
        kernel_size=3,
        strides=2,
        name="lds_conv",
        normalization=NORMALIZATION,
        normalization_kwargs=NORMALIZATION_KWARGS,
    )

    x = depthwise_separable_conv(
        x,
        filters=24,
        kernel_size=3,
        strides=2,
        name="lds_dsconv1",
        normalization=NORMALIZATION,
        normalization_kwargs=NORMALIZATION_KWARGS,
    )

    x = depthwise_separable_conv(
        x,
        filters=SHORTCUT_CHANNELS,
        kernel_size=3,
        strides=2,
        name="lds_dsconv2",
        normalization=NORMALIZATION,
        normalization_kwargs=NORMALIZATION_KWARGS,
    )

    return x


def build_microseg(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), num_classes=NUM_CLASSES):
    """Build the MicroSeg model.

    Architecture:

    96 x 96 x 3 input
      -> Learning-to-downsample encoder (strides 2, 2, 2)
      -> 12 x 12 x 32 shortcut features
      -> Four inverted-residual blocks at 12 x 12 (expansion factor 4)
      -> Feature fusion with the shortcut
      -> Depthwise-separable refinement block
      -> 1 x 1 classifier convolution (four logits per low-resolution cell)
      -> Bilinear upsample to 96 x 96 x 4 logits
    """

    height, width, channels = input_shape
    if channels != 3:
        raise ValueError(f"Expected a three-channel RGB input, got {channels}.")
    if height % DOWNSAMPLE_FACTOR or width % DOWNSAMPLE_FACTOR:
        raise ValueError(
            f"Input {height} x {width} must be divisible by {DOWNSAMPLE_FACTOR}."
        )

    inputs = layers.Input(shape=input_shape, name="image")

    # --------------------------------------------------------
    # STAGE 1: LEARNING TO DOWNSAMPLE (shortcut features)
    # --------------------------------------------------------

    shortcut = learning_to_downsample(inputs)

    # --------------------------------------------------------
    # STAGE 2: LOW-RESOLUTION CONTEXT FEATURES
    # --------------------------------------------------------

    context = shortcut
    for index in range(1, NUM_RESIDUAL_BLOCKS + 1):
        context = inverted_residual(
            context,
            output_channels=CONTEXT_CHANNELS,
            stride=1,
            expansion_factor=EXPANSION_FACTOR,
            name=f"irb{index}",
            normalization=NORMALIZATION,
            normalization_kwargs=NORMALIZATION_KWARGS,
        )

    # --------------------------------------------------------
    # STAGE 3: FEATURE FUSION
    # --------------------------------------------------------

    fused = layers.Concatenate(name="fusion_concat")([shortcut, context])
    fused = depthwise_separable_conv(
        fused,
        filters=CONTEXT_CHANNELS,
        kernel_size=3,
        strides=1,
        name="fusion_refine",
        normalization=NORMALIZATION,
        normalization_kwargs=NORMALIZATION_KWARGS,
    )

    # --------------------------------------------------------
    # STAGE 4: CLASSIFIER AND UPSAMPLE
    # --------------------------------------------------------

    low_res_logits = layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        strides=1,
        padding="same",
        name="classifier_logits",
    )(fused)

    outputs = layers.UpSampling2D(
        size=DOWNSAMPLE_FACTOR,
        interpolation="bilinear",
        name="upsample_logits",
    )(low_res_logits)

    model = Model(inputs=inputs, outputs=outputs, name="MicroSeg")

    return model


if __name__ == "__main__":

    print("\nBuilding MicroSeg four-class model...")

    model = build_microseg()

    print("\n========== MICROSEG TEST ==========")

    input_shape = (None, INPUT_HEIGHT, INPUT_WIDTH, 3)
    output_shape = tuple(model.compute_output_shape(input_shape))

    print("\nInput shape:")
    print(input_shape)

    print("\nOutput shape:")
    print(output_shape)

    total_params = model.count_params()

    print("\nTotal parameters:")
    print(f"{total_params:,}")

    expected_output_shape = (None, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES)

    if output_shape != expected_output_shape:
        raise ValueError(
            f"Unexpected output shape: {output_shape}. "
            f"Expected: {expected_output_shape}"
        )

    if total_params >= 120_000:
        raise ValueError(
            f"Parameter count is unexpectedly high: {total_params:,}"
        )

    print("\nMicroSeg architecture built successfully.")

    print("\n========== MODEL SUMMARY ==========\n")

    model.summary()
