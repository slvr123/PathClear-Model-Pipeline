import math

from keras import layers, Model


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 1024
NUM_CLASSES = 19


# ============================================================
# STANDARD CONVOLUTION BLOCK
# ============================================================

def conv_bn_relu(
    x,
    filters,
    kernel_size=3,
    strides=1,
    dilation_rate=1,
    groups=1,
    name=None
):
    """
    Conv2D -> BatchNorm -> ReLU
    """

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        dilation_rate=dilation_rate,
        groups=groups,
        use_bias=False,
        name=f"{name}_conv" if name else None
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn" if name else None
    )(x)

    x = layers.ReLU(
        name=f"{name}_relu" if name else None
    )(x)

    return x


# ============================================================
# DEPTHWISE SEPARABLE CONVOLUTION
# ============================================================

def depthwise_separable_conv(
    x,
    filters,
    kernel_size=3,
    strides=1,
    name=None
):
    """
    Fast-SCNN depthwise separable convolution.

    Important:

    Depthwise Conv
        ↓
    BatchNorm
        ↓
    Pointwise Conv
        ↓
    BatchNorm
        ↓
    ReLU

    There is intentionally NO activation between the
    depthwise and pointwise convolution.
    """

    # --------------------------------------------------------
    # DEPTHWISE
    # --------------------------------------------------------

    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_depthwise" if name else None
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_depthwise_bn" if name else None
    )(x)


    # --------------------------------------------------------
    # POINTWISE
    # --------------------------------------------------------

    x = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_pointwise" if name else None
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_pointwise_bn" if name else None
    )(x)

    x = layers.ReLU(
        name=f"{name}_relu" if name else None
    )(x)

    return x


# ============================================================
# INVERTED RESIDUAL BOTTLENECK
# ============================================================

def inverted_residual(
    x,
    output_channels,
    stride,
    expansion_factor=6,
    name=None
):
    """
    MobileNetV2-style inverted residual block used by Fast-SCNN.

    Expansion 1x1 Conv
            ↓
           ReLU
            ↓
    Depthwise 3x3 Conv
            ↓
           ReLU
            ↓
    Projection 1x1 Conv
            ↓
          Linear
            ↓
    Optional residual connection
    """

    input_channels = int(x.shape[-1])

    expanded_channels = (
        input_channels * expansion_factor
    )

    residual = x


    # ========================================================
    # EXPANSION
    # ========================================================

    x = layers.Conv2D(
        filters=expanded_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_expand_conv" if name else None
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_expand_bn" if name else None
    )(x)

    # Fast-SCNN paper uses ReLU, NOT ReLU6.
    x = layers.ReLU(
        name=f"{name}_expand_relu" if name else None
    )(x)


    # ========================================================
    # DEPTHWISE CONVOLUTION
    # ========================================================

    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_depthwise" if name else None
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_depthwise_bn" if name else None
    )(x)

    x = layers.ReLU(
        name=f"{name}_depthwise_relu" if name else None
    )(x)


    # ========================================================
    # LINEAR PROJECTION
    # ========================================================

    x = layers.Conv2D(
        filters=output_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_project_conv" if name else None
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_project_bn" if name else None
    )(x)

    # IMPORTANT:
    # No activation here.
    # This is the linear bottleneck.


    # ========================================================
    # RESIDUAL CONNECTION
    # ========================================================

    if (
        stride == 1
        and input_channels == output_channels
    ):
        x = layers.Add(
            name=f"{name}_add" if name else None
        )([
            residual,
            x
        ])

    return x


# ============================================================
# STAGE 1:
# LEARNING TO DOWNSAMPLE
# ============================================================

def learning_to_downsample(x):
    """
    Paper structure:

    Input
        512 x 1024 x 3

    Conv2D, stride 2
        256 x 512 x 32

    DSConv, stride 2
        128 x 256 x 48

    DSConv, stride 2
         64 x 128 x 64

    Output is at 1/8 resolution.
    """

    # --------------------------------------------------------
    # NORMAL CONVOLUTION
    # --------------------------------------------------------

    x = conv_bn_relu(
        x,
        filters=32,
        kernel_size=3,
        strides=2,
        name="downsample_conv"
    )


    # --------------------------------------------------------
    # DSConv 32 -> 48
    # --------------------------------------------------------

    x = depthwise_separable_conv(
        x,
        filters=48,
        kernel_size=3,
        strides=2,
        name="downsample_dsconv1"
    )


    # --------------------------------------------------------
    # DSConv 48 -> 64
    # --------------------------------------------------------

    x = depthwise_separable_conv(
        x,
        filters=64,
        kernel_size=3,
        strides=2,
        name="downsample_dsconv2"
    )

    return x


# ============================================================
# BOTTLENECK STAGE
# ============================================================

def bottleneck_stage(
    x,
    output_channels,
    num_blocks,
    first_stride,
    name
):
    """
    Group of inverted residual blocks.

    Only the first block applies first_stride.
    """

    x = inverted_residual(
        x,
        output_channels=output_channels,
        stride=first_stride,
        expansion_factor=6,
        name=f"{name}_block1"
    )

    for i in range(1, num_blocks):

        x = inverted_residual(
            x,
            output_channels=output_channels,
            stride=1,
            expansion_factor=6,
            name=f"{name}_block{i + 1}"
        )

    return x


# ============================================================
# PYRAMID POOLING MODULE
# ============================================================

def pyramid_pooling(
    x,
    output_channels=128
):
    """
    Pyramid Pooling Module.

    Pooling bins:
        1 x 1
        2 x 2
        3 x 3
        6 x 6

    Each pooled branch becomes 32 channels.

    Original features:
        128 channels

    Four branches:
        4 x 32 = 128

    Concatenated:
        256 channels

    Final 1x1 convolution:
        128 channels
    """

    height = int(x.shape[1])
    width = int(x.shape[2])

    bins = [1, 2, 3, 6]

    branch_channels = (
        output_channels // len(bins)
    )

    branches = [x]


    # --------------------------------------------------------
    # CREATE FOUR CONTEXT BRANCHES
    # --------------------------------------------------------

    for bin_size in bins:

        pool_height = math.ceil(
            height / bin_size
        )

        pool_width = math.ceil(
            width / bin_size
        )


        # ----------------------------------------------------
        # POOL
        # ----------------------------------------------------

        branch = layers.AveragePooling2D(
            pool_size=(
                pool_height,
                pool_width
            ),
            strides=(
                pool_height,
                pool_width
            ),
            padding="same",
            name=f"ppm_pool_{bin_size}"
        )(x)


        # ----------------------------------------------------
        # 1x1 CONVOLUTION
        # ----------------------------------------------------

        branch = conv_bn_relu(
            branch,
            filters=branch_channels,
            kernel_size=1,
            strides=1,
            name=f"ppm_branch_{bin_size}"
        )


        # ----------------------------------------------------
        # UPSAMPLE BACK TO FEATURE MAP SIZE
        # ----------------------------------------------------

        branch = layers.Resizing(
            height,
            width,
            interpolation="bilinear",
            name=f"ppm_resize_{bin_size}"
        )(branch)

        branches.append(branch)


    # --------------------------------------------------------
    # CONCATENATE
    # --------------------------------------------------------

    x = layers.Concatenate(
        axis=-1,
        name="ppm_concat"
    )(branches)


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # 1x1 convolution.
    #
    # Our previous version incorrectly used 3x3 here,
    # which added roughly 260k unnecessary parameters.
    # --------------------------------------------------------

    x = conv_bn_relu(
        x,
        filters=output_channels,
        kernel_size=1,
        strides=1,
        name="ppm_output"
    )

    return x


# ============================================================
# STAGE 2:
# GLOBAL FEATURE EXTRACTOR
# ============================================================

def global_feature_extractor(x):
    """
    Paper layout:

    Bottleneck:
        t = 6
        c = 64
        n = 3
        s = 2

    Bottleneck:
        t = 6
        c = 96
        n = 3
        s = 2

    Bottleneck:
        t = 6
        c = 128
        n = 3
        s = 1

    Then:
        Pyramid Pooling Module
    """

    # --------------------------------------------------------
    # 64 CHANNEL STAGE
    #
    # 64 x 128
    #      ↓
    # 32 x 64
    # --------------------------------------------------------

    x = bottleneck_stage(
        x,
        output_channels=64,
        num_blocks=3,
        first_stride=2,
        name="gfe_64"
    )


    # --------------------------------------------------------
    # 96 CHANNEL STAGE
    #
    # 32 x 64
    #      ↓
    # 16 x 32
    # --------------------------------------------------------

    x = bottleneck_stage(
        x,
        output_channels=96,
        num_blocks=3,
        first_stride=2,
        name="gfe_96"
    )


    # --------------------------------------------------------
    # 128 CHANNEL STAGE
    #
    # stays:
    # 16 x 32
    # --------------------------------------------------------

    x = bottleneck_stage(
        x,
        output_channels=128,
        num_blocks=3,
        first_stride=1,
        name="gfe_128"
    )


    # --------------------------------------------------------
    # PYRAMID POOLING
    # --------------------------------------------------------

    x = pyramid_pooling(
        x,
        output_channels=128
    )

    return x


# ============================================================
# STAGE 3:
# FEATURE FUSION MODULE
# ============================================================

def feature_fusion(
    high_resolution,
    low_resolution
):
    """
    Fast-SCNN Feature Fusion Module.

    High-resolution branch:
        1/8 resolution

    Low-resolution branch:
        1/32 resolution

    Resolution difference:
        X = 4

    Paper specifies a depthwise convolution using:
        dilation = X = 4
    """

    target_height = int(
        high_resolution.shape[1]
    )

    target_width = int(
        high_resolution.shape[2]
    )


    # ========================================================
    # LOW-RESOLUTION / SEMANTIC BRANCH
    # ========================================================

    # --------------------------------------------------------
    # UPSAMPLE x4
    # --------------------------------------------------------

    low = layers.Resizing(
        target_height,
        target_width,
        interpolation="bilinear",
        name="fusion_low_resize"
    )(low_resolution)


    # --------------------------------------------------------
    # DILATED DEPTHWISE CONVOLUTION
    #
    # The paper specifies dilation X.
    #
    # X = 4 because:
    #
    # 1/32 -> 1/8
    # --------------------------------------------------------

    low = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=4,
        use_bias=False,
        name="fusion_low_depthwise"
    )(low)

    low = layers.BatchNormalization(
        name="fusion_low_depthwise_bn"
    )(low)

    low = layers.ReLU(
        name="fusion_low_depthwise_relu"
    )(low)


    # --------------------------------------------------------
    # POINTWISE PROJECTION
    #
    # No activation here.
    # --------------------------------------------------------

    low = layers.Conv2D(
        filters=128,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=True,
        name="fusion_low_pointwise"
    )(low)


    # ========================================================
    # HIGH-RESOLUTION / SPATIAL BRANCH
    # ========================================================

    high = layers.Conv2D(
        filters=128,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=True,
        name="fusion_high_pointwise"
    )(high_resolution)


    # ========================================================
    # FUSION
    # ========================================================

    x = layers.Add(
        name="feature_fusion_add"
    )([
        high,
        low
    ])

    x = layers.ReLU(
        name="feature_fusion_relu"
    )(x)

    return x


# ============================================================
# STAGE 4:
# CLASSIFIER
# ============================================================

def classifier(
    x,
    num_classes
):
    """
    Fast-SCNN classifier:

    DSConv
       ↓
    DSConv
       ↓
    Dropout
       ↓
    1x1 Conv
       ↓
    num_classes logits
    """

    # --------------------------------------------------------
    # DSConv 1
    # --------------------------------------------------------

    x = depthwise_separable_conv(
        x,
        filters=128,
        kernel_size=3,
        strides=1,
        name="classifier_dsconv1"
    )


    # --------------------------------------------------------
    # DSConv 2
    # --------------------------------------------------------

    x = depthwise_separable_conv(
        x,
        filters=128,
        kernel_size=3,
        strides=1,
        name="classifier_dsconv2"
    )


    # --------------------------------------------------------
    # DROPOUT
    # --------------------------------------------------------

    x = layers.Dropout(
        rate=0.1,
        name="classifier_dropout"
    )(x)


    # --------------------------------------------------------
    # CLASS LOGITS
    #
    # NO SOFTMAX here.
    #
    # Later we will use:
    #
    # SparseCategoricalCrossentropy(from_logits=True)
    # --------------------------------------------------------

    x = layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=True,
        name="classifier_logits"
    )(x)


    # --------------------------------------------------------
    # UPSAMPLE FROM 1/8 TO FULL INPUT RESOLUTION
    # --------------------------------------------------------

    x = layers.Resizing(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        interpolation="bilinear",
        name="output_resize"
    )(x)

    return x


# ============================================================
# BUILD COMPLETE FAST-SCNN
# ============================================================

def build_fast_scnn(
    input_shape=(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3
    ),
    num_classes=NUM_CLASSES
):
    """
    Build Fast-SCNN.

    Input:
        512 x 1024 x 3

    Output:
        512 x 1024 x 19
    """

    inputs = layers.Input(
        shape=input_shape,
        name="image"
    )


    # ========================================================
    # STAGE 1
    # LEARNING TO DOWNSAMPLE
    # ========================================================

    high_resolution_features = (
        learning_to_downsample(
            inputs
        )
    )


    # ========================================================
    # STAGE 2
    # GLOBAL FEATURE EXTRACTOR
    # ========================================================

    low_resolution_features = (
        global_feature_extractor(
            high_resolution_features
        )
    )


    # ========================================================
    # STAGE 3
    # FEATURE FUSION
    # ========================================================

    fused_features = feature_fusion(
        high_resolution_features,
        low_resolution_features
    )


    # ========================================================
    # STAGE 4
    # CLASSIFIER
    # ========================================================

    outputs = classifier(
        fused_features,
        num_classes
    )


    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Fast_SCNN"
    )

    return model


# ============================================================
# TEST MODEL
# ============================================================

# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("\nBuilding paper-aligned Fast-SCNN...")

    model = build_fast_scnn()

    print("\n========== FAST-SCNN TEST ==========")


    # --------------------------------------------------------
    # GET MODEL SHAPES
    # --------------------------------------------------------
    input_shape = (
        None,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3
    )

    output_shape = tuple(
        model.compute_output_shape(input_shape)
    )


    print("\nInput shape:")
    print(input_shape)

    print("\nOutput shape:")
    print(output_shape)


    # --------------------------------------------------------
    # PARAMETER COUNT
    # --------------------------------------------------------

    total_params = model.count_params()

    print("\nTotal parameters:")
    print(f"{total_params:,}")


    # --------------------------------------------------------
    # EXPECTED SHAPES
    # --------------------------------------------------------

    expected_input_shape = (
        None,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3
    )

    expected_output_shape = (
        None,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        NUM_CLASSES
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if input_shape != expected_input_shape:
        raise ValueError(
            f"Unexpected input shape: {input_shape}. "
            f"Expected: {expected_input_shape}"
        )

    if output_shape != expected_output_shape:
        raise ValueError(
            f"Unexpected output shape: {output_shape}. "
            f"Expected: {expected_output_shape}"
        )

    if total_params >= 1_250_000:
        raise ValueError(
            f"Parameter count is unexpectedly high: "
            f"{total_params:,}"
        )


    print("\nFast-SCNN architecture built successfully.")

    print("\n========== MODEL SUMMARY ==========\n")

    model.summary()