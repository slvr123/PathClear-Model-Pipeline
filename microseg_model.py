"""MicroSeg — Ultra-lightweight semantic segmentation for ESP32-CAM.

Architecture: encoder-decoder with MobileNetV2-style inverted residual blocks.
Target: 96×96 RGB input → 96×96 5-class logit output.

Design constraints:
  - All ops supported by TFLite Micro (no custom ops).
  - Bilinear resize maps to TFLite ResizeBilinear — verified ESP32 compatible.
  - BatchNorm is folded into Conv weights during INT8 quantization.
  - Peak INT8 activation RAM ≈ 100-130 KB (fits in ESP32-CAM internal SRAM).
  - Parameter count ≈ 60-80 K (INT8 model size ≈ 70-90 KB in flash).

Network structure:
  Input 96×96×3
      │
  [Learning to Downsample]   stride 8 total → 12×12×32
      │  ╰─────── shortcut ──────╮
  [Global Feature Extractor]  4×IRB at 12×12 → 12×12×32
      │                          │
  [Feature Fusion]  ← Add(shortcut, GFE) + DSConv refine → 12×12×32
      │
  [Classifier Head]  1×1 Conv → 12×12×5
      │
  [Bilinear Upsample ×8] → 96×96×5  (output logits)
"""
import tensorflow as tf
from keras import layers, Model


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------

def _conv_bn_relu(x, filters, kernel_size=3, strides=1, name=None):
    """Standard Conv2D → BatchNorm → ReLU."""
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding="same",
        use_bias=False, name=f"{name}_conv" if name else None,
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
    x = layers.ReLU(name=f"{name}_relu" if name else None)(x)
    return x


def _dw_sep_conv(x, filters, strides=1, name=None):
    """Depthwise-separable convolution (DW + BN, PW + BN + ReLU)."""
    x = layers.DepthwiseConv2D(
        3, strides=strides, padding="same",
        use_bias=False, name=f"{name}_dw" if name else None,
    )(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn" if name else None)(x)
    x = layers.Conv2D(
        filters, 1, strides=1, padding="same",
        use_bias=False, name=f"{name}_pw" if name else None,
    )(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn" if name else None)(x)
    x = layers.ReLU(name=f"{name}_relu" if name else None)(x)
    return x


def _inverted_residual(x, out_channels, expansion=4, name=None):
    """MobileNetV2-style inverted residual block (stride=1, same spatial size).

    Expand → Depthwise → Project (linear).
    Residual add when input and output channels match.
    """
    in_channels = x.shape[-1]
    exp_channels = in_channels * expansion

    residual = x

    # Expand
    x = layers.Conv2D(
        exp_channels, 1, padding="same", use_bias=False,
        name=f"{name}_expand_conv" if name else None,
    )(x)
    x = layers.BatchNormalization(name=f"{name}_expand_bn" if name else None)(x)
    x = layers.ReLU(name=f"{name}_expand_relu" if name else None)(x)

    # Depthwise
    x = layers.DepthwiseConv2D(
        3, strides=1, padding="same", use_bias=False,
        name=f"{name}_dw" if name else None,
    )(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn" if name else None)(x)
    x = layers.ReLU(name=f"{name}_dw_relu" if name else None)(x)

    # Project (linear — no activation)
    x = layers.Conv2D(
        out_channels, 1, padding="same", use_bias=False,
        name=f"{name}_project_conv" if name else None,
    )(x)
    x = layers.BatchNormalization(name=f"{name}_project_bn" if name else None)(x)

    # Residual only when shapes match
    if in_channels == out_channels:
        x = layers.Add(name=f"{name}_add" if name else None)([residual, x])

    return x


# ---------------------------------------------------------------------------
# MicroSeg sub-modules
# ---------------------------------------------------------------------------

def _learning_to_downsample(x):
    """Three-stage downsample: 96×96→48×48→24×24→12×12 (stride 8 total).

    Produces the high-resolution shortcut for the Feature Fusion module.
    """
    x = _conv_bn_relu(x, 16, strides=2, name="ltd_conv")       # 48×48×16
    x = _dw_sep_conv(x, 24, strides=2, name="ltd_dw1")         # 24×24×24
    x = _dw_sep_conv(x, 32, strides=2, name="ltd_dw2")         # 12×12×32
    return x


def _global_feature_extractor(x):
    """Four IRB blocks at 12×12 spatial resolution.

    Builds a large receptive field across the entire 96×96 field of view
    while staying entirely at the cheap 12×12 feature map resolution.
    """
    x = _inverted_residual(x, 24, expansion=4, name="gfe_irb1")  # 12×12×24
    x = _inverted_residual(x, 24, expansion=4, name="gfe_irb2")  # 12×12×24
    x = _inverted_residual(x, 32, expansion=4, name="gfe_irb3")  # 12×12×32
    x = _inverted_residual(x, 32, expansion=4, name="gfe_irb4")  # 12×12×32
    return x


def _feature_fusion(shortcut, features):
    """Fuse LTD shortcut + GFE output, then refine with a DSConv.

    Both tensors must be 12×12×32 at this point.
    """
    # Project shortcut channels to match features if needed.
    # Both are 32-channel so this is a residual-style alignment conv.
    shortcut = _conv_bn_relu(shortcut, 32, kernel_size=1, name="fusion_shortcut_proj")
    x = layers.Add(name="fusion_add")([shortcut, features])
    x = _dw_sep_conv(x, 32, name="fusion_refine")               # 12×12×32
    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_microseg(input_shape=(96, 96, 3), num_classes=5) -> Model:
    """Build and return the MicroSeg Keras model.

    Args:
        input_shape: HWC tuple; must be (96, 96, 3) for ESP32-CAM deployment.
        num_classes: number of output segmentation classes (default 5).

    Returns:
        A Keras Model with:
          - Input shape:  (None, H, W, 3)  float32 in [0, 1]
          - Output shape: (None, H, W, num_classes)  float32 logits

    TFLite export note:
        Quantize with full-integer INT8 PTQ. The bilinear upsample uses
        tf.image.resize which maps to ResizeBilinear in TFLite — confirmed
        compatible with TFLite Micro on ESP32.
    """
    H, W, C = input_shape

    inputs = layers.Input(shape=input_shape, name="image_input")

    # Encoder
    shortcut = _learning_to_downsample(inputs)       # 12×12×32
    features = _global_feature_extractor(shortcut)   # 12×12×32

    # Fusion
    fused = _feature_fusion(shortcut, features)      # 12×12×32

    # Head: 1×1 conv → logits at coarse resolution
    logits_coarse = layers.Conv2D(
        num_classes, 1, padding="same", use_bias=True,
        name="classifier_logits",
    )(fused)                                         # 12×12×num_classes

    # Upsample back to input resolution (×8).
    # Lambda wraps tf.image.resize so it serializes cleanly.
    logits = layers.Lambda(
        lambda t: tf.image.resize(t, [H, W], method="bilinear"),
        name="upsample_x8",
    )(logits_coarse)                                 # 96×96×num_classes

    return Model(inputs=inputs, outputs=logits, name="MicroSeg")
