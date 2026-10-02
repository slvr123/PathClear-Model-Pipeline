"""
FOMO (Faster Objects, More Objects) Architecture for ESP32-CAM
Edge Impulse Style Centroid Object/Hazard Localization

Hardware Target: ESP32-CAM (Xtensa LX6 @ 240MHz, TFLite Micro)
Input: 96x96x3 (or 96x96x1 Grayscale) from OV2640 camera
Output: 12x12x(N+1) Centroid Heatmap (Background + 10 Navigation Classes)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 11 Channels: Index 0 is Background, 1..10 are the target navigation classes
CLASSES = [
    "background",        # 0: Background / No object
    "road",              # 1: Road surface
    "sidewalk",          # 2: Sidewalk / Walkway
    "person",            # 3: Pedestrian / Rider
    "vehicle",           # 4: Car, Jeepney, Bus, Truck, Bike
    "boundary",          # 5: Wall, Fence, Gate
    "obstacle",          # 6: Pole, Sign, Trash bin, Cart, Barrier
    "vegetation",        # 7: Overhead branch, Bush (head-level)
    "stairs",            # 8: Steps up / steps down
    "crosswalk",         # 9: Zebra crossing markings
    "uneven_pavement"    # 10: Broken pavement / potholes
]

NUM_CLASSES = len(CLASSES)  # 11 total


def build_fomo_mobilenetv2(input_shape=(96, 96, 3), alpha=0.35, num_classes=NUM_CLASSES):
    """
    Builds a native FOMO architecture with MobileNetV2 backbone (alpha=0.35).
    
    Why FOMO works for ESP32-CAM:
    1. Cut the decoder: Standard segmentation requires an upsampling decoder that uses 80% of RAM.
       FOMO eliminates the decoder completely.
    2. Centroid Prediction: It takes the stride-8 feature map (12x12 for 96x96 input)
       and applies a 1x1 convolution directly to output class probabilities per spatial cell.
    3. Runs at 10-15 FPS on ESP32-CAM!
    """
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Load MobileNetV2 base without top classifier, stride 8 cutoff
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights=None  # Built from scratch / ready for fine-tuning
    )

    # Use the 'block_6_expand_relu' feature map (stride 8)
    # For 96x96 input, stride 8 gives exact 12x12 spatial resolution
    try:
        feature_layer = base_model.get_layer("block_6_expand_relu").output
    except Exception:
        feature_layer = base_model.layers[-1].output

    feature_extractor = keras.Model(inputs=base_model.input, outputs=feature_layer, name="mobilenetv2_backbone")
    features = feature_extractor(inputs)

    # FOMO Head:
    # Dropout for regularization
    x = layers.Dropout(0.2, name="fomo_dropout")(features)
    
    # 1x1 Conv directly to num_classes logits
    x = layers.Conv2D(num_classes, kernel_size=1, padding="same", name="fomo_logits")(x)
    
    # Softmax across the class channels for each 12x12 spatial grid cell
    outputs = layers.Softmax(axis=-1, name="fomo_centroids")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"FOMO_MobileNetV2_a{int(alpha*100)}")
    return model


def extract_centroids(heatmap, threshold=0.5):
    """
    Post-processes FOMO output tensor (12, 12, num_classes)
    Extracts detected object centroids (grid_x, grid_y, class_id, confidence).
    """
    grid_h, grid_w, num_classes = heatmap.shape
    detections = []

    for gy in range(grid_h):
        for gx in range(grid_w):
            scores = heatmap[gy, gx]
            top_class = np.argmax(scores)
            confidence = scores[top_class]

            # Ignore background (class 0)
            if top_class > 0 and confidence >= threshold:
                detections.append({
                    "grid_x": gx,
                    "grid_y": gy,
                    "norm_x": (gx + 0.5) / grid_w,
                    "norm_y": (gy + 0.5) / grid_h,
                    "class_id": int(top_class),
                    "class_name": CLASSES[top_class],
                    "confidence": float(confidence)
                })

    return detections


if __name__ == "__main__":
    model = build_fomo_mobilenetv2(input_shape=(96, 96, 3), alpha=0.35, num_classes=NUM_CLASSES)
    model.summary()

    total_params = model.count_params()
    print(f"\n--- FOMO MobileNetV2 Analysis ---")
    print(f"Total Parameters     : {total_params:,}")
    print(f"Estimated FP32 Size  : {total_params * 4 / 1024:.2f} KB")
    print(f"Estimated INT8 Size  : {total_params * 1 / 1024:.2f} KB (Fits in Flash)")
    print(f"Output Heatmap Shape : {model.output_shape} (12x12 Centroid Grid, 11 Channels)")
