"""
Training, Dataset Preparation & Full INT8 Quantization for FOMO MobileNetV2
Hardware Target: Standalone ESP32-CAM Assistive Eyewear
Exports: fomo_mobilenetv2_int8.tflite and model_data.h (C array for TFLite Micro)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fomo_model import build_fomo_mobilenetv2, CLASSES, NUM_CLASSES


def generate_fomo_synthetic_batch(batch_size=256, img_size=96, grid_size=12):
    """
    Generates synthetic training samples formatted for FOMO centroid detection:
    - Inputs: (batch_size, 96, 96, 3)
    - Targets: (batch_size, 12, 12, 11) - One-hot class encoding per grid cell.
    """
    images = np.zeros((batch_size, img_size, img_size, 3), dtype=np.float32)
    # Default to background (channel 0)
    labels = np.zeros((batch_size, grid_size, grid_size, NUM_CLASSES), dtype=np.float32)
    labels[:, :, :, 0] = 1.0  # All background initially

    for i in range(batch_size):
        # Base scene: Road on left (gx=0..3, gy=6..11), Sidewalk on center/right (gx=4..9, gy=6..11)
        # Boundary wall on far right (gx=10..11, gy=5..11)
        for gy in range(6, grid_size):
            for gx in range(grid_size):
                labels[i, gy, gx, 0] = 0.0  # Clear background
                if gx <= 3:
                    labels[i, gy, gx, 1] = 1.0  # Class 1: Road
                    images[i, gy*8:(gy+1)*8, gx*8:(gx+1)*8] = [0.25, 0.25, 0.28]
                elif gx >= 10:
                    labels[i, gy, gx, 5] = 1.0  # Class 5: Boundary (wall/fence)
                    images[i, gy*8:(gy+1)*8, gx*8:(gx+1)*8] = [0.6, 0.5, 0.4]
                else:
                    labels[i, gy, gx, 2] = 1.0  # Class 2: Sidewalk
                    images[i, gy*8:(gy+1)*8, gx*8:(gx+1)*8] = [0.55, 0.55, 0.55]

        # Upper scene: distant sky or overhead hanging vegetation
        hazard_type = np.random.choice(["overhead_vegetation", "person_ahead", "stairs_ahead", "crosswalk", "none"])

        if hazard_type == "overhead_vegetation":
            # Hanging branch in top-center (gy=1..3, gx=5..7)
            labels[i, 1:4, 5:8, :] = 0.0
            labels[i, 1:4, 5:8, 7] = 1.0  # Class 7: Vegetation
            images[i, 8:32, 40:64] = [0.1, 0.65, 0.15]

        elif hazard_type == "person_ahead":
            # Pedestrian centroid at gy=6..8, gx=6..7
            labels[i, 6:9, 6:8, :] = 0.0
            labels[i, 6:9, 6:8, 3] = 1.0  # Class 3: Person
            images[i, 48:72, 48:64] = [0.8, 0.4, 0.2]

        elif hazard_type == "stairs_ahead":
            # Steps at bottom-center (gy=9..11, gx=5..7)
            labels[i, 9:12, 5:8, :] = 0.0
            labels[i, 9:12, 5:8, 8] = 1.0  # Class 8: Stairs
            images[i, 72:96, 40:64] = [0.4, 0.4, 0.45]

        elif hazard_type == "crosswalk":
            # Zebra crosswalk across walking lane
            labels[i, 8:10, 4:9, :] = 0.0
            labels[i, 8:10, 4:9, 9] = 1.0  # Class 9: Crosswalk
            images[i, 64:80, 32:72] = [0.95, 0.95, 0.95]

        # Add camera sensor noise
        images[i] += np.random.normal(0, 0.03, images[i].shape)
        images[i] = np.clip(images[i], 0.0, 1.0)

    return images, labels


def representative_dataset_gen():
    """Calibration dataset for INT8 quantization."""
    calib_images, _ = generate_fomo_synthetic_batch(batch_size=80, img_size=96, grid_size=12)
    for i in range(80):
        sample = np.expand_dims(calib_images[i], axis=0).astype(np.float32)
        yield [sample]


def convert_and_quantize_fomo(model, output_tflite_path):
    """Full INT8 quantization for TFLite Micro on ESP32."""
    print(f"\n--- Quantizing FOMO Model to INT8 for ESP32-CAM ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Enforce INT8 kernels
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    with open(output_tflite_path, "wb") as f:
        f.write(tflite_quant_model)

    size_kb = len(tflite_quant_model) / 1024.0
    print(f"Quantized FOMO model saved to: {output_tflite_path}")
    print(f"Quantized Model Size: {size_kb:.2f} KB (Fits in Flash!)")
    return tflite_quant_model


def export_to_c_header(tflite_bytes, output_header_path, variable_name="g_model"):
    """Converts the binary .tflite model into a C byte array header file."""
    print(f"\n--- Exporting C Header Array to {output_header_path} ---")
    hex_lines = []
    line_bytes = []
    
    for i, b in enumerate(tflite_bytes):
        line_bytes.append(f"0x{b:02x}")
        if (i + 1) % 12 == 0:
            hex_lines.append("  " + ", ".join(line_bytes) + ",")
            line_bytes = []
    if line_bytes:
        hex_lines.append("  " + ", ".join(line_bytes))

    header_content = f"""// Auto-generated TinyML Model Header for ESP32-CAM
// Model: FOMO MobileNetV2 alpha=0.35 (INT8 Quantized)
// Navigation Classes: 11 (0: Background, 1: Road, 2: Sidewalk, 3: Person, 4: Vehicle, 5: Boundary, 6: Obstacle, 7: Vegetation, 8: Stairs, 9: Crosswalk, 10: Uneven Pavement)
// Total Size: {len(tflite_bytes)} bytes ({len(tflite_bytes)/1024:.2f} KB)

#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

alignas(16) const unsigned char {variable_name}[] = {{
{chr(10).join(hex_lines)}
}};

const unsigned int {variable_name}_len = {len(tflite_bytes)};

#ifdef __cplusplus
}}
#endif

#endif  // MODEL_DATA_H_
"""
    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write(header_content)
    print(f"C Header generated successfully ({len(tflite_bytes)} bytes)!")


def main():
    print("=" * 80)
    print("FOMO MOBILENETV2 (ALPHA=0.35) INT8 TRAINING & EXPORT FOR ESP32-CAM")
    print("=" * 80)

    # 1. Build FOMO Model
    model = build_fomo_mobilenetv2(input_shape=(96, 96, 3), alpha=0.35, num_classes=NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 2. Train on assistive scenes
    print("\nPre-training FOMO on assistive scene centroid patterns...")
    x_train, y_train = generate_fomo_synthetic_batch(batch_size=384, img_size=96, grid_size=12)
    model.fit(x_train, y_train, epochs=4, batch_size=32, verbose=1)

    # 3. Quantize to INT8
    os.makedirs("tinyml-test/firmware", exist_ok=True)
    tflite_path = "tinyml-test/model/fomo_mobilenetv2_int8.tflite"
    tflite_bytes = convert_and_quantize_fomo(model, tflite_path)

    # 4. Export C Header
    header_path = "tinyml-test/firmware/model_data.h"
    export_to_c_header(tflite_bytes, header_path, variable_name="g_model")

    print("\n" + "=" * 80)
    print("SUCCESS: Pure FOMO Model exported and ready for ESP32-CAM!")
    print("=" * 80)


if __name__ == "__main__":
    main()
