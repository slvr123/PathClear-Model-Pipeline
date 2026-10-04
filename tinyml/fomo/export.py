"""Full INT8 TFLite export with training-only calibration and PC invocation check."""
import argparse
import json
import time
import tempfile

import numpy as np

from config import CLASS_NAMES, SETTINGS, output_directory
from dataset import load_rgb, load_arrays, validated_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True, help="Calibration uses only this manifest's train split")
    parser.add_argument("--allow-synthetic", action="store_true", help="Plumbing-only calibration/export")
    parser.add_argument("--output")
    args = parser.parse_args()
    records, dataset_report = validated_records(args.manifest, args.allow_synthetic)
    import tensorflow as tf
    from model import load_model
    model, metadata = load_model(args.model)
    if str(args.model).endswith(".tflite"):
        parser.error("Export requires the original .keras checkpoint")
    if metadata["synthetic"] and not args.allow_synthetic:
        parser.error("Synthetic-trained model requires --allow-synthetic")
    all_train = [r for r in records if r["split"] == "train"]
    order = np.random.default_rng(SETTINGS.seed).permutation(len(all_train))[:SETTINGS.calibration_images]
    train = [all_train[i] for i in order]
    calibration, _ = load_arrays(train)
    def representative():
        for pixels in calibration:
            yield [pixels.astype(np.float32)[None]/255.0]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    directory = output_directory(args.output, "export")
    # Keras conversion creates an intermediate SavedModel. Keep its temporary files here too.
    with tempfile.TemporaryDirectory(prefix="conversion-", dir=directory) as scratch:
        previous_tempdir = tempfile.tempdir
        try:
            tempfile.tempdir = scratch
            float_binary = converter.convert()
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            binary = converter.convert()
        finally:
            tempfile.tempdir = previous_tempdir
    interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
    interpreter.allocate_tensors()
    input_info, output_info = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if input_info["dtype"] != np.int8 or output_info["dtype"] != np.int8:
        raise RuntimeError("Export did not produce INT8 I/O")
    scale, zero = input_info["quantization"]
    sample = calibration[:1].astype(np.float32)/255.0
    quantized = np.clip(np.rint(sample / scale + zero), -128, 127).astype(np.int8)
    interpreter.set_tensor(input_info["index"], quantized)
    interpreter.invoke()
    latencies = []
    for _ in range(5):
        start = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - start) * 1000)
    out_scale, out_zero = output_info["quantization"]
    output = (interpreter.get_tensor(output_info["index"]).astype(np.float32) - out_zero) * out_scale
    reference = model(sample, training=False).numpy()
    grid_size = metadata['contract']['grid_size']
    if output.shape != (1, grid_size, grid_size, len(CLASS_NAMES)+1) or not np.isfinite(output).all():
        raise RuntimeError("Invalid exported output")
    from inference import decode_grid, decoding_policy
    # Quantized softmax may not sum to exactly one; renormalize for this PC decoding check.
    normalized = output[0] / np.maximum(output[0].sum(axis=-1, keepdims=True), 1e-8)
    frame = load_rgb(train[0]["path"])
    detections = decode_grid(normalized, frame.shape[1], frame.shape[0])
    report = {"contract": metadata["contract"], "model_bytes": len(binary),
              "decoding": decoding_policy(),
              "float_tflite_bytes": len(float_binary),
              "keras_checkpoint_bytes": __import__("pathlib").Path(args.model).stat().st_size,
              "synthetic": dataset_report["synthetic"] or metadata["synthetic"],
              "sanity": metadata["sanity"], "calibration_train_images": len(train),
              "calibration_ids": [r["id"] for r in train],
              "input_shape": input_info["shape"].tolist(), "output_shape": output_info["shape"].tolist(),
              "input_quantization": [scale, zero], "output_quantization": [out_scale, out_zero],
              "pc_int8_invoke_ms_median": float(np.median(latencies)),
              "pc_int8_invoke_ms_p95": float(np.percentile(latencies, 95)),
              "sample_max_probability_difference": float(np.max(np.abs(output - reference))),
              "sample_detections": detections,
              "embedded_flash_requirement": None, "embedded_ram_requirement": None,
              "esp32_latency_ms": None, "esp32_fps": None,
              "limitations": "PC check only; compare float/INT8 on reviewed test data and profile the physical board."}
    (directory / "model.int8.tflite").write_bytes(binary)
    (directory / "model.float32.tflite").write_bytes(float_binary)
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Export: {directory}")


if __name__ == "__main__":
    main()
