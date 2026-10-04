# My FOMO board benchmark protocol

I have no runnable ESP32 camera firmware project in this repository: the inspection found no `.ino`, `platformio.ini`, or ESP-IDF camera application.
I have not compiled, flashed, or measured a physical board in this experiment.
I keep hardware figures unknown until I can collect them on the actual device.

## What I know from the frozen baseline

| Item | Verified value |
|---|---:|
| INT8 model file | 70,736 bytes |
| INT8 input tensor | 160 x 160 x 3 = 76,800 bytes |
| INT8 output tensor | 20 x 20 x 3 = 1,200 bytes |
| Prior paired desktop model-call median / p95 | 1.85 / 2.44 ms |
| Prior paired desktop model plus NMS decoding median / p95 | 2.07 / 2.82 ms |

I retain the timing source at `artifacts/grid-size-benchmark-20261004/report.json`.
It uses one TFLite CPU thread, 20 warm-ups and 200 measured calls over 100 prepared inputs.
Model-call timing includes input quantization and output copying/dequantization.
These desktop figures exclude camera acquisition, resizing, MicroSeg, and decision logic; they are not ESP32 latency or FPS.
The no-NMS policy changes desktop decoding work and requires its own timing record.
The file size and the two I/O tensors do not describe the full runtime memory requirement.

My selected weight-10 v2 candidate also has a 70,736-byte INT8 file and the same I/O tensor dimensions.
The final paired desktop run records 1.15 / 2.18 ms median / p95 for its model call and 1.34 / 2.49 ms with cells/0.70 decoding.
Frozen v1 in that same run records 1.34 / 2.54 ms with NMS/0.60 decoding.
I use `artifacts/v2-policy-benchmark-isolated-20261004/report.json` for this comparison and make no hardware speedup claim.

## What I will measure later

1. I record the board, camera, CPU frequency, actual detected PSRAM capacity, power supply, ESP-IDF version, TFLite Micro version, optimized-kernel settings, model SHA-256, and decoder/threshold.
2. I first use the experimental AI Thinker ESP32-CAM for FOMO alone.
   My preferred upgrade remains the XIAO ESP32-S3 Sense with its Sense camera setup and 8 MB PSRAM; ESP32-S3 DevKitC N8R8 is acceptable for development.
3. I pin the exact firmware/runtime versions and verify every operator in this exported model is supported by that build before claiming successful inference.
4. I replay a fixed input tensor and compare output classes, probabilities, and decoded coordinates with the desktop export.
   I use the export report's quantization scales and zero points rather than assuming an int8 conversion formula.
   I retain the same RGB order, direct bilinear resize, embedded model scaling, and output-softmax normalization policy.
5. I use a separate recording build with `tflite::RecordingMicroInterpreter`, invoke the model, and call `PrintAllocations()`.
   I record arena allocation, persistent allocations, and scratch/head use, then verify a production interpreter build at the chosen arena size with margin.
   I search for the smallest arena that allocates and repeatedly invokes successfully, recording failures rather than hiding them.
   Recording itself adds overhead, so I use the production build for final latency and memory claims.
   This follows the [TFLite Micro memory guide](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md).
6. I log heap values before camera initialization, after camera initialization, after interpreter allocation, and during the sustained loop.
   I record `heap_caps_get_free_size`, `heap_caps_get_minimum_free_size`, and `heap_caps_get_largest_free_block` separately for `MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT` and `MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT`.
   I record static memory from `idf.py size` and task stack high-water marks as well.
   I distinguish boot-lifetime heap minima from the local benchmark interval and do not add arena usage twice when interpreting total memory.
   The [ESP-IDF heap API](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/mem_alloc.html) defines these counters.
7. I initialize the camera with one frame buffer first and record pixel format, resolution, buffer location, buffer count, and capture failures.
   I repeat with the intended final capture configuration, returning every acquired frame buffer.
   Multiple frame buffers affect memory and capture behavior, as documented by the [Espressif camera driver](https://github.com/espressif/esp32-camera).
8. I time capture/wait, decode/resize/quantize, `Invoke()`, output decoding, and the complete FOMO loop separately using `esp_timer_get_time()`.
   I collect 20 warm-ups followed by at least 200 timed frames and a ten-minute sustained run.
   I log median, p95, maximum, failures, dropped/stale frames, and completed fresh frames divided by elapsed wall time.
   Reciprocal model latency alone is not camera-to-result FPS.
   The timer returns microseconds; its behavior is documented in the [ESP Timer API](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/esp_timer.html).
9. I later measure decision-logic overhead and real false warnings per minute on reviewed, route-separated footage.
   I leave MicroSeg coexistence, its scheduling, and its combined peak RAM unmeasured here because MicroSeg work is outside this request.
   VL53L1X distance data remains a separate future input to decision logic.

## Measurement record

I save board results in a new dated folder under `tinyml/fomo/artifacts/`, with raw serial logs and one row per frame.
I use these CSV columns: `frame_id,capture_us,preprocess_us,invoke_us,decode_us,total_us,internal_free_bytes,internal_min_free_bytes,internal_largest_block_bytes,psram_free_bytes,psram_min_free_bytes,psram_largest_block_bytes,detections,status`.
I save run-level metadata separately: board/runtime versions, model hash, arena capacity and recorded use, camera configuration, warm-up count, measured-frame count, duration, reset/failure counts, and static memory.
I label tensor arena, peak runtime RAM, camera-buffer interaction, physical latency, FPS, MicroSeg coexistence, and decision-logic overhead as unknown until measured.
