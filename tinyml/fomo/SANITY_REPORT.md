# My FOMO pipeline sanity check, 2026-10-03

I verified this local experiment on `tinyml-fomo` using the existing Windows CPU environment.
I did not use Philippine photographs, Cityscapes, the groupmate's CNN, or the ESP32-CAM.
The model is explicitly FOMO-inspired and is not an official Edge Impulse training result.

| Check | Result |
|---|---|
| Dataset inspection | 13 synthetic images, 12 boxes, no structural errors |
| Splits | 7 train, 3 validation, 3 test images |
| Training subset | 6 images, 3 weight updates across 2 partial/full passes |
| Model input/output | `160 x 160 x 3` RGB to `20 x 20 x 4` class probabilities |
| Model parameters | 20,820, including nontrainable parameters |
| Initial/final weighted train loss | `1.386295` / `1.368533` |
| Maximum trainable-weight change | `0.003005` |
| Checkpoint | Saved and reloaded successfully; inference ran |
| Tests | 16 passed, including crop edges, class IDs, grid collisions, leakage, matching, and write guards |
| CLI entry points | Inspector, trainer, evaluator, inference, and exporter help commands passed |
| Missing dataset guard | Training without an explicit dataset or synthetic sanity flag was refused |
| INT8 export | Integer-only conversion, INT8 input/output, successful PC invocation |
| INT8 model file | 66,424 bytes, about 64.9 KiB |
| One-sample float/INT8 probability difference | Maximum absolute difference about `0.001961` |
| Synthetic held-out float and INT8 results | No detections at `0.60`; 3 false negatives, zero recall/F1, undefined precision |
| PC INT8 invocation | Median about 1.77 ms for 5 warmed calls during export |
| PC INT8 evaluation model call | Median about 1.94 ms across 3 synthetic test frames |
| ESP32 flash, RAM, latency, FPS | Not measured |

The fixture uses differently colored rectangles as stand-ins for the three classes, plus a negative image.
These shapes do not establish performance on actual people, vehicles, or obstacles.
Training samples and validation samples had similar losses, but this small artificial dataset cannot establish generalization.

The loss is a penalty for assigning the wrong class probability to a grid cell, with extra weight on object-center cells.
An almost uniform four-class prediction has a loss close to `log(4) = 1.386`.
The small reduction and nonzero weight changes show that optimization ran.
I did not observe useful detection learning at the agreed threshold.
There is no basis here for claiming good accuracy or choosing a deployment model.

The crop and nonempty-detection handoff were checked with controlled probability grids because the three-update model produced no detections.
Those checks verified original-frame coordinate scaling, top-three sorting, merging of adjacent same-class cells, RGB order, normalization, and padding at frame edges.
They do not establish that model-generated crops will contain sufficient detail for the CNN in real scenes.

TensorFlow reported that native Windows GPU support was unavailable; this bounded run completed on the CPU.
The converter printed warnings, including one about input statistics, but required integer built-in operators and produced verified INT8 I/O.
The installed `tf.lite.Interpreter` also printed a deprecation warning while successfully executing the exported model.
I kept the working environment unchanged; future dependency changes need separate validation.

Raw artifacts are local and ignored by Git:

- `artifacts/sanity-check-20261003/model.json`: training settings, manifest digest, loss history, and checkpoint check.
- `artifacts/inspection-check-20261003/`: inspection report and six annotated previews, visually checked.
- `artifacts/evaluation-check-20261003/report.json`: original Keras model detection results.
- `artifacts/export-verified-20261003/`: INT8 model, scales, and conversion report.
- `artifacts/evaluation-int8-20261003/report.json`: exported model detection results.

I need reviewed local instance boxes, agreed obstacle rules, independent route/session splits, and a decision about pretrained initialization or official Edge Impulse training before a real experiment.
I then compare held-out float/INT8 center detection results and measure the physical camera-to-warning path.

Git verification showed no changes to any previously tracked file.
Both branch tips remained at `53ac532`; only new files under `tinyml/fomo/` were untracked.
Generated images and model binaries were confirmed ignored.
I did not commit, push, merge, reset, rebase, or change the stable Fast-SCNN pipeline.
