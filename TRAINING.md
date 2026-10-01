# Cityscapes training

I preserved the Fast-SCNN architecture, raw logits, 512 x 1024 resolution, Windows environment, and WSL GPU launcher.
I use the existing loader's batch-size default, with an optional `--batch-size` override passed directly to `make_dataset`.
The WSL launcher defaults to batch size 1; Windows retains batch size 2.

## Controlled test

From the project directory in WSL:

```bash
bash run_wsl.sh -u train.py --smoke-test
```

I cap smoke mode at one epoch, 50 training batches and 10 validation batches.
Smaller explicit step counts are allowed for diagnostics.
Without flags, training also defaults to one epoch with 50/10 batches.
I use `--full-dataset` explicitly to process all training and validation batches.
The loader shuffles training image paths before loading; validation uses a stable order.
A short validation subset is only an infrastructure check and may not represent all Cityscapes classes or cities.

## Metrics and loss

I pass pixel accuracy, mIoU and Dice through Keras `weighted_metrics`.
The loader replaces void ID 255 with a safe placeholder target and assigns it weight zero, so ignored pixels do not become road pixels in these metrics or the loss.
All predictions are obtained from the largest of the 19 raw logits.

Pixel accuracy is correct valid pixels divided by all valid pixels.
For each class, IoU is TP / (TP + FP + FN).
Dice is 2 TP / (2 TP + FP + FN).
I accumulate confusion counts over all evaluated batches, then macro-average IoU and Dice across classes present in either valid targets or valid predictions.
Classes absent from both are excluded, while false-positive-only classes contribute zero.
An entirely ignored evaluation returns zero metrics.
The confusion matrices use float64 to accept float sample weights without warnings and retain precision across large pixel counts.

I retained `SparseCategoricalCrossentropy(from_logits=True)` with its existing default reduction.
Zero-weight pixels contribute no loss, but the loss denominator includes all pixel positions, including ignored positions.
Thus the reported loss is not a mean over valid pixels alone and can vary with void coverage.
I retained this behavior for continuity with the working sanity test.

Training updates weights; Keras validation uses inference behavior without optimizer updates.
Smoke mode explicitly checks every model weight before and after validation, including BatchNorm statistics, and verifies a training weight update.
Training metrics monitor learning; thesis results require held-out validation/test evaluation, not training scores.

## Saved outputs and resume

I save the following under `checkpoints/` by default:

- `best.keras`: full model and optimizer from the highest validation mIoU.
- `last.keras`: full model and optimizer from the most recent completed epoch.
- `history.csv`: epoch-level train/validation loss, pixel accuracy, mIoU, Dice and learning rate.
- `state.json`: completed epoch count, best score, optimizer step count and learning-rate scheduler counters.
- `run_from_epoch_XXXX.json`: CLI settings for each invocation.

I refuse to overwrite a nonempty output directory without explicit `--resume`.
Resume requires the matching checkpoint, state and history files in that directory.
It restores optimizer state, epoch numbering, the best-score threshold and scheduler patience; `--epochs` means additional epochs.
The loader's random shuffle stream is not restored exactly, so resume is not bit-for-bit replay.
Only completed epochs are resumable; a partially completed epoch is not saved.

```bash
bash run_wsl.sh -u train.py --resume --epochs 5 --batch-size 1
```

I use ReduceLROnPlateau to halve learning rate after three epochs without meaningful validation mIoU improvement, down to 0.000001.
I use TerminateOnNaN and reject non-finite epoch metrics before saving checkpoints.
I keep mixed precision disabled and preserve GPU memory growth.
Generated models and the default output directory are excluded by `.gitignore`.
For another output location, I keep generated artifacts out of version control as well.

## Next longer experiment

For a fresh five-epoch baseline using all 2,975 training images and all 500 validation images:

```bash
bash run_wsl.sh -u train.py --epochs 5 --full-dataset --batch-size 1 --output-dir checkpoints/baseline-5epochs
```

I have not started this longer experiment.
I use a separate output directory so smoke-subset scores are not compared with full-validation scores when selecting the best model.

Metric and checkpoint API references: [Keras segmentation metrics](https://keras.io/api/metrics/segmentation_metrics/) and [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/).

## Verified smoke result, 2026-09-12

I ran `bash run_wsl.sh -u train.py --smoke-test` with batch size 1 at 512 x 1024.
The epoch processed 50 training batches and 10 validation batches in approximately 61 seconds, including compilation.
All 2,975 training pairs and 500 validation pairs were found without missing masks.

| Metric | Training | Validation subset |
|---|---:|---:|
| Loss | 1.320006 | 2.376966 |
| Pixel accuracy | 0.571163 | 0.417999 |
| mIoU | 0.115066 | 0.023222 |
| Macro Dice | 0.158022 | 0.032753 |

I verified training changed the inspected weight tensor by up to 0.02155414, and validation changed no model weights.
TensorFlow created GPU:0 for the RTX 3050 Ti and compiled the training cluster with XLA for CUDA.
There was no out-of-memory error.
Both model files saved successfully at 14,349,277 bytes each and reloaded successfully with optimizer iteration 50.
I also verified scheduler-state restoration and the model's original 1,160,707 parameters and output shape.
The metric tests passed on Windows CPU and WSL; Windows CLI help also passed.
I did not run another Cityscapes training epoch to test resume end to end.

The old float-weight/integer-metric warning is resolved.
The smoke log included CUDA warmup timing and XLA ignored-assert messages; the run completed with finite outputs.
I removed a redundant Keras shuffle warning after the run by setting `fit(shuffle=False)` because the training dataset already shuffles itself.
I have not rerun training after that argument-only change.
The raw smoke log is saved in `checkpoints/smoke.log`.

## PathClear preparation, 2026-09-29

I keep the commands above as the original nineteen-class Cityscapes workflow.
I do not use `train.py` to train the new ten-class taxonomy.
I prepared separate remapping, weight-transfer, and loss/metric helpers, tested without optimizer updates or a new checkpoint.
I describe the results and exact commands in [the ten-class preparation report](docs/PATHCLEAR_10CLASS_PREPARATION.md).
I will review [the annotation guide](docs/PATHCLEAR_ANNOTATION_GUIDE.md) on a real local pilot before selecting the local preprocessing and training configuration.
I have not implemented a local-data training command or started local training in this preparation stage.

## Seven-class Cityscapes adaptation, 2026-09-29

After discussing the missing classes, I authorized a separate three-batch smoke check followed by one full Cityscapes epoch if the checks pass.
I keep the original 19-class baseline and its training command unchanged.
I use `train_pathclear_cityscapes.py` for this new stage.
It always runs exactly one epoch; smoke mode contains only three training images and two validation images.
Full mode uses all 2,975 training images and 500 validation images, at batch size 1.
Each run starts fresh from the original feature weights and a newly initialized classifier; the smoke model is not resumed into the full run.

I preserve ten output channels and attach a serialized `ActiveClasses` layer that disables IDs 7-9 during both loss computation and prediction.
I train only `classifier_logits` with Adam at 0.001; all feature layers and BatchNorm statistics stay frozen.
I normalize loss over valid pixels within each image; epoch loss averages those image losses.
I accumulate accuracy, IoU, and Dice counts globally across valid pixels.
The ten-slot confusion matrix must have zero rows and columns for IDs 7-9, so reported macro scores cover the supported classes with nonzero union.
These seven-class scores are not directly comparable to the original 19-class baseline scores.

```powershell
wsl -d Ubuntu -- bash ./run_wsl.sh -m unittest -v test_pathclear_stages test_pathclear_pipeline test_training_metrics
wsl -d Ubuntu -- bash ./run_wsl.sh -u train_pathclear_cityscapes.py --mode smoke --output-dir checkpoints/pathclear_10class/cityscapes_smoke_2026-09-29
wsl -d Ubuntu -- bash ./run_wsl.sh -u train_pathclear_cityscapes.py --mode full --output-dir checkpoints/pathclear_10class/cityscapes_1epoch_2026-09-29
```

I refuse to overwrite a nonempty experiment directory or write inside the original baseline experiment.
I record configuration, input paths, source hashes, initial validation, training history, final validation per class, and a persistent run log beside the checkpoints.
I save `best.keras` and `last.keras` after the one completed epoch passes integrity checks; both represent that same epoch.
I verify exact checkpoint and optimizer reload, unchanged frozen/reserved weights, and unchanged hashes for the original checkpoints, training state, history, and core scripts.
I stop after this bounded experiment and review its results before adding epochs or unfreezing feature layers.

### Re-enabling my local classes later

I import `pathclear_stages` before loading the model so Keras can deserialize the saved output-selection layer.
When reviewed local labels and a local training loader are ready, I can remove that layer without changing the learned weights:

```python
import keras
from pathclear_stages import enable_local_classes

stage = keras.models.load_model("checkpoints/pathclear_10class/cityscapes_1epoch_2026-09-29/best.keras")
local = enable_local_classes(stage)
```

This helper returns all ten raw logits and a fresh optimizer while preserving the current feature freeze policy.
It does not train, save a local model, or supply a local dataset.
I tested that all three reserved outputs receive gradients after reactivation using a synthetic model, and verify weight-preserving removal on the real trained model.
I still need local annotations for all ten classes and a separately scoped training experiment.

I completed this smoke check and the single full epoch successfully.
I recorded metrics, per-class results, checkpoint locations, and visual limitations in [the adaptation report](docs/PATHCLEAR_CITYSCAPES_STAGE_2026-09-29.md).

## Continuing the saved adaptation model

I subsequently requested ten additional epochs, continuing from epoch 1 through epoch 11.
I use the separate `continue_pathclear_cityscapes.py` runner to preserve the original one-epoch experiment and its historical scripts.
The continuation restores `last.keras` with the saved Adam state, validates its epoch/update counts, and checks that recomputed validation matches the saved result before training.
I keep batch size 1, Adam at 0.001, frozen features, and seven active output classes.
The random stream is restarted from the recorded seed, so I do not claim bit-for-bit equivalence to an uninterrupted dropout/shuffle stream.

```powershell
wsl -d Ubuntu -- bash ./run_wsl.sh -m unittest -v test_pathclear_continuation test_pathclear_stages test_pathclear_pipeline test_training_metrics
wsl -d Ubuntu -- bash ./run_wsl.sh -u continue_pathclear_cityscapes.py --resume-from checkpoints/pathclear_10class/cityscapes_1epoch_2026-09-29 --epochs 10 --output-dir checkpoints/pathclear_10class/cityscapes_epochs2_11_2026-09-29
```

Here `--epochs` means additional epochs.
I carry the previous best checkpoint into the new experiment and replace it only when validation mIoU improves.
I save the latest completed model and optimizer each epoch, together with matching hashes and counters in `state.json`.
I preserve cumulative history and per-epoch validation confusion matrices, metrics, and class scores.
I verify frozen features and reserved outputs before each checkpoint save.
The runner rejects checkpoint/state/history mismatches rather than silently restarting.
I can resume a completed epoch from a continuation directory into another new output directory; partial epochs are not resumable.
I keep the three local-only classes inactive until a separately scoped local-data experiment.

I completed the requested continuation through epoch 11, selected epoch 11 as best, and stopped training.
I recorded the full history, class results, checks, and checkpoint location in [the continuation report](docs/PATHCLEAR_EPOCHS_2_11_2026-09-29.md).

I then explicitly requested five more epochs with the same settings, resuming from epoch 11 through epoch 16.
I completed this run and stopped at epoch 16, which is the best saved checkpoint at 68.5992% validation mean IoU and 92.6262% pixel accuracy.
I saved it at `checkpoints/pathclear_10class/cityscapes_epochs12_16_2026-09-29/best.keras`, preserving the earlier experiments.
I recorded the epoch results, class scores, and verification in [the five-epoch continuation report](docs/PATHCLEAR_EPOCHS_12_16_2026-09-29.md).

## My first TFLite integration export

On 2026-09-30, I requested a baseline export to test app integration.
I exported the preserved epoch-16 adaptation model to float32 TFLite, retaining ten output slots and seven active classes without training or quantization.
I saved the model and developer handoff in [the TFLite bundle](output/tflite/pathclear_epoch16_float32/README.md).
I verified prediction agreement against Keras on three Cityscapes images and two synthetic inputs, checked the packaged reference input independently, and preserved the original model hashes.
I still need to test the app's runtime, preprocessing, performance and live-camera integration on the target phone.

```powershell
wsl -d Ubuntu -- bash ./run_wsl.sh -u export_pathclear_tflite.py --output-dir output/tflite/pathclear_epoch16_float32
wsl -d Ubuntu -- bash ./run_wsl.sh output/tflite/pathclear_epoch16_float32/verify_bundle.py
```

The export command refuses to overwrite the completed bundle.
