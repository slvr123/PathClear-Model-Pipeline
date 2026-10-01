# PathClear Cityscapes adaptation: first epoch

Date: 2026-09-29.
Status: one full adaptation epoch completed and verified; training stopped at the authorized boundary.
After discussing the missing classes, I approved learning the seven Cityscapes-supported categories first and enabling the three local-only classes when labeled local data is ready.
I authorized a small smoke check followed by one full epoch if verification passed.

## What I now have

I have a saved intermediate Fast-SCNN model with ten output slots and seven active predictions: Road, Sidewalk, Person, Vehicle, Boundary, Obstacle, and Vegetation.
I keep Stairs, Crosswalk, and Uneven Pavement disabled and untrained during this stage.
I saved the full-run checkpoint at `checkpoints/pathclear_10class/cityscapes_1epoch_2026-09-29/best.keras`.
I also saved `last.keras`; because I ran exactly one epoch, both contain the same completed training state.
I did not export TFLite or train on Philippine images.

I retain the original epoch-19 checkpoint at `checkpoints/baseline-5epochs/best.keras` unchanged.
I verified unchanged hashes for that checkpoint, the original `last.keras`, `state.json`, `history.csv`, and the original architecture, loader, training, and metric scripts.
The original best-checkpoint SHA-256 remains `47a248f9c3ee5f38194eb7071c035b4bc1c3c01254bec8bb44bbc240409321d7`.
The new full-run best-checkpoint SHA-256 is `5980875305d59a2c9d53584e27d8dc198ce30d22efa016dbb0033ed101dc1481`.

## My experiment

| Setting | Value |
|---|---|
| Starting features | I transferred 86 learned layers, 214 tensors, from the epoch-19 Cityscapes baseline. |
| Classifier | I initialized a fresh ten-output `classifier_logits` layer. |
| Trainable weights | I trained only the classifier; its seven active outputs contain 903 scalar weights/biases. |
| Reserved outputs | I preserved the 387 scalar values for IDs 7-9 exactly. |
| Features and BatchNorm | I kept every transferred weight and moving statistic frozen. |
| Training data | I used all 2,975 Cityscapes training images with automatic trainId remapping. |
| Validation data | I used all 500 validation images before and after training. |
| Image shape / batch size | I used 512x1024 RGB and batch size 1. |
| Optimizer / learning rate | I used a fresh Adam optimizer at 0.001. |
| Epochs / updates | I ran one full epoch, exactly 2,975 optimizer updates. |
| Seed | I used 42. |
| Runtime | I used TensorFlow 2.21.0 and Keras 3.15.1 through the existing WSL launcher on the RTX 3050 Ti. |
| Duration | I measured about 228 seconds inside the run, including initial evaluation, training, final evaluation, saving, and reload checks. |

I first ran a separate three-image training smoke check with two validation images and verified its checkpoint reload.
I did not resume that smoke model into the full run; the full run started independently from the original features and the same random seed.
I saved the smoke artifacts under `checkpoints/pathclear_10class/cityscapes_smoke_2026-09-29/`.

## Validation results

I compare the new classifier before and after adaptation on the same 500-image validation split.
The before column is the newly initialized classifier on transferred features, not the original trained 19-class baseline.

| Metric | Before adaptation | After one epoch |
|---|---:|---:|
| Pixel accuracy | 48.04% | 92.37% |
| Mean IoU | 11.90% | 67.63% |
| Macro Dice | 16.91% | 78.82% |
| Loss | 1.36749 | 0.25162 |

| Active class | Validation IoU |
|---|---:|
| Road | 94.45% |
| Sidewalk | 63.73% |
| Person | 53.73% |
| Vehicle | 82.00% |
| Boundary | 35.21% |
| Obstacle | 52.80% |
| Vegetation | 91.50% |

I compute IoU, Dice, and accuracy from accumulated valid-pixel counts; the absent classes have zero truth and prediction counts.
I exclude source void, building, terrain, and sky pixels from loss and metrics.
I normalize loss over valid pixels per image and average image losses over the epoch.
I do not compare these overall scores directly against the original 19-class results because categories, ignored regions, and the loss denominator differ.
I have not established Philippine performance, performance on classes 7-9, or an advantage over a fairly remapped original-baseline prediction comparison.

## How I keep local classes available

I attach a serialized `ActiveClasses` layer that replaces logits for IDs 7-9 with a fixed low value during the Cityscapes stage.
Those outputs cannot win predictions or receive training gradients, while the model retains its ten-channel output shape.
I tested the loss against the corresponding seven-class loss and verified zero gradient to the three inactive outputs.
I verified their actual classifier weights remained exactly unchanged throughout the real run.

I can use `enable_local_classes()` from `pathclear_stages.py` later to remove this output-selection layer and return all ten raw logits without modifying learned weights.
I tested that classes 7-9 receive learning updates again on a synthetic local-label example, and verified weight-preserving reactivation on the real saved model without training it.
I have not saved an activated local model or run local training.
I will use reviewed local examples covering all ten classes and a separately approved local experiment.

## Verification and observed issues

I passed all 21 unit tests, including three new stage tests and the 18 existing mapping/metric/pilot checks.
I verified that both smoke and full runs changed only the active classifier weights, left validation weights unchanged, and preserved the original experiment files.
I reloaded the full model and optimizer and compared their arrays exactly with the saved training state.
I confirmed output shape `(1, 512, 1024, 10)` and predictions restricted to IDs 0-6.
I observed no out-of-memory failure or non-finite training metrics.
TensorFlow reported peak allocator usage of 381,915,648 bytes, about 364 MiB; this is not total device/process memory or a mobile deployment benchmark.

I corrected a unit-test tolerance from seven to six decimal places because a metric returned 0.99999976 for an expected 1.0.
I observed existing framework warnings about an assertion skipped by XLA and a NumPy/Keras array conversion deprecation.
The dataset-stage assertions run outside the compiled training loss, and the explicit tests and integrity checks passed.

## Prediction inspection

I selected the first validation filename in each of Frankfurt, Lindau, and Munster before examining the predictions.
I generated and visually inspected all three comparisons, with original image, remapped truth, full prediction, and valid-pixel error map.
These are diagnostic examples, not a representative sample or an additional training run.

I saw broad road, sidewalk, vehicle, and vegetation regions being recognized, with errors around thin structures, pedestrians, vehicles, and surface boundaries.
I also saw buildings and sky assigned to supported categories, often Obstacle, because ignored content still receives model predictions even though it is absent from the loss and score.
I cannot interpret those ignored-region predictions as reliable obstacle detections.
Boundary is the weakest class in the full validation result and deserves inspection before extending training or using this model in a warning system.

![Frankfurt prediction comparison](../checkpoints/pathclear_10class/cityscapes_1epoch_2026-09-29/previews/frankfurt_comparison.png)

I saved all three comparisons and prediction ID masks in the experiment's `previews/` directory.
Preview inference preserved the checkpoint hash and all model weights.

## Commands and files

```powershell
wsl -d Ubuntu -- bash ./run_wsl.sh -m unittest -v test_pathclear_stages test_pathclear_pipeline test_training_metrics
wsl -d Ubuntu -- bash ./run_wsl.sh -u train_pathclear_cityscapes.py --mode smoke --output-dir checkpoints/pathclear_10class/cityscapes_smoke_2026-09-29
wsl -d Ubuntu -- bash ./run_wsl.sh -u train_pathclear_cityscapes.py --mode full --output-dir checkpoints/pathclear_10class/cityscapes_1epoch_2026-09-29
wsl -d Ubuntu -- bash ./run_wsl.sh -u preview_pathclear_stage.py checkpoints/pathclear_10class/cityscapes_1epoch_2026-09-29/best.keras --output-dir checkpoints/pathclear_10class/cityscapes_1epoch_2026-09-29/previews
```

I added `pathclear_stages.py`, `test_pathclear_stages.py`, `train_pathclear_cityscapes.py`, `preview_pathclear_stage.py`, and this report.
I updated `TRAINING.md`, `LOCAL_DATASET_PLAN.md`, and the historical preparation report to distinguish this authorized adaptation run from the earlier preparation-only stage.
I left the original nineteen-class executable scripts unchanged.

Each experiment contains configuration and source hashes, the selected input paths, transfer details, initial validation, CSV history, per-class validation, a confusion matrix, the completed result, and a persistent run log.
The scripts reject nonempty output directories, so repeating the displayed commands against these completed folders will refuse to overwrite them.
The current adaptation runner deliberately has no resume or multiple-epoch option; continuing this checkpoint needs a separately scoped continuation change.

## My next decision

I will review the per-class results and error images before authorizing more epochs or unfreezing features.
I will collect and annotate the local pilot in parallel, with deliberate coverage of Stairs, Crosswalk, and Uneven Pavement.
I will keep this intermediate checkpoint as a separate experimental stage and retain the original baseline for a fair later comparison.
