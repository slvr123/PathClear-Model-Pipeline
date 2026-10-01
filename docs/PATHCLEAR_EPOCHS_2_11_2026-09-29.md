# PathClear continuation through epoch 11

Date: 2026-09-29.
Status: I completed the ten additional epochs I requested and stopped training at epoch 11.
I continued the seven-active-class Cityscapes adaptation from its saved epoch-1 model and Adam optimizer state.
I preserved the original nineteen-class baseline and the separate epoch-1 experiment.

## My result

I selected epoch 11 as the best checkpoint by validation mean IoU across all 500 Cityscapes validation images.
I saved it at `checkpoints/pathclear_10class/cityscapes_epochs2_11_2026-09-29/best.keras`.
My `last.keras` is also epoch 11.
Both checkpoint hashes are `4a64d771cb8401d213df1f259626e5f89cc9822552de3de0f4d7c1b74dabaa34`.

| Validation metric | Epoch 1 | Epoch 11 |
|---|---:|---:|
| Mean IoU | 67.63% | 68.56% |
| Pixel accuracy | 92.37% | 92.62% |
| Macro Dice | 78.82% | 79.59% |
| Loss | 0.25162 | 0.24712 |

I gained 0.93 percentage points of validation mean IoU over epoch 1.
The final epoch added about 0.0095 percentage points over epoch 10, so improvements were becoming small with this frozen-feature configuration.
I do not treat the higher training scores as a substitute for validation performance.

## Epoch history

I used all 2,975 training images and 500 validation images per epoch, at batch size 1.
The row numbering here starts at 1; the underlying Keras CSV stores zero-based epoch indices.

| Epoch | Training loss | Validation loss | Validation mean IoU | Validation pixel accuracy |
|---|---:|---:|---:|---:|
| 1, source | 0.21070 | 0.25162 | 67.6318% | 92.3694% |
| 2 | 0.15361 | 0.24986 | 68.0877% | 92.4859% |
| 3 | 0.15002 | 0.24887 | 68.2656% | 92.5335% |
| 4 | 0.14863 | 0.24822 | 68.3542% | 92.5607% |
| 5 | 0.14788 | 0.24776 | 68.4181% | 92.5776% |
| 6 | 0.14740 | 0.24751 | 68.4561% | 92.5894% |
| 7 | 0.14707 | 0.24745 | 68.4890% | 92.5968% |
| 8 | 0.14681 | 0.24731 | 68.5141% | 92.6030% |
| 9 | 0.14667 | 0.24726 | 68.5318% | 92.6084% |
| 10 | 0.14653 | 0.24712 | 68.5494% | 92.6129% |
| 11 | 0.14640 | 0.24712 | 68.5589% | 92.6159% |

![Training and validation history](../checkpoints/pathclear_10class/cityscapes_epochs2_11_2026-09-29/training_history.png)

## Class results

| Class | Epoch-1 IoU | Epoch-11 IoU |
|---|---:|---:|
| Road | 94.45% | 94.63% |
| Sidewalk | 63.73% | 64.50% |
| Person | 53.73% | 55.34% |
| Vehicle | 82.00% | 82.94% |
| Boundary | 35.21% | 36.54% |
| Obstacle | 52.80% | 54.33% |
| Vegetation | 91.50% | 91.63% |

I improved all seven class IoUs, but Boundary remained the weakest category.
I kept Stairs, Crosswalk, and Uneven Pavement inactive and untrained; these results do not measure them.
I excluded building, sky, terrain, and source void pixels from scores and loss as specified in my mapping.
Those pixels still receive predictions, so this score does not establish reliable obstacle detection on ignored content.
I have not measured Philippine/local accuracy or made a directly comparable assessment against the original nineteen-class overall score.

I visually inspected the same first-by-filename validation examples from Frankfurt, Lindau, and Munster used after epoch 1.
I saw similar broad segmentations and persistent errors around thin structures, boundaries, and ignored backgrounds.
The Frankfurt sample's valid-pixel accuracy decreased from 95.79% to 95.38%, while the Lindau and Munster examples improved.
I therefore do not interpret the improved aggregate metrics as improvement on every scene.
I saved the comparisons under the continuation experiment's `previews/` directory and verified that preview inference preserved the checkpoint and weights.

## What I preserved and checked

I restored epoch 1 at optimizer update 2,975, without recompiling or resetting Adam.
I recomputed its full validation result before continuing and checked it against the saved result.
I added 29,750 optimizer updates and finished at update 32,725, corresponding to eleven adaptation epochs.
I kept the learning rate at 0.001, trained only `classifier_logits`, and left all feature layers and BatchNorm statistics frozen.
I checked every frozen tensor and the reserved classifier values at each completed epoch.
Validation did not modify model weights.

I saved the latest completed checkpoint and matching state each epoch and replaced the best checkpoint only when validation mean IoU improved.
I carried the previous best into the new directory so a worse continuation could not erase it.
I preserved cumulative history and per-epoch validation confusion matrices and class scores.
I reloaded the final checkpoint and optimizer and compared their arrays exactly.
I reloaded the best checkpoint, reevaluated all 500 validation images, and matched its saved selection score.
I also checked that removing the class-selection layer preserves the trained weights and exposes all ten outputs for a future local stage, without running local training.

I passed all 23 tests before starting the real continuation.
The two new continuation tests checked an Adam update after save/reload against uninterrupted training, and verified retention/replacement of the best checkpoint with controlled validation examples.
The real run completed without a non-finite metric or out-of-memory failure.
I measured 1,913.86 seconds, about 31.9 minutes, including resume verification, training, checkpoint saves, and final validation checks.
I restarted the recorded random seed, so this continuation is not a claim of bit-for-bit replay of an uninterrupted dropout/shuffle stream.

I verified the original baseline best-checkpoint hash remains `47a248f9c3ee5f38194eb7071c035b4bc1c3c01254bec8bb44bbc240409321d7`.
I verified the source epoch-1 best-checkpoint hash remains `5980875305d59a2c9d53584e27d8dc198ce30d22efa016dbb0033ed101dc1481`.
I also checked the protected source experiment files and original core scripts against their saved hashes.

## Files and commands

I added `continue_pathclear_cityscapes.py`, `test_pathclear_continuation.py`, `plot_pathclear_history.py`, and this report.
I updated the training guide and local dataset plan with the explicitly requested continuation.
I generalized the preview figure title so it does not incorrectly label later checkpoints as one-epoch models.
I kept the original baseline and the earlier one-epoch training runner unchanged.

```powershell
wsl -d Ubuntu -- bash ./run_wsl.sh -m unittest -v test_pathclear_continuation test_pathclear_stages test_pathclear_pipeline test_training_metrics
wsl -d Ubuntu -- bash ./run_wsl.sh -u continue_pathclear_cityscapes.py --resume-from checkpoints/pathclear_10class/cityscapes_1epoch_2026-09-29 --epochs 10 --output-dir checkpoints/pathclear_10class/cityscapes_epochs2_11_2026-09-29
wsl -d Ubuntu -- bash ./run_wsl.sh -u preview_pathclear_stage.py checkpoints/pathclear_10class/cityscapes_epochs2_11_2026-09-29/best.keras --output-dir checkpoints/pathclear_10class/cityscapes_epochs2_11_2026-09-29/previews
wsl -d Ubuntu -- bash ./run_wsl.sh plot_pathclear_history.py checkpoints/pathclear_10class/cityscapes_epochs2_11_2026-09-29/history.csv checkpoints/pathclear_10class/cityscapes_epochs2_11_2026-09-29/training_history.png
```

I keep machine-readable results in `result.json`, resumable counters/hashes in `state.json`, and cumulative values in `history.csv` beside the checkpoints.
I saved the persistent console log as `run.log` and the recorded settings and source hashes in `config.json`.
The scripts refuse to overwrite an existing nonempty output directory.

## My next decision

I will keep epoch 11 as the current best intermediate model.
Given the small late-epoch gains, I will review remaining errors and prioritize local collection and annotation before requesting another unchanged training block.
Any unfreezing of feature layers, local-data training, or TFLite export remains a separate step.
