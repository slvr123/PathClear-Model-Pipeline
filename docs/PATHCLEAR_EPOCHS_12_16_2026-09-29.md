# PathClear continuation through epoch 16

Date: 2026-09-29.
I requested exactly five additional epochs to check whether the small validation gains would continue.
I resumed the epoch-11 checkpoint and Adam optimizer, completed epochs 12-16, and stopped.
The run took 1,080.21 seconds, about 18 minutes, including validation and checkpoint verification.

## My results

I selected epoch 16 as the best checkpoint by mean IoU on all 500 Cityscapes validation images.
I saved it at `checkpoints/pathclear_10class/cityscapes_epochs12_16_2026-09-29/best.keras`.
The final `last.keras` contains the same checkpoint.

| Epoch | Validation pixel accuracy | Validation mean IoU | Validation loss |
|---|---:|---:|---:|
| 11, source | 92.6159% | 68.5589% | 0.247116 |
| 12 | 92.6180% | 68.5706% | 0.247079 |
| 13 | 92.6207% | 68.5858% | 0.247051 |
| 14 | 92.6234% | 68.5888% | 0.247002 |
| 15 | 92.6248% | 68.5980% | 0.246964 |
| 16, best | 92.6262% | 68.5992% | 0.246961 |

I gained 0.0403 percentage points of mean IoU and 0.0104 percentage points of pixel accuracy over epoch 11.
Macro Dice increased from 79.5924% to 79.6268%.
Every added epoch improved validation mean IoU, but the final epoch added only 0.0011 percentage points.
These results support diminishing returns from this unchanged classifier-only training setup.
They do not establish a hard performance ceiling or a statistically significant improvement on unseen local scenes.

![Cumulative adaptation history](../checkpoints/pathclear_10class/cityscapes_epochs12_16_2026-09-29/training_history.png)

## Class results

| Class | Epoch-11 IoU | Epoch-16 IoU |
|---|---:|---:|
| Road | 94.6344% | 94.6397% |
| Sidewalk | 64.5030% | 64.5218% |
| Person | 55.3379% | 55.4348% |
| Vehicle | 82.9397% | 82.9679% |
| Boundary | 36.5363% | 36.6162% |
| Obstacle | 54.3303% | 54.3720% |
| Vegetation | 91.6306% | 91.6416% |

All seven class IoUs improved slightly; Boundary remains the weakest category.
Stairs, Crosswalk, and Uneven Pavement remain inactive and untrained pending annotated local data.
Building, sky, terrain, and source void pixels remain excluded from loss and scores under the existing mapping.
I do not interpret this Cityscapes validation result as Philippine accuracy or a measure of the three missing classes.

## What I verified

I used all 2,975 training images per epoch, batch size 1, input resolution 512 by 1024, Adam at 0.001, and the existing frozen-feature configuration.
I restored optimizer update 32,725, added exactly 14,875 updates, and finished at update 47,600.
I reproduced the source validation scores before training.
I verified that feature weights, BatchNorm statistics, and the three reserved classifier outputs stayed unchanged at every completed epoch.
I reloaded the final model and optimizer and compared their saved arrays exactly.
I reevaluated the saved best model and matched its selection score.
I checked that removing the active-class gate preserves the weights for a future ten-class stage, without training that stage.
I verified protected source experiment files and original core scripts against their recorded hashes.
The original baseline and epoch-11 experiment remain unchanged.
The run completed without a non-finite metric or out-of-memory failure.
I restarted the recorded random seed, so I do not claim exact equivalence to an uninterrupted random stream.

The best and last checkpoint SHA-256 is `0789ec26e12f801b79eaf59c47e3e9b0fa638e8ef5d205e0ee006973fa585b95`.
I saved cumulative history, per-epoch confusion matrices and class scores, settings, resume verification, logs, state, and final results beside the checkpoints.

## Reproduction and next decision

```powershell
wsl -d Ubuntu -- bash ./run_wsl.sh -u continue_pathclear_cityscapes.py --resume-from checkpoints/pathclear_10class/cityscapes_epochs2_11_2026-09-29 --epochs 5 --output-dir checkpoints/pathclear_10class/cityscapes_epochs12_16_2026-09-29
wsl -d Ubuntu -- bash ./run_wsl.sh plot_pathclear_history.py checkpoints/pathclear_10class/cityscapes_epochs12_16_2026-09-29/history.csv checkpoints/pathclear_10class/cityscapes_epochs12_16_2026-09-29/training_history.png
```

The runner refuses to overwrite this completed experiment.
I retain epoch 16 as the best saved intermediate checkpoint from the completed runs.
Codex recommends stopping unchanged training blocks because the five added epochs produced only a 0.0403-point mean-IoU gain.
Any feature-layer fine-tuning, local-data training, or TFLite export remains a separate step.
