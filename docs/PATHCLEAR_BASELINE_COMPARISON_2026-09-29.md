# Original baseline and seven-class adaptation comparison

Date: 2026-09-29.
I compared the saved training histories and validation confusion counts without training either model or changing any checkpoint.
I distinguish each experiment's historical scores from a comparison using the same seven-category label mapping.

## Why my recent gains are small

I trained the original nineteen-class network from random initialization, allowing the feature layers and classifier to learn together.
For adaptation, I reused the epoch-19 features and froze them, training only the seven active output channels of the final 1x1 classifier.
Those active channels contain 903 weights and biases; the other three channels remain inactive.
My current experiment therefore adjusts a small classifier using a fixed representation of each image.
It cannot improve that representation while the feature layers remain frozen.

My newly initialized classifier started at 11.90% validation mean IoU and reached 67.63% after the first adaptation epoch.
It then improved to 68.56% by epoch 11.
Most of the initial classifier learning had already happened before the ten-epoch continuation.
The flattening curve is consistent with diminishing returns under the current configuration, although I have not proven an absolute performance ceiling.

## Historical validation scores

I read both `history.csv` files and convert their zero-based epoch indices to one-based numbering below.
The original model uses nineteen classes; adaptation uses seven active classes and excludes additional ground-truth regions.
I do not use this table to claim that a raw 68.56% adaptation score is directly better than the original 42.30% score.

| Stage | Epoch | Validation pixel accuracy | Validation mean IoU |
|---|---:|---:|---:|
| Original, 19 classes | 1 | 65.04% | 13.25% |
| Original, 19 classes | 5 | 84.75% | 32.51% |
| Original, 19 classes | 10 | 86.69% | 37.49% |
| Original, 19 classes | 11 | 86.06% | 36.26% |
| Original, 19 classes | 19, best | 88.67% | 42.30% |
| Original, 19 classes | 20, last | 88.02% | 41.41% |
| Adaptation, 7 active classes | 1 | 92.37% | 67.63% |
| Adaptation, 7 active classes | 5 | 92.58% | 68.42% |
| Adaptation, 7 active classes | 11, best | 92.62% | 68.56% |

I did not start the adaptation from an untrained visual feature extractor; its first epoch follows the learned features from nineteen original training epochs.
I therefore do not interpret eleven adaptation epochs as fewer total learning stages than the original experiment.
The original history also records a learning-rate reduction from 0.001 to 0.0005 beginning at epoch 18, while the adaptation stayed at 0.001.
I observed validation fluctuations in the original run and a drop after its best epoch, so further epochs do not guarantee better generalization.

![Historical validation curves with separate label sets](../output/comparisons/cityscapes_baseline_vs_adaptation_2026-09-29/historical_curves.png)

## Comparison with the same seven-category scoring

I remapped the saved epoch-19 baseline confusion counts into the seven supported categories.
I excluded ground-truth building, sky, and terrain, as the adaptation does.
When the old model predicted one of those excluded categories on a valid Road/Sidewalk/etc. pixel, I counted that prediction as an error rather than excluding the pixel.
I verified matching true-pixel support for every class: both models are scored on 169,243,720 valid pixels from the same 500-image validation split at 512x1024.

| Model | Pixel accuracy | Mean IoU | Macro Dice |
|---|---:|---:|---:|
| Original epoch 19, predictions remapped | 89.96% | 63.39% | 74.86% |
| Adaptation epoch 11 | 92.62% | 68.56% | 79.59% |

Under this protocol, the adaptation is higher by 2.66 percentage points in pixel accuracy and 5.17 points in mean IoU.
I remapped the old model's already-selected predicted class, not its probability distribution.
Summing probabilities across merged source classes could produce different predictions; I have not tested that alternative adapter here.
The adapted model also restricts predictions to the seven active categories, while the old model can predict categories outside them.
This is a comparison of the current adapted model against directly remapping the original predictions, not proof that the entire difference comes from training alone.
This validation split has been used for model selection, so I do not describe the comparison as an independent final-test result.

| Class | Original epoch 19, remapped IoU | Adaptation epoch 11 IoU |
|---|---:|---:|
| Road | 94.47% | 94.63% |
| Sidewalk | 62.83% | 64.50% |
| Person | 48.78% | 55.34% |
| Vehicle | 81.47% | 82.94% |
| Boundary | 34.56% | 36.54% |
| Obstacle | 31.94% | 54.33% |
| Vegetation | 89.66% | 91.63% |

## Whether I should continue unchanged

I measured a 0.93-point mean-IoU gain from adaptation epoch 1 to 11.
From epoch 5 to 11, the gain was only about 0.14 points; from epoch 10 to 11 it was about 0.0095 points.
These are percentage-point changes, not relative percentages.
Validation is still improving slightly, so I do not claim that the run is clearly overfitting or that further gains are impossible.
However, the evidence does not support expecting large gains from another unchanged block of classifier-only training.

Codex proposed keeping epoch 11 and, if I want further Cityscapes improvement before local data arrives, considering a separately scoped experiment that unfreezes selected later feature layers at a much lower learning rate while keeping BatchNorm inference behavior controlled.
I have not approved or started that experiment through this comparison request.
This follows the optional post-classifier fine-tuning stage described in the [Keras transfer-learning guide](https://keras.io/guides/transfer_learning/), with validation needed to establish whether it actually helps this model.
Local collection and reviewed labels remain necessary to teach Stairs, Crosswalk, and Uneven Pavement and measure Philippine performance.

## Evidence and reproduction

I used the original baseline history, its saved evaluation metadata and confusion CSV, the adaptation history, and the saved epoch-11 validation confusion matrix.
I verified the original and adapted checkpoint hashes against their evaluation records and matched the newly computed adaptation score to the recorded score.
I saved detailed numbers and the remapped original confusion matrix under `output/comparisons/cityscapes_baseline_vs_adaptation_2026-09-29/`.
I added `compare_pathclear_histories.py` to reproduce the calculation; it refuses to overwrite existing comparison outputs.

```powershell
wsl -d Ubuntu -- bash ./run_wsl.sh compare_pathclear_histories.py
```

I ran no new inference or training for this comparison.
