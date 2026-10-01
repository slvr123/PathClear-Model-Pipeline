# PathClear ten-class preparation

Date: 2026-09-29.
Status: preparation and sanity checks complete; local data collection and training remain pending.
Historical scope: this report describes the preparation stage before any adaptation training.
Later on 2026-09-29, I authorized a separate seven-active-class Cityscapes experiment, documented in [the training guide](../TRAINING.md).
I subsequently completed that bounded experiment and recorded it in [the adaptation report](PATHCLEAR_CITYSCAPES_STAGE_2026-09-29.md).
Codex proposed preparing the ten-class pipeline while waiting for reviewed local data; I confirmed that sequence on 2026-09-29.
I built a temporary ten-class model in memory to verify transfer and inference, then discarded it without saving a model checkpoint.
I did not train, fine-tune, export TFLite, or change the baseline experiment.

## My two model stages

| Stage | Purpose | Current state |
|---|---|---|
| Cityscapes baseline | I learn general urban visual features and retain a thesis reference using the original 19 trainIds. | I retain `checkpoints/baseline-5epochs/best.keras`, selected at epoch 19 after 20 completed epochs. |
| PathClear model | I adapt those features to ten navigation-oriented classes with local supervision. | I have tested the preparation code; I have no trained or saved ten-class model yet. |

My baseline is a full Keras model with model weights, compile configuration, and optimizer state, rather than a weights-only file.
Its sibling `state.json` and `history.csv` record training progress and scheduler/checkpoint-selection information.
For transfer, I load it with `compile=False` and reuse only compatible model weights.
I do not transfer optimizer momentum or epoch counters to the changed classification task.

I verified that the baseline file hash before and after the check is:

```text
47a248f9c3ee5f38194eb7071c035b4bc1c3c01254bec8bb44bbc240409321d7
```

I keep the original file at `checkpoints/baseline-5epochs/best.keras` without renaming or modifying it.
I store preparation artifacts under `checkpoints/pathclear_10class/sanity/`.
That folder contains a JSON report and mask/preview PNGs only, not a model checkpoint.

## What I inspected

| Existing file | Finding |
|---|---|
| `fast_scnn.py` | I found learning-to-downsample, global feature extraction, feature fusion, and a classifier with two depthwise-separable blocks, dropout, a 1x1 logits convolution, and bilinear output resizing. |
| `fast_scnn.py` | I found `NUM_CLASSES = 19` as the default, with `build_fast_scnn(num_classes=...)` already configurable; output geometry remains fixed at 512x1024. |
| `dataset_config.py` | I found the external Cityscapes root under `C:/datasets/cityscapes` and its WSL equivalent. |
| `dataset_loader.py` | I found trainId PNG pairing, bilinear RGB resizing, nearest-neighbor mask resizing, division by 255, and source void pixels replaced by target 0 with zero sample weights. |
| `train.py` | I found 19-class training, validation through `fit`, weighted metrics, full-model checkpoints, and explicit resume state handling. |
| `segmentation_metrics.py` | I found sample-weighted pixel accuracy, global-confusion-matrix MeanIoU, and registered MacroDice. |
| `evaluate_baseline.py` | I found the frozen 19-class evaluation path, explicit 19-class metric arrays, and checkpoint-preservation checks. |
| `infer_ph_samples.py` | I found qualitative 19-class inference with fixed preprocessing and no local ground truth. |
| `test_pipeline.py`, `test_training_metrics.py` | I found the baseline shape check and weighted-metric tests. |
| `run_wsl.sh`, `TRAINING.md` | I confirmed the existing WSL environment and baseline commands. |
| `LOCAL_DATASET_PLAN.md`, `OUTDOOR_MOBILITY_ROADMAP.md`, `docs/PATHCLEAR_HANDOFF_2026-09-22.md` | I retained the pilot, route/session split, frozen baseline, and pending preprocessing decisions. |
| Baseline `state.json`, `reload_epoch20.json`, and `best.keras` | I confirmed 20 completed epochs, best epoch 19, and the existing model artifact. |

I left the architecture, baseline loader, training, metrics, inference, evaluation, and launcher files unchanged.
Their 19-class assumptions remain valid for the original experiment.

## My taxonomy and mapping

| PathClear ID | Class | Cityscapes trainIds mapped here |
|---|---|---|
| 0 | Road | 0 road |
| 1 | Sidewalk | 1 sidewalk |
| 2 | Person | 11 person, 12 rider |
| 3 | Vehicle | 13 car, 14 truck, 15 bus, 16 train, 17 motorcycle, 18 bicycle |
| 4 | Boundary | 3 wall, 4 fence |
| 5 | Obstacle | 5 pole, 6 traffic light, 7 traffic sign |
| 6 | Vegetation | 8 vegetation |
| 7 | Stairs | None |
| 8 | Crosswalk | None |
| 9 | Uneven Pavement | None |
| 255 | Ignore | 2 building, 9 terrain, 10 sky, source void and unknown IDs |

I apply this lookup to trainIds, not the original Cityscapes labelIds.
I never edit source annotation PNGs.
I restore the baseline loader's zero-weight void pixels to 255 before mapping, so they cannot become valid Road pixels.
I keep raw remapped masks in 0-6/255, then provide safe targets and zero weights together to Keras.
I preserve local IDs 7-9 when checking local masks; I never fabricate them from Cityscapes.

## My transfer method

I build the existing architecture with ten outputs and match each layer by name and type.
I reject unexpected layer-set, tensor-count, or tensor-shape differences instead of silently skipping them.
I transfer 86 layers containing 214 weight tensors and 1,158,256 scalar weight values, including BatchNorm moving statistics and the classifier's feature-processing blocks.
I verify every transferred array with exact numerical equality against the loaded baseline.
I list every transferred layer and its tensor count in `sanity_report.json`.
Layers without weights are listed separately; they are not failed transfers.

I reinitialize only `classifier_logits`:

- I replace its kernel shape `(1, 1, 128, 19)` with `(1, 1, 128, 10)`.
- I replace its bias shape `(19,)` with `(10,)`.

I considered averaging old classifier weights for merged classes and chose a fresh output layer.
A merged probability corresponds to a sum of source probabilities; averaging linear weights does not reproduce that operation or its calibration.
Directly copying some channels while inventing the others would also leave an untrained mixed classifier.
I therefore retain all compatible features and leave classifier learning for supervised local data.
I freeze reused layers, including BatchNorm, in the prepared model as a starting configuration for a future classifier-training stage.
I have not selected or run the later backbone fine-tuning schedule.

## My ignore-label and metric checks

I turn mask ID 255 into target 0 only alongside an exactly zero pixel weight.
I use `SparseCategoricalCrossentropy(from_logits=True, reduction="mean_with_sample_weight")` for the new pipeline, normalizing over valid pixel weights.
I verified that ignored predictions do not change the loss, ignored pixels have zero gradient, and an all-ignored batch produces finite zero loss and zero gradients.
The historical baseline uses a different denominator that includes ignored image positions; I leave that historical behavior unchanged and do not compare the two loss numbers directly.

I reuse the existing metrics with `num_classes=10` and register them through `weighted_metrics`, so zero-weight pixels do not enter accuracy or confusion counts.
IoU and Dice include a class when it has ground-truth or predicted support; false positives for an absent class still count as errors.
I do not describe Cityscapes scores for classes 7-9 as a measurement of their recognition ability because Cityscapes supplies no positive examples for them.
I treat the temporary model's test-batch metric values only as evidence that evaluation executes, not as PathClear performance results.

## Checks I executed

I ran these commands from the project directory through Ubuntu WSL:

```powershell
wsl -d Ubuntu -- bash ./run_wsl.sh -m unittest -v test_pathclear_pipeline test_training_metrics
wsl -d Ubuntu -- bash ./run_wsl.sh -u check_pathclear_pipeline.py
wsl -d Ubuntu -- bash ./run_wsl.sh validate_pathclear_pilot.py datasets/pathclear_pilot/manifest.json
```

| Check | Result |
|---|---|
| Synthetic mapping, loss, metrics, serialization, pilot validation, and existing baseline metric tests | I passed all 18 tests. |
| All Cityscapes trainIds, source void, and unknown integer IDs | I matched the explicit mapping and produced no IDs 7-9. |
| Real Cityscapes sample | I loaded `frankfurt_000000_000294` through the TensorFlow dataset path. |
| Real forward pass | I obtained `(1, 512, 1024, 10)` with finite logits. |
| Native and resized remapped masks | I observed exactly `[0, 1, 2, 3, 4, 5, 6, 255]`. |
| Independent remapping comparison | I matched the NumPy raw-mask remap with the TensorFlow loader pixel for pixel after nearest-neighbor resize. |
| Transfer | I verified exact equality for all 214 transferred weight tensors. |
| Evaluation preservation | I verified that inference/evaluation changed no model weights and the new optimizer remained at iteration zero. |
| Source preservation | I verified unchanged hashes for the baseline, source image, and source annotation. |
| Preview | I visually inspected the original image, remapped ground truth, overlay, and readable legend; mapped road, sidewalk, vehicles, people, vegetation, and poles align with source regions. |
| Empty local manifest | I received zero samples, missing class coverage, and a false readiness flag as expected. |

I corrected one test-fixture issue during verification: Keras accuracy required a tensor rather than the raw TensorFlow variable used for the gradient test.
The final run passed without changing the existing baseline metric implementation.

![Inspected mapping preview](../checkpoints/pathclear_10class/sanity/mapping_preview.png)

## Files I changed or created

I updated only these existing documentation files:

- `LOCAL_DATASET_PLAN.md`
- `OUTDOOR_MOBILITY_ROADMAP.md`
- `TRAINING.md`

I created these source and documentation files:

- `pathclear_taxonomy.py`: class IDs, names, colors, and deterministic remapping.
- `pathclear_pipeline.py`: safe targets, Cityscapes adapter, new loss configuration, and strict transfer.
- `validate_pathclear_pilot.py`: native-mask, provenance-field, duplicate-file, split, review, and coverage checks.
- `check_pathclear_pipeline.py`: one-image inference/evaluation and preservation checks without model saving.
- `test_pathclear_pipeline.py`: 16 new tests; the two existing metric tests also ran.
- `datasets/pathclear_pilot/manifest.json`: empty collection template.
- `docs/PATHCLEAR_ANNOTATION_GUIDE.md`: draft boundaries, label precedence, native mask format, and review workflow.
- `docs/PATHCLEAR_10CLASS_PREPARATION.md`: this report.

I generated `sanity_report.json`, `remapped_native_ids.png`, `remapped_model_ids.png`, and `mapping_preview.png` under `checkpoints/pathclear_10class/sanity/`.
The sanity script refuses to overwrite an existing nonempty output folder; I use `--output-dir` with a new folder if another run is needed.
This workspace has no Git repository metadata, so I did not create a commit or claim a Git diff verification.

## What I still need

I need actual local clips/images and reviewed masks, especially multiple independent examples of Stairs, Crosswalk, and Uneven Pavement.
I will check the draft annotation rules on roughly 20-30 diverse frames before scaling collection and labeling.
I will resolve the pending aspect-ratio comparison before implementing the local training loader; today's real-image check retains the established Cityscapes geometry.
I have no local-data training command or saved ten-class model yet.
I will select a bounded experiment and training configuration separately once the local pilot is usable.
I will use an explicit shared-class evaluation protocol when comparing the 19-class baseline and ten-class model; I will not compare their overall mIoU as if their label spaces were identical.
I will not count ignored terrain, drains, water, animals, or other uncovered hazards as evidence that a route is clear.
