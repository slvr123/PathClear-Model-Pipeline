# My frozen Cityscapes FOMO v2 development checkpoint

I froze this selected weight-10 candidate on 2026-10-04, after diagnosing `baseline_v1` and comparing controlled loss-weight and decoding changes.
I use the version identifier `cityscapes_v2` and the local annotated tag `fomo-cityscapes-v2` to identify this checkpoint.
I preserve the original model files byte for byte, without retraining or exporting them again.

## What I froze

```text
160 x 160 RGB image, bilinear direct resize
  -> MobileNetV2 alpha 0.35 through block_6_expand_relu
  -> 32-channel head and 20 x 20 x 3 softmax grid
  -> winning Person / Vehicle cells
  -> individual-cell decoder, no neighborhood suppression
  -> confidence threshold 0.70, uncapped detections
```

I use public detection IDs `Person=0` and `Vehicle=1`.
The model's internal channels are `background=0`, `person=1`, and `vehicle=2`.
I keep these detection classes separate from MicroSeg's scene/navigation classes.
FOMO and MicroSeg independently receive camera input; their outputs meet later in PathClear decision logic, optionally with VL53L1X ToF data.
I do not include a secondary classifier or a third CNN.

I normalize RGB bytes to float32 `[0,1]`; the model contains its own `2*x-1` rescaling.
The INT8 adapter quantizes using the saved input scale and zero point, then dequantizes and renormalizes the softmax output.
I preserve the full preprocessing and quantization contract in [config.json](config.json).
The model has 20,787 parameters and the INT8 file occupies **70,736 bytes**.
File size does not establish total firmware flash, peak RAM, or embedded FPS.

## Validation results and selection

| Candidate | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original `baseline_v1`, neighborhood suppression at 0.60 | 70.65% | 25.43% | 37.40% |
| Original weights, individual cells at 0.80 | 72.09% | 33.51% | 45.75% |
| Selected weight-10 v2, individual cells at 0.70 | **71.89%** | **38.15%** | **49.85%** |
| Weight-40 comparison, individual cells at 0.90 | 75.36% | 23.54% | 35.87% |

I selected the strongest tested balanced candidate using the same rule for each model: highest recall among tested thresholds whose overall precision reached 70%.
Decoding alone improved the original weights substantially; controlled weight-10 retraining improved the selected operating point further, with unchanged INT8 file size.
I did not select the 40 x 40 grid experiment because its tested configuration lost precision and F1 despite a small recall increase.
Its shallower feature extractor also changed, so I do not interpret it as a pure resolution comparison.
I retain that experiment and all previous artifacts in their existing locations.

I measured **Person recall 32.66%**, **Vehicle recall 41.64%**, and **recall for objects at least 8 input pixels across 62.81%**.
Object size here means the smaller bounding-box dimension after resizing into the 160 x 160 input space.
The selected model matched 3,860 objects, produced 1,509 scored false positives, missed 6,259 objects, and produced 1,072 ignored predictions.
The at-least-8-pixel group contains 2,194 objects, of which 1,378 were matched.
I use same-class, confidence-ordered, one-to-one centroid matching within eight input pixels.
Unmatched predictions in the original void/group cells are ignored; all labeled objects remain in the recall denominator.

These are Cityscapes development-validation results, not final PathClear real-world performance.
The model still misses many objects and must not yet be treated as reliable navigation perception.
I used this validation split for checkpoint and operating-point selection, so it is not an independent test.
ESP32-CAM compatibility, RAM, latency, FPS, and independently recorded walking-footage results remain unmeasured.
I reserve that independent footage evaluation for the next stage.

## Files and provenance

I copied `best.keras`, `best.json`, and `history.json` from `tinyml/fomo/artifacts/v2-weight10-training-20261004/`.
I copied both TFLite models and `report.json` from `tinyml/fomo/artifacts/v2-weight10-export-20261004/`.
I keep `best.json` and `report.json` beside the models because the current loader requires their tensor contracts.
Those original metadata files describe historical checkpoint selection/export using `cell_nms` at 0.60; the frozen operating policy is explicitly `cells` at 0.70 in `config.json` and the commands below.
I did not rewrite the historical metadata to conceal that distinction.
The float32 TFLite file is preserved for provenance; the existing CLI adapter accepts the full INT8 TFLite model.

I record every canonical file's origin and the original manifest hashes in [source_artifacts.json](source_artifacts.json).
[metrics.json](metrics.json) preserves exact counts and class/size results; [verification.json](verification.json) records the fresh check of the copied model.
[split_ids.json](split_ids.json) preserves all 2,975 training and 500 validation image IDs in their original order.
[environment.json](environment.json) records known software versions and explicitly marks missing build details.
[SHA256SUMS.txt](SHA256SUMS.txt) covers every canonical file except the checksum list itself.

I retain the full local artifact directories, while committing curated reports, visual error examples, prediction previews, histories, and comparisons in their existing locations.
I exclude raw Cityscapes images/masks, generated RGB caches and large manifests, temporary caches, and redundant experimental model copies from this commit.
The preserved conversion/refinement code, ordered split IDs, manifest fingerprints, review receipts, and commands allow data preparation to be repeated from a separately available Cityscapes installation.
Original absolute paths in copied historical metadata are historical evidence; `source_artifacts.json` provides repository-relative references.

## Running this frozen model

I run these PowerShell commands from the repository root with the packages pinned in `tinyml/fomo/requirements.txt`.
The example image is an existing Cityscapes validation image; I can replace `$imagePath` with another RGB image.

```powershell
$imagePath = 'C:/datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'
& .\.venv\Scripts\python.exe -B tinyml/fomo/inference.py `
  --model tinyml/fomo/checkpoints/cityscapes_v2/model.int8.tflite `
  --image $imagePath --decoder cells --threshold 0.70 --max-detections 0

& .\.venv\Scripts\python.exe -B tinyml/fomo/evaluate.py `
  --model tinyml/fomo/checkpoints/cityscapes_v2/model.int8.tflite `
  --manifest tinyml/fomo/artifacts/cityscapes-data-20261004/manifest.json `
  --split validation --decoder cells --threshold 0.70 --radius 8 --samples 0

& .\.venv\Scripts\python.exe -B tinyml/fomo/verify_checkpoint.py
& .\.venv\Scripts\python.exe -B -m unittest discover -s tinyml/fomo -p 'test_*.py'
```

The evaluation command creates a fresh artifact directory when `--output` is omitted.
The checkpoint verifier checks canonical hashes and settings without loading training data or writing files.

## Reproduction record for later experiments

I trained using the center-target manifest, object weight 10, seed 42, batch size 32, Adam at initial learning rate 0.001, horizontal flips, and a 20-epoch budget.
Training completed 19 epochs and selected epoch 14 by validation F1 with `cell_nms` at threshold 0.60.
The learning rate halved after two non-improving epochs, with a floor of 0.00001; early stopping used five non-improving epochs.
I calibrated the existing INT8 export on 100 training images; their exact IDs are retained in `report.json`.
I selected the final individual-cell threshold afterward using the recorded controlled comparison.

The original training command was:

```powershell
wsl -d Ubuntu -- bash ./run_wsl.sh -B -u tinyml/fomo/train.py `
  --manifest tinyml/fomo/artifacts/cityscapes-center-targets-20261004/manifest.json `
  --grid-size 20 --object-weight 10 --epochs 20 --batch-size 32 `
  --output tinyml/fomo/artifacts/v2-weight10-training-20261004
```

I record that command for provenance, not as permission to rerun training.
The original output directory already exists, so the scripts reject reusing it.
Future authorized reproduction must use a new artifact directory and retain these settings.
I can regenerate the data with `prepare_cityscapes.py --source C:/datasets/cityscapes`, then `refine_targets.py --manifest <new-original-manifest> --grid-size 20`.
Each creates a new directory; I inspect the previews and use `inspect_dataset.py --manifest <new-manifest> --review-note <actual-review-findings>` before training.
I check the resulting IDs against `split_ids.json` and preserve the official split.
Generated manifest bytes can differ because paths and source timestamps are machine-specific.
For a future export, `export.py --model <new-run>/best.keras --manifest <reviewed-center-manifest>` uses the recorded training-only calibration policy and a fresh output directory.
I do not promise bitwise-identical retraining across different hardware or numerical-library builds; the copied frozen model and its hashes are the exact reference.

## Immutability and Git identity

I treat this entire directory, its model weights, preprocessing, mappings, split, and selected operating point as immutable.
Training, conversion, and evaluation writers already reject output paths outside `tinyml/fomo/artifacts/` and refuse existing run directories.
Regression tests cover rejection of this frozen folder and preservation of an existing experiment.
This is an application-level guard and hash check, not filesystem write protection.
Future experiments may change code or settings only in separately named runs and checkpoints, without overwriting this directory or previous artifacts.
I can retrieve the original implementation through the annotated tag even after future source changes.

`config.json` records the Git commit reference `refs/tags/fomo-cityscapes-v2^{commit}`.
A commit cannot contain its own final hash without changing that hash, so I resolve the recorded reference after committing:

```powershell
git -c safe.directory=C:/Users/Asus/OneDrive/Desktop/Code/PathClear rev-parse 'fomo-cityscapes-v2^{commit}'
```

I also retain the resolved commit hash in the local post-commit receipt under `tinyml/fomo/artifacts/freeze-audit-20261004/`.
The tag is local and must not be moved or overwritten.

## Experiment reports

- [Baseline v1 and v2 investigation](../../BASELINE_V1_V2_REPORT.md)
- [20 x 20 versus 40 x 40 comparison](../../GRID_RESOLUTION_REPORT.md)
- [Controlled v2 experiment plan](../../V2_EXPERIMENT_PLAN.md)
- [Physical ESP32 benchmark protocol](../../ESP32_BENCHMARK_PROTOCOL.md)
- [Independent FOMO and MicroSeg architecture](../../STANDALONE_ARCHITECTURE.md)
- [Selected operating points](../../artifacts/v2-operating-points-20261004/selected.md)
- [Baseline visual error review](../../artifacts/baseline-v1-diagnostics-20261004/review/index.html)
