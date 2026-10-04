# MicroSeg custom CNN plan

## Purpose and status

This document describes the four-class, lightweight, dense-segmentation
CNN for PathClear. It is a separate experiment from the existing Fast-SCNN
pipeline and from the experimental FOMO centroid-detection direction.

**Status: smoke test and bounded pilots completed; no full experiment run.** The `microseg/`
package contains the four-class label taxonomy, model, Cityscapes and Kaggle
Crosswalk loaders, and a guarded trainer. The trainer saves a reproducible
video-grouped Crosswalk split, evaluates held-out real Crosswalk clips
separately from Cityscapes validation, checks finite metrics/weight updates,
and keeps checkpoints and logs under a non-overwriting run directory.
Quantization and deployment are not implemented.

The agreed purpose of the Kaggle import is to add supervision for the
Crosswalk class to the custom CNN. Its binary masks are not intended to label
the rest of each image as Background or to replace labels from the existing
training sources.

## Target behavior

- Perform dense semantic segmentation: produce a class prediction for every
  input pixel, not object centroids or bounding boxes.
- Start with a small `96 x 96 x 3` RGB input and a four-channel output at the
  same spatial resolution.
- Keep the model small enough to investigate INT8 deployment on ESP32-CAM.
- Preserve existing supervision for the non-Crosswalk classes when adding
  Crosswalk-only examples.
- Treat Kaggle mask background pixels as **ignored supervision**, not as
  class 0. Only marked Crosswalk pixels contribute a Kaggle loss.

This is a navigation perception component, not by itself a guarantee that a
route is safe or a complete mobility system.

## Four-class output taxonomy

Output IDs are zero-based and must stay consistent across source masks, model
channels, metrics, visualization, and any firmware integration.

| ID | Class | Intended source/supervision |
|---:|---|---|
| 0 | Walkable | Existing pedestrian-surface annotations; Cityscapes sidewalk (trainId 1) is a compatible source. |
| 1 | Non-walkable | Existing roadway annotations; Cityscapes road (trainId 0) is a compatible source. |
| 2 | Boundary | Existing boundary annotations; Cityscapes wall (3) and fence (4) are compatible sources. |
| 3 | Crosswalk | FPVCrosswalk2025 binary masks and any reviewed local annotations. |
| 255 | Ignore | Pixels excluded from loss and metrics. |

Person, Vehicle, Obstacle, Vegetation, Stairs, and Uneven Pavement map to the
ignored ID rather than a Background class. Background was dropped because it
had no supervision source; it can be reintroduced later as a separate taxonomy
change once local annotations exist.

The intended data plan is that existing Cityscapes data handles classes 0-2 and
the Kaggle dataset fills the Crosswalk gap (class 3). Before training, inspect
the actual prepared source masks and manifests to confirm that each class
claimed as supervised has positive examples. The Kaggle source covers class 3
only; it does not provide labels for the other three classes.

**Crosswalk conflict policy:** ignore Cityscapes pixels that are visibly part
of a crosswalk, while keeping the remaining Cityscapes road pixels as
Non-walkable (MicroSeg ID 1). Kaggle FPVCrosswalk2025 supplies positive
Crosswalk supervision (ID 3) to the same four-class model; its unmarked mask
pixels remain ignored. The combined trainer interleaves Cityscapes and Kaggle
batches rather than training a second model.

Cityscapes' `labelIds` and `trainIds` do not distinguish crosswalk markings
from the surrounding road. The loader now accepts separate reviewed,
image-aligned binary ignore masks and applies them after semantic remapping.
The agreed annotation covers the full visibly bounded crossing footprint,
including gaps between stripes, matching the intended Kaggle Crosswalk-region
supervision. The desktop workflow is documented in
[`MICROSEG_CITYSCAPES_ANNOTATION.md`](MICROSEG_CITYSCAPES_ANNOTATION.md).
The default directory convention is
`<cityscapes-root>/ignore_masks/{split}/{city}/{image-stem}_ignore.png`.
Masks use `0` for normal supervision and `1` or `255` for ignored pixels; their
native dimensions must match both image and semantic mask. Supplying
`--cityscapes-ignore-root` requires a mask for every train and validation
image. An explicit Cityscapes manifest can instead specify an `ignore_mask`
path per pair. The original Cityscapes masks remain unchanged.

No crosswalk-specific annotations have been prepared in the local Cityscapes
dataset yet. Do not ignore all road pixels as a shortcut, since that would
remove the Cityscapes source for Non-walkable supervision. Prepare and review
the masks before another training pilot.

## Candidate CNN architecture

The initial architecture is a compact encoder/fusion/head network inspired by
MobileNetV2 and Fast-SCNN building blocks:

```text
96 x 96 x 3 input
  -> Learning-to-downsample encoder (strides 2, 2, 2)
  -> 12 x 12 x 32 shortcut features
  -> Four inverted-residual blocks at 12 x 12 (expansion factor 4, 40 channels)
  -> Feature fusion with the shortcut (concatenate)
  -> Depthwise-separable refinement block (40 channels)
  -> 1 x 1 classifier convolution (four logits per low-resolution cell)
  -> Bilinear upsample to 96 x 96 x 4 logits
```

The candidate encoder uses a standard convolution followed by two
depthwise-separable downsampling blocks. The low-resolution feature extractor
uses four inverted-residual blocks with expansion factor 4. The fusion stage
combines encoder and context features before a depthwise-separable refinement
block. The classifier emits raw logits; softmax belongs in the loss or
post-processing, not in the model output.

Initial design targets from the prior architecture sketch:

- Input/output spatial size: `96 x 96`.
- Output channels: exactly four.
- Rough parameter estimate: approximately 60-80 thousand; the implemented
  model measures 65,124 parameters and keeps a structural regression test.
- Rough INT8 model-size estimate: approximately 70-90 KB.
- Rough peak activation estimate: approximately 100-130 KB.

These memory and size figures are estimates, not measurements or verified
ESP32-CAM guarantees. Validate actual parameter count, TFLite conversion,
operator support, tensor-arena use, latency, and camera-memory interaction on
the target hardware before making a deployment claim. Bilinear resize support
must likewise be confirmed in the chosen TFLite Micro runtime.

## Kaggle dataset

- Dataset: [FPVCrosswalk2025: First-Person View Crosswalk](https://www.kaggle.com/datasets/jocelyndumlao/fpvcrosswalk2025-first-person-view-crosswalk)
- Primary source page: [Mendeley Data, dataset mcr2jwk5bp](https://data.mendeley.com/datasets/mcr2jwk5bp/1)
- Kaggle listing license: **CC0: Public Domain**. Recheck the terms and
  attribution notes when downloading the exact dataset version.
- Description reports 3,000 synthetic and 300 real first-person-view images,
  each paired with a binary segmentation mask. The dataset is organized by
  source and environmental conditions; the listing describes weather and
  lighting variation.
- The annotations mark crosswalk regions as quadrilaterals.
- The extracted archive was inspected at `C:\datasets` on 2026-10-04:
  `Real-world dataset` and `Synthetic dataset` each contain condition folders
  with parallel `images` and `masks` subfolders. RGB files are `.jpg`; masks
  have matching stems with `_mask.jpg`. There are 300 real pairs and 3,000
  synthetic pairs by folder counts. `Fine-tuning seed images` contains 131
  unpaired images and is excluded from supervised segmentation training.
- The implemented importer found all 3,300 image/mask pairs and intentionally
  excluded the 131 unpaired seed images. A TensorFlow decode check loaded
  five samples across the sorted collection as 96 x 96 RGB with aligned
  640 x 360 source image/mask dimensions. Their JPEG masks included values
  from the low range (0-8) and high range (248-255); the existing `>127`
  foreground threshold produced both Crosswalk (ID 3) and zero-weight ignored
  pixels for each sampled mask. Visually inspect representative overlays
  before a training run; five samples do not prove every mask is correctly
  oriented or aligned.
- Cityscapes in `C:\datasets\cityscapes` contains 2,975 training and 500
  validation RGB images with matching `*_gtFine_labelIds.png` masks, but no
  `*_gtFine_labelTrainIds.png` masks. MicroSeg now accepts both formats
  directly without rewriting source annotations. Original labelIds 7/8 map to
  Non-walkable/Walkable, 12/13 map to Boundary, and all other labels are
  ignored. Format detection runs on the TensorFlow path tensor so the
  `tf.data` pipeline selects the same mapping as direct loader calls. A real
  train batch and validation batch were decoded; the first train mask at
  96 x 96 contained 120 Walkable and 3,690 Non-walkable pixels, while the
  first validation mask contained 749 Walkable, 2,763 Non-walkable, and 10
  Boundary pixels. Crosswalk is not provided by Cityscapes.

Suggested local root:

```text
C:\datasets\crosswalks\
```

Use `KAGGLE_CROSSWALK_ROOT` to select a different extraction directory. In
WSL, the default is `/mnt/c/datasets/crosswalks`; set the variable before
launching the script if you extract elsewhere. `.env.example` documents the
setting but is not automatically loaded by Python. Keep the downloaded archive
and extracted images outside version control.

## Importer and mask contract

The importer recursively supports common paired-file layouts: mirrored
image/mask directories, nested source/condition folders, and same-directory
mask suffixes. The archive's exact pairing convention must still be verified
after extraction. Current pairing behavior:

1. Recursively find image and mask files in the extracted root.
2. Pair each image with its corresponding mask using relative path/stem
   information, including separate mirrored image/mask directories and nested
   source/condition folders if used by the archive.
3. Support the actual image and mask formats discovered in the archive.
4. Report total images, valid pairs, unpaired files, and ambiguous matches.
   Stop with a useful error instead of silently dropping mismatches.
5. Preserve a deterministic pair ordering before training shuffles examples.

After the format is observed, add explicit archive-specific pairing rules if
the current conventions do not match. Verify a sample of pairs visually before
building the training dataset.

Mask transformation for the Crosswalk source:

| Source mask pixel | Target ID | Sample weight |
|---|---:|---:|
| Greater than 127 | 3 (Crosswalk) | 1 |
| 127 or less | 255 (Ignore) | 0 |

The threshold tolerates lossy/binary mask encodings. Confirm the mask
foreground polarity and value range from real files before relying on it. Use
nearest-neighbor resize for masks and bilinear resize for RGB images. Resize
images and masks with exactly the same geometry.

The negative/unmarked region in a Kaggle mask is **not** relabeled as
Background, Walkable, or Non-walkable. Its zero sample weight prevents it from
changing the loss and metrics. This lets the dataset supply Crosswalk
supervision without overwriting labels or making false claims about the
surrounding pixels.

The checked extraction contains 300 labeled real images across cloudy, night,
rainy, and sunny conditions, plus 3,000 labeled synthetic images. The trainer
holds out 20% of real source-video groups per condition for Crosswalk
validation; all frames from one video stay in the same split. Synthetic
examples remain training-only. The split manifest is saved with each run.

## Combining sources and training

Build a unified training example as:

```text
(96 x 96 x 3 float RGB image, 96 x 96 integer class mask,
 96 x 96 float sample-weight map)
```

Normalize image values to `[0, 1]`. Targets must contain only IDs 0-3 or
ignore ID 255; replace ignored targets with a safe in-range placeholder only
when paired with exactly zero sample weight.

Train with sparse categorical cross-entropy from logits and pixel sample
weights. Use weighted metrics so ignored pixels do not count in accuracy,
IoU, or Dice. Include tests that prove ignored pixels have no effect on loss
or metrics.

`microseg.dataset.make_combined_dataset` interleaves Cityscapes and Kaggle
batches for the training split; the default selects 35% Crosswalk batches
(`MICROSEG_CROSSWALK_WEIGHT`) and the remainder from Cityscapes. The trainer
can sample real and synthetic Crosswalk streams separately, defaulting to an
equal source ratio. Both sampling ratios are configurable and recorded with
each run. Validation remains Cityscapes-only for multiclass metrics, with
held-out real Crosswalk positive-pixel recall reported separately. Treat
sampling proportions as experiment settings, not a settled optimum.

### Cross-source normalization and audited label balance

The first five-epoch mixed-source pilot predicted Non-walkable for every
validation pixel. This was not a label-mapping failure: a full pass through the
same 96 x 96 loader used for training confirmed that all four classes have the
intended support across the two sources. Cityscapes train contains 13.81%
Walkable, 82.71% Non-walkable, and 3.47% Boundary among supervised pixels;
KPV train contains 477,275 real and 8,909,992 synthetic supervised Crosswalk
pixels. KPV unmarked pixels have zero weight and are not mislabeled as another
class.

The decisive diagnostic was to evaluate the trained checkpoint with
normalization in inference mode versus per-batch training mode. In inference
mode, both Cityscapes and held-out KPV batches predicted 100% Non-walkable.
Using current batch statistics instead produced all four predictions on both
domains (32 Cityscapes images: 10.8% Walkable, 47.7% Non-walkable, 9.1%
Boundary, 32.4% Crosswalk; 32 KPV images: 11.7%, 27.3%, 12.0%, and 49.0%,
respectively). The Fast-SCNN helper blocks used BatchNormalization, whose
moving statistics were shared while training alternated between visually
different Cityscapes and KPV domains. The observed train/inference mismatch
therefore implicates cross-source BatchNormalization statistics as a primary
cause of the all-Non-walkable inference collapse.

MicroSeg now uses GroupNormalization with eight channel groups instead of
BatchNormalization, avoiding source-dependent running statistics. Fast-SCNN
continues to default to BatchNormalization and is unchanged in behavior. This
fix requires training a new MicroSeg checkpoint from scratch; old checkpoints
retain the old architecture and normalization. The source mix remains
imbalanced within Cityscapes (Boundary is only 3.47% of supervised train
pixels), so class-weighting may still be needed.

A fresh GroupNormalization pilot was run from scratch for five epochs, 50 steps
per epoch, with batch size 8. It used 2,975 Cityscapes train pairs, 500
Cityscapes validation pairs, and a leakage-safe Crosswalk split of 3,242 train
pairs (242 real and 3,000 synthetic) plus 58 held-out real images. The run used
the CPU because TensorFlow could not initialize CUDA in the WSL environment.
The saved best checkpoint predicted multiple classes in inference mode on both
validation domains, so the original all-Non-walkable inference collapse did not
recur:

- Cityscapes validation prediction IoU was 0.227 Walkable, 0.854 Non-walkable,
  and 0.172 Boundary. Crosswalk has no Cityscapes ground truth; its IoU is
  unsupported there. The reported four-class mean IoU was 0.313.
- On the 110,004 supervised positive pixels in held-out real Crosswalk masks,
  76.0% were predicted as Crosswalk. This equals positive-pixel recall, not
  precision: KPV unmarked pixels are ignored, so this validation cannot measure
  Crosswalk false positives.
- Held-out Crosswalk recall varied across epochs: 86.5%, 89.8%, 92.6%, 87.2%,
  then 76.0%. Cityscapes mean IoU improved each epoch, reaching 0.313 in epoch
  5, so selecting the checkpoint by Cityscapes mean IoU does not optimize
  Crosswalk recall.

This bounded result supports GroupNormalization as a remedy for the observed
normalization-related collapse, but it is not evidence of deployment-ready
quality. In particular, Boundary IoU remains low, and positive-only Crosswalk
evaluation cannot quantify false alarms. Do not begin a longer training run
until the intended Boundary semantics and a validation approach that covers
Crosswalk negatives are settled.

The prior candidate plan starts from random initialization. If transfer
learning is considered instead, specify and test compatible layers and
initialization separately; do not imply the existing Fast-SCNN checkpoint
loads into a different architecture.

The trainer entry point is `python -m microseg.train`. Its `--smoke-test`
option trains on exactly one Cityscapes batch and one Crosswalk batch, then
checks Cityscapes validation plus one held-out real Crosswalk batch. It asserts
that weights changed during training and did not change during validation.
This smoke test completed on the WSL GTX 1050 Ti using TensorFlow 2.21.0. The
training loop completed two steps, both validation paths completed, and the
model weights changed. The initial held-out Crosswalk metrics were zero from
random initialization; smoke-test scores are not performance results. Keep
the cached pair manifests and run output outside the repository, preferably on
WSL's Linux filesystem to avoid slow repeated Windows-mount metadata lookups.
For example, from WSL:

```bash
bash run_wsl.sh -m microseg.train \
  --smoke-test --batch-size 1 \
  --split-manifest /path/to/crosswalk_split.json \
  --cityscapes-manifest /path/to/cityscapes_pairs.json \
  --output-dir /path/to/microseg_runs/smoke
```

Five bounded pilots completed, each with 5 epochs, 50 steps per epoch, and
batch size 8. They saved separate checkpoints, histories, and split manifests
outside the repository. The original 0.35 Crosswalk batch rate, including a
50/50 real/synthetic source-balanced run, left held-out real Crosswalk Dice,
IoU, and pixel accuracy at zero. Raising Crosswalk batch sampling to 0.65
alone also failed. Combining 0.65 Crosswalk sampling with 50/50 real/synthetic
sampling caused unstable class collapse: validation briefly predicted
Crosswalk everywhere, then reverted to Non-walkable. A separate 0.50/50-50
pilot again ended at Cityscapes mean IoU 0.2811 and zero Crosswalk metrics.

The mixed-data probe confirmed that the original flat Crosswalk stream yielded
16 Crosswalk batches in a 50-batch sample, with Crosswalk pixels representing
27.2% of supervised pixels. Of 3,242 Crosswalk training pairs, only 242 are
real and 3,000 synthetic, so the flat stream under-sampled real images. The
trainer now supports independent real/synthetic streams and explicit
`--crosswalk-weight` and `--crosswalk-real-weight` controls. However, the
short source-balanced pilots still did not produce stable class behavior.
Crosswalk validation labels only positive pixels; its metrics therefore
measure recall on marked pixels and cannot measure false positives on ignored
regions. Do not start a long run yet; first review class predictions and
develop a loss/sampling strategy that preserves Cityscapes classes while
improving Crosswalk recall.

The smoke test is not a full training experiment and its scores are not model
quality claims.

Recommended safeguards for the trainer:

- Set random seeds and record source versions, split manifests, and run
  configuration.
- Save best and latest checkpoints, history, and resumable state under a
  dedicated output directory.
- Refuse accidental overwrite of a non-empty run directory.
- Stop on non-finite loss or metrics.
- Support a short smoke run that checks a real batch, output shape, finite
  loss, a training weight update, and unchanged model weights during
  validation.
- Keep smoke-test scores separate from full validation and final test scores.

## Splits, validation, and reporting

Kaggle describes source and condition groupings, but this plan does not assume
that the download already provides a suitable train/validation/test split.
Create and save an explicit split manifest after inspecting the archive.

- Preserve a held-out real-image evaluation subset; do not report synthetic
  validation results as real-world performance.
- Keep images from the same source scene, recording, or near-duplicate group
  together in a single split.
- Avoid choosing thresholds or training settings based on the final test set.
- Report real and synthetic results separately and, where data permits, break
  results down by weather/lighting condition.
- Report per-class IoU, Dice, support, and confusion counts, especially
  Crosswalk support. Also inspect overlays and boundary alignment.
- Record the exact resize procedure and image/mask dimensions.
- Compare against a clearly identified baseline trained/evaluated on the
  same held-out data where feasible.

Because ignored pixels do not participate in evaluation, show valid-pixel
coverage alongside metrics. A high metric on a small valid area is not enough
to claim reliable segmentation.

## Implementation sequence and acceptance checks

1. **Inspect the archive.** Record its directory tree, image/mask extensions,
   counts, mask dimensions, pixel-value distribution, and several aligned
   image/mask previews.
2. **Implement pairing.** Add tests for the observed layout, duplicate
   basenames in distinct conditions, missing masks, and ambiguous matches.
3. **Verify preprocessing.** Assert image/mask spatial alignment after resize,
   valid target IDs, ignored-pixel zero weights, and Crosswalk-positive
   examples in the resulting batches.
4. **Implement the model.** Test input/output shapes, finite forward pass,
   five output channels, and measured parameter count.
5. **Implement training.** Test weighted loss/metrics and the smoke-run
   safeguards before a longer run.
6. **Run a controlled experiment.** Save code revision, seeds, split
   manifests, source sampling proportions, checkpoints, logs, and per-class
   results.
7. **Evaluate deployment feasibility.** Convert to the target TFLite format,
   verify operator compatibility, measure quantized output quality, then
   benchmark RAM and latency on the physical board.

No training or deployment is considered verified until these checks are
actually run and their results are recorded.

## Scope boundaries

- This plan is for four-class per-pixel segmentation, not the existing
  19-class Fast-SCNN baseline, the ten-class PathClear taxonomy, or FOMO
  centroid detection.
- It does not change the existing Fast-SCNN training pipeline or checkpoint.
- It does not treat Crosswalk as safe passage by itself.
- It does not infer depth, obstacle distance, collision timing, or sensor
  reliability from image segmentation.
- It does not claim embedded performance from parameter-count or desktop
  estimates.

## Current repository state

On the `custom-cnn` branch, the repository contains the Fast-SCNN PathClear
pipeline plus the `microseg/` four-class taxonomy, model, 96 x 96 Cityscapes
loader, and Kaggle Crosswalk loader. Kaggle pairing and sampled decoding are
verified for all 3,300 labeled pairs. Cityscapes has 2,975 training and 500
validation image/mask pairs using original `labelIds`; MicroSeg remaps these
in memory and continues to prefer `labelTrainIds` if both formats are
available. Loader tests cover both label formats, including automatic
detection inside `tf.data`; real train and validation batches have been read
and their weighted class counts match the independently remapped mask values.
The trainer's WSL smoke test and five initial sampling pilots are recorded
above. A subsequent five-epoch GroupNormalization pilot is documented in the
normalization section: inference no longer collapses to Non-walkable, but KPV
false positives remain unevaluable and Boundary IoU is low. Cityscapes
ignore-mask loading and manifest support have focused regression tests; the
candidate-selection/annotation review queue and handoff are described in
[`MICROSEG_ONBOARDING.md`](MICROSEG_ONBOARDING.md). Do not start a longer run
until Boundary semantics and Crosswalk-negative evaluation are settled. INT8
export and firmware remain unimplemented. Keep generated datasets,
checkpoints, and caches outside Git.
