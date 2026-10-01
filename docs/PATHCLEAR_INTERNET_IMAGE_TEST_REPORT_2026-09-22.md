# PathClear: five internet images - baseline inference test report

Run date: 2026-09-21.
Report date: 2026-09-22.

## Purpose

I tested the unchanged epoch-19 Fast-SCNN baseline on five supplied internet photographs described as Philippine street scenes.
This was a qualitative inspection of all five inputs, not a labeled benchmark or a live moving-pedestrian test.

## Method

I used best.keras with 19 Cityscapes output classes, batch size 1, bilinear resizing to 512 x 1024, and RGB normalization by 255.
Predictions were restored to original image dimensions with nearest-neighbor interpolation for display.
Inference used training=False.

## Verified integrity

The recorded run left model weights, optimizer variables, and the checkpoint hash unchanged.
Optimizer updates stayed at 56,525.
The complete inference-and-artifact script took 32.84 seconds; this is not a live camera-to-warning latency benchmark.

## Accuracy limitation

These images have no ground-truth segmentation masks, so I cannot report local Pixel Accuracy, IoU, Dice, or loss.
The 88.67% Pixel Accuracy, 42.30% mIoU, and 54.03% Macro Dice are from the separate 500-image Cityscapes validation evaluation.
Predicted class area and confidence are not accuracy.

## Overall interpretation

I observed plausible large road, sky, building, or vegetation regions, but substantial errors in sidewalks, pedestrians, boundaries, and clutter.
These outputs do not support using the current model alone to guide pedestrian movement.

## Limits and next step

The sample is only five unlabeled photographs, and their locations/source URLs were not independently verified.
Direct resizing distorts their aspect ratios.
I will compare aspect-ratio-preserving inference before attributing all failures to unfamiliar local scenes, then collect and label a local pilot.
No retraining or new preprocessing comparison was performed for this report.

## Sample 1: Busy urban street

Original size: 3648 x 2736 pixels.
Buildings, roadside trees, narrow sidewalks, pedestrians, cars, motorcycles, and pedal-powered vehicles appear in the scene.

![Sample 1: original, prediction and overlay](../checkpoints/baseline-5epochs/ph_samples_2026-09-21/sample_01_comparison.png)

### What appears to work

I can recognize broad road, building, sky, and tree regions in the output.

### Main visible errors

I see fragmented pedestrians and vehicles, extensive false person/rider/vehicle labels across storefronts, and poor sidewalk separation.
Some sign predictions spread onto unrelated surfaces.

### Interpretation for PathClear

The model captures the broad scene layout but does not reliably preserve the people and walking-space boundaries needed for obstacle awareness.
Recognizing the road is not enough to identify an unobstructed pedestrian route.

## Sample 2: Street beside a stone wall

Original size: 1400 x 1400 pixels.
A large stone wall borders a very narrow sidewalk, with cars on the adjacent road and trees overhead.

![Sample 2: original, prediction and overlay](../checkpoints/baseline-5epochs/ph_samples_2026-09-21/sample_02_comparison.png)

### What appears to work

Much of the road and sky is coherent, and broad foliage regions are recognizable.

### Main visible errors

Large portions of the stone wall become vegetation or rider.
The narrow sidewalk is not consistently retained, and the visible cars contain mixed labels.

### Interpretation for PathClear

The wall errors are a strong example of false object predictions.
The model does not reliably distinguish the narrow walking strip from surrounding surfaces in this photograph.

## Sample 3: Curved neighborhood street

Original size: 2048 x 1536 pixels.
A curved asphalt road passes pedestrians, gates, posters, corrugated structures, and an uneven raised pavement edge.

![Sample 3: original, prediction and overlay](../checkpoints/baseline-5epochs/ph_samples_2026-09-21/sample_03_comparison.png)

### What appears to work

Most foreground asphalt is labeled road, and part of the central pedestrian receives the person label.

### Main visible errors

The right pavement merges with road and other classes.
Gates, structures, and posters receive extensive traffic-sign or vegetation labels.
Pedestrian outlines are incomplete.

### Interpretation for PathClear

The dominant road region makes the prediction appear more successful than the smaller regions warrant.
Pavement boundaries and pedestrian completeness remain weak.
Cityscapes labels also do not describe surface condition or every local hazard.

## Sample 4: Tree-lined pedestrian sidewalk

Original size: 600 x 900 pixels.
A paved sidewalk runs between trees and a fence, alongside a separate roadway. This is the portrait-format sample.

![Sample 4: original, prediction and overlay](../checkpoints/baseline-5epochs/ph_samples_2026-09-21/sample_04_comparison.png)

### What appears to work

Much of the sky and dense foliage is recognizable.

### Main visible errors

The main sidewalk is labeled road.
Tree trunks contain false person/rider labels, and parts of foliage and the fence receive unrelated labels, including traffic sign.

### Interpretation for PathClear

This is a particularly relevant failure for PathClear because the main walking surface is misclassified.
Portrait-to-2:1 resizing introduces the greatest aspect-ratio distortion among these samples, so I cannot isolate the cause without a preprocessing comparison.

## Sample 5: Commercial sidewalk with a stall

Original size: 1024 x 768 pixels.
A sidewalk contains a vendor stall, utility poles, storefronts, pedestrians, parked or roadside vehicles, and wet pavement.

![Sample 5: original, prediction and overlay](../checkpoints/baseline-5epochs/ph_samples_2026-09-21/sample_05_comparison.png)

### What appears to work

Much of the sky is recognizable, and some large background surfaces remain distinct.

### Main visible errors

The main sidewalk becomes road.
The stall and sections of the building receive vehicle labels.
Poles, signs, pedestrians, and pavement boundaries are fragmented or confused.

### Interpretation for PathClear

The output is unreliable for distinguishing walking space from obstacles in this scene.
The 19-class model has no dedicated vendor-stall or puddle class, so those hazards cannot be inferred merely from the presence of a colored segmentation region.

## Artifacts and reproducibility

Source run: `checkpoints/baseline-5epochs/ph_samples_2026-09-21/`.
The folder contains all five original photographs, raw train-ID masks, colored masks, overlays, comparisons, `gallery.md`, and `inference_manifest.json`.
Inference helper: `infer_ph_samples.py`.
Frozen checkpoint: `checkpoints/baseline-5epochs/best.keras`.
Checkpoint SHA-256: `47a248f9c3ee5f38194eb7071c035b4bc1c3c01254bec8bb44bbc240409321d7`.
This report reuses the completed inference outputs; it does not run inference again or change the model.
