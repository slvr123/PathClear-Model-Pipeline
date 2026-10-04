# MicroSeg onboarding: current state

This page is the handoff point for continuing the MicroSeg experiment. The
work is on the `custom-cnn` branch. Start with the [full experiment plan](./MICROSEG_CUSTOM_CNN_PLAN.md)
for taxonomy, data provenance, and implementation details.

## What is in place

- A four-class, 96 x 96 segmentation model: Walkable (0), Non-walkable (1),
  Boundary (2), Crosswalk (3). ID 255 is Ignore, not a model output.
- Cityscapes supervises road, sidewalk, wall, and fence. KPV supplies positive
  Crosswalk pixels only; unmarked pixels are ignored.
- Candidate selection and the human-reviewed Cityscapes ignore-mask workflow
  are implemented. Original Cityscapes masks remain unchanged.
- MicroSeg uses eight-group GroupNormalization. Shared Fast-SCNN blocks keep
  their BatchNormalization default.
- A five-epoch GroupNorm pilot completed on WSL CPU. Its best checkpoint and
  manifests are outside Git at
  `/home/nacsi/microseg_runs/pilot_groupnorm_20261004`.
- The pilot no longer predicted only Non-walkable at inference. Best-checkpoint
  Cityscapes validation mIoU was 0.313. Held-out KPV positive-pixel recall
  varied by epoch, peaking at 0.926 and ending at 0.760. Boundary IoU was
  0.172.

The visual grid is [MicroSeg GroupNorm pilot examples](./images/microseg_groupnorm_pilot_examples.png).
Regenerate it from the repository root in WSL:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
bash run_wsl.sh -m scripts.visualize_microseg_pilot
```

The renderer reads `best.keras`, the saved split manifests, and the local
Cityscapes/KPV datasets. It performs inference only and writes the image under
`docs/images/`.

## Important limitations

- KPV validation labels only positive Crosswalk pixels. Its reported recall
  does not measure false positives or precision; obtain/prepare valid
  Crosswalk-negative labels before using it to tune false-alarm behavior.
- Boundary currently maps Cityscapes wall/fence, which is not necessarily the
  intended curb or road-edge concept. Confirm that class definition before
  another training run.
- Cityscapes class imbalance remains relevant, particularly the low Boundary
  pixel share. The short pilot does not establish a settled sampling or
  class-weighting strategy.
- This is a bounded experiment, not a deployment-ready model. Quantization,
  TFLite compatibility, and ESP32-CAM memory/latency remain unverified.
- TensorFlow could not initialize CUDA in WSL for this pilot, so training and
  the example-rendering command should use CPU unless GPU setup is separately
  verified.

## Next steps

1. Review the visualization for representative alignment errors; it shows
   deterministic validation samples, not a curated success-only gallery.
2. Decide whether Boundary means wall/fence, curb, road edge, or a different
   explicitly annotated concept.
3. Establish evaluation data that includes labeled non-Crosswalk pixels, so
   Crosswalk false positives can be measured separately from positive-pixel
   recall.
4. Only after those decisions, design a controlled experiment for Boundary
   balance and Crosswalk precision/recall. Do not extend training by epochs
   alone.
5. Run the focused tests before changing training behavior:

   ```bash
   bash run_wsl.sh -m unittest tests.test_microseg_model tests.test_microseg_train
   bash run_wsl.sh -m unittest tests.test_microseg_dataset tests.test_microseg_taxonomy
   ```

   Expand to the candidate-selector and ignore-annotator tests when changing
   those workflows.

Keep datasets, checkpoints, caches, and run outputs outside Git. Commit source,
tests, concise documentation, and the generated example image only.
