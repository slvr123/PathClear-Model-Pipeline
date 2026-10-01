# My TinyML/FOMO experiment

I am exploring lightweight standalone inference as a fallback/basic mode for PathClear smart glasses.
I keep Fast-SCNN as the main semantic segmentation pipeline.
This folder contains a handoff scaffold, not a working FOMO model or validated mobility mode.

FOMO classifies locations on an output grid and reports approximate object centroids rather than full object outlines or reliable object extents.
It does not inherently provide full walkable-area segmentation.
Objects whose centers occupy the same output cell can be indistinguishable.
The Edge Impulse workflow accepts object bounding boxes and derives centroid supervision from them.
I use the [official FOMO design explanation](https://www.edgeimpulse.com/blog/announcing-fomo-faster-objects-more-objects/) as the starting reference.

## My first steps

I first review `../configs/pathclear_classes.yaml` and `../docs/DATASET_SETUP.md` with my groupmate.
On a machine with the existing WSL environment and external Cityscapes data, my first runnable baseline check is:

```bash
bash run_wsl.sh test_pipeline.py
```

I run it from the repository root, not from this folder.
There is no TinyML command to run yet.
I agree a small object-label pilot and the exact board/camera before implementing a converter or trainer.
I record preprocessing, class IDs, quantization, memory, and measured latency with each future experiment.
TinyML detection IDs may need a separate background class and must not silently reuse segmentation IDs.

| Folder | Intended work |
|---|---|
| `training/` | Isolated FOMO training configuration and scripts; generated outputs stay in `outputs/`. |
| `preprocessing/` | Reviewed annotation conversion and matching image/label transforms. |
| `export/` | Quantization/export tooling; generated artifacts stay in `generated/`. |
| `esp32/` | Board-specific camera and inference experiments. |

I can later commit a small reviewed final `.tflite` outside generated directories; the repository does not globally ignore TFLite files.
I keep local datasets, experimental checkpoints, credentials, and device build products out of Git.

## My annotation recommendation

I inspected `pathclear_taxonomy.py` and the loader's 19-class Cityscapes `labelTrainIds` masks.
Those semantic masks identify pixel classes but do not preserve individual object identity.
Connected components can merge touching people/vehicles or split occluded objects, so their centers are only candidates for manual review.
I prefer original instance annotations where available and compatible, or reviewed object boxes, before centroid training.
I do not implement a full converter until object definitions and output format are agreed.

| PathClear class | Current Cityscapes source | My FOMO assessment |
|---|---|---|
| Person | person 11, rider 12 | Good initial candidate with reviewed instances; rider/person overlap needs a labeling rule. |
| Vehicle | car 13, truck 14, bus 15, train 16, motorcycle 17, bicycle 18 | Good candidate if object instances and scale remain visible after resizing. |
| Obstacle | pole 5, traffic light 6, traffic sign 7 | Conditional: discrete objects are plausible, but these source labels do not establish a path obstruction. |
| Boundary | wall 3, fence 4 | Poor general centroid target; long boundaries require spatial extent. |
| Road / Sidewalk | road 0, sidewalk 1 | Poor fit: broad surfaces need segmentation rather than one object center. |
| Vegetation | vegetation 8 | Poor general fit; foliage regions are not consistent discrete objects. |
| Stairs / Crosswalk | No source class | Possible custom experiments only after defining one staircase/crossing region and reviewing labels; centroid output loses extent. |
| Uneven pavement | No source class | Requires local defect annotations; no automatic Cityscapes conversion is supported. |

I start with Person and Vehicle, then consider clearly defined discrete obstacles.
I keep stairs and crosswalk exploratory until annotation compatibility and resolution are tested.
I do not infer distance, safe passage, or complete navigation readiness from centroid detections.
