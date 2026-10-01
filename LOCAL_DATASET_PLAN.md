# PathClear local dataset plan

I confirmed the original sequence on 2026-09-21 after Codex proposed it during our baseline evaluation discussion.
On 2026-09-29, I first confirmed preparing the ten-class pipeline while waiting for a labeled local pilot.
Later in the same discussion, I approved a separate Cityscapes adaptation stage that learns the seven supported categories before local data is ready.
I authorized a three-batch smoke test followed by one full epoch if checks pass, with classes 7-9 inactive until local training.
I will keep the epoch-19 Cityscapes baseline frozen as the reference.

1. **Check my annotation rules on a small pilot.**
   I use the ten fixed IDs and the draft overlap rules in [the annotation guide](docs/PATHCLEAR_ANNOTATION_GUIDE.md).
2. **Collect local footage at the intended camera position.**
   I include clear paths, narrow sidewalks, vehicles, pedestrians, barriers, discrete obstacles, vegetation, stairs, crossings, and visibly damaged pavement.
   I include intact pavement, ordinary road markings, shadows, and wet patches as contrasting examples.
3. **Split by route and session before tuning.**
   I keep neighboring frames and repeat views of the same location together, reserving independent routes for validation and final testing.
4. **Inspect the frozen baseline on development footage.**
   I compare direct resizing with aspect-ratio-preserving preprocessing before attributing errors entirely to the local setting.
   I do not inspect final-test predictions to choose preprocessing.
5. **Annotate and review roughly 20-30 diverse pilot frames, then expand.**
   I settle class boundaries first and check coverage across routes and classes, rather than assume this initial count is enough to train.
6. **Propose a separate, bounded fine-tuning experiment after labeled data is ready.**
   I compare the candidate with the frozen baseline on compatible classes using the same held-out local images and explicit label mapping.

## My next collection session

I will preserve complete clips and note the route, session, camera, mounting height/angle, time, and lighting.
The existing roadmap proposes 30-50 short clips across 5-8 routes and 100-200 candidate frames; these are planning estimates, not required dataset sizes or proof of sufficiency.
I will start smaller if needed to settle annotation rules, while deliberately finding examples of Stairs, Crosswalk, and Uneven Pavement.
I will include multiple instances and views, not count neighboring frames as independent evidence.
I will store clips under `datasets/pathclear_pilot/clips/`, selected images under `images/`, and ID masks under `masks/` relative to that pilot folder.
I can create those folders when I have real files to place in them.
I will add each selected image to [the empty manifest](datasets/pathclear_pilot/manifest.json), using `mask: null` until its annotation exists.
I will follow the format and review rules in [the annotation guide](docs/PATHCLEAR_ANNOTATION_GUIDE.md).

## What is ready now

I have preparation helpers for deterministic Cityscapes remapping, strict baseline weight transfer, and ten-class loss/metrics.
I have verified these with synthetic tests and one real Cityscapes image without training or saving a new model.
I recorded the checks, file inventory, and limitations in [the preparation report](docs/PATHCLEAR_10CLASS_PREPARATION.md).
I subsequently completed the authorized smoke check and one full seven-active-class Cityscapes epoch, with results in [the adaptation report](docs/PATHCLEAR_CITYSCAPES_STAGE_2026-09-29.md).
I now have a saved intermediate model while local-only classes remain untrained and disabled.
The original nineteen-class training and evaluation scripts remain unchanged.

I can check the pilot manifest from WSL with:

```bash
bash run_wsl.sh validate_pathclear_pilot.py datasets/pathclear_pilot/manifest.json
```

I currently have zero entries in the new local manifest.
The five previously inspected internet photographs remain qualitative examples, not a labeled local dataset or a source of local accuracy, IoU, or Dice.
After completing the smoke check and one full adaptation epoch, I requested ten additional Cityscapes epochs, continuing through epoch 11 with the same frozen-feature setup described in [the training guide](TRAINING.md).
I completed those ten epochs and retained epoch 11 as the best intermediate checkpoint; I recorded the results in [the continuation report](docs/PATHCLEAR_EPOCHS_2_11_2026-09-29.md).
I subsequently requested and completed five additional epochs with the same settings, stopping at epoch 16 and retaining its best checkpoint.
The extra run added 0.0403 percentage points of validation mean IoU; I recorded it in [the five-epoch continuation report](docs/PATHCLEAR_EPOCHS_12_16_2026-09-29.md).
I have not authorized feature-layer fine-tuning or custom-data training.
