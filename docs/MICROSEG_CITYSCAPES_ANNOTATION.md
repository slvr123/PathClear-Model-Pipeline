# Cityscapes Crosswalk ignore-mask review

## Purpose

Create reviewed, image-aligned masks that exclude Cityscapes crosswalk regions
from MicroSeg loss without changing the original Cityscapes annotations.
Ordinary road pixels remain Non-walkable. Kaggle continues to provide
positive-only Crosswalk supervision to the same four-class model.

The annotation tool creates **Ignore** masks; it does not relabel Cityscapes
pixels as Crosswalk.

## Annotation policy

- Mark the **full visibly bounded crossing footprint**, including the gaps
  between painted stripes. This matches the intended Crosswalk-region
  supervision in the Kaggle source.
- Include only pixels that can be confidently identified as part of a
  pedestrian crossing. Do not mark lane lines, arrows, stop lines, unrelated
  road paint, or uncertain regions.
- Do not extend a polygon through an occluder if the crossing boundary cannot
  be identified confidently.
- If no crosswalk footprint is visible, skip the image; no mask file is needed.
- If uncertain whether a road marking is a crosswalk, leave it unmasked and
  flag it for human discussion; do not guess.

## Output format

For an image such as:

```text
leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
```

the expected mask is:

```text
ignore_masks/train/aachen/aachen_000000_000019_ignore.png
```

The mask is a single-channel 8-bit PNG at the original image size:

- `0`: normal Cityscapes supervision
- `255`: reviewed pixels excluded from loss and metrics

The original Cityscapes semantic masks are never modified. Images without a
positive ignore mask remain unchanged in training; missing masks are not
treated as reviewed and do not create empty files.

## Select likely candidates

Run the selector separately for each split:

```powershell
.\.venv\Scripts\python.exe -m scripts.select_cityscapes_candidates --split train `
  --output C:\datasets\cityscapes\microseg_review_candidates\train_v2.json
.\.venv\Scripts\python.exe -m scripts.select_cityscapes_candidates --split val `
  --output C:\datasets\cityscapes\microseg_review_candidates\val_v2.json
```

It uses Cityscapes road labels plus two visual signals in the lower road area:
repeated bright markings and broad transverse bands measured relative to the
road's local luminance. This adaptive contrast signal is intended to reduce
misses in dark scenes; it is still a heuristic. The JSON output contains
ranked `candidates` and `scored_images` for audit and rejected-image sampling.
It records the total, selected, and rejected counts. Adjust the threshold with
`--score-threshold`; a lower threshold generally sends more images for review.
Scores are not probabilities. The selector is candidate mining only, not a
crosswalk detector or a replacement for human verification.

The current selector does not use MicroSeg predictions. Previous pilots showed
unstable class collapse, so those outputs are not a reliable mining signal
until model behavior improves.

The heuristic may select lane paint, stop lines, glare, or other road markings;
these are expected false positives for a human to reject. It can also miss
faded, occluded, small, or unusual crossings. Low-scoring images are not
certified crosswalk-free.

## Run the desktop tool

Run from the repository root in the configured Windows Python environment.
To review the ranked candidate queue:

```powershell
.\.venv\Scripts\python.exe -m scripts.annotate_cityscapes_ignore --split train `
  --candidates-json C:\datasets\cityscapes\microseg_review_candidates\train_v2.json
```

Repeat for validation:

```powershell
.\.venv\Scripts\python.exe -m scripts.annotate_cityscapes_ignore --split val `
  --candidates-json C:\datasets\cityscapes\microseg_review_candidates\val_v2.json
```

The default output is `C:\datasets\cityscapes\ignore_masks`. To use another
output directory, pass `--ignore-root`, but keep it inside the Cityscapes
dataset root so the trainer can safely resolve it.

Controls:

- Left-click to add polygon vertices.
- Right-click or press Enter to finish the current polygon.
- **Undo polygon** removes the active point or last polygon added in this
  editing session.
- **Clear mask** removes previously saved regions and new regions for the
  current image; save afterward to remove an existing positive mask.
- **Save positive mask** saves the current ignore region without changing
  images.
- **Save & next** saves a positive mask and advances.
- **Skip / next** moves on without creating a mask.
- **Previous** navigates backward. Unsaved edits prevent navigation.

With a candidate file, the tool opens only the ranked candidates and starts at
the first unreviewed candidate. It stores confirmed/rejected progress beside
the candidate JSON as `<split>.review.json`, bound to the exact candidate file.
The progress file keeps separate reviewed, confirmed, and rejected counts, so
skipping an image does not require writing an empty training mask.
Candidates without a crosswalk need no mask file; click **Skip / next** or
**Previous** to reject or pass them. Confirmed regions get a positive mask
when saved. Existing masks can be opened and edited; use **Clear mask** and
save to remove a saved region. Saved masks are written atomically.

To estimate missed-crosswalk risk, also review a deterministic sample of
rejected images:

```powershell
.\.venv\Scripts\python.exe -m scripts.annotate_cityscapes_ignore --split train `
  --candidates-json C:\datasets\cityscapes\microseg_review_candidates\train_v2.json `
  --review-rejected-sample 50
```

This audit samples from the entire scored non-candidate set. If a rejected
sample contains a crossing, annotate it and lower the selector threshold or
improve the heuristic before relying on the queue. Review high-, medium-, and
low-scoring candidates as well. Record totals for candidates reviewed,
confirmed crossings, rejected candidates, rejected samples reviewed, and
missed crossings; use the reviewed sample to estimate candidate recall, not to
claim perfect coverage.

## Recommended review workflow

1. Run the selector and record its total/candidate/rejected counts.
2. Review candidates across score bands and confirm polygon placement,
   image scaling, and the full-footprint rule.
3. Review a deterministic sample of rejected images and record any missed
   crossings. A miss shows the selector is not adequate for that split;
   lowering a threshold helps only if those images have nonzero scores.
4. Save positive masks only for confirmed crossings. Do not create empty masks
   for rejected/no-crosswalk images.
5. Periodically inspect saved overlays by reopening them in the tool.
6. Treat the sample used to tune a selector as development data, not as an
   independent recall estimate. Audit a fresh sample before deciding whether
   to rely on the revised queue.
7. Start a pilot only after candidate recall and the remaining label-conflict
   risk have been reviewed and the bounded smoke test passes.

The local Cityscapes collection has 2,975 training and 500 validation images.
At threshold `0.08`, the original and revised selectors ranked:

| Version | Split | Total | Candidates | Rejected by heuristic |
|---|---|---:|---:|---:|
| Original | Train | 2,975 | 373 (12.5%) | 2,602 (87.5%) |
| Original | Validation | 500 | 128 (25.6%) | 372 (74.4%) |
| Revised | Train | 2,975 | 1,523 (51.2%) | 1,452 (48.8%) |
| Revised | Validation | 500 | 288 (57.6%) | 212 (42.4%) |

These are workload counts only. A human audit of 50 rejected training images
then found 14 confirmed crossings (28%), so the original selector had an
unacceptable observed miss rate. The revised selector adds adaptive contrast
and repeated transverse-band scoring. On that same development sample it
ranked 11 of those 14 misses, but also ranked 12 of the 36 images humans had
rejected. These figures are for tuning only—not an independent recall estimate
or proof that the revised selector is reliable. Fresh rejected-image auditing
is required before relying on its candidate list. The revised report selected
1,523 images, but existing review progress was carried forward: the annotation
tool will skip the 396 already-reviewed images that are in that candidate set,
leaving 1,127 new candidates in the queue. Review the fresh rejected sample
before processing that larger queue.

## Use masks for training

With the default output root, pass the root to the trainer:

```bash
bash run_wsl.sh -m microseg.train \
  --cityscapes-ignore-root /mnt/c/datasets/cityscapes/ignore_masks \
  --smoke-test \
  --output-dir /home/nacsi/microseg_runs/smoke-with-cityscapes-ignore
```

The trainer applies available positive masks and leaves images without masks
unchanged. Its run manifest records the masks used. Keep run outputs outside
the repository, preferably on WSL's Linux filesystem. Do not start a longer
pilot until candidate recall has been assessed and the smoke run passes.
