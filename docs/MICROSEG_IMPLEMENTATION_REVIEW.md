# MicroSeg implementation compatibility review

## Summary

The reconciled plan matches the existing direction: one shared, four-class
segmentation model is trained with complementary Cityscapes and Kaggle
Crosswalk supervision. The loader now accepts reviewed Cityscapes ignore masks
without changing the existing label mappings or loss mechanism. The dataset
still needs reviewed masks to be supplied; the local annotations have not yet
been prepared, and training has not been rerun with this feature.

This review compares the plan with the implementation as it stood on
2026-10-04. It is a compatibility/status note, not evidence that a full
training run is ready or that the model has good segmentation performance.

## Implemented and aligned

| Area | Current behavior |
|---|---|
| Shared model | A single MicroSeg CNN predicts Walkable, Non-walkable, Boundary, and Crosswalk. |
| Output layout | The Keras model emits channels-last logits with shape `[B, H, W, 4]`. It does not emit `[B, 4, H, W]`. |
| Ignore label | ID `255` is not a predicted class. The loader replaces ignored targets with a safe in-range ID and supplies zero sample weight at those pixels. The weighted sparse categorical loss and weighted metrics exclude them. This is not implemented through a named `ignore_index` argument. |
| Cityscapes taxonomy | Existing Cityscapes `labelIds` and `trainIds` mappings are preserved. In particular, road maps to Non-walkable, sidewalk to Walkable, and the existing wall/fence labels map to Boundary. |
| Kaggle mask behavior | Pixels above the current foreground threshold map to Crosswalk. All other Kaggle mask pixels are ignored, not treated as Walkable, Non-walkable, or Background. |
| Cityscapes ignore masks | Optional binary masks can now be paired with Cityscapes images. Ignored pixels become zero-weight targets after the normal semantic remapping. |
| Boundary policy | Boundary currently comes from the existing Cityscapes wall/fence mappings. The pipeline does not generate boundaries morphologically. |
| Mixed training | Cityscapes and Kaggle batches feed the same model. Kaggle is used for training, while the current Cityscapes multiclass validation remains separate. |
| Smoke test | The trainer has a bounded smoke mode that uses one Cityscapes batch and one Kaggle Crosswalk batch, checks that weights changed, and checks that Cityscapes validation did not change the weights. This is a pipeline check, not a model-quality result. |

## Gaps and plan clarifications

### 1. Reviewed Cityscapes ignore masks must still be prepared

The loader accepts a separate image-aligned ignore mask. The agreed policy is
to ignore reviewed Cityscapes crosswalk pixels while retaining Non-walkable
supervision for the remaining road pixels. Do not ignore every road pixel as a
substitute.

The default directory convention is
`<cityscapes-root>/ignore_masks/{split}/{city}/{image-stem}_ignore.png`,
mirroring the Cityscapes image split/city folders. The ignore root must be
inside the Cityscapes dataset root. Masks must use the same native dimensions
as their image and semantic mask and contain only `0`, `1`, or `255`; nonzero
pixels are ignored. Passing `--cityscapes-ignore-root` applies available
positive masks; images without one are left unchanged, not claimed as reviewed.
Alternatively, an explicit Cityscapes JSON manifest may include an
`ignore_mask` path for individual pairs. The trainer saves these paths in its
run manifest so resume uses the same masks.

### 2. Kaggle provides positive-only Crosswalk supervision

The current Kaggle masks do not label the rest of an image as Walkable,
Non-walkable, or Boundary. Only marked Crosswalk pixels are supervised; all
unmarked pixels remain ignored. The plan should not imply that Kaggle supplies
the other three classes unless new, reviewed annotations and mappings are
introduced separately.

### 3. Keep the existing Boundary policy separate

The existing Boundary representation is sourced from Cityscapes wall/fence
labels. There is no dynamic or morphological boundary-generation step in the
current loader. Preserve this behavior as part of Crosswalk integration. Any
proposal to derive boundaries from other classes should be designed, documented,
and evaluated as a separate change.

### 4. Dataset pixel-count reporting is not implemented

The current pipeline does not report per-source pixel counts and percentages
for all classes plus Ignore. If this reporting is added, distinguish:

- **Raw source counts:** counts over each dataset's examples, reported
  separately for Cityscapes and Kaggle.
- **Sampled training counts:** counts observed after the configured
  interleaving and source-sampling policy.

These are different quantities because the training pipeline samples and
repeats streams. Do not describe a mixture sampling ratio as the raw combined
dataset distribution.

### 5. Smoke and integrity checks should be distinguished

The existing smoke test covers a small end-to-end training path, weight
updates, and validation weight stability. The focused MicroSeg regression
suite passes all 51 tests, including new checks for mask pairing, alignment,
invalid mask values, zero sample weights, loss invariance at ignored pixels,
weighted metric invariance, manifest paths, and preservation of ordinary
Cityscapes road supervision. Annotation and selector test results are tracked
in their own test modules.
The smoke test has not yet been rerun with the new mask path. Per-source raw
class-distribution reporting is also not implemented. Additional checks for:

- allowed target IDs and target/weight shape agreement;
- Kaggle unlabeled-pixel ignore behavior;
- expected Crosswalk-positive pixels after resize;
- continued consistency of the existing Boundary mapping.

Only claim these checks are covered after they have been implemented and run.

### 6. Held-out Kaggle Crosswalk metrics have a coverage limitation

The held-out real Kaggle masks supervise Crosswalk-positive pixels only. The
current evaluation can measure performance on those marked pixels, but ignored
regions cannot establish false-positive behavior there. These metrics are not
a complete road-versus-crosswalk precision or IoU assessment. For that,
evaluate on data with fully labeled Crosswalk and non-Crosswalk regions.

## Recommended implementation sequence

1. Keep the four output classes, class IDs, Cityscapes mappings, and current
   Boundary policy unchanged.
2. Specify and implement a separate, reviewed Cityscapes ignore-mask format
   and pairing rule.
3. Apply that mask consistently to Cityscapes training and validation targets
   and sample weights.
4. Add tests proving ignored pixels stay unsupervised and Kaggle unmarked
   pixels are never converted into semantic negatives.
5. Add per-source raw class-distribution reporting, clearly separated from
   sampled training statistics.
6. Prepare and review Cityscapes ignore annotations, then run the dataset
   integrity checks and the existing bounded smoke test.
7. Only then consider another bounded training pilot; the previous pilots
   showed unstable class collapse, so this note does not authorize a long run.

## Relevant implementation files

- [`microseg/model.py`](../microseg/model.py) — four-class model and output
  shape.
- [`microseg/taxonomy.py`](../microseg/taxonomy.py) — class IDs and
  Cityscapes mappings.
- [`microseg/dataset.py`](../microseg/dataset.py) — Cityscapes/Kaggle loading,
  ignore weights, and mixed dataset sampling.
- [`microseg/train.py`](../microseg/train.py) — smoke checks, training,
  validation, and run configuration.
- [`scripts/annotate_cityscapes_ignore.py`](../scripts/annotate_cityscapes_ignore.py)
  and [`MICROSEG_CITYSCAPES_ANNOTATION.md`](MICROSEG_CITYSCAPES_ANNOTATION.md)
  — reviewed ignore-mask drawing tool and workflow.
- [`tests/test_microseg_dataset.py`](../tests/test_microseg_dataset.py) —
  loader and dataset behavior tests.
- [`tests/test_microseg_train.py`](../tests/test_microseg_train.py) — trainer
  split and manifest tests.
- [`tests/test_cityscapes_ignore_annotator.py`](../tests/test_cityscapes_ignore_annotator.py)
  — annotation mask path, rasterization, and encoding tests.
- [`tests/test_cityscapes_candidate_selector.py`](../tests/test_cityscapes_candidate_selector.py)
  — verifies candidate scores and candidate/rejected classification on
  constructed scenes.
