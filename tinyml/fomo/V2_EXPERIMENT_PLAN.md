# My controlled 20 x 20 v2 experiments

I completed the frozen INT8 threshold sweep and generated 50 stratified TP, FP, and FN examples before any new training.
I inspected tiny-person, larger-person, larger-vehicle, and false-person examples against both the original image and actual resized input.
My complete diagnostic artifacts are in `artifacts/baseline-v1-diagnostics-20261004/`.
The similarly named `baseline-v1-analysis-20261004/` is an incomplete first diagnostic attempt whose gallery failed on a cache-path field; it is not the final report.

## Fixed comparison

I preserve `baseline_v1` in its original paths and identify it by the fingerprints in [the frozen record](artifacts/baseline-v1-diagnostics-20261004/baseline_v1.json).
I retain 160 x 160 RGB input, a 20 x 20 grid, Person/Vehicle, the exact Cityscapes splits, center-mask targets, ImageNet initialization, seed 42, batch size 32, horizontal flips, architecture, and the existing learning-rate/stopping rules.
I use a maximum of 20 epochs and select checkpoints by validation centroid F1 at confidence 0.60, as in v1.
I export with the same 100 training calibration images and compare INT8 predictions on all 500 validation images with unchanged matching, NMS, ignore handling, and no detection cap.
I report threshold changes separately from changes in trained weights.
This is a single-seed development comparison on repeatedly consulted validation data, not independent test performance.

## Operating-point experiment: no training

- Hypothesis: the 0.60 confidence filter hides useful detections.
- Change: evaluate 0.20 through 0.80 in 0.05 steps using the saved INT8 probability grids, with live-model spot checks.
- Expected result: lower thresholds recover more objects.
- Risk: more false warnings and more unknown-region predictions reach decision logic.

The observed highest F1 is at 0.35, but this adds 937 scored false positives for 482 extra matches relative to 0.60.
I provisionally recommend 0.55 for a modest recall-oriented development comparison, while preserving 0.60 as the frozen reference and default.
At 0.55, I gain 144 matches with 232 additional scored false positives.
There is no experimentally validated warning-cost function or deployment-safe threshold yet.

## Weight 40 experiment

- Hypothesis: background supervision still dominates after object weighting and discourages some object predictions.
- Change: double only the positive-cell loss coefficient from 20 to 40.
- Expected result: higher recall, preferably with improved F1 at comparable precision.
- Risk: false positives, class confusion, and poorly calibrated probabilities may increase.

The training targets contain 1,040,719 background cells and 30,973 positive cells, or 33.60 background cells per positive cell.
With per-image normalization, weight 20 gives object cells about 0.55 times the total background weight mass.
These masses are coefficients, not measured gradient contributions, and do not prove that imbalance causes low recall.

## Weight 10 experiment

- Hypothesis: some crowded-scene misses coexist with overconfident foreground and duplicate predictions; stronger object weighting may be counterproductive.
- Change: halve only the positive-cell loss coefficient from 20 to 10.
- Expected result: better precision or a better precision-recall tradeoff after threshold adjustment.
- Risk: lower recall and additional missed people.

I run these two bracketing weights as the smallest useful test of the imbalance hypothesis.
Each starts independently from the same ImageNet initialization and has a separate output directory.
Validation loss uses the selected coefficient too, so I compare centroid metrics rather than raw loss values across coefficients.

## Separate NMS diagnostic

The saved-prediction probe associates 1,938 misses with a same-class candidate suppressed by NMS, and 3,040 with an eligible prediction already used for another object.
I therefore also test the existing `cells` decoder, which disables same-class neighborhood suppression, at 0.60 and 0.55 using the identical saved grids.
This explicitly changes one inference factor and is a separate diagnostic, not the primary model-comparison protocol.
The hypothesis is that an eight-pixel suppression neighborhood merges nearby real instances.
I expect higher recall but risk multiple predictions for one object and more false positives.
I do not silently change the default decoder or claim that geometric associations establish the cause of every miss.

The initial no-NMS check recovered many more objects at the cost of lower precision.
Before inspecting trained-candidate outcomes, I extended this separate inference sweep to 0.95 to find an operating point retaining at least 70% precision on development validation.
Among the tested no-NMS points meeting that floor, 0.80 has the highest recall: 72.09% precision, 33.51% recall, and 45.75% F1.
I lock this as the common secondary protocol for both trained candidates, alongside the unchanged primary protocol of NMS at 0.60.
The 70% precision floor is a development comparison choice, not a proven safety requirement.
This supersedes the provisional 0.55 recommendation for the unchanged decoder; it does not change frozen v1 or application defaults.

## Final confidence-comparable check

After completing the primary and locked secondary comparisons, I observed that weight 10 favored precision and weight 40 favored recall.
Because loss weighting can change score calibration, I add a final equal-rule comparison before selecting a candidate.
I apply the identical no-NMS threshold grid, 0.20 through 0.95 in steps of 0.05, to every saved INT8 model.
I select the highest-recall point meeting the same predeclared 70% development-precision floor for each model, breaking ties by precision.
This changes no trained weights and does not replace the fixed-threshold tables.
I report it separately as additional validation tuning, with no independent-test claim.

## Deferred factors

I do not simultaneously change capacity, augmentation, loss formulation, or the learning-rate schedule.
The baseline losses are still declining slowly, but its missing historical training-F1 curve prevents a confident underfitting diagnosis.
I first measure the saved checkpoint on the unaugmented training split to assess its current train/validation gap.
I defer longer training and larger features until this smaller diagnostic experiment is assessed.
I do not modify MicroSeg, Fast-SCNN, datasets, firmware, or the 40 x 40 experiment.
