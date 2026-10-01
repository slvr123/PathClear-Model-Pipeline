# PathClear outdoor mobility roadmap

Date: 2026-09-22.
Status: proposed execution plan, following my agreement to include moving-obstacle awareness in the outdoor-mobility scope.
Codex developed the detailed steps below at my request; specific experiment settings and acceptance thresholds remain proposals until reviewed.
This expands LOCAL_DATASET_PLAN.md without replacing its six-step sequence.
Creating this plan does not start training, change labels, or finalize hardware.

Update, 2026-09-29: I supplied a ten-class taxonomy and confirmed preparing its pipeline without training until a labeled local pilot is available.
I keep this roadmap's original proposals below for context; my current label definitions and collection sequence are in [the annotation guide](docs/PATHCLEAR_ANNOTATION_GUIDE.md) and [local dataset plan](LOCAL_DATASET_PLAN.md).
I recorded the completed preparation checks in [the preparation report](docs/PATHCLEAR_10CLASS_PREPARATION.md).

## Intended behavior

I want PathClear to identify obstructions in the immediate walking corridor and update warnings as people and other obstacles enter, remain in, or leave it.
I am not promising pedestrian trajectory prediction, exact collision timing, automatic steering, or a guarantee that the route is safe.
I will distinguish no obstruction observed from a verified safe route.
I will explicitly represent unknown or stale sensor observations.
Camera direction can differ from walking direction, particularly with a head-mounted camera, so a screen-center region cannot automatically establish the user's path.

## Existing foundation

I keep checkpoints/baseline-5epochs/best.keras, epoch 19, frozen as the reference.
The baseline already has full Cityscapes evaluation and five qualitative internet-image examples.
Its local sidewalk and pedestrian errors require attention before warnings can be trusted.
The present files establish segmentation and offline evaluation, not a tested live mobility system.
The stated sensor name is provisionally interpreted as VL53L0X; I will confirm the marking and module before wiring or fixing requirements.
Other hardware, camera mounting, processing board, and feedback components remain to be documented.

## Phase 1: Define a bounded first prototype

I will start with controlled daylight walking-path scenarios with one pedestrian and a stationary obstacle.
I will document supported lighting, camera orientation, surfaces, operating distances, and exclusions rather than treat outdoor mobility as an unlimited claim.
I will inventory the camera, processor, ToF model/count, mounting, optional IMU, power system, and feedback device.
I will define obstruction events and distinguish people alongside the corridor from people occupying it.
I will choose measurable requirements for warning delay, missed events, false alerts, warning-clear delay, and handling lost data before tuning against the held-out tests.
The requirements will be justified by the intended operating conditions and observed hardware limits, not invented accuracy targets.

Deliverables: system scope, hardware inventory, scenario definitions, and draft acceptance criteria.
Exit check: I can explain which conditions the prototype supports and what evidence is needed to demonstrate each behavior.

## Phase 2: Verify inference geometry without training

I will compare the existing direct 1024 x 512 resize with a documented aspect-ratio-preserving method on the same five supplied photographs and initial local samples.
Padding is a candidate; I will record the padding value and resized dimensions, remove padding from output before restoring masks, and use nearest-neighbor interpolation for class IDs.
Padding also changes model input context and may reduce useful resolution, so I will not assume it improves predictions.
I will keep the original images, checkpoint, normalization, and labels fixed and verify checkpoint hashes before and after.
I will compare sidewalk boundaries, pedestrians, thin objects, and false detections qualitatively.
Numerical comparison requires reviewed masks on a development subset; model confidence is not accuracy.

Deliverables: side-by-side comparisons, preprocessing metadata, and a selected provisional capture/inference procedure.
Exit check: no unexplained coordinate mismatch, crop loss, or overlay misalignment remains.

## Phase 3: Collect an initial local pilot

I will use the intended mounting height and angle where possible and clearly mark any substitute-camera footage.
My proposed starting collection is 30-50 short clips of roughly 10-20 seconds across 5-8 routes and multiple sessions.
These are planning quantities, not proof of statistical sufficiency or enough training data.
I will include clear and obstructed sidewalks, narrow passages, buildings, trees, parked motorcycles, stalls, poles, shade, and backlighting within the proposed operating scope.
I will include people approaching, crossing both directions, stopping ahead, moving away, walking alongside, briefly occluding one another, and changing speed.
I will capture normal head turns and walking motion as well as a stable camera.
I will preserve original videos, capture timestamps, route/session IDs, camera settings, and mounting notes.
When the sensor capture rig is available, I will log time-aligned ranges and validity/status information alongside frames.
I will initially select about 100-200 diverse frames for inspection, with a smaller 20-30-frame annotation pilot before expanding labels.
I will use controlled participants for deliberate interaction scenarios and keep raw identifiable footage access limited rather than publish it by default.

Deliverables: indexed local clips, representative frames, and a coverage checklist.
Exit check: footage is usable at the intended viewpoint and includes both failures and clear examples.

## Phase 4: Split, annotate, and measure the frozen baseline

I will assign routes/sessions to development, validation, and held-out testing before extensive annotation or model tuning.
I will keep adjacent frames and repeat recordings of the same scene together wherever practical.
An approximate 60/20/20 allocation is a starting proposal only; route separation and scenario coverage take priority over exact counts.
With too few independent routes, I will collect more rather than claim a strong held-out result.
I will not inspect or tune against the final test split during development.
I will document class definitions, occlusion rules, boundaries, and ignore/unknown handling using the small annotation pilot.
I will preserve 19-class compatibility for baseline comparison where labels apply.
I will not force puddles, stalls, or other unsupported hazards into misleading classes; I will document their treatment explicitly.
Any decision to introduce new output classes requires a separate label/model plan and fair comparison on shared classes.
I will separately annotate dynamic event intervals such as obstruction present, obstruction cleared, and observation uncertain; these are not segmentation classes.
I will quality-check pedestrian and sidewalk masks especially carefully.
I will run the frozen baseline and report global and per-class IoU/Dice/recall/support, error images, and results by scene type.

Deliverables: annotation guide, split manifests, reviewed masks, event annotations, and local baseline results.
Exit check: labels and split boundaries are consistent, and local performance is measured without test leakage.

## Phase 5: Characterize hardware in parallel with data work

I will test VL53L0X readings against measured distances, for example 0.3, 0.5, 1.0, 1.5, and 2.0 m, where the device reports valid data.
I will test indoor reference conditions, outdoor shade, and daylight, varied clothing/targets, and more than one object in view.
I will record error, invalid-reading rate, range update intervals, and the distance over which performance is dependable.
The manufacturer's up-to-2-m specification is not an assured outdoor working range.
I will map the sensor's viewing area into camera coordinates at several distances and characterize ambiguous target association.
I will measure complete capture-to-warning delay on the intended processor, including communication and feedback, not infer it from laptop model timing.
I will determine how head motion affects the monitored corridor before promising that camera-forward means walking-forward.
I will retain this sensor for controlled experiments while deciding whether longer range, more spatial zones, different sensing, or narrower supported conditions are required.
I will not assume an IMU alone provides walking direction or per-person distance.

Deliverables: outdoor ranging report, alignment checks, timing logs, and hardware decision with rationale.
Exit check: the chosen sensor arrangement supports the specified prototype conditions, or the hardware/scope is revised explicitly.

## Phase 6: Improve recognition in a separate experiment

Only after reviewed local labels and a measured baseline will I propose a bounded fine-tuning run.
I will specify the starting checkpoint, data version, number of epochs, settings, checkpoint-selection rule, and output folder before execution.
The first experiment will use a controlled change rather than combine resolution, architecture, class weights, and augmentation changes at once.
If I introduce geometric augmentation, image and mask transformations must match and class masks must use appropriate discrete interpolation.
I will select models using validation results and retain the original baseline unchanged.
I will evaluate sidewalk/person recall and IoU, rare-class support, false positives, and temporal consistency, not just overall pixel accuracy.
I will check retention on Cityscapes if the model is intended to preserve those broader capabilities.
I will use the held-out local test after selecting the candidate and report limitations and scene coverage.

Deliverables: separately versioned candidate and baseline comparison.
Exit check: relevant validation gains are repeatable enough to justify integration and do not conceal major regressions.

## Phase 7: Build continuously updated obstruction warnings

I will implement timestamped camera input, segmentation, corridor interpretation, calibrated distance association, short-term obstacle continuity, and warning output.
I will decide whether simple region continuity is sufficient for the bounded one-person prototype before choosing an individual tracker for more complex scenes.
If I add a separate person detector/tracker, it will be measured for recognition, latency, and compatibility rather than assumed necessary or free of cost.
I will distinguish no obstruction observed, obstruction observed, near obstruction, and uncertain/stale observations internally.
These internal states do not all require spoken messages; feedback must avoid excessive alerts.
I will escalate on credible near-obstacle evidence without imposing a long smoothing delay.
I will use time-based persistence to clear warnings, because a fixed frame count changes meaning when frame rate varies.
I will not interpret invalid range, lost tracks, or an object leaving the sensor beam as automatically confirming a clear path.
I will suppress person-specific distance claims when association is ambiguous and use a general obstruction/uncertainty indication instead.
I will determine warning thresholds from the verified range and end-to-end delay, supported walking conditions, and supervised feedback testing.
For direct approach only, a planning distance budget can include relative closing speed times total delay plus a margin; this is not a general crossing-pedestrian collision model.

Deliverables: replayable live prototype, event logs, and documented warning transitions.
Exit check: warnings respond consistently to entry/exit, uncertainty is handled, and timing meets the agreed development requirements.

## Phase 8: Evaluate the integrated system

I will first replay recorded sequences, then use supervised controlled walking tests without relying on the prototype to prevent contact.
I will test empty paths, stationary obstacles, approach, crossing, walking alongside, stopping ahead, speeding up to leave, occlusion, head turns, poor lighting, and sensor loss.
I will repeat scenarios across sessions and reserve independent clips/routes for final evaluation.
I will count missed obstruction events, event recall, false alerts per minute, warning-on delay, warning-clear delay, and warning oscillations.
I will also report range validity, capture-to-output latency including high-percentile delays, dropped frames, sustained temperature/power behavior, and scenario-specific failures.
I will compare results with the predeclared criteria and report failures rather than adjust the final-test definitions to obtain a pass.
Passing controlled tests establishes that bounded demonstration, not unrestricted outdoor navigation readiness.

Deliverables: scenario report, demonstration recordings, supported-condition statement, and remaining limitations.

## Immediate order

1. Confirm the hardware inventory and bounded prototype behavior.
2. Run the unchanged-model preprocessing comparison.
3. Prepare route/session naming, capture instructions, and annotation rules.
4. Collect a small pilot with complete clips preserved.
5. Start sensor characterization alongside annotation and local baseline evaluation.
6. Propose fine-tuning only after the preceding evidence is available.
7. Integrate and evaluate moving-obstacle warnings before making outdoor-mobility performance claims.

## References

The [ST VL53L0X datasheet](https://www.st.com/resource/en/datasheet/vl53l0x.pdf) specifies up-to-2-m ranging and describes performance/configuration conditions.
The [TensorFlow segmentation tutorial](https://www.tensorflow.org/tutorials/images/segmentation) demonstrates paired image/mask processing relevant to augmentation.
All collection quantities, milestone organization, and exit checks above are project proposals rather than manufacturer guarantees or validated safety thresholds.
