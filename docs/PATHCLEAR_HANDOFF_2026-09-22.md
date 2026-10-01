# PathClear: progress and next steps
Status handoff | 22 September 2026 | Prepared for my project group and AI continuity

## Current position
I have completed the 20-epoch Cityscapes baseline, evaluated its best checkpoint, and inspected its predictions on five supplied Philippine street photographs from the internet.
I am keeping epoch 19 frozen as the reference while preparing local data collection and moving-obstacle integration.
No training beyond epoch 20, local fine-tuning, or live pedestrian-warning demonstration has been completed.

## Verified baseline
| Item | Result |
| --- | --- |
| Completed training | 20 epochs; 59,500 optimizer updates in final resume state |
| Best checkpoint | best.keras, epoch 19, selected by validation mIoU |
| Best-checkpoint optimizer updates | 56,525 |
| Evaluation | All 500 Cityscapes validation images; batch size 1 |
| Input / output | 512 x 1024 RGB; 19 output classes |
| Pixel Accuracy | 88.6698% |
| Mean IoU | 0.422975 (42.30%) |
| Macro Dice | 0.540340 (54.03%) |
| Validation loss | 0.341819 |

The checkpoint loaded successfully, and evaluation left model weights, optimizer variables, and its file hash unchanged.
Metrics matched epoch-19 history within 0.000005.
Void pixels were excluded from segmentation metrics using the existing pipeline; loss retained its original weighted reduction.
The evaluation loop took 28.83 seconds; the initial evaluation script took 50.71 seconds including loading and initial artifacts.
These timings do not establish live camera-to-warning latency.

## Evaluation outputs
I generated per-class metrics, raw and normalized confusion matrices, per-image metrics, all 500 predicted masks, and 12 diagnostic visual examples.
The visual examples included weak, middle, and strong image-score groups plus targeted failures.
This completes a measured baseline, not proof of outdoor navigation readiness.


## Recognition strengths and weaknesses
| Class | Validation IoU | Interpretation |
| --- | --- | --- |
| Road | 94.16% | Strong on Cityscapes validation |
| Sidewalk | 60.79% | Substantial boundary and road confusion remains |
| Person | 40.52% | Weak for a pedestrian-mobility application |
| Bicycle | 42.42% | Incomplete shapes and class confusion |
| Rider | 15.64% | Weak recognition |
| Motorcycle | 7.80% | Very weak recognition |

The five strongest classes were road, vegetation, sky, car, and building.
The five weakest were train, motorcycle, rider, traffic light, and truck.
Large classes dominate pixel accuracy, so 88.67% pixel accuracy is not 88.67% navigation reliability.
The history discussion identified worsening validation at epoch 20 despite stronger training performance; I decided to stop unchanged continuation and retain epoch 19.

## Five internet-image inference checks
I ran all five supplied images through the unchanged checkpoint and saved originals, class-ID masks, color segmentations, overlays, and a gallery.
Broad road, sky, and vegetation regions sometimes looked plausible, but sidewalks frequently became road.
Pedestrians were fragmented, while parts of walls, storefronts, and tree trunks received false person, rider, vehicle, or traffic-sign labels.
These photographs are qualitative samples, not my custom dataset, and have no ground-truth masks.
I cannot calculate their local accuracy, IoU, or Dice.

The existing direct resize to 1024 x 512 changes their aspect ratios, especially the portrait photograph.
I have identified an aspect-ratio-preserving comparison as the next diagnostic step; it has not been run yet.
I cannot attribute all errors to unfamiliar Philippine scenes before checking preprocessing and viewpoint effects.

## Example: the main sidewalk becomes road
The comparison below shows original, predicted segmentation, and overlay.
Purple denotes road; the main pedestrian pavement was incorrectly assigned this class.

![Sidewalk inference example](../checkpoints/baseline-5epochs/ph_samples_2026-09-21/sample_04_comparison.png)


## Outdoor mobility includes moving people
I confirmed that PathClear should respond to pedestrians occupying or entering the immediate walking corridor.
I did not promise explicit pedestrian trajectory prediction or exact collision timing.
The intended behavior is to update obstruction warnings as the scene changes, including when someone stops ahead or moves away.
No obstruction observed must not be presented as a guarantee that the route is safe.

## What remains missing
The current implementation provides segmentation and offline inference/evaluation.
It does not yet demonstrate camera-distance association, temporal obstacle continuity, walking-corridor interpretation, or a validated live warning system.
I need to account for camera/head movement because the center of the camera view does not always represent walking direction.
I need warning logic that escalates on credible near-obstacle evidence and clears only after consistent observations.
Lost detections, stale frames, or invalid ranges must remain uncertain rather than automatically clearing a warning.
Fine-tuning can improve recognition, but cannot by itself deliver these system behaviors.

## Hardware decision remains open
I reported the current sensor as VL530X; it was provisionally interpreted as VL53L0X and its marking still needs confirmation.
If confirmed, the VL53L0X is a single-zone sensor with an advertised maximum range of up to 2 m, not 4 m.
Its useful outdoor range must be measured under actual conditions.
One zone cannot directly provide separate simultaneous distances to multiple pedestrians.
I will use it for controlled experiments before deciding whether the sensor arrangement or supported scope must change.
The camera, processing board, mounting, optional IMU, power, and feedback hardware still need a complete inventory.

## What changed in the plan
I retained the six-step local dataset sequence and expanded it to include complete video recordings, sensor characterization, temporal warning behavior, and event-level testing.
This is a scoped extension toward changing-obstacle awareness, not a restart or replacement of the baseline.
The eight-phase roadmap contains proposed execution details; numerical targets and hardware choices are not finalized.
No model architecture, training labels, or training hyperparameters were changed during these evaluation and planning tasks.


## 1. Define the first demonstration
I will document a bounded daylight prototype with controlled stationary obstacles and one pedestrian before adding more complex scenes.
I will inventory hardware and define what counts as an obstruction, a clear observation, and uncertainty.
I will establish measurable acceptance criteria before tuning on final evaluation scenarios.
Output: scope statement, hardware list, and scenario checklist.

## 2. Compare inference geometry without training
I will compare direct resizing with a documented aspect-ratio-preserving method using the same unchanged checkpoint and images.
I will verify coordinate mapping, mask restoration, and alignment, and record any padding.
I will not assume padding improves results because it also changes context and useful resolution.
Output: side-by-side diagnostic report and provisional preprocessing procedure.

## 3. Collect a local pilot with complete clips
My proposed starting quantity is 30-50 clips of roughly 10-20 seconds across 5-8 routes and multiple sessions.
These quantities are planning estimates, not proof of enough training data.
I will use the intended camera height/angle and include clear paths, narrow sidewalks, parked motorcycles, poles, stalls, shade, and backlighting.
Pedestrian scenes will include approach, crossing, walking alongside, stopping, leaving, speed changes, and brief occlusion.
I will preserve full clips, timestamps, route/session IDs, and mounting notes, then select roughly 100-200 diverse frames for inspection.
When available, the capture rig will record synchronized distance values and validity/status information.
Output: indexed pilot collection and coverage checklist.

## 4. Split, annotate, and measure the local baseline
I will split by route/session and keep neighboring frames together before tuning.
I will reserve independent final-test footage and avoid using it to choose preprocessing, models, or warning thresholds.
I will start with a proposed 20-30-frame annotation pilot to settle class boundaries, occlusions, and unknown/ignore rules.
I will not force unsupported hazards into misleading Cityscapes classes; new output labels require a separate explicit decision.
I will annotate obstruction entry/exit events separately from segmentation masks.
Output: annotation guide, split manifests, reviewed labels, and a measured frozen local baseline.


## 5. Characterize the hardware in parallel
I will measure range error, invalid readings, update rate, camera alignment, and target ambiguity under controlled outdoor conditions.
I will measure capture-to-warning delay on the intended processor rather than extrapolate laptop inference timing.
Output: evidence to retain the sensor, change the arrangement, or narrow the supported operating conditions.

## 6. Propose a separate fine-tuning experiment
After local labels and baseline measurements exist, I will specify a bounded experiment with documented settings and separate outputs.
I will compare sidewalk/person metrics, false positives, and regressions while retaining epoch 19 unchanged.
Output: a measured candidate model; no run is authorized merely by this roadmap.

## 7. Integrate continuously updated warnings
I will connect camera input, recognition, corridor interpretation, distance association, short-term obstacle continuity, and feedback.
I will start with bounded one-person scenarios and assess whether an individual tracker is needed for more complex scenes.
Warnings must not flicker, clear on sensor failure, or assign a distance to a person without adequate association evidence.
Output: replayable prototype and timestamped warning logs.

## 8. Test the whole system
I will begin with recorded playback and supervised controlled tests, without relying on the prototype to prevent contact.
I will measure missed obstruction events, false alerts per minute, warning-on delay, warning-clear delay, warning oscillations, range validity, and processing latency.
I will include empty paths, static obstacles, approach, crossing, parallel walking, stopping, leaving, occlusion, head turns, and sensor loss.
Output: scenario results and a supported-condition statement, not a claim of unrestricted outdoor readiness.

## Immediate priority and task ownership
I need to confirm the exact hardware and capture viewpoint with my group.
The next software task is the frozen-model resize comparison, followed by a capture guide and pilot annotation rules.
Local collection and sensor characterization can then proceed alongside one another.
No project roles, calendar deadlines, purchases, new classes, or further training have been finalized.

## Reference documents
LOCAL_DATASET_PLAN.md preserves the agreed six-step sequence.
OUTDOOR_MOBILITY_ROADMAP.md expands the implementation phases and exit checks.
The evaluation report and PH sample gallery contain the supporting results.
Sensor reference: https://www.st.com/resource/en/datasheet/vl53l0x.pdf


## AI continuation details

- Actual source repository: `C:/Users/Asus/OneDrive/Desktop/Code/PathClear`.
- The session workspace `C:/Users/Asus/OneDrive/Documents/ChatGPT/pathClear` is not the source repository.
- Frozen checkpoint: `checkpoints/baseline-5epochs/best.keras`.
- Final resume state: `checkpoints/baseline-5epochs/state.json` records 20 epochs and 59,500 updates; the best checkpoint itself has 56,525 updates.
- Baseline evaluation: `checkpoints/baseline-5epochs/evaluation/metrics.json`, `evaluation_summary.md`, per-class CSVs, confusion data, and qualitative gallery.
- Five-photo outputs: `checkpoints/baseline-5epochs/ph_samples_2026-09-21/`, including `gallery.md` and `inference_manifest.json`.
- Added helpers: `evaluate_baseline.py`, `render_evaluation.py`, `write_evaluation_report.py`, and `infer_ph_samples.py`.
- Checkpoint SHA-256: `47a248f9c3ee5f38194eb7071c035b4bc1c3c01254bec8bb44bbc240409321d7`.
- Existing WSL launcher: `bash run_wsl.sh`; WSL repository is `/mnt/c/Users/Asus/OneDrive/Desktop/Code/PathClear`.
- Inspect current files before editing; preserve known-good Windows/WSL training and dataset code.
- Do not start additional training, alter labels/architecture, or implement proposed hardware changes as part of this handoff.
- Next pending diagnostic: aspect-ratio-preserving inference comparison, not yet performed.
- Use Sean's name, avoid em dashes, and write project decision/handoff documents in first person.
- Authority: completed results were checked against current JSON artifacts and source files on 2026-09-22; scope decisions come from the conversation; phase quantities and detailed acceptance criteria remain proposals.
