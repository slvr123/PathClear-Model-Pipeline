#!/usr/bin/env bash
# Does the flood deadband earn its place in practice?
# Replays the weight-0 configuration (the run that flooded to 98.79% of the
# frame) under --score-mode harmonic_flood, so the penalty fires against a real
# flood rather than being inferred from history.csv arithmetic. Same 800x5 budget
# as the original pilot-nohorizon-20261005 so the two are directly comparable.
cd /mnt/c/Users/ALLAN/dev/PathClear-Model-Pipeline || exit 1

name=pilot-neg0-harmonicflood-20261005
run_dir="/home/nacsi/microseg_runs/$name"
# The log lives OUTSIDE run_dir: train.py refuses to start when its
# --output-dir is non-empty, so nothing may be written there first.
log="/home/nacsi/microseg_runs/$name.train.log"
rm -rf "$run_dir"
: > "$log"

echo "=== $name neg=0.0 score_mode=harmonic_flood ===" >> "$log"
PYTHONUNBUFFERED=1 bash run_wsl.sh -m microseg.train \
  --epochs 5 \
  --batch-size 8 \
  --steps-per-epoch 800 \
  --crosswalk-weight 0.5 \
  --crosswalk-real-weight 0.5 \
  --crosswalk-negative-weight 0.0 \
  --crosswalk-horizon-fraction 0.45 \
  --augment \
  --score-mode harmonic_flood \
  --cityscapes-score-weight 0.5 \
  --frozen-metric-patience 3 \
  --cityscapes-manifest /home/nacsi/microseg_runs/pilot_groupnorm_20261004/cityscapes_pairs.json \
  --split-manifest /home/nacsi/microseg_runs/crosswalk_split.json \
  --output-dir "$run_dir" >> "$log" 2>&1
echo "EXIT=$?" >> "$log"