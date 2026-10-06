#!/usr/bin/env bash
# Step-budget curve point at 400x5, now that harmonic_flood is the default.
# Brackets the two budgets already measured at the same total sample count as the
# baseline (200x5 = 1000 samples) and the 800-step run (800x5 = 4000 samples):
#   200x5 -> Cityscapes 0.4482, Crosswalk full-frame IoU 0.6675, coverage 28.4%
#   800x5 -> Cityscapes 0.3838, Crosswalk full-frame IoU 0.8175, coverage 22.1%
# Everything else is pinned to the s200 reference so only --steps-per-epoch moves.
# Augmentation stays off: the s200 A/B showed a 0.0021 mIoU gap, inside noise.
cd /mnt/c/Users/ALLAN/dev/PathClear-Model-Pipeline || exit 1

name=pilot-neg025-s400-augoff-20261005
run_dir="/home/nacsi/microseg_runs/$name"
# The log lives OUTSIDE run_dir: train.py refuses to start when its
# --output-dir is non-empty, so nothing may be written there first.
log="/home/nacsi/microseg_runs/$name.train.log"
rm -rf "$run_dir"
: > "$log"

echo "=== $name steps=400 neg=0.25 score_mode=default(harmonic_flood) ===" >> "$log"
PYTHONUNBUFFERED=1 bash run_wsl.sh -m microseg.train \
  --epochs 5 \
  --batch-size 8 \
  --steps-per-epoch 400 \
  --crosswalk-weight 0.5 \
  --crosswalk-real-weight 0.5 \
  --crosswalk-negative-weight 0.25 \
  --crosswalk-horizon-fraction 0.45 \
  --no-augment \
  --cityscapes-score-weight 0.5 \
  --frozen-metric-patience 3 \
  --cityscapes-manifest /home/nacsi/microseg_runs/pilot_groupnorm_20261004/cityscapes_pairs.json \
  --split-manifest /home/nacsi/microseg_runs/crosswalk_split.json \
  --output-dir "$run_dir" >> "$log" 2>&1
echo "EXIT=$?" >> "$log"