#!/usr/bin/env bash
# Compare checkpoint-selection metrics on the 200x5 configuration.
# Arm A: harmonic mean only. Arm B: harmonic mean plus the flood deadband.
# Everything else is pinned to the s200-augoff settings so the only difference
# between the arms, and against pilot-neg025-s200-augoff-20261005, is the metric.
cd /mnt/c/Users/ALLAN/dev/PathClear-Model-Pipeline || exit 1

run_one() {
  local name="$1"
  local mode="$2"
  local log="/home/nacsi/microseg_runs/$name.train.log"
  local run_dir="/home/nacsi/microseg_runs/$name"
  # The log lives OUTSIDE run_dir: train.py refuses to start when its
  # --output-dir is non-empty, so nothing may be written there first.
  rm -rf "$run_dir"
  : > "$log"

  echo "=== $name score_mode=$mode ===" >> "$log"
  PYTHONUNBUFFERED=1 bash run_wsl.sh -m microseg.train \
    --epochs 5 \
    --batch-size 8 \
    --steps-per-epoch 200 \
    --crosswalk-weight 0.5 \
    --crosswalk-real-weight 0.5 \
    --crosswalk-negative-weight 0.25 \
    --crosswalk-horizon-fraction 0.45 \
    --no-augment \
    --score-mode "$mode" \
    --cityscapes-score-weight 0.5 \
    --frozen-metric-patience 3 \
    --cityscapes-manifest /home/nacsi/microseg_runs/pilot_groupnorm_20261004/cityscapes_pairs.json \
    --split-manifest /home/nacsi/microseg_runs/crosswalk_split.json \
    --output-dir "$run_dir" >> "$log" 2>&1
  echo "EXIT=$?" >> "$log"
}

run_one pilot-s200-harmonic-20261005 harmonic
run_one pilot-s200-harmonicflood-20261005 harmonic_flood

echo "ALL_DONE" >> /home/nacsi/microseg_runs/pilot-s200-harmonicflood-20261005.train.log