#!/usr/bin/env bash
# Step-budget x augmentation isolation, holding --crosswalk-negative-weight 0.25.
# Both arms use 200 steps x 5 epochs = 1000 samples, matching the baseline run's
# 50 x 20 = 1000, so effective sample budget matches the 0.4679 mIoU reference.
# Negative supervision stays on in both arms because the weight-0 ablation showed
# it is what prevents the Crosswalk flood (coverage 98.8% without it).
cd /mnt/c/Users/ALLAN/dev/PathClear-Model-Pipeline || exit 1

run_one() {
  local name="$1"
  local steps="$2"
  local aug_flag="$3"

  local run_dir="/home/nacsi/microseg_runs/$name"
  # The log lives OUTSIDE run_dir: train.py refuses to start when its
  # --output-dir is non-empty, so nothing may be written there first.
  local log="/home/nacsi/microseg_runs/$name.train.log"
  rm -rf "$run_dir"
  : > "$log"

  echo "=== $name steps=$steps aug=$aug_flag ===" >> "$log"
  PYTHONUNBUFFERED=1 bash run_wsl.sh -m microseg.train \
    --epochs 5 \
    --batch-size 8 \
    --steps-per-epoch "$steps" \
    --crosswalk-weight 0.5 \
    --crosswalk-real-weight 0.5 \
    --crosswalk-negative-weight 0.25 \
    --crosswalk-horizon-fraction 0.45 \
    "$aug_flag" \
    --cityscapes-score-weight 0.5 \
    --frozen-metric-patience 3 \
    --cityscapes-manifest /home/nacsi/microseg_runs/pilot_groupnorm_20261004/cityscapes_pairs.json \
    --split-manifest /home/nacsi/microseg_runs/crosswalk_split.json \
    --output-dir "$run_dir" >> "$log" 2>&1
  echo "EXIT=$?" >> "$log"
}

run_one pilot-neg025-s200-augoff-20261005 200 --no-augment
run_one pilot-neg025-s200-augon-20261005 200 --augment

echo "ALL_DONE" >> /home/nacsi/microseg_runs/pilot-neg025-s200-augoff-20261005.train.log