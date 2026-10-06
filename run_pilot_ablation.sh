#!/usr/bin/env bash
cd /mnt/c/Users/ALLAN/dev/PathClear-Model-Pipeline || exit 1
exec bash run_wsl.sh -m microseg.train \
  --epochs 5 \
  --batch-size 8 \
  --steps-per-epoch 800 \
  --crosswalk-weight 0.5 \
  --crosswalk-real-weight 0.5 \
  --crosswalk-negative-weight 0.0 \
  --crosswalk-horizon-fraction 0.45 \
  --augment \
  --cityscapes-score-weight 0.5 \
  --frozen-metric-patience 3 \
  --cityscapes-manifest /home/nacsi/microseg_runs/pilot_groupnorm_20261004/cityscapes_pairs.json \
  --split-manifest /home/nacsi/microseg_runs/crosswalk_split.json \
  --output-dir /home/nacsi/microseg_runs/pilot-nohorizon-20261005