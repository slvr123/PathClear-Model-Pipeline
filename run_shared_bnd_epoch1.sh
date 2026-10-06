#!/bin/bash
# One-epoch PathClear adaptation under the shared Boundary map (buildings
# included). CPU fallback; expect hours. Run detached via run_wsl.sh -u.
set -euo pipefail
cd /mnt/c/Users/ALLAN/dev/PathClear-Model-Pipeline
run_dir=/home/nacsi/pathclear_runs/shared_bnd_epoch1_20261006
log=${run_dir}.train.log
rm -rf "$run_dir"
bash run_wsl.sh -u scripts/train_pathclear_cityscapes.py \
  --mode full --allow-cpu \
  --output-dir "$run_dir" >> "$log" 2>&1
echo "EXIT=$?" >> "$log"
