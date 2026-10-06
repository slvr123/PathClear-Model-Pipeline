#!/usr/bin/env bash
# Run the MicroSeg dataset+train unit tests detached (survives wsl.exe exit).
cd /mnt/c/Users/ALLAN/dev/PathClear-Model-Pipeline || exit 1
log=/mnt/c/Users/ALLAN/dev/PathClear-Model-Pipeline/.verify.txt
: > "$log"
export PYTHONUNBUFFERED=1
timeout 1500 bash run_wsl.sh -m unittest tests.test_microseg_dataset tests.test_microseg_train >> "$log" 2>&1
echo "EXIT=$?" >> "$log"