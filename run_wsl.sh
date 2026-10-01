#!/usr/bin/env bash
# Run the existing scripts with the separate Linux TensorFlow environment.
set -euo pipefail
venv_path="${PATHCLEAR_VENV:-$HOME/venvs/pathclear}"
source "$venv_path/bin/activate"
site_packages="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export LD_LIBRARY_PATH="$site_packages/nvidia/cusolver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$site_packages/nvidia/cuda_nvcc/bin:$PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PATHCLEAR_BATCH_SIZE="${PATHCLEAR_BATCH_SIZE:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")"
exec python "$@"
