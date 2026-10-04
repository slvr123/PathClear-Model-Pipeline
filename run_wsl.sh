#!/usr/bin/env bash
# Run the existing scripts with the separate Linux TensorFlow environment.
set -euo pipefail
venv_path="${PATHCLEAR_VENV:-$HOME/venvs/pathclear}"
source "$venv_path/bin/activate"
site_packages="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
nvidia_lib_dirs=()
for lib_dir in "$site_packages"/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        nvidia_lib_dirs+=("$lib_dir")
    fi
done
if ((${#nvidia_lib_dirs[@]})); then
    nvidia_lib_path="$(IFS=:; printf '%s' "${nvidia_lib_dirs[*]}")"
    export LD_LIBRARY_PATH="$nvidia_lib_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PATH="$site_packages/nvidia/cuda_nvcc/bin:$PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PATHCLEAR_BATCH_SIZE="${PATHCLEAR_BATCH_SIZE:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD:$PWD/scripts:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}"
# Resolve old bare script names so recorded experiment commands still work.
args=("$@")
for index in "${!args[@]}"; do
    arg="${args[$index]}"
    if [[ "$arg" == *.py && "$arg" != */* && ! -f "$arg" ]]; then
        for folder in scripts tests; do
            if [[ -f "$folder/$arg" ]]; then
                args[$index]="$folder/$arg"
                break
            fi
        done
        break
    fi
done
exec python "${args[@]}"
