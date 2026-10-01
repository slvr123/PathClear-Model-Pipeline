# WSL GPU setup

I kept TensorFlow 2.21.0 and the existing Windows `.venv`.
The GPU detection failure came from the shared-library search path: TensorFlow could not find `libcusolver.so.11`, although the installed library and its dependencies loaded by absolute path.
Adding the installed cuSOLVER directory to `LD_LIBRARY_PATH` restored GPU detection and a GPU matrix multiplication.

I use the separate Linux environment at `/home/asus/venvs/pathclear` through this launcher:

```bash
cd /mnt/c/Users/Asus/OneDrive/Desktop/Code/PathClear
bash run_wsl.sh test_pipeline.py
bash run_wsl.sh -u train.py
```

The launcher also makes the installed CUDA compiler tools available on `PATH`, enables incremental GPU memory allocation, and defaults to batch size 1.
These settings apply to the launched process.
Ordinary Python commands outside the launcher do not receive its library-path fix.
I installed Matplotlib and Pillow in the Linux environment for the existing loader imports.

I retained the 512 x 1024 resolution and the existing model, forward-pass script, and one-batch training logic.
Windows defaults remain `C:/datasets/cityscapes` and batch size 2.
Linux defaults to `/mnt/c/datasets/cityscapes`.
I can override the dataset location with `CITYSCAPES_ROOT` and batch size with `PATHCLEAR_BATCH_SIZE`.

This remains a sanity check, not full training or held-out evaluation.
The existing MeanIoU warning about float weights and integer metric state still needs attention before full training.

## Verification on 2026-09-12

I verified all 2,975 training image/mask pairs were found with no missing masks.
The forward pass produced float32 logits shaped `(1, 512, 1024, 19)`.
The one-batch training check completed with loss `2.870524`, mIoU `0.016118`, pixel accuracy `0.059429`, and maximum weight change `0.000999990850687027`.
TensorFlow reported an XLA-compiled CUDA cluster, and an NVIDIA-SMI sample during training showed 36% GPU utilization.
The detected GPU name is RTX 3050 Ti Laptop GPU with 4 GB VRAM.
I also checked the Windows loader still imports with its original path and batch size 2.
File hashes confirmed `fast_scnn.py`, `test_pipeline.py`, and `train.py` stayed unchanged.
