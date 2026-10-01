# My local dataset setup

I keep Cityscapes outside the repository at `C:\datasets\cityscapes`.
I do not commit its images, masks, archives, or local PathClear photographs.

```text
C:\datasets\cityscapes\
  archives/
  leftImg8bit/
    train/<city>/*_leftImg8bit.png
    val/<city>/*_leftImg8bit.png
    test/<city>/*_leftImg8bit.png
  gtFine/
    train/<city>/*_gtFine_labelTrainIds.png
    val/<city>/*_gtFine_labelTrainIds.png
    test/
```

I expect 2,975 training pairs and 500 validation pairs in this prepared dataset.
The loader uses `labelTrainIds`, with classes 0 through 18 and ignore ID 255.
Cityscapes test labels are withheld; I do not expect usable test training masks merely because `gtFine/test` exists.
`validate_cityscapes()` checks the four train/validation directories; `collect_pairs()` checks actual image/mask pairing.

I configure the root through `CITYSCAPES_ROOT`, which `dataset_config.py` reads and `dataset_loader.py` imports.
The defaults are `C:/datasets/cityscapes` on Windows and `/mnt/c/datasets/cityscapes` in WSL.
I set the variable before launching Python when using another location.
`.env.example` documents the setting; Python does not automatically load a `.env` file.

```powershell
$env:CITYSCAPES_ROOT = 'C:/datasets/cityscapes'
```

```bash
export CITYSCAPES_ROOT=/mnt/c/datasets/cityscapes
bash run_wsl.sh tests/test_pipeline.py
```

I use that command for one real batch and a forward pass, without training or checkpoint writes.
The existing WSL Python environment is a local prerequisite and is not included in Git.
I follow [WSL_GPU.md](WSL_GPU.md) for its setup.
`scripts/inspect_cityscapes.py` remains a Windows-specific visual diagnostic with a hardcoded example image.

For Philippine data, I keep the root configurable through the existing pilot manifest/path arguments rather than adding a machine-specific path to source code.
I follow [LOCAL_DATASET_PLAN.md](LOCAL_DATASET_PLAN.md) and the [annotation guide](PATHCLEAR_ANNOTATION_GUIDE.md), preserving route/session splits.
I keep local manifests and images outside version control and share reviewed data separately under the applicable dataset permissions.
