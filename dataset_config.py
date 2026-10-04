"""Shared Cityscapes location for future PathClear training scripts."""

import os
from pathlib import Path

DEFAULT_CITYSCAPES_ROOT = (
    "C:/datasets/cityscapes" if os.name == "nt" else "/mnt/c/datasets/cityscapes"
)
CITYSCAPES_ROOT = Path(os.environ.get("CITYSCAPES_ROOT", DEFAULT_CITYSCAPES_ROOT))

DEFAULT_KAGGLE_CROSSWALK_ROOT = (
    "C:/datasets/crosswalks" if os.name == "nt" else "/mnt/c/datasets/crosswalks"
)
KAGGLE_CROSSWALK_ROOT = Path(
    os.environ.get("KAGGLE_CROSSWALK_ROOT", DEFAULT_KAGGLE_CROSSWALK_ROOT)
)


def validate_cityscapes():
    """Return the root once the training and validation folders exist."""
    required = [CITYSCAPES_ROOT / kind / split
                for kind in ("leftImg8bit", "gtFine")
                for split in ("train", "val")]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Cityscapes needs extraction. Missing: " + ", ".join(missing))
    return CITYSCAPES_ROOT
