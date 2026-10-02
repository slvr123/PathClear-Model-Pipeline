"""Shared dataset root configuration for MicroSeg training sources.

Set environment variables to override defaults:
    CITYSCAPES_ROOT        — path to Cityscapes dataset root
    KAGGLE_CROSSWALK_ROOT  — path to extracted FPV Crosswalk dataset root
    LOCAL_PILOT_ROOT       — path to PathClear local Philippine pilot dataset
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Cityscapes (primary source for Walkable, Non-walkable, Boundary)
# ---------------------------------------------------------------------------

_DEFAULT_CITYSCAPES = (
    "C:/datasets/cityscapes" if os.name == "nt" else "/mnt/c/datasets/cityscapes"
)
CITYSCAPES_ROOT = Path(os.environ.get("CITYSCAPES_ROOT", _DEFAULT_CITYSCAPES))


# ---------------------------------------------------------------------------
# Kaggle FPV Crosswalk dataset (sole source for Crosswalk class)
# Dataset: jocelyndumlao/fpvcrosswalk2025-first-person-view-crosswalk
# License: CC0: Public Domain
# ---------------------------------------------------------------------------

_DEFAULT_CROSSWALK = (
    "C:/datasets/fpv_crosswalk" if os.name == "nt" else "/mnt/c/datasets/fpv_crosswalk"
)
KAGGLE_CROSSWALK_ROOT = Path(os.environ.get("KAGGLE_CROSSWALK_ROOT", _DEFAULT_CROSSWALK))


# ---------------------------------------------------------------------------
# Local PathClear pilot (Philippine street footage, all 5 classes)
# ---------------------------------------------------------------------------

_DEFAULT_LOCAL = (
    "C:/datasets/pathclear_pilot" if os.name == "nt" else "/mnt/c/datasets/pathclear_pilot"
)
LOCAL_PILOT_ROOT = Path(os.environ.get("LOCAL_PILOT_ROOT", _DEFAULT_LOCAL))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_cityscapes() -> Path:
    """Raise FileNotFoundError if Cityscapes train/val folders are absent."""
    required = [
        CITYSCAPES_ROOT / kind / split
        for kind in ("leftImg8bit", "gtFine")
        for split in ("train", "val")
    ]
    missing = [str(p) for p in required if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Cityscapes extraction missing. Set CITYSCAPES_ROOT or check:\n"
            + "\n".join(missing)
        )
    return CITYSCAPES_ROOT


def validate_crosswalk() -> Path:
    """Raise FileNotFoundError if the Kaggle crosswalk root is absent."""
    if not KAGGLE_CROSSWALK_ROOT.is_dir():
        raise FileNotFoundError(
            f"Kaggle FPV Crosswalk dataset not found at {KAGGLE_CROSSWALK_ROOT}.\n"
            "Download from: https://www.kaggle.com/datasets/"
            "jocelyndumlao/fpvcrosswalk2025-first-person-view-crosswalk\n"
            "Then set KAGGLE_CROSSWALK_ROOT to the extracted folder."
        )
    return KAGGLE_CROSSWALK_ROOT
