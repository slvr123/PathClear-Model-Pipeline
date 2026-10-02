"""MicroSeg 5-class taxonomy for ESP32-CAM lightweight semantic segmentation.

Source of truth for class IDs, Cityscapes remapping, and visualization colors.
No TensorFlow dependency — safe to import anywhere including unit tests.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

CLASS_NAMES = ("Background", "Walkable", "Non-walkable", "Boundary", "Crosswalk")
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = 5
IGNORE_LABEL = 255  # Excluded from loss and metrics

# ---------------------------------------------------------------------------
# Cityscapes trainId → MicroSeg ID remapping
# ---------------------------------------------------------------------------
# Keys are Cityscapes trainIds (0-18) from *_labelTrainIds.png masks.
# Any trainId absent from this dict maps to IGNORE_LABEL by default.
#
# Design rationale:
# - Road (0) → Non-walkable: vehicular surface, dangerous for pedestrians.
# - Sidewalk (1) → Walkable: primary safe pedestrian surface.
# - Wall (3), Fence (4) → Boundary: lateral navigation anchors.
# - All dynamic objects (person, vehicle) → Ignore: we model surfaces, not
#   moving agents. Dynamic obstacles are handled by the ToF reflex layer.
# - Vegetation, terrain, sky, structures → Ignore: learned from local data.

CITYSCAPES_TO_MICROSEG = {
    0:  2,    # road          → Non-walkable
    1:  1,    # sidewalk      → Walkable
    2:  255,  # building      → Ignore
    3:  3,    # wall          → Boundary
    4:  3,    # fence         → Boundary
    5:  255,  # pole          → Ignore
    6:  255,  # traffic light → Ignore
    7:  255,  # traffic sign  → Ignore
    8:  255,  # vegetation    → Ignore
    9:  255,  # terrain       → Ignore
    10: 255,  # sky           → Ignore
    11: 255,  # person        → Ignore
    12: 255,  # rider         → Ignore
    13: 255,  # car           → Ignore
    14: 255,  # truck         → Ignore
    15: 255,  # bus           → Ignore
    16: 255,  # train         → Ignore
    17: 255,  # motorcycle    → Ignore
    18: 255,  # bicycle       → Ignore
}

# ---------------------------------------------------------------------------
# Visualization colors  (RGB)
# ---------------------------------------------------------------------------

COLORS = np.array([
    ( 64,  64,  64),   # 0  Background   — dark grey
    (144, 238, 144),   # 1  Walkable      — light green
    (220,  50,  50),   # 2  Non-walkable  — red
    (255, 165,   0),   # 3  Boundary      — orange
    (100, 149, 237),   # 4  Crosswalk     — cornflower blue
], dtype=np.uint8)

# Firmware-facing constant: IDs that the ESP32 inference loop actively uses.
ACTIVE_CLASS_IDS = tuple(range(NUM_CLASSES))


# ---------------------------------------------------------------------------
# Lookup table helpers
# ---------------------------------------------------------------------------

def cityscapes_lookup() -> np.ndarray:
    """Return a 256-element uint8 LUT: Cityscapes trainId → MicroSeg class ID.

    Usage:
        lut = cityscapes_lookup()
        microseg_mask = lut[cityscapes_train_id_mask]
    """
    lut = np.full(256, IGNORE_LABEL, dtype=np.uint8)
    for src, dst in CITYSCAPES_TO_MICROSEG.items():
        lut[src] = dst
    return lut


def remap_cityscapes(mask: np.ndarray) -> np.ndarray:
    """Apply cityscapes_lookup to an integer mask array."""
    mask = np.asarray(mask)
    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError("Masks must contain integer class IDs, not float values.")
    valid = (mask >= 0) & (mask < 256)
    return np.where(valid, cityscapes_lookup()[np.clip(mask, 0, 255)], IGNORE_LABEL).astype(np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Return an H×W×3 uint8 RGB image from an H×W integer class-ID mask."""
    mask = np.asarray(mask)
    valid = (mask >= 0) & (mask < NUM_CLASSES)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[valid] = COLORS[mask[valid]]
    return rgb
