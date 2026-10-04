"""MicroSeg four-class label IDs and remapping, independent of TensorFlow.

The five-class sketch in docs/MICROSEG_CUSTOM_CNN_PLAN.md collapses to four
classes: Background is left unsupervised, so its output channel is dropped and
the remaining PathClear classes become ignored supervision (ID 255).

IDs are zero-based and must stay consistent across source masks, model
channels, metrics, and any firmware integration.
"""
import numpy as np

from pathclear_taxonomy import (
    IGNORE_LABEL,
    cityscapes_lookup as pathclear_cityscapes_lookup,
)

CLASS_NAMES = ("Walkable", "Non-walkable", "Boundary", "Crosswalk")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Ten-class PathClear ID -> four-class MicroSeg ID.
# Anything absent (Person, Vehicle, Obstacle, Vegetation, Stairs, Uneven
# Pavement) keeps its ignored status.
PATHCLEAR_TO_MICROSEG = {
    1: 0,  # Sidewalk -> Walkable
    0: 1,  # Road -> Non-walkable
    4: 2,  # Boundary -> Boundary
    8: 3,  # Crosswalk -> Crosswalk
}

# Cityscapes original labelId -> MicroSeg ID. These are annotation IDs from
# *_gtFine_labelIds.png, not trainIds. All unsupported/void IDs stay ignored.
CITYSCAPES_LABEL_IDS_TO_MICROSEG = {
    7: 1,   # road -> Non-walkable
    8: 0,   # sidewalk -> Walkable
    12: 2,  # wall -> Boundary
    13: 2,  # fence -> Boundary
}

COLORS = np.array([
    (0, 180, 80),      # 0 Walkable
    (128, 64, 128),    # 1 Non-walkable
    (220, 20, 60),     # 2 Boundary
    (255, 255, 100),   # 3 Crosswalk
], dtype=np.uint8)


def microseg_lookup():
    lookup = np.full(256, IGNORE_LABEL, dtype=np.uint8)
    for source, target in PATHCLEAR_TO_MICROSEG.items():
        lookup[source] = target
    return lookup


def remap_pathclear_ids(mask):
    """Map ten-class PathClear IDs to four-class MicroSeg IDs.

    Unknown and ignored integer IDs stay ignored. Masks must contain integer
    IDs, not probabilities or RGB colors.
    """
    mask = np.asarray(mask)
    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError("Masks must contain integer IDs, not probabilities or RGB colors.")
    valid = (mask >= 0) & (mask < 256)
    return np.where(valid, microseg_lookup()[np.clip(mask, 0, 255)], IGNORE_LABEL).astype(np.uint8)


def cityscapes_lookup():
    """Compose the existing Cityscapes->PathClear table with the four-class step.

    The TensorFlow loader and :func:`remap_cityscapes_ids` share this one table.
    """
    return microseg_lookup()[pathclear_cityscapes_lookup()]


def cityscapes_label_ids_lookup():
    """Return a 256-entry LUT for original Cityscapes labelIds masks."""
    lookup = np.full(256, IGNORE_LABEL, dtype=np.uint8)
    for source, target in CITYSCAPES_LABEL_IDS_TO_MICROSEG.items():
        lookup[source] = target
    return lookup


def remap_cityscapes_ids(mask):
    """Map Cityscapes trainIds directly to the four MicroSeg classes."""
    mask = np.asarray(mask)
    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError("Masks must contain integer IDs, not probabilities or RGB colors.")
    valid = (mask >= 0) & (mask < 256)
    return np.where(valid, cityscapes_lookup()[np.clip(mask, 0, 255)], IGNORE_LABEL).astype(np.uint8)


def remap_cityscapes_label_ids(mask):
    """Map original Cityscapes labelIds to MicroSeg classes.

    Only road, sidewalk, wall, and fence have MicroSeg targets. All other
    original Cityscapes labels, including void and dynamic objects, are ignored.
    """
    mask = np.asarray(mask)
    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError("Masks must contain integer IDs, not probabilities or RGB colors.")
    valid = (mask >= 0) & (mask < 256)
    return np.where(
        valid,
        cityscapes_label_ids_lookup()[np.clip(mask, 0, 255)],
        IGNORE_LABEL,
    ).astype(np.uint8)


def colorize(mask):
    mask = np.asarray(mask)
    valid = (mask >= 0) & (mask < NUM_CLASSES)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[valid] = COLORS[mask[valid]]
    return rgb
