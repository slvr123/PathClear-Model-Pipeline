"""PathClear label IDs and Cityscapes trainId conversion, independent of TF."""
import numpy as np

CLASS_NAMES = (
    "Road", "Sidewalk", "Person", "Vehicle", "Boundary", "Obstacle",
    "Vegetation", "Stairs", "Crosswalk", "Uneven Pavement",
)
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_LABEL = 255
CITYSCAPES_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)
# These keys are Cityscapes trainIds, NOT the original labelIds.
# Boundary (ID 4) is building + wall + fence: facades are the dominant
# vertical surface and the shared lateral anchor for shorelining, matching
# the MicroSeg labelId mapping (building/wall/fence -> Boundary).
CITYSCAPES_TO_PATHCLEAR = {
    0: 0, 1: 1, 2: 4, 3: 4, 4: 4, 5: 5, 6: 5, 7: 5,
    8: 6, 9: 255, 10: 255, 11: 2, 12: 2, 13: 3, 14: 3,
    15: 3, 16: 3, 17: 3, 18: 3,
}
CITYSCAPES_SUPPORTED_IDS = tuple(range(7))
LOCAL_ONLY_IDS = (7, 8, 9)
COLORS = np.array([
    (128, 64, 128), (244, 35, 232), (220, 20, 60), (0, 90, 220),
    (130, 110, 80), (255, 170, 0), (65, 155, 60), (0, 205, 205),
    (255, 255, 100), (180, 95, 20),
], dtype=np.uint8)


def cityscapes_lookup():
    lookup = np.full(256, IGNORE_LABEL, dtype=np.uint8)
    for source, target in CITYSCAPES_TO_PATHCLEAR.items():
        lookup[source] = target
    return lookup


def remap_cityscapes_train_ids(mask):
    """Return a new mask; void and unknown integer IDs stay ignored."""
    mask = np.asarray(mask)
    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError("Masks must contain integer IDs, not probabilities or RGB colors.")
    valid = (mask >= 0) & (mask < 256)
    return np.where(valid, cityscapes_lookup()[np.clip(mask, 0, 255)], IGNORE_LABEL).astype(np.uint8)


def colorize(mask):
    mask = np.asarray(mask)
    valid = (mask >= 0) & (mask < NUM_CLASSES)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[valid] = COLORS[mask[valid]]
    return rgb
