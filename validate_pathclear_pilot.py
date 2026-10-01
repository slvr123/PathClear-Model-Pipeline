"""Check a local pilot manifest and native-resolution ID masks without training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pathclear_taxonomy import CLASS_NAMES, NUM_CLASSES, IGNORE_LABEL

SPLITS = ("train", "val", "test", "unassigned")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def contained_file(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Image/mask paths must be nonempty paths relative to the manifest.")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing file or path outside pilot root: {value}")
    return path


def validate_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("samples"), list):
        raise ValueError("Expected schema_version=1 and a samples list.")
    root = manifest_path.parent
    seen_ids, image_hashes, routes, sessions = set(), set(), {}, {}
    counts = {split: np.zeros(NUM_CLASSES, dtype=np.int64) for split in SPLITS}
    frames = {split: np.zeros(NUM_CLASSES, dtype=np.int64) for split in SPLITS}
    labeled = reviewed = ignored = 0
    split_counts = {split: 0 for split in SPLITS}
    for sample in manifest["samples"]:
        for field in ("sample_id", "image", "route_id", "session_id", "camera", "source"):
            if not isinstance(sample.get(field), str) or not sample[field].strip():
                raise ValueError(f"Each sample requires a nonempty {field}.")
        identity = sample["sample_id"]
        if identity in seen_ids:
            raise ValueError(f"Duplicate sample_id: {identity}")
        seen_ids.add(identity)
        split = sample.get("split")
        if split not in SPLITS:
            raise ValueError(f"Invalid split for {identity}: {split}")
        split_counts[split] += 1
        for group, key in ((routes, "route_id"), (sessions, "session_id")):
            value = sample[key]
            if value in group and group[value] != split:
                raise ValueError(f"Split leakage: {key} {value} appears in different splits.")
            group[value] = split
        image_path = contained_file(root, sample["image"])
        digest = sha256(image_path)
        if digest in image_hashes:
            raise ValueError(f"Duplicate image bytes: {identity}")
        image_hashes.add(digest)
        with Image.open(image_path) as image:
            image_size = image.size
            image.verify()
        if not isinstance(sample.get("reviewed"), bool):
            raise ValueError(f"reviewed must be true or false: {identity}")
        if sample["reviewed"] and not sample.get("reviewer", "").strip():
            raise ValueError(f"Reviewed masks need a reviewer: {identity}")
        if sample.get("mask") is None:
            if sample["reviewed"]:
                raise ValueError(f"A missing mask cannot be reviewed: {identity}")
            continue
        mask_path = contained_file(root, sample["mask"])
        with Image.open(mask_path) as image:
            if image.format != "PNG" or image.mode not in ("L", "P") or image.size != image_size:
                raise ValueError(f"Mask must be an 8-bit ID PNG matching image dimensions: {identity}")
            mask = np.array(image)
        values = np.unique(mask)
        if not np.all((values < NUM_CLASSES) | (values == IGNORE_LABEL)):
            raise ValueError(f"Invalid mask IDs for {identity}: {values.tolist()}")
        valid = mask != IGNORE_LABEL
        if not np.any(valid):
            raise ValueError(f"Mask has no labeled pixels: {identity}")
        pixels = np.bincount(mask[valid], minlength=NUM_CLASSES)
        counts[split] += pixels
        frames[split] += pixels > 0
        ignored += int(np.sum(~valid))
        labeled += 1
        reviewed += int(sample["reviewed"])
    missing = {split: [CLASS_NAMES[i] for i in range(NUM_CLASSES) if counts[split][i] == 0]
               for split in SPLITS[:3]}
    complete = bool(seen_ids) and labeled == reviewed == len(seen_ids) and not split_counts["unassigned"]
    complete = complete and all(not missing[split] for split in SPLITS[:3])
    return {
        "samples": len(seen_ids), "labeled": labeled, "reviewed": reviewed,
        "split_counts": split_counts, "ignored_pixels": ignored,
        "pixels_by_split": {s: dict(zip(CLASS_NAMES, counts[s].tolist())) for s in SPLITS},
        "frames_by_split": {s: dict(zip(CLASS_NAMES, frames[s].tolist())) for s in SPLITS},
        "missing_classes": missing, "coverage_and_review_checks_pass": bool(complete),
        "training_authorized": False,
        "note": "Presence and review checks are not proof of sufficient data or model quality.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_manifest(args.manifest), indent=2))
