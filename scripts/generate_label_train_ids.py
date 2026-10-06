"""Generate Cityscapes labelTrainIds masks from the distributed labelIds masks.

The local Cityscapes copy ships only ``*_gtFine_labelIds.png`` files, but the
Fast-SCNN loader expects ``*_gtFine_labelTrainIds.png``. This script derives
them with the official Cityscapes labelId -> trainId table (pure LUT remap,
no interpolation), writing each output next to its source. Existing outputs
are never overwritten, so the script is idempotent.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_config import CITYSCAPES_ROOT

# Official Cityscapes mapping, from the released labels definition:
# road(7)->0, sidewalk(8)->1, building(11)->2, wall(12)->3, fence(13)->4,
# pole(17)->5, traffic light(19)->6, traffic sign(20)->7, vegetation(21)->8,
# terrain(22)->9, sky(23)->10, person(24)->11, rider(25)->12, car(26)->13,
# truck(27)->14, bus(28)->15, train(31)->16, motorcycle(32)->17,
# bicycle(33)->18. Everything else (including void) is 255.
_LABEL_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    31: 16, 32: 17, 33: 18,
}
_VALID_TRAIN_IDS = frozenset(range(19)) | {255}


def label_to_train_lookup():
    """Return the 256-entry labelId -> trainId table (default 255)."""
    lookup = np.full(256, 255, dtype=np.uint8)
    for source, target in _LABEL_TO_TRAIN.items():
        lookup[source] = target
    return lookup


def convert_split(root, split, lookup, dry_run=False):
    """Convert every labelIds mask in a split; return (written, skipped)."""
    mask_dir = Path(root) / "gtFine" / split
    sources = sorted(mask_dir.rglob("*_gtFine_labelIds.png"))
    written, skipped = 0, 0
    for source in sources:
        target = source.with_name(
            source.name.replace("_labelIds.png", "_labelTrainIds.png")
        )
        if target.exists():
            skipped += 1
            continue
        if dry_run:
            written += 1
            continue
        label_ids = np.asarray(Image.open(source))
        if label_ids.dtype != np.uint8:
            raise ValueError(f"Expected 8-bit mask: {source}")
        train_ids = lookup[label_ids]
        Image.fromarray(train_ids).save(target)
        written += 1
    return written, skipped


def verify_split(root, split):
    """Check every generated file holds only 0..18 or 255."""
    mask_dir = Path(root) / "gtFine" / split
    generated = sorted(mask_dir.rglob("*_gtFine_labelTrainIds.png"))
    checked, bad = 0, 0
    for path in generated:
        ids = np.asarray(Image.open(path))
        if ids.dtype != np.uint8 or not set(np.unique(ids)) <= _VALID_TRAIN_IDS:
            print(f"BAD: {path}")
            bad += 1
        checked += 1
    return checked, bad


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--split", nargs="*", default=["train", "val"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else Path(CITYSCAPES_ROOT)
    lookup = label_to_train_lookup()
    total_written, total_skipped = 0, 0
    for split in args.split:
        if args.verify_only:
            checked, bad = verify_split(root, split)
            print(f"[{split}] checked={checked} bad={bad}")
            if bad:
                raise SystemExit(1)
            continue
        written, skipped = convert_split(root, split, lookup, args.dry_run)
        print(f"[{split}] written={written} skipped_existing={skipped}")
        total_written += written
        total_skipped += skipped
    if not args.verify_only:
        for split in args.split:
            checked, bad = verify_split(root, split)
            print(f"[{split}] verify checked={checked} bad={bad}")
            if bad:
                raise SystemExit(1)
    print(f"done written={total_written} skipped={total_skipped}")


if __name__ == "__main__":
    main()

