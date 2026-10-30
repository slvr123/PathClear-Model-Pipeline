"""Derive Person/Vehicle boxes from original Cityscapes instance IDs, read-only sources."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from config import CLASS_NAMES, CITYSCAPES_CLASS_MAP, SETTINGS, output_directory
from dataset import preprocess

# Cityscapes official label IDs, not train IDs. Groups use the plain semantic ID.
VOID_IDS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 255)


def center_ignore_cells(mask, grid_size=SETTINGS.grid_size):
    """Ignore unknown semantic labels at prediction centers, not anywhere in a cell."""
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or grid_size < 1:
        raise ValueError("Expected an integer mask and positive grid size")
    height, width = mask.shape
    rows = np.floor((np.arange(grid_size)+.5)*height/grid_size).astype(int)
    cols = np.floor((np.arange(grid_size)+.5)*width/grid_size).astype(int)
    centers = mask[np.ix_(rows, cols)]
    ignored = np.isin(centers, list(VOID_IDS)+list(CITYSCAPES_CLASS_MAP)) | np.isin(centers//1000, (29, 30))
    return np.argwhere(ignored).tolist()


def extract_instances(mask):
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("instanceIds must be a two-dimensional integer image")
    height, width = mask.shape
    ys, xs = np.nonzero(mask >= 1000)
    values = mask[ys, xs]
    ids, inverse, sizes = np.unique(values, return_inverse=True, return_counts=True)
    xmin, ymin = np.full(len(ids), width), np.full(len(ids), height)
    xmax, ymax = np.zeros(len(ids), dtype=int), np.zeros(len(ids), dtype=int)
    np.minimum.at(xmin, inverse, xs)
    np.maximum.at(xmax, inverse, xs)
    np.minimum.at(ymin, inverse, ys)
    np.maximum.at(ymax, inverse, ys)
    annotations, excluded = [], Counter()
    for index, value in enumerate(ids):
        value = int(value)
        semantic, instance = divmod(value, 1000)
        if semantic not in CITYSCAPES_CLASS_MAP:
            excluded["unrelated_individual_instances"] += 1
            continue
        annotations.append({"class_id": CITYSCAPES_CLASS_MAP[semantic],
                            "bbox": [int(xmin[index]), int(ymin[index]),
                                     int(xmax[index]-xmin[index]+1), int(ymax[index]-ymin[index]+1)],
                            "source_instance_id": value, "source_semantic_id": semantic,
                            "source_instance_number": instance, "visible_pixels": int(sizes[index])})
    # No guessed boxes for unresolved groups. Ignore their pixels, plus official void areas.
    group = np.isin(mask, list(CITYSCAPES_CLASS_MAP))
    void = np.isin(mask, VOID_IDS) | np.isin(mask // 1000, (29, 30))
    ignored_y, ignored_x = np.nonzero(group | void)
    g = SETTINGS.grid_size
    cells = np.unique((ignored_y * g // height) * g + ignored_x * g // width)
    excluded["group_pixels"] = int(group.sum())
    excluded["void_pixels"] = int(void.sum())
    excluded["group_semantic_regions"] = sum(bool(np.any(mask == v)) for v in CITYSCAPES_CLASS_MAP)
    return annotations, [[int(c // g), int(c % g)] for c in cells], dict(excluded)


def convert_one(job):
    index, split, image_path, mask_path = job
    try:
        image_stat, mask_stat = image_path.stat(), mask_path.stat()
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
        height, width = rgb.shape[:2]
        if mask.shape != (height, width):
            raise ValueError("Image and instance mask dimensions differ")
        annotations, ignore_cells, excluded = extract_instances(mask)
        image_id = image_path.name.removesuffix("_leftImg8bit.png")
        city, sequence = image_id.rsplit("_", 2)[:2]
        record = {"id": image_id, "file": image_path.as_posix(), "split": split,
                  "group": f"{city}-{sequence}", "width": width, "height": height,
                  "annotations": annotations, "ignore_cells": ignore_cells,
                  "cache_file": f"{split}.rgb.npy", "cache_index": index,
                  "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                  "source_mask": mask_path.as_posix(),
                  "source_image_stat": [image_stat.st_size, image_stat.st_mtime_ns],
                  "source_mask_stat": [mask_stat.st_size, mask_stat.st_mtime_ns],
                  "conversion_stats": excluded}
        small = np.rint(preprocess(rgb) * 255).astype(np.uint8)
        return record, small, None
    except (OSError, ValueError) as exc:
        return None, None, {"image": str(image_path), "mask": str(mask_path), "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Existing Cityscapes root, read-only")
    parser.add_argument("--output", help="Fresh artifacts directory; original images are referenced, not copied")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Optional bounded sample count per split")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    source = args.source.resolve()
    if not (source / "leftImg8bit").is_dir() or not (source / "gtFine").is_dir():
        parser.error("source must contain leftImg8bit/ and gtFine/")
    jobs_by_split, source_counts = {}, {}
    for source_split, target_split in (("train", "train"), ("val", "validation")):
        images = sorted((source / "leftImg8bit" / source_split).glob("*/*_leftImg8bit.png"))
        source_counts[target_split] = len(images)
        if not images:
            parser.error(f"No source images found in {source_split}")
        if args.limit:
            # Spread the pilot across the split rather than use one city's first frames.
            indices = np.linspace(0, len(images)-1, min(args.limit, len(images)), dtype=int)
            images = [images[i] for i in indices]
        jobs_by_split[target_split] = [(i, target_split, path,
            source / "gtFine" / source_split / path.parent.name /
            path.name.replace("_leftImg8bit.png", "_gtFine_instanceIds.png")) for i, path in enumerate(images)]
    directory = output_directory(args.output, "cityscapes")
    records, failures, totals = [], [], Counter()
    started = time.perf_counter()
    for split, jobs in jobs_by_split.items():
        cache = np.lib.format.open_memmap(directory / f"{split}.rgb.npy", mode="w+", dtype=np.uint8,
                                          shape=(len(jobs), SETTINGS.image_size, SETTINGS.image_size, 3))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for number, (record, pixels, error) in enumerate(executor.map(convert_one, jobs), start=1):
                if error:
                    failures.append(error)
                else:
                    cache[record["cache_index"]] = pixels
                    records.append(record)
                    totals.update(record["conversion_stats"])
                if number % 100 == 0 or number == len(jobs):
                    print(f"{split}: {number}/{len(jobs)} images, {len(failures)} errors", flush=True)
        cache.flush()
        del cache
    manifest = {"schema_version": 2, "class_names": list(CLASS_NAMES), "synthetic": False,
                "reviewed": False, "source": "Cityscapes fine instanceIds, original label IDs",
                "source_root": source.as_posix(), "source_counts": source_counts,
                "mapping": CITYSCAPES_CLASS_MAP, "split_policy": "official train and val retained; test labels withheld",
                "cache_preprocessing": "RGB uint8 bilinear direct resize 160x160; divide by255 at training",
                "images": records}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {"source_images": source_counts, "converted_images": len(records),
               "annotation_count": sum(len(r["annotations"]) for r in records),
               "classes": dict(Counter(CLASS_NAMES[a["class_id"]] for r in records for a in r["annotations"])),
               "images_without_relevant_objects": sum(not r["annotations"] for r in records),
               "invalid_missing_files": failures, "skipped": dict(totals),
               "seconds": time.perf_counter()-started, "manifest": str(directory / "manifest.json")}
    (directory / "conversion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
