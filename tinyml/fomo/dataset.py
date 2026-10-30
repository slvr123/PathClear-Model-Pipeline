"""Read reviewed object boxes and derive center-cell targets, without TensorFlow."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from config import CLASS_NAMES, GRID_CUT_POINTS, SETTINGS


def load_rgb(path):
    with Image.open(path) as image:
        image.load()
        # Explicit orientation policy: do not rotate based on EXIF metadata.
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def preprocess(frame):
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Frame must be H x W x 3 RGB uint8, with EXIF already resolved.")
    return np.asarray(Image.fromarray(frame).resize(
        (SETTINGS.image_size, SETTINGS.image_size), Image.Resampling.BILINEAR),
        dtype=np.float32) / 255.0


def resolve_path(value, base=None):
    """Allow a Windows-authored manifest to be read by the existing WSL environment."""
    value = str(value).replace("\\", "/")
    if os.name != "nt" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + value[2:]
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        value = value[5].upper() + ":/" + value[7:]
    path = Path(value)
    return (Path(base) / path).resolve() if base and not path.is_absolute() else path.resolve()


def centroid_target(annotations, width, height, ignore_cells=None, grid_size=SETTINGS.grid_size):
    if grid_size not in GRID_CUT_POINTS:
        raise ValueError("Supported output grids are 20 and 40")
    target = np.zeros((grid_size, grid_size), dtype=np.int32)
    for row, col in ignore_cells or []:
        target[row, col] = -1
    occupied = {}
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        col = int((x + w / 2) / width * grid_size)
        row = int((y + h / 2) / height * grid_size)
        cell = (row, col)
        channel = annotation["class_id"] + 1
        if cell in occupied and occupied[cell] != channel:
            occupied[cell] = -1  # Mixed classes cannot both be learned by one cell.
        else:
            occupied[cell] = channel
        target[row, col] = occupied[cell]
    return target


def inspect_manifest(manifest_path, verify_sources=False):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    counts = Counter()
    splits = Counter()
    split_classes = {s: Counter() for s in ("train", "validation", "test")}
    valid = []
    seen_ids, seen_paths, group_splits, content_splits = set(), set(), {}, {}
    records = manifest.get("images", [])
    if manifest.get("schema_version") not in (1, 2):
        errors.append("schema_version must be 1 or 2")
    if manifest.get("class_names") != list(CLASS_NAMES):
        errors.append(f"class_names must be exactly {list(CLASS_NAMES)}")
    if not isinstance(manifest.get("synthetic"), bool) or not isinstance(manifest.get("reviewed"), bool):
        errors.append("synthetic and reviewed must be explicit booleans")
    if not isinstance(records, list) or not records:
        errors.append("images must be a nonempty list")
        records = []
    annotation_count = 0
    collisions = Counter()
    cache_arrays = {}
    empty_images = 0
    missing_images = broken_images = invalid_annotations = 0
    for index, record in enumerate(records):
        label = f"images[{index}]"
        before = len(errors)
        try:
            if not isinstance(record, dict):
                raise ValueError("image record must be an object")
            image_id, group, split = record["id"], record["group"], record["split"]
            if not all(isinstance(v, str) and v.strip() for v in (image_id, group)):
                raise ValueError("id and group must be nonempty strings")
            if split not in split_classes:
                raise ValueError("split must be train, validation, or test")
            splits[split] += 1
            if image_id in seen_ids:
                errors.append(f"{label}: duplicate id {image_id}")
            seen_ids.add(image_id)
            if group in group_splits and group_splits[group] != split:
                errors.append(f"{label}: group {group!r} occurs across splits")
            group_splits[group] = split
            file = record["file"]
            if not isinstance(file, str) or not file:
                raise ValueError("file must be a nonempty path")
            image_path = resolve_path(file, path.parent)
            if image_path in seen_paths:
                errors.append(f"{label}: duplicate image path")
            seen_paths.add(image_path)
            if not image_path.is_file():
                missing_images += 1
                raise ValueError(f"missing image: {image_path}")
            try:
                if "cache_file" in record:
                    cache_path = resolve_path(record["cache_file"], path.parent)
                    if cache_path not in cache_arrays:
                        cache_arrays[cache_path] = np.load(cache_path, mmap_mode="r", allow_pickle=False)
                    cache = cache_arrays[cache_path]
                    if cache.dtype != np.uint8 or cache.shape[1:] != (160, 160, 3):
                        raise ValueError("Invalid cached input shape/type")
                    if type(record["cache_index"]) is not int or not 0 <= record["cache_index"] < len(cache):
                        raise ValueError("Invalid cache index")
                    width, height = record["width"], record["height"]
                    fingerprint = record["pixel_sha256"]
                    for src, expected in ((image_path, record["source_image_stat"]),
                                           (resolve_path(record["source_mask"]), record["source_mask_stat"])):
                        stat = src.stat()
                        if [stat.st_size, stat.st_mtime_ns] != expected:
                            raise ValueError(f"Source changed since conversion: {src}")
                    if verify_sources:
                        rgb = load_rgb(image_path)
                        if rgb.shape[:2] != (height, width) or hashlib.sha256(rgb.tobytes()).hexdigest() != fingerprint:
                            raise ValueError("Source pixels differ from conversion record")
                        if not np.array_equal(np.rint(preprocess(rgb)*255).astype(np.uint8), cache[record["cache_index"]]):
                            raise ValueError("Cached resized pixels differ from source preprocessing")
                else:
                    rgb = load_rgb(image_path)
                    height, width = rgb.shape[:2]
                    fingerprint = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
            except (OSError, ValueError, KeyError) as exc:
                broken_images += 1
                raise ValueError(f"broken image/cache: {exc}") from exc
            if fingerprint in content_splits and content_splits[fingerprint] != split:
                errors.append(f"{label}: duplicate image content across splits")
            content_splits[fingerprint] = split
            annotations = record["annotations"]
            if not isinstance(annotations, list):
                raise ValueError("annotations must be a list, empty only for reviewed negatives")
            annotation_count += len(annotations)
            empty_images += not annotations
            for number, annotation in enumerate(annotations):
                try:
                    class_id = annotation["class_id"]
                    if type(class_id) is not int or not 0 <= class_id < len(CLASS_NAMES):
                        raise ValueError(f"class_id must be an integer in 0..{len(CLASS_NAMES)-1}")
                    box = annotation["bbox"]
                    if not isinstance(box, list) or len(box) != 4 or not all(
                        type(v) in (int, float) and math.isfinite(v) for v in box
                    ):
                        raise ValueError("bbox must contain four finite numbers [x,y,width,height]")
                    x, y, w, h = box
                    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                        raise ValueError(f"bbox lies outside {width}x{height} or has nonpositive size")
                    counts[CLASS_NAMES[class_id]] += 1
                    split_classes[split][CLASS_NAMES[class_id]] += 1
                except (KeyError, TypeError, ValueError) as exc:
                    invalid_annotations += 1
                    errors.append(f"{label} annotation {number}: {exc}")
            if len(errors) == before:
                try:
                    ignore_cells = record.get("ignore_cells", [])
                    target_ignore_cells = record.get("target_ignore_cells", ignore_cells)
                    grid_size = record.get("target_grid_size", SETTINGS.grid_size)
                    if grid_size not in GRID_CUT_POINTS or (grid_size != 20 and "target_ignore_cells" not in record):
                        raise ValueError("A finer target grid requires regenerated target_ignore_cells")
                    for cells, size in ((ignore_cells, 20), (target_ignore_cells, grid_size)):
                        if any(not isinstance(c, list) or len(c) != 2 or any(type(v) is not int or not 0 <= v < size for v in c) for c in cells):
                            raise ValueError("ignore_cells must be valid [row,column] pairs for their grid")
                    target = centroid_target(annotations, width, height, target_ignore_cells, grid_size)
                    occupied = Counter((int((a["bbox"][1]+a["bbox"][3]/2)/height*grid_size),
                                        int((a["bbox"][0]+a["bbox"][2]/2)/width*grid_size)) for a in annotations)
                    collision_cells = [cell for cell, n in occupied.items() if n > 1]
                    collisions["images_with_collisions"] += bool(collision_cells)
                    collisions["collision_cells"] += len(collision_cells)
                    collisions["extra_centers_in_shared_cells"] += sum(occupied[c]-1 for c in collision_cells)
                    collisions["mixed_class_cells_ignored"] += sum(target[c] == -1 for c in collision_cells)
                    collisions["centers_in_mixed_cells"] += sum(occupied[c] for c in collision_cells if target[c] == -1)
                except ValueError as exc:
                    errors.append(f"{label}: {exc}")
            if len(errors) == before:
                valid.append(dict(record, path=image_path, width=width, height=height,
                                  manifest_dir=path.parent))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{label}: {exc}")
    report = {"images": len(records), "annotations": annotation_count,
              "classes": {c: counts[c] for c in CLASS_NAMES},
              "splits": {s: splits[s] for s in split_classes},
              "split_classes": {s: {c: v[c] for c in CLASS_NAMES} for s, v in split_classes.items()},
              "missing_images": missing_images, "broken_images": broken_images,
              "invalid_annotations": invalid_annotations, "errors": errors,
              "source_images": manifest.get("source_counts", len(records)),
              "converted_images": len(valid), "images_without_relevant_objects": empty_images,
              "average_objects_per_image": annotation_count / len(records) if records else 0,
              "grid_diagnostics": {k: int(v) for k, v in collisions.items()},
              "synthetic": manifest.get("synthetic"), "reviewed": manifest.get("reviewed")}
    return manifest, valid, report


def validated_records(manifest_path, allow_synthetic=False):
    manifest, records, report = inspect_manifest(manifest_path)
    if report["errors"]:
        raise ValueError("Dataset invalid:\n" + "\n".join(report["errors"]))
    if not manifest["reviewed"]:
        review_path = Path(manifest_path).resolve().parent / "review.json"
        review = json.loads(review_path.read_text()) if review_path.is_file() else {}
        digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        if not review.get("visual_review_note") or review.get("manifest_sha256") != digest or not review.get("valid"):
            raise ValueError("Dataset must be inspected and visually reviewed before training/evaluation/export.")
    if manifest["synthetic"] and not allow_synthetic:
        raise ValueError("Synthetic data is allowed only with the explicit sanity/synthetic flag.")
    for split in ("train", "validation"):
        if not report["splits"][split]:
            raise ValueError(f"Missing {split} split; define session-separated splits first.")
    missing_classes = [c for c in CLASS_NAMES if not report["split_classes"]["train"][c]]
    if missing_classes:
        raise ValueError(f"Training split lacks classes: {missing_classes}")
    return records, report


def batch_arrays(records):
    images, targets = load_arrays(records)
    return images.astype(np.float32)/255.0, targets


def load_arrays(records, grid_size=None):
    """Small uint8 input cache avoids decoding 2-megapixel originals every epoch."""
    caches, images, targets = {}, [], []
    sizes = {r.get("target_grid_size", SETTINGS.grid_size) for r in records}
    if len(sizes) != 1 or (grid_size is not None and sizes != {grid_size}):
        raise ValueError("Target grid differs from requested model; regenerate masks, never resize cached ignore cells")
    grid_size = sizes.pop()
    for record in records:
        if "cache_file" in record:
            path = resolve_path(record["cache_file"], record["manifest_dir"])
            if path not in caches:
                caches[path] = np.load(path, mmap_mode="r", allow_pickle=False)
            images.append(caches[path][record["cache_index"]])
        else:
            images.append(np.rint(preprocess(load_rgb(record["path"]))*255).astype(np.uint8))
        targets.append(centroid_target(record["annotations"], record["width"], record["height"],
                                       record.get("target_ignore_cells", record.get("ignore_cells")), grid_size))
    return np.stack(images), np.stack(targets)
