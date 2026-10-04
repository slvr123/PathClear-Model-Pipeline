"""Rank Cityscapes images for human review of possible crosswalk regions.

This heuristic emits review candidates only. It never creates training masks
or treats a low score as evidence that an image contains no crosswalk.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_config import CITYSCAPES_ROOT
from microseg.dataset import collect_cityscapes_pairs

_DEFAULT_OUTPUT_ROOT = CITYSCAPES_ROOT / "microseg_review_candidates"
_MAX_ANALYSIS_SIZE = (1024, 512)
_MIN_LUMINANCE = 65
_ROAD_LUMINANCE_MARGIN = 18
_MAX_SATURATION = 140
_LEGACY_LUMINANCE_THRESHOLD = 155
_LEGACY_MAX_SATURATION = 90
_MIN_TRANSITIONS = 4
_MIN_TRANSVERSE_BANDS = 4
_MIN_ROW_ROAD_FRACTION = 0.12
_MIN_BRIGHT_ROW_FRACTION = 0.08
_DEFAULT_SCORE_THRESHOLD = 0.08


def _load_cityscapes_road_mask(mask_path, size):
    with Image.open(mask_path) as source:
        mask = source.convert("L")
        if mask_path.endswith("_labelIds.png"):
            road_value = 7
        elif mask_path.endswith("_labelTrainIds.png"):
            road_value = 0
        else:
            raise ValueError(f"Unrecognized Cityscapes mask filename: {mask_path}")
        mask = mask.resize(size, Image.Resampling.NEAREST)
        return np.asarray(mask) == road_value


def _to_analysis_image(image_path):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail(_MAX_ANALYSIS_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image)


def _repeated_marking_score(bright, road):
    """Measure the prevalence of road scanlines with repeated bright bands."""
    best_fraction = 0.0
    for axis in (0, 1):
        scan_bright = np.moveaxis(bright, axis, 0)
        scan_road = np.moveaxis(road, axis, 0)
        overlap = scan_road[1:] & scan_road[:-1]
        transitions = (
            (scan_bright[1:] != scan_bright[:-1])
            & overlap
        )
        transition_count = np.sum(transitions, axis=0)
        road_count = np.sum(scan_road, axis=0)
        eligible = road_count >= 8
        if not np.any(eligible):
            continue
        # Count scanlines that encounter several high-contrast bright bands.
        repeated = transition_count[eligible] >= _MIN_TRANSITIONS
        best_fraction = max(best_fraction, float(np.mean(repeated)))
    return best_fraction


def _transverse_band_score(bright, road):
    """Score repeated bright bands spanning a substantial part of road rows."""
    road_per_row = np.sum(road, axis=1)
    bright_fraction = np.sum(bright, axis=1) / np.maximum(road_per_row, 1)
    active_rows = (
        (road_per_row >= max(8, int(road.shape[1] * _MIN_ROW_ROAD_FRACTION)))
        & (bright_fraction >= _MIN_BRIGHT_ROW_FRACTION)
    )
    transitions = np.diff(np.concatenate(([False], active_rows, [False])).astype(np.int8))
    band_count = int(np.count_nonzero(transitions == 1))
    if band_count < _MIN_TRANSVERSE_BANDS:
        return 0.0
    coverage = float(np.max(bright_fraction))
    return min(1.0, band_count / 5.0) * min(1.0, coverage / 0.15)


def score_cityscapes_pair(image_path, mask_path):
    """Return transparent road-visibility and visual repeated-marking scores."""
    image = _to_analysis_image(image_path)
    height, width = image.shape[:2]
    road = _load_cityscapes_road_mask(mask_path, (width, height))
    road_pixels = int(np.count_nonzero(road))
    road_fraction = float(np.mean(road))
    if road_pixels == 0:
        return {
            "score": 0.0,
            "road_fraction": road_fraction,
            "bright_road_fraction": 0.0,
            "repeated_marking_score": 0.0,
        }

    rgb = image.astype(np.int32)
    luminance = (
        299 * rgb[..., 0] + 587 * rgb[..., 1] + 114 * rgb[..., 2]
    ) // 1000
    saturation_proxy = (
        np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
    )
    lower_road = road.copy()
    lower_road[: height // 3, :] = False
    if np.count_nonzero(lower_road) < 64:
        lower_road = road
    road_luminance = luminance[lower_road]
    adaptive_threshold = max(
        _MIN_LUMINANCE,
        int(np.median(road_luminance)) + _ROAD_LUMINANCE_MARGIN,
    )
    adaptive_bright_road = (
        lower_road
        & (luminance >= adaptive_threshold)
        & (saturation_proxy <= _MAX_SATURATION)
    )
    legacy_bright_road = (
        lower_road
        & (luminance >= _LEGACY_LUMINANCE_THRESHOLD)
        & (saturation_proxy <= _LEGACY_MAX_SATURATION)
    )
    bright_fraction = float(
        np.count_nonzero(adaptive_bright_road)
        / max(1, np.count_nonzero(lower_road))
    )
    legacy_bright_fraction = float(
        np.count_nonzero(legacy_bright_road)
        / max(1, np.count_nonzero(lower_road))
    )
    repeated_score = _repeated_marking_score(legacy_bright_road, lower_road)
    score = max(
        repeated_score * min(1.0, legacy_bright_fraction / 0.08),
        _transverse_band_score(adaptive_bright_road, lower_road),
    )
    return {
        "score": float(score),
        "road_fraction": road_fraction,
        "bright_road_fraction": bright_fraction,
        "repeated_marking_score": repeated_score,
    }


def select_candidates(pairs, root, score_threshold):
    """Score all images, marking which should enter the human-review queue."""
    root = Path(root)
    scored = []
    for image_path, mask_path in pairs:
        image = Path(image_path)
        metrics = score_cityscapes_pair(str(image), str(mask_path))
        record = {
            "image": image.relative_to(root).as_posix(),
            "mask": Path(mask_path).relative_to(root).as_posix(),
            **metrics,
        }
        record["candidate"] = bool(
            metrics["score"] >= score_threshold and metrics["road_fraction"] > 0
        )
        record["review_status"] = "pending"
        scored.append(record)
    scored.sort(key=lambda item: (-item["score"], item["image"]))
    return scored


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=_DEFAULT_SCORE_THRESHOLD,
        help="Minimum heuristic score for human review (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path (default: <Cityscapes root>/microseg_review_candidates/<split>.json)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    output = (
        args.output
        if args.output is not None
        else _DEFAULT_OUTPUT_ROOT / f"{args.split}.json"
    )
    image_paths, mask_paths = collect_cityscapes_pairs(args.split)
    pairs = list(zip(image_paths, mask_paths))
    scored_images = select_candidates(
        pairs, CITYSCAPES_ROOT, args.score_threshold
    )
    candidates = [item for item in scored_images if item["candidate"]]
    report = {
        "schema_version": 1,
        "dataset_root": str(CITYSCAPES_ROOT.resolve()),
        "split": args.split,
        "selector": "cityscapes-road-adaptive-transverse-bands-v2",
        "score_threshold": args.score_threshold,
        "total_images": len(pairs),
        "candidate_count": len(candidates),
        "rejected_count": len(scored_images) - len(candidates),
        "reviewed_count": 0,
        "confirmed_count": 0,
        "candidates": candidates,
        "scored_images": scored_images,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Cityscapes {args.split}: {len(candidates)} review candidates "
        f"of {len(pairs)} images (score threshold {args.score_threshold:.3f})"
    )
    print(f"Candidate list: {output}")


if __name__ == "__main__":
    main()
