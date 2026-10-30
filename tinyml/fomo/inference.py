"""Independent FOMO camera inference for downstream PathClear decision logic."""
import argparse
import json
import math
import time

import numpy as np

from config import CLASS_NAMES, GRID_CUT_POINTS, SETTINGS
from dataset import load_rgb, preprocess


DECODERS = ("cell_nms", "connected", "cells")


def decoding_policy(decoder=None, nms_radius_pixels=SETTINGS.nms_radius_pixels):
    selected = decoder or SETTINGS.decoder
    nms_enabled = selected == "cell_nms"
    return {"version": 3, "decoder": selected,
            "threshold": SETTINGS.confidence_threshold, "max_detections": SETTINGS.max_detections,
            "nms_enabled": nms_enabled,
            "nms_distance": "same class, Chebyshev distance in 160x160 input pixels" if nms_enabled else "disabled",
            "nms_radius_input_pixels": nms_radius_pixels if nms_enabled else 0.0,
            "consumer": "PathClear decision logic", "creates_classifier_crops": False}


def decode_grid(probabilities, frame_width, frame_height, threshold=None, max_detections=None,
                decoder=None, nms_radius_pixels=SETTINGS.nms_radius_pixels):
    """Decode center candidates; legacy connected merging remains explicitly selectable."""
    grid = np.asarray(probabilities)
    if grid.ndim != 3 or grid.shape[0] not in GRID_CUT_POINTS or grid.shape != (grid.shape[0], grid.shape[0], len(CLASS_NAMES)+1) or not np.isfinite(grid).all():
        raise ValueError("Expected finite 20x20x3 or 40x40x3 probabilities")
    g = grid.shape[0]
    if not math.isfinite(nms_radius_pixels) or nms_radius_pixels < 0:
        raise ValueError("NMS radius must be finite and nonnegative")
    nms_cells = math.floor(nms_radius_pixels*g/SETTINGS.image_size)
    if (grid < 0).any() or (grid > 1).any() or not np.allclose(grid.sum(axis=-1), 1, atol=1e-3):
        raise ValueError("Grid must contain softmax probabilities, not logits")
    threshold = SETTINGS.confidence_threshold if threshold is None else threshold
    limit = SETTINGS.max_detections if max_detections is None else max_detections
    decoder = SETTINGS.decoder if decoder is None else decoder
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}")
    if not 0 <= threshold <= 1 or not all(math.isfinite(v) and v > 0 for v in (frame_width, frame_height)):
        raise ValueError("Invalid threshold or frame dimensions")
    if type(limit) is not int or limit < 0:
        raise ValueError("max_detections must be a nonnegative integer (0 means uncapped)")
    winners = grid.argmax(axis=-1)
    detections = []
    if decoder != "connected":
        candidates = []
        for row, col in zip(*np.where((winners > 0) & (grid.max(axis=-1) >= threshold))):
            channel = int(winners[row, col])
            candidates.append((float(grid[row, col, channel]), channel, int(row), int(col)))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        suppressed = np.zeros_like(grid, dtype=bool)
        for confidence, channel, row, col in candidates:
            if decoder == "cell_nms" and suppressed[row, col, channel]:
                continue
            detections.append({"centroid_x": float((col+.5)/g*frame_width),
                               "centroid_y": float((row+.5)/g*frame_height),
                               "coarse_class_id": channel-1, "class_name": CLASS_NAMES[channel-1],
                               "confidence": confidence})
            if decoder == "cell_nms":
                suppressed[max(0,row-nms_cells):min(g,row+nms_cells+1),
                           max(0,col-nms_cells):min(g,col+nms_cells+1), channel] = True
        return detections[:limit] if limit else detections
    for channel in range(1, len(CLASS_NAMES)+1):
        mask = (winners == channel) & (grid[:, :, channel] >= threshold)
        visited = set()
        for row, col in zip(*np.where(mask)):
            start = (int(row), int(col))
            if start in visited:
                continue
            visited.add(start)
            pending, component = [start], []
            while pending:
                r, c = pending.pop()
                component.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < g and 0 <= nc < g and mask[nr, nc] and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            pending.append((nr, nc))
            # Mean cell centers, not object extent; confidence is the strongest cell.
            cy, cx = np.mean(component, axis=0) + 0.5
            detections.append({"centroid_x": float(cx / g * frame_width),
                               "centroid_y": float(cy / g * frame_height),
                               "coarse_class_id": channel - 1,
                               "class_name": CLASS_NAMES[channel - 1],
                               "confidence": float(max(grid[r, c, channel] for r, c in component))})
    detections.sort(key=lambda d: (-d["confidence"], d["coarse_class_id"], d["centroid_y"], d["centroid_x"]))
    return detections[:limit] if limit else detections


def crop_around_centroid(frame, centroid_x, centroid_y, crop_size=SETTINGS.crop_size):
    """Legacy inspection helper only, unused by the standalone inference pipeline."""
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 frame")
    if type(crop_size) is not int or crop_size <= 0:
        raise ValueError("crop_size must be a positive integer")
    if not all(math.isfinite(v) for v in (centroid_x, centroid_y)):
        raise ValueError("Centroid must be finite")
    height, width = frame.shape[:2]
    if not (0 <= centroid_x <= width and 0 <= centroid_y <= height):
        raise ValueError("Centroid must be within frame coordinates")
    left, top = math.floor(centroid_x - crop_size / 2), math.floor(centroid_y - crop_size / 2)
    right, bottom = left + crop_size, top + crop_size
    x1, y1, x2, y2 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    crop[y1 - top:y2 - top, x1 - left:x2 - left] = frame[y1:y2, x1:x2]
    return crop.astype(np.float32) / 255.0


def predict(model, frame, threshold=None, max_detections=None, decoder=None):
    batch = preprocess(frame)[None]
    start = time.perf_counter()
    grid = np.asarray(model(batch, training=False))[0]
    elapsed = (time.perf_counter() - start) * 1000
    return decode_grid(grid, frame.shape[1], frame.shape[0], threshold, max_detections, decoder), elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--decoder", choices=DECODERS, default=SETTINGS.decoder)
    parser.add_argument("--threshold", type=float, default=SETTINGS.confidence_threshold,
                        help="Explicit experimental operating point; default remains 0.60")
    parser.add_argument("--max-detections", type=int, default=SETTINGS.max_detections,
                        help="0 keeps all detections; warning selection belongs to decision logic")
    args = parser.parse_args()
    from model import load_model
    model, metadata = load_model(args.model)
    frame = load_rgb(args.image)
    predict(model, frame, threshold=args.threshold, max_detections=args.max_detections, decoder=args.decoder)
    detections, elapsed = predict(model, frame, threshold=args.threshold, max_detections=args.max_detections, decoder=args.decoder)
    policy = dict(decoding_policy(args.decoder), threshold=args.threshold, max_detections=args.max_detections)
    print(json.dumps({"module": "fomo", "taxonomy": list(CLASS_NAMES), "decoding": policy,
                      "coordinate_space": "original frame pixels, continuous edge coordinates",
                      "frame_width": frame.shape[1], "frame_height": frame.shape[0],
                      "detections": detections,
                      "pc_model_call_ms": elapsed, "synthetic_training": metadata["synthetic"]}, indent=2))


if __name__ == "__main__":
    main()
