"""Render deterministic validation examples from a saved MicroSeg pilot."""
import argparse
import json
import textwrap
from pathlib import Path

import keras
import numpy as np
from PIL import Image, ImageDraw

from dataset_config import CITYSCAPES_ROOT, KAGGLE_CROSSWALK_ROOT
from microseg.dataset import load_crosswalk_pair, load_pair
from microseg.taxonomy import CLASS_NAMES, COLORS

_DEFAULT_RUN_DIR = Path.home() / "microseg_runs" / "pilot_groupnorm_20261004"
_TILE_SIZE = 256
_PADDING = 12
_BACKGROUND = (25, 29, 36)
_IGNORE_COLOR = (80, 84, 92)


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=_DEFAULT_RUN_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/microseg_groupnorm_pilot_examples.png"),
    )
    parser.add_argument("--cityscapes-root", type=Path, default=CITYSCAPES_ROOT)
    parser.add_argument("--crosswalk-root", type=Path, default=KAGGLE_CROSSWALK_ROOT)
    return parser.parse_args(argv)


def _require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required pilot artifact does not exist: {path}")


def _sample_indices(count, limit=3):
    if count < 1:
        raise ValueError("Cannot select examples from an empty validation split.")
    return np.linspace(0, count - 1, num=min(count, limit), dtype=int).tolist()


def _load_city_example(model, image_path, mask_path, ignore_path):
    image, target, weight = load_pair(
        str(image_path),
        str(mask_path),
        ignore_mask_path=str(ignore_path) if ignore_path else None,
    )
    return _example(
        model,
        image.numpy(),
        target.numpy(),
        weight.numpy() > 0,
        Path(image_path).name,
    )


def _load_crosswalk_example(model, image_path, mask_path):
    image, target, weight = load_crosswalk_pair(str(image_path), str(mask_path))
    return _example(
        model,
        image.numpy(),
        target.numpy(),
        weight.numpy() > 0,
        Path(image_path).name,
    )


def _example(model, image, target, valid, name):
    logits = model(np.expand_dims(image, axis=0), training=False)
    prediction = np.argmax(np.asarray(logits), axis=-1)[0].astype(np.uint8)
    return {
        "name": name,
        "image": np.clip(image * 255.0, 0, 255).astype(np.uint8),
        "target": target.astype(np.uint8),
        "valid": valid,
        "prediction": prediction,
    }


def _colorize(mask, valid=None):
    mask = np.asarray(mask)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    classes = (mask >= 0) & (mask < len(CLASS_NAMES))
    rgb[classes] = COLORS[mask[classes]]
    if valid is not None:
        rgb[~valid] = _IGNORE_COLOR
    else:
        rgb[~classes] = _IGNORE_COLOR
    return rgb


def _overlay(image, labels, valid=None):
    colors = _colorize(labels, valid)
    if valid is None:
        visible = np.ones(labels.shape, dtype=bool)
    else:
        visible = valid
    result = image.copy()
    blended = (
        image[visible].astype(np.float32) * 0.48
        + colors[visible].astype(np.float32) * 0.52
    )
    result[visible] = np.clip(blended, 0, 255).astype(np.uint8)
    result[~visible] = colors[~visible]
    return result


def _tile(array):
    return Image.fromarray(array).resize(
        (_TILE_SIZE, _TILE_SIZE), resample=Image.Resampling.NEAREST
    )


def _render(examples, output):
    columns = (
        "Input",
        "Ground truth\n(gray = ignored)",
        "Predicted classes",
        "Prediction overlay",
    )
    label_width = 170
    header_height = 100
    row_height = _TILE_SIZE + 46
    width = label_width + len(columns) * (_TILE_SIZE + _PADDING) + _PADDING
    height = header_height + len(examples) * row_height + 86
    canvas = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    draw.text((16, 14), "MicroSeg GroupNorm pilot - inference examples", fill="white")
    draw.text(
        (16, 42),
        "96x96 inputs enlarged with nearest-neighbor; deterministic validation samples",
        fill=(205, 210, 218),
    )
    for column, title in enumerate(columns):
        x = label_width + column * (_TILE_SIZE + _PADDING)
        draw.multiline_text((x, 72), title, fill=(230, 230, 230), spacing=2)

    for row, example in enumerate(examples):
        top = header_height + row * row_height
        draw.text(
            (16, top + 8),
            example["domain"],
            fill=(255, 220, 130),
        )
        draw.multiline_text(
            (16, top + 30),
            "\n".join(textwrap.wrap(example["name"], width=23, max_lines=3)),
            fill=(220, 225, 232),
            spacing=2,
        )
        target = example["target"]
        valid = example["valid"]
        prediction = example["prediction"]
        tiles = (
            example["image"],
            _colorize(target, valid),
            _colorize(prediction),
            _overlay(example["image"], prediction),
        )
        for column, array in enumerate(tiles):
            x = label_width + column * (_TILE_SIZE + _PADDING)
            canvas.paste(_tile(array), (x, top))

    legend_top = height - 65
    draw.text((16, legend_top), "Class colors:", fill="white")
    x = 120
    for class_id, name in enumerate(CLASS_NAMES):
        color = tuple(int(value) for value in COLORS[class_id])
        draw.rectangle((x, legend_top + 2, x + 18, legend_top + 20), fill=color)
        draw.text((x + 25, legend_top + 2), name, fill=(230, 230, 230))
        x += 190
    draw.rectangle((x, legend_top + 2, x + 18, legend_top + 20), fill=_IGNORE_COLOR)
    draw.text((x + 25, legend_top + 2), "Ignore", fill=(230, 230, 230))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main(argv=None):
    args = _arguments(argv)
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "run_config.json"
    city_manifest_path = run_dir / "cityscapes_pairs.json"
    crosswalk_manifest_path = run_dir / "crosswalk_split.json"
    checkpoint_path = run_dir / "best.keras"
    for path in (
        config_path,
        city_manifest_path,
        crosswalk_manifest_path,
        checkpoint_path,
    ):
        _require_file(path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    cityscapes_root = args.cityscapes_root
    crosswalk_root = args.crosswalk_root
    if not cityscapes_root.is_dir():
        raise FileNotFoundError(f"Cityscapes root does not exist: {cityscapes_root}")
    if not crosswalk_root.is_dir():
        raise FileNotFoundError(f"Crosswalk root does not exist: {crosswalk_root}")

    city_manifest = json.loads(city_manifest_path.read_text(encoding="utf-8"))
    city_entries = city_manifest.get("splits", {}).get("val")
    if not isinstance(city_entries, list) or not city_entries:
        raise ValueError(f"Cityscapes validation split is empty: {city_manifest_path}")
    crosswalk_manifest = json.loads(
        crosswalk_manifest_path.read_text(encoding="utf-8")
    )
    crosswalk_entries = [
        item for item in crosswalk_manifest.get("items", [])
        if item.get("split") == "validation" and item.get("source") == "real"
    ]
    if not crosswalk_entries:
        raise ValueError(
            f"Held-out real Crosswalk split is empty: {crosswalk_manifest_path}"
        )

    model = keras.models.load_model(checkpoint_path, compile=False)
    examples = []
    for index in _sample_indices(len(city_entries)):
        entry = city_entries[index]
        image_path = cityscapes_root / entry["image"]
        mask_path = cityscapes_root / entry["mask"]
        ignore_path = cityscapes_root / entry["ignore_mask"] if entry.get(
            "ignore_mask"
        ) else None
        example = _load_city_example(model, image_path, mask_path, ignore_path)
        example["domain"] = "Cityscapes"
        examples.append(example)

    for index in _sample_indices(len(crosswalk_entries)):
        entry = crosswalk_entries[index]
        example = _load_crosswalk_example(
            model,
            crosswalk_root / entry["image"],
            crosswalk_root / entry["mask"],
        )
        example["domain"] = "Held-out KPV"
        examples.append(example)

    output = args.output.resolve()
    _render(examples, output)
    print(
        f"Saved {len(examples)} inference examples to {output} "
        f"(checkpoint: {checkpoint_path}; epoch count: "
        f"{config.get('epochs_requested', 'unknown')})."
    )


if __name__ == "__main__":
    main()
