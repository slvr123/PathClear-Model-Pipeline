"""Inspect one fixed validation example per Cityscapes city, without training."""
import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from dataset_loader import collect_pairs
from pathclear_pipeline import BASELINE, load_cityscapes_pair, sha256
from pathclear_stages import GATE_NAME  # Also register the serialized layer.
from pathclear_taxonomy import CLASS_NAMES, COLORS, colorize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(BASELINE.parent.resolve()):
        raise ValueError("Preview output must be outside the original baseline experiment.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    before_hash = sha256(args.checkpoint)
    model = keras.models.load_model(args.checkpoint, compile=False)
    assert tuple(model.get_layer(GATE_NAME).active_ids) == tuple(range(7))
    before_weights = model.get_weights()
    images, masks = collect_pairs("val")
    selected = {}
    for image_path, mask_path in zip(images, masks):
        selected.setdefault(Path(image_path).parent.name, (image_path, mask_path))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for city, (image_path, mask_path) in selected.items():
        image, target, weight = load_cityscapes_pair(image_path, mask_path)
        logits = model(image[None], training=False)
        prediction = tf.argmax(logits[0], axis=-1).numpy().astype(np.uint8)
        assert prediction.max() < 7
        valid = weight.numpy() > 0
        truth = np.where(valid, target.numpy(), 255).astype(np.uint8)
        errors = np.zeros((*truth.shape, 3), dtype=np.uint8)
        errors[valid & (prediction == truth)] = (145, 145, 145)
        errors[valid & (prediction != truth)] = (235, 35, 35)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
        panels = (image.numpy(), colorize(truth), colorize(prediction), errors)
        titles = ("Original", "Remapped ground truth", "Prediction: seven active classes", "Error: red | correct: gray | ignored: black")
        for ax, panel, title in zip(axes, panels, titles):
            ax.imshow(panel)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"Cityscapes adaptation | {city} | First sample by filename")
        handles = [Patch(color=COLORS[i] / 255, label=CLASS_NAMES[i]) for i in range(7)]
        handles.append(Patch(color="black", label="Ignore (ground truth only)"))
        fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=9)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.85, bottom=0.14, wspace=0.03)
        fig.savefig(output / f"{city}_comparison.png", dpi=120)
        plt.close(fig)
        Image.fromarray(prediction).save(output / f"{city}_prediction_ids.png")
        rows.append({"city": city, "image": image_path, "mask": mask_path,
                     "predicted_ids": np.unique(prediction).tolist(),
                     "valid_pixel_accuracy": float(np.mean(prediction[valid] == truth[valid]))})
    for before, after in zip(before_weights, model.get_weights()):
        np.testing.assert_array_equal(before, after)
    assert sha256(args.checkpoint) == before_hash
    report = {"checkpoint_sha256": before_hash, "weights_unchanged": True, "samples": rows,
              "selection": "First validation filename in each city; diagnostic examples, not a representative sample.",
              "warning": "Sky/building/terrain/void predictions are unsupervised and excluded from scores. No local Philippine accuracy is measured."}
    (output / "preview_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
