"""Bounded preparation check: one real image, no training, no model checkpoint."""
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
from pathclear_pipeline import BASELINE, load_cityscapes_pair, sha256, transfer_baseline, compile_pathclear
from pathclear_taxonomy import CLASS_NAMES, COLORS, IGNORE_LABEL, colorize, remap_cityscapes_train_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pathclear_10class/sanity"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(BASELINE.parent.resolve()):
        raise ValueError("Sanity artifacts must be outside the baseline experiment.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory.")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.utils.set_random_seed(42)
    model, report = transfer_baseline()
    compile_pathclear(model)
    image_paths, mask_paths = collect_pairs("val")
    image_path, mask_path = Path(image_paths[0]), Path(mask_paths[0])
    source_hashes = {str(p): sha256(p) for p in (BASELINE, image_path, mask_path)}
    # Exercise tf.data graph execution, including source void restoration.
    dataset = tf.data.Dataset.from_tensor_slices(([str(image_path)], [str(mask_path)]))
    images, targets, weights = next(iter(dataset.map(load_cityscapes_pair).batch(1)))
    before_weights = model.get_weights()
    logits = model(images, training=False)
    assert tuple(logits.shape) == (1, 512, 1024, 10)
    assert bool(tf.reduce_all(tf.math.is_finite(logits)))
    result = model.test_on_batch(images, targets, sample_weight=weights, return_dict=True)
    assert all(np.isfinite(value) for value in result.values())
    for before, after in zip(before_weights, model.get_weights()):
        np.testing.assert_array_equal(before, after)
    assert int(model.optimizer.iterations.numpy()) == 0
    restored = np.where(weights[0].numpy() > 0, targets[0].numpy(), IGNORE_LABEL).astype(np.uint8)
    observed = np.unique(restored).tolist()
    assert set(observed) <= {0, 1, 2, 3, 4, 5, 6, 255}
    with Image.open(mask_path) as source:
        raw = np.array(source)
    mapped = remap_cityscapes_train_ids(raw)
    # Independently compare the graph loader with the raw-mask NumPy mapping.
    resized = tf.image.resize(mapped[..., None], [512, 1024], method="nearest")[..., 0]
    np.testing.assert_array_equal(resized.numpy().astype(np.uint8), restored)
    for path, before in source_hashes.items():
        assert sha256(path) == before
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mapped).save(output / "remapped_native_ids.png")
    Image.fromarray(restored).save(output / "remapped_model_ids.png")
    image = images[0].numpy()
    rgb = colorize(restored)
    overlay = image.copy()
    valid = restored != IGNORE_LABEL
    overlay[valid] = 0.55 * image[valid] + 0.45 * rgb[valid] / 255
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, array, title in zip(axes, (image, rgb, overlay),
                               ("Cityscapes RGB", "Remapped ground truth (not a prediction)", "Label overlay")):
        ax.imshow(array)
        ax.set_title(title)
        ax.axis("off")
    handles = [Patch(color=COLORS[i] / 255, label=f"{i}: {name}") for i, name in enumerate(CLASS_NAMES)]
    handles.append(Patch(color="black", label="255: Ignore"))
    fig.legend(handles=handles, loc="lower center", ncol=6)
    fig.suptitle("PathClear mapping check | Cityscapes provides no labels for classes 7, 8, 9")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.19, wspace=0.03)
    fig.savefig(output / "mapping_preview.png", dpi=130)
    plt.close(fig)
    report.update({
        "sample_image": str(image_path), "sample_mask": str(mask_path),
        "observed_model_mask_ids": observed, "observed_native_mask_ids": np.unique(mapped).tolist(),
        "source_hashes_before": source_hashes,
        "source_hashes_after": {path: sha256(path) for path in source_hashes},
        "baseline_unchanged": sha256(BASELINE) == report["baseline_sha256"],
        "forward_shape": list(logits.shape), "forward_finite": True,
        "inference_and_evaluation_preserved_weights": True, "optimizer_iterations": 0,
        "evaluation_values_for_plumbing_only": {key: float(value) for key, value in result.items()},
        "warning": "Fresh classifier; these values do not measure a trained PathClear model. Classes 7-9 have no Cityscapes supervision.",
    })
    (output / "sanity_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "transferred_layer_count", "transferred_weight_tensor_count", "transferred_parameters",
        "reinitialized_layers", "forward_shape", "observed_model_mask_ids", "baseline_unchanged",
        "training_performed", "model_saved")}, indent=2))
    print(f"Sanity artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
