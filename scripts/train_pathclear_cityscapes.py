"""Run exactly one head-only Cityscapes adaptation epoch, or a three-batch smoke test."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import keras
import numpy as np
import tensorflow as tf

from dataset_loader import collect_pairs
from pathclear_pipeline import BASELINE, load_cityscapes_pair, sha256, transfer_baseline
from pathclear_stages import cityscapes_model, enable_local_classes, GATE_NAME
from pathclear_taxonomy import CLASS_NAMES, CITYSCAPES_SUPPORTED_IDS

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def stage_pair(image, mask):
    image, labels, weights = load_cityscapes_pair(image, mask)
    tf.debugging.assert_less(labels, len(CITYSCAPES_SUPPORTED_IDS),
                             message="Cityscapes stage cannot consume local-only labels")
    return image, labels, weights


def dataset_for(images, masks, training, seed):
    data = tf.data.Dataset.from_tensor_slices((images, masks))
    if training:
        data = data.shuffle(len(images), seed=seed, reshuffle_each_iteration=False)
    return data.map(stage_pair, num_parallel_calls=2, deterministic=True).batch(1).prefetch(1)


class Checks(keras.callbacks.Callback):
    def __init__(self, output, total_steps):
        super().__init__()
        self.output, self.total_steps = output, total_steps

    def on_train_begin(self, logs=None):
        self.started = time.perf_counter()
        self.frozen = {layer.name: layer.get_weights() for layer in self.model.layers
                       if layer.name != "classifier_logits" and layer.weights}
        self.head = self.model.get_layer("classifier_logits").get_weights()

    def on_train_batch_end(self, batch, logs=None):
        if not all(np.isfinite(value) for value in (logs or {}).values()):
            raise FloatingPointError(f"Non-finite training metrics at batch {batch + 1}")
        if batch == 0 or (batch + 1) % 100 == 0 or batch + 1 == self.total_steps:
            progress = {"batch": batch + 1, "total_batches": self.total_steps,
                        "elapsed_seconds": time.perf_counter() - self.started,
                        "metrics": {key: float(value) for key, value in (logs or {}).items()}}
            write_json(self.output / "progress.json", progress)
            print(f"Training {batch + 1}/{self.total_steps}: " + json.dumps(progress["metrics"]), flush=True)

    def on_test_begin(self, logs=None):
        self.before_validation = self.model.get_weights()
        print("Validation started.", flush=True)

    def on_test_end(self, logs=None):
        for before, after in zip(self.before_validation, self.model.get_weights()):
            np.testing.assert_array_equal(before, after)
        print("Validation preserved all model weights.", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        if not all(np.isfinite(value) for value in (logs or {}).values()):
            raise FloatingPointError("Non-finite epoch metrics; model will not be saved.")

    def on_train_end(self, logs=None):
        for name, before in self.frozen.items():
            for old, new in zip(before, self.model.get_layer(name).get_weights()):
                np.testing.assert_array_equal(old, new, err_msg=f"Frozen layer changed: {name}")
        after = self.model.get_layer("classifier_logits").get_weights()
        for old, new in zip(self.head, after):
            np.testing.assert_array_equal(old[..., 7:], new[..., 7:], err_msg="Reserved output weights changed")
        self.max_head_change = max(float(np.max(np.abs(old[..., :7] - new[..., :7])))
                                   for old, new in zip(self.head, after))
        if self.max_head_change <= 0:
            raise AssertionError("Active classifier weights did not change.")
        print(f"Verified: feature weights and outputs 7-9 unchanged; active head change {self.max_head_change:.8f}.", flush=True)


def confusion_matrix(model):
    for container in model.metrics:
        for metric in getattr(container, "metrics", []):
            if metric.name == "mean_iou":
                return metric.total_cm.numpy()
    raise RuntimeError("MeanIoU confusion matrix not found.")


def run(args, output):
    started = time.perf_counter()
    protected = [BASELINE, BASELINE.parent / "last.keras", BASELINE.parent / "state.json",
                 BASELINE.parent / "history.csv"]
    protected += [ROOT / name for name in ("fast_scnn.py", "dataset_loader.py", "train.py", "segmentation_metrics.py")]
    before_hashes = {str(path): sha256(path) for path in protected}
    keras.utils.set_random_seed(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("Expected the verified WSL GPU; stopping before training.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    pairs = {split: collect_pairs(split) for split in ("train", "val")}
    for split, expected in (("train", 2975), ("val", 500)):
        if len(pairs[split][0]) != expected:
            raise ValueError(f"Expected {expected} complete {split} pairs, found {len(pairs[split][0])}.")
    if args.mode == "smoke":
        # Deterministic, independently bounded smoke subset, never resumed into full run.
        pairs = {split: (images[:limit], masks[:limit])
                 for split, limit in (("train", 3), ("val", 2)) for images, masks in [pairs[split]]}
    write_json(output / "dataset_manifest.json", {
        split: [{"image": image, "train_id_mask": mask} for image, mask in zip(*pair)]
        for split, pair in pairs.items()})
    train_data = dataset_for(*pairs["train"], training=True, seed=args.seed)
    val_data = dataset_for(*pairs["val"], training=False, seed=args.seed)
    steps = len(pairs["train"][0])
    config = {"mode": args.mode, "seed": args.seed, "epochs": 1, "batch_size": 1,
              "train_images": steps, "validation_images": len(pairs["val"][0]),
              "active_ids": list(CITYSCAPES_SUPPORTED_IDS), "reserved_ids": [7, 8, 9],
              "feature_layers_frozen": True, "learning_rate": 0.001, "optimizer": "Adam",
              "input_shape": [512, 1024, 3], "output_channels": 10,
              "loss_reduction": "mean_with_sample_weight", "ignore_label": 255,
              "tensorflow": tf.__version__, "keras": keras.__version__, "devices": [str(gpu) for gpu in gpus],
              "baseline_hashes_before": before_hashes,
              "source_hashes": {name: sha256(ROOT / name) for name in (
                  "pathclear_stages.py", "pathclear_pipeline.py", "pathclear_taxonomy.py", "scripts/train_pathclear_cityscapes.py")}}
    write_json(output / "config.json", config)
    print(json.dumps(config, indent=2), flush=True)
    raw, transfer = transfer_baseline()
    write_json(output / "transfer.json", transfer)
    model = cityscapes_model(raw)
    if {weight.path.split("/")[0] for weight in model.trainable_weights} != {"classifier_logits"}:
        raise AssertionError("Only classifier_logits must be trainable.")
    initial_weights = model.get_weights()
    print("Evaluating initialized classifier before training.", flush=True)
    initial = model.evaluate(val_data, verbose=0, return_dict=True)
    for before, after in zip(initial_weights, model.get_weights()):
        np.testing.assert_array_equal(before, after)
    write_json(output / "initial_validation.json", initial)
    print("Initial validation: " + json.dumps(initial), flush=True)
    checks = Checks(output, steps)
    history = model.fit(train_data, validation_data=val_data, epochs=1, verbose=2, shuffle=False,
                        callbacks=[checks, keras.callbacks.CSVLogger(str(output / "history.csv"))])
    if int(model.optimizer.iterations) != steps:
        raise AssertionError("Unexpected optimizer update count.")
    cm = confusion_matrix(model)
    if np.any(cm[7:, :]) or np.any(cm[:, 7:]):
        raise AssertionError("An inactive class entered evaluation.")
    rows = []
    for i in range(7):
        truth, predicted, tp = float(cm[i].sum()), float(cm[:, i].sum()), float(cm[i, i])
        union = truth + predicted - tp
        rows.append({"id": i, "class": CLASS_NAMES[i], "true_pixels": int(truth), "predicted_pixels": int(predicted),
                     "iou": tp / union if union else None, "dice": 2 * tp / (truth + predicted) if truth + predicted else None})
    write_json(output / "validation_per_class.json", rows)
    np.save(output / "validation_confusion.npy", cm)
    # Exactly one epoch: best and last refer to the same completed model.
    model.save(output / "last.keras")
    model.save(output / "best.keras")
    images, _, _ = next(iter(val_data))
    expected_logits = model(images, training=False).numpy()
    loaded = keras.models.load_model(output / "best.keras")
    np.testing.assert_allclose(loaded(images, training=False), expected_logits, rtol=0, atol=0)
    for before, after in zip(model.get_weights(), loaded.get_weights()):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(model.optimizer.variables, loaded.optimizer.variables):
        np.testing.assert_array_equal(before.numpy(), after.numpy())
    assert tuple(loaded.get_layer(GATE_NAME).active_ids) == tuple(range(7))
    assert expected_logits.shape == (1, 512, 1024, 10)
    assert np.max(np.argmax(expected_logits, axis=-1)) < 7
    local = enable_local_classes(loaded)
    assert local.output_shape == (None, 512, 1024, 10)
    assert not any(layer.name == GATE_NAME for layer in local.layers)
    for before, after in zip(loaded.get_weights(), local.get_weights()):
        np.testing.assert_array_equal(before, after)
    after_hashes = {str(path): sha256(path) for path in protected}
    if before_hashes != after_hashes:
        raise AssertionError("Protected baseline file changed.")
    report = {"status": "complete", "mode": args.mode, "completed_epochs": 1,
              "optimizer_iterations": int(model.optimizer.iterations), "elapsed_seconds": time.perf_counter() - started,
              "initial_validation": initial, "final_metrics": {key: float(value[-1]) for key, value in history.history.items()},
              "active_ids": list(range(7)), "untrained_reserved_ids": [7, 8, 9], "forward_shape": list(expected_logits.shape),
              "frozen_layers_unchanged": True, "reserved_output_weights_unchanged": True,
              "validation_preserved_weights": True, "max_active_head_change": checks.max_head_change,
              "checkpoint_reload_exact": True, "optimizer_reload_exact": True,
              "local_reactivation_verified_without_training": True,
              "baseline_files_unchanged": before_hashes == after_hashes, "baseline_hashes_after": after_hashes,
              "best_checkpoint": str(output / "best.keras"), "checkpoint_sha256": sha256(output / "best.keras"),
              "gpu_memory": tf.config.experimental.get_memory_info("GPU:0"),
              "note": "Seven-class Cityscapes adaptation only. No local training or TFLite export. Loss/IoU are not comparable to the original 19-class scores."}
    write_json(output / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(BASELINE.parent.resolve()):
        raise ValueError("Output must be outside the original baseline experiment.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            try:
                run(args, output)
            except Exception as error:
                write_json(output / "failure.json", {"status": "failed", "error": repr(error)})
                raise


if __name__ == "__main__":
    main()
