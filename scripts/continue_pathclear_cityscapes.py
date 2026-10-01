"""Resume a saved seven-class adaptation run into a new, bounded experiment."""
import argparse
import contextlib
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import keras
import numpy as np
import tensorflow as tf

from dataset_loader import collect_pairs
from pathclear_pipeline import BASELINE, sha256
from pathclear_stages import GATE_NAME, enable_local_classes
from pathclear_taxonomy import CLASS_NAMES
from scripts.train_pathclear_cityscapes import Checks, Tee, confusion_matrix, dataset_for, write_json

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model_atomic(model, path):
    temporary = path.with_name(path.stem + ".pending.keras")
    model.save(temporary)
    temporary.replace(path)


def assert_optimizer_state(model, state):
    if int(model.optimizer.iterations) != state["optimizer_iterations"]:
        raise ValueError("Checkpoint optimizer steps disagree with epoch state.")


def read_resume_state(source):
    config = read_json(source / "config.json")
    required = {"mode": "full", "batch_size": 1, "train_images": 2975, "validation_images": 500,
                "active_ids": list(range(7)), "reserved_ids": [7, 8, 9], "feature_layers_frozen": True,
                "learning_rate": 0.001, "input_shape": [512, 1024, 3], "output_channels": 10,
                "loss_reduction": "mean_with_sample_weight", "ignore_label": 255}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Source is not the expected full-dataset, frozen-feature Cityscapes stage.")
    if (source / "state.json").is_file():
        state = read_json(source / "state.json")
        for name in ("last", "best"):
            if sha256(source / f"{name}.keras") != state[f"{name}_sha256"]:
                raise ValueError(f"{name} checkpoint hash disagrees with state.")
    else:
        result = read_json(source / "result.json")
        if result.get("status") != "complete" or result["completed_epochs"] != 1:
            raise ValueError("Legacy adaptation source must be the completed single-epoch run.")
        if sha256(source / "best.keras") != result["checkpoint_sha256"]:
            raise ValueError("Source best checkpoint hash mismatch.")
        state = {"completed_epochs": 1, "optimizer_iterations": result["optimizer_iterations"],
                 "best_epoch": 1, "best_val_mean_iou": result["final_metrics"]["val_mean_iou"],
                 "last_metrics": result["final_metrics"]}
    completed = state["completed_epochs"]
    if completed < 1 or state["optimizer_iterations"] != completed * 2975:
        raise ValueError("Expected one full training pass per completed epoch.")
    with (source / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in history] != list(range(completed)):
        raise ValueError("Source history and checkpoint epochs do not match.")
    if not 1 <= state["best_epoch"] <= completed or not np.isfinite(state["best_val_mean_iou"]):
        raise ValueError("Invalid best-checkpoint state.")
    return config, state


def scores(cm):
    if np.any(cm[7:, :]) or np.any(cm[:, 7:]):
        raise AssertionError("Reserved classes entered the validation confusion matrix.")
    rows = []
    for i in range(7):
        truth, predicted, tp = float(cm[i].sum()), float(cm[:, i].sum()), float(cm[i, i])
        union = truth + predicted - tp
        rows.append({"id": i, "class": CLASS_NAMES[i], "true_pixels": int(truth),
                     "predicted_pixels": int(predicted), "iou": tp / union if union else None,
                     "dice": 2 * tp / (truth + predicted) if truth + predicted else None})
    return rows


class EpochCheckpointer(Checks):
    def __init__(self, output, steps, state, protected_hashes):
        super().__init__(output, steps)
        self.best = state["best_val_mean_iou"]
        self.best_epoch = state["best_epoch"]
        self.protected_hashes = protected_hashes
        self.state = dict(state)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        print(f"Starting continuation epoch {self.epoch}.", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        if batch == 0 or (batch + 1) % 100 == 0 or batch + 1 == self.total_steps:
            progress = read_json(self.output / "progress.json")
            progress["epoch"] = self.epoch
            write_json(self.output / "progress.json", progress)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        # Check all frozen/reserved arrays before saving every completed epoch.
        super().on_train_end(logs)
        if int(self.model.optimizer.iterations) != self.total_steps * (epoch + 1):
            raise AssertionError("Optimizer count does not match the completed full epochs.")
        if any(sha256(path) != expected for path, expected in self.protected_hashes.items()):
            raise AssertionError("An original experiment file changed.")
        cm = confusion_matrix(self.model)
        per_class = scores(cm)
        metrics = {key: float(value) for key, value in logs.items()}
        improved = metrics["val_mean_iou"] > self.best
        save_model_atomic(self.model, self.output / "last.keras")
        if improved:
            save_model_atomic(self.model, self.output / "best.keras")
            self.best = metrics["val_mean_iou"]
            self.best_epoch = epoch + 1
        np.save(self.output / f"epoch_{epoch + 1:02d}_validation_confusion.npy", cm)
        write_json(self.output / f"epoch_{epoch + 1:02d}_validation.json", {"metrics": metrics, "per_class": per_class})
        self.state = {"mode": "full", "completed_epochs": epoch + 1,
                      "optimizer_iterations": int(self.model.optimizer.iterations),
                      "best_epoch": self.best_epoch, "best_val_mean_iou": self.best,
                      "last_metrics": metrics, "last_sha256": sha256(self.output / "last.keras"),
                      "best_sha256": sha256(self.output / "best.keras")}
        write_json(self.output / "state.pending.json", self.state)
        (self.output / "state.pending.json").replace(self.output / "state.json")
        print(f"Saved epoch {epoch + 1}; best epoch {self.best_epoch}, validation mIoU {self.best:.6f}.", flush=True)


def run(args, output):
    started = time.perf_counter()
    source = args.resume_from.resolve()
    source_config, state = read_resume_state(source)
    protected = [source / name for name in ("last.keras", "best.keras", "history.csv", "config.json")]
    protected += [path for path in (source / "state.json", source / "result.json") if path.exists()]
    protected += [BASELINE.parent / name for name in ("best.keras", "last.keras", "state.json", "history.csv")]
    protected += [ROOT / name for name in ("fast_scnn.py", "dataset_loader.py", "train.py", "segmentation_metrics.py")]
    protected_hashes = {str(path): sha256(path) for path in protected}
    keras.utils.set_random_seed(source_config["seed"])
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("Expected the established WSL GPU before continuation.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model = keras.models.load_model(source / "last.keras")
    assert_optimizer_state(model, state)
    if tuple(model.get_layer(GATE_NAME).active_ids) != tuple(range(7)):
        raise ValueError("Source must keep classes 7-9 inactive.")
    if model.input_shape != (None, 512, 1024, 3) or model.output_shape != (None, 512, 1024, 10):
        raise ValueError("Unexpected checkpoint input/output geometry.")
    if {w.path.split("/")[0] for w in model.trainable_weights} != {"classifier_logits"}:
        raise ValueError("Only the final classifier may be trainable.")
    if not np.isclose(float(model.optimizer.learning_rate), 0.001):
        raise ValueError("Unexpected source learning rate.")
    if (source / "state.json").exists() is False:
        # Legacy run saved best and last at the same epoch; verify their state.
        legacy_best = keras.models.load_model(source / "best.keras")
        for a, b in zip(model.get_weights(), legacy_best.get_weights()):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(model.optimizer.variables, legacy_best.optimizer.variables):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
        del legacy_best
    pairs = {split: collect_pairs(split) for split in ("train", "val")}
    for split, count in (("train", 2975), ("val", 500)):
        if len(pairs[split][0]) != count:
            raise ValueError(f"Incomplete {split} image/mask pairs.")
    dataset_manifest = {split: [{"image": i, "train_id_mask": m} for i, m in zip(*pair)]
                        for split, pair in pairs.items()}
    if dataset_manifest != read_json(source / "dataset_manifest.json"):
        raise ValueError("Dataset file list differs from the source experiment.")
    write_json(output / "dataset_manifest.json", dataset_manifest)
    config = dict(source_config)
    config.update({"resume_from": str(source), "initial_epoch": state["completed_epochs"],
                   "additional_epochs": args.epochs, "epochs": state["completed_epochs"] + args.epochs,
                   "optimizer_restored": True, "protected_hashes_before": protected_hashes,
                   "continuation_source_sha256": sha256(Path(__file__)),
                   "tensorflow": tf.__version__, "keras": keras.__version__})
    write_json(output / "config.json", config)
    # Carry the previous best forward, even if all continuation epochs are worse.
    shutil.copy2(source / "best.keras", output / "best.keras")
    shutil.copy2(source / "history.csv", output / "history.csv")
    train_data = dataset_for(*pairs["train"], training=True, seed=config["seed"])
    val_data = dataset_for(*pairs["val"], training=False, seed=config["seed"])
    before = model.get_weights()
    initial = model.evaluate(val_data, verbose=0, return_dict=True)
    for a, b in zip(before, model.get_weights()):
        np.testing.assert_array_equal(a, b)
    assert_optimizer_state(model, state)
    for name, value in initial.items():
        np.testing.assert_allclose(value, state["last_metrics"]["val_" + name], rtol=1e-5, atol=1e-6)
    write_json(output / "resume_verification.json", {"source_state": state, "recomputed_validation": initial,
                                                    "optimizer_iterations_restored": int(model.optimizer.iterations)})
    print(f"Resumed epoch {state['completed_epochs']}, optimizer step {int(model.optimizer.iterations)}; validation matches.", flush=True)
    print(f"Training {args.epochs} additional epochs through epoch {config['epochs']}.", flush=True)
    checks = EpochCheckpointer(output, 2975, state, protected_hashes)
    model.fit(train_data, validation_data=val_data, initial_epoch=state["completed_epochs"],
              epochs=config["epochs"], verbose=2, shuffle=False,
              callbacks=[checks, keras.callbacks.CSVLogger(str(output / "history.csv"), append=True)])
    if checks.state["completed_epochs"] != config["epochs"]:
        raise AssertionError("Requested continuation did not complete.")
    reloaded = keras.models.load_model(output / "last.keras")
    assert_optimizer_state(reloaded, checks.state)
    for a, b in zip(model.get_weights(), reloaded.get_weights()):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(model.optimizer.variables, reloaded.optimizer.variables):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    best = keras.models.load_model(output / "best.keras")
    if int(best.optimizer.iterations) != checks.best_epoch * 2975:
        raise AssertionError("Best checkpoint epoch and optimizer steps disagree.")
    weights = best.get_weights()
    best_validation = best.evaluate(val_data, verbose=0, return_dict=True)
    for a, b in zip(weights, best.get_weights()):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(best_validation["mean_iou"], checks.best, rtol=1e-5, atol=1e-6)
    best_scores = scores(confusion_matrix(best))
    write_json(output / "best_validation_per_class.json", best_scores)
    local = enable_local_classes(best)
    assert local.output_shape == (None, 512, 1024, 10)
    for a, b in zip(weights, local.get_weights()):
        np.testing.assert_array_equal(a, b)
    if any(sha256(path) != digest for path, digest in protected_hashes.items()):
        raise AssertionError("An original experiment file changed.")
    result = {"status": "complete", "mode": "full", "additional_epochs": args.epochs,
              "completed_epochs": checks.state["completed_epochs"], "initial_epoch": state["completed_epochs"],
              "optimizer_iterations": int(model.optimizer.iterations), "best_epoch": checks.best_epoch,
              "best_validation": best_validation, "final_metrics": checks.state["last_metrics"],
              "elapsed_seconds": time.perf_counter() - started, "best_checkpoint": str(output / "best.keras"),
              "checkpoint_sha256": sha256(output / "best.keras"), "last_checkpoint_sha256": sha256(output / "last.keras"),
              "checkpoint_and_optimizer_reload_exact": True, "frozen_and_reserved_weights_unchanged": True,
              "validation_preserved_weights": True, "source_experiments_unchanged": True,
              "local_reactivation_verified_without_training": True,
              "note": "Seven-class, frozen-feature Cityscapes continuation. No local training or TFLite export."}
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True, help="Additional epochs, not the total")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    output = args.output_dir.resolve()
    if output.is_relative_to(BASELINE.parent.resolve()) or output.is_relative_to(args.resume_from.resolve()):
        raise ValueError("Continuation must use a separate output directory outside the source experiments.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
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
