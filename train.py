"""Bounded Cityscapes training with explicit checkpoint resume."""
import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from dataset_loader import BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES, make_dataset
from fast_scnn import build_fast_scnn
from segmentation_metrics import make_metrics


def build_training_model():
    model = build_fast_scnn(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=NUM_CLASSES,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=make_metrics(NUM_CLASSES),
    )
    return model


class VerifyFinite(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite {name}: {value}; checkpoint not saved.")


class SmokeChecks(keras.callbacks.Callback):
    """Prove validation preserves all model weights, including BatchNorm state."""
    def on_train_begin(self, logs=None):
        self.before_training = self.model.trainable_weights[0].numpy().copy()

    def on_test_begin(self, logs=None):
        self.before_validation = self.model.get_weights()

    def on_test_end(self, logs=None):
        for before, after in zip(self.before_validation, self.model.get_weights()):
            np.testing.assert_array_equal(before, after)
        print("Validation left all model weights unchanged.", flush=True)

    def on_train_end(self, logs=None):
        change = np.max(np.abs(self.before_training - self.model.trainable_weights[0].numpy()))
        if not np.isfinite(change) or change == 0:
            raise AssertionError("Training did not produce a finite weight update.")
        print(f"Verified training weight change: {change:.8f}", flush=True)


class SaveState(keras.callbacks.Callback):
    def __init__(self, path, best_checkpoint, scheduler):
        super().__init__()
        self.path, self.best_checkpoint, self.scheduler = path, best_checkpoint, scheduler

    def on_epoch_end(self, epoch, logs=None):
        state = {
            "completed_epochs": epoch + 1,
            "best_val_mean_iou": float(self.best_checkpoint.best),
            "optimizer_iterations": int(self.model.optimizer.iterations.numpy()),
            "scheduler": {
                "best": float(self.scheduler.best),
                "wait": int(self.scheduler.wait),
                "cooldown_counter": int(self.scheduler.cooldown_counter),
            },
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


class RestoreScheduler(keras.callbacks.Callback):
    def __init__(self, scheduler, state):
        super().__init__()
        self.scheduler, self.state = scheduler, state

    def on_train_begin(self, logs=None):
        # ReduceLROnPlateau resets its counters in its own on_train_begin.
        for name, value in self.state.items():
            setattr(self.scheduler, name, value)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=positive, default=1, help="Additional epochs, including on resume")
    parser.add_argument("--batch-size", type=positive, default=BATCH_SIZE)
    parser.add_argument("--steps-per-epoch", type=positive, default=50)
    parser.add_argument("--validation-steps", type=positive, default=10)
    parser.add_argument("--full-dataset", action="store_true", help="Use all train/validation batches each epoch")
    parser.add_argument("--smoke-test", action="store_true", help="Cap at 1 epoch, 50 train and 10 validation batches")
    parser.add_argument("--resume", action="store_true", help="Explicitly restore last.keras and state from output directory")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.smoke_test and args.full_dataset:
        parser.error("--smoke-test and --full-dataset cannot be combined")
    if args.smoke_test:
        args.epochs = 1
        args.steps_per_epoch = min(args.steps_per_epoch, 50)
        args.validation_steps = min(args.validation_steps, 10)
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.resolve()
    state = {}
    if args.resume:
        for name in ("last.keras", "best.keras", "state.json", "history.csv"):
            if not (output / name).is_file():
                raise FileNotFoundError(f"Resume requires {output / name}")
        state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty. Use --resume or a new --output-dir.")

    keras.utils.set_random_seed(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Devices: {gpus or 'CPU'}; batch size: {args.batch_size}; resolution: 512 x 1024", flush=True)
    train_data = make_dataset("train", training=True, batch_size=args.batch_size)
    val_data = make_dataset("val", training=False, batch_size=args.batch_size)
    if not args.full_dataset:
        # take() gives each epoch a fresh iterator and keeps validation bounded.
        train_data = train_data.take(args.steps_per_epoch)
        val_data = val_data.take(args.validation_steps)
    print(f"Batches per epoch: train={int(train_data.cardinality())}, validation={int(val_data.cardinality())}", flush=True)

    initial_epoch = state.get("completed_epochs", 0)
    if args.resume:
        print(f"Resuming {output / 'last.keras'} after {initial_epoch} completed epochs.", flush=True)
        model = keras.models.load_model(output / "last.keras")
        if int(model.optimizer.iterations.numpy()) != state["optimizer_iterations"]:
            raise ValueError("Checkpoint/state mismatch; use a consistent epoch-end checkpoint.")
    else:
        print("Starting from random weights.", flush=True)
        model = build_training_model()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / f"run_from_epoch_{initial_epoch:04d}.json"
    config_path.write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")

    scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor="val_mean_iou", mode="max", factor=0.5, patience=3,
        min_lr=1e-6, verbose=1,
    )
    best = keras.callbacks.ModelCheckpoint(
        str(output / "best.keras"), monitor="val_mean_iou", mode="max",
        save_best_only=True, initial_value_threshold=state.get("best_val_mean_iou"), verbose=1,
    )
    callbacks = [
        keras.callbacks.TerminateOnNaN(), VerifyFinite(), scheduler,
        RestoreScheduler(scheduler, state.get("scheduler", {})), best,
        keras.callbacks.ModelCheckpoint(str(output / "last.keras")),
        keras.callbacks.CSVLogger(str(output / "history.csv"), append=args.resume),
        SaveState(output / "state.json", best, scheduler),
    ]
    if args.smoke_test:
        callbacks.append(SmokeChecks())
    history = model.fit(
        train_data, validation_data=val_data,
        initial_epoch=initial_epoch, epochs=initial_epoch + args.epochs,
        callbacks=callbacks, verbose=2, shuffle=False,
    )
    print("Final epoch:", {key: float(value[-1]) for key, value in history.history.items()}, flush=True)
    print(f"Saved checkpoints, resume state and CSV history in {output}", flush=True)


if __name__ == "__main__":
    main()
