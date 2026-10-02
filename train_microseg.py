"""MicroSeg training script — 5-class lightweight segmentation for ESP32-CAM.

Usage examples:
    # Smoke test (1 epoch, 3 train batches, 3 val batches — no GPU required):
    bash run_wsl.sh train_microseg.py --smoke-test

    # Full training from scratch:
    bash run_wsl.sh train_microseg.py --epochs 30 --output-dir checkpoints/microseg_v1

    # Resume from last checkpoint:
    bash run_wsl.sh train_microseg.py --resume --output-dir checkpoints/microseg_v1 --epochs 10
"""
import argparse
import json
import numpy as np
from pathlib import Path

import keras
import tensorflow as tf

from micro_dataset_loader import BATCH_SIZE, make_dataset
from microseg_model import build_microseg
from microseg_taxonomy import NUM_CLASSES
from segmentation_metrics import make_metrics   # reused unchanged from Fast-SCNN era

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96


# ---------------------------------------------------------------------------
# Callbacks (reused patterns from train.py)
# ---------------------------------------------------------------------------

class VerifyFinite(keras.callbacks.Callback):
    """Abort training if any logged metric is non-finite."""
    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"Non-finite {name}={value} at epoch {epoch}; checkpoint not saved."
                )


class SmokeChecks(keras.callbacks.Callback):
    """Verify weights update during training and stay frozen during validation."""
    def on_train_begin(self, logs=None):
        self._before = self.model.trainable_weights[0].numpy().copy()

    def on_test_begin(self, logs=None):
        self._val_before = self.model.get_weights()

    def on_test_end(self, logs=None):
        for before, after in zip(self._val_before, self.model.get_weights()):
            np.testing.assert_array_equal(before, after)
        print("Validation left all model weights unchanged.", flush=True)

    def on_train_end(self, logs=None):
        delta = np.max(np.abs(self._before - self.model.trainable_weights[0].numpy()))
        if not np.isfinite(delta) or delta == 0:
            raise AssertionError("Training did not produce a finite weight update.")
        print(f"Verified training weight change: {delta:.8f}", flush=True)


class SaveState(keras.callbacks.Callback):
    """Write a JSON resume-state file atomically after every epoch."""
    def __init__(self, path: Path, best_cb, scheduler):
        super().__init__()
        self.path = path
        self.best_cb = best_cb
        self.scheduler = scheduler

    def on_epoch_end(self, epoch, logs=None):
        state = {
            "completed_epochs": epoch + 1,
            "best_val_mean_iou": float(self.best_cb.best),
            "optimizer_iterations": int(self.model.optimizer.iterations.numpy()),
            "scheduler": {
                "best": float(self.scheduler.best),
                "wait": int(self.scheduler.wait),
                "cooldown_counter": int(self.scheduler.cooldown_counter),
            },
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)


class RestoreScheduler(keras.callbacks.Callback):
    """Restore ReduceLROnPlateau internal state after a resume."""
    def __init__(self, scheduler, state: dict):
        super().__init__()
        self.scheduler = scheduler
        self.state = state

    def on_train_begin(self, logs=None):
        for key, value in self.state.items():
            setattr(self.scheduler, key, value)


# ---------------------------------------------------------------------------
# Model build + compile
# ---------------------------------------------------------------------------

def build_training_model() -> keras.Model:
    model = build_microseg(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                           num_classes=NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="mean_with_sample_weight",
        ),
        weighted_metrics=make_metrics(NUM_CLASSES),
    )
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=_positive, default=1,
                   help="Epochs to train (added on top of resumed state).")
    p.add_argument("--steps-per-epoch", type=_positive, default=100)
    p.add_argument("--validation-steps", type=_positive, default=20)
    p.add_argument("--batch-size", type=_positive, default=BATCH_SIZE)
    p.add_argument("--full-dataset", action="store_true",
                   help="Use all available batches each epoch.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Cap at 1 epoch, 3 train + 3 val batches.")
    p.add_argument("--no-crosswalk", action="store_true",
                   help="Train on Cityscapes only (useful when Kaggle data absent).")
    p.add_argument("--resume", action="store_true",
                   help="Restore last.keras and state.json from output directory.")
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/microseg"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.smoke_test and args.full_dataset:
        p.error("--smoke-test and --full-dataset cannot be combined.")
    if args.smoke_test:
        args.epochs = 1
        args.steps_per_epoch = 3
        args.validation_steps = 3
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.resolve()

    # Resume validation
    state = {}
    if args.resume:
        for name in ("last.keras", "best.keras", "state.json", "history.csv"):
            if not (output / name).is_file():
                raise FileNotFoundError(f"Resume requires {output / name}")
        state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"{output} is not empty. Use --resume or a new --output-dir."
        )

    keras.utils.set_random_seed(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(
        f"Devices: {gpus or 'CPU'}  |  batch={args.batch_size}  |  "
        f"resolution={IMAGE_HEIGHT}×{IMAGE_WIDTH}  |  classes={NUM_CLASSES}",
        flush=True,
    )

    use_crosswalk = not args.no_crosswalk
    train_ds = make_dataset("train", training=True,
                            batch_size=args.batch_size,
                            use_crosswalk=use_crosswalk)
    val_ds = make_dataset("val", training=False,
                          batch_size=args.batch_size,
                          use_crosswalk=False)

    if not args.full_dataset:
        train_ds = train_ds.take(args.steps_per_epoch)
        val_ds = val_ds.take(args.validation_steps)

    print(
        f"Batches per epoch: train={int(train_ds.cardinality())}  "
        f"val={int(val_ds.cardinality())}",
        flush=True,
    )

    initial_epoch = state.get("completed_epochs", 0)
    if args.resume:
        print(f"Resuming {output / 'last.keras'} after {initial_epoch} epochs.",
              flush=True)
        model = keras.models.load_model(output / "last.keras")
        if int(model.optimizer.iterations.numpy()) != state["optimizer_iterations"]:
            raise ValueError("Checkpoint / state mismatch; use a consistent checkpoint.")
    else:
        print("Starting from random weights.", flush=True)
        model = build_training_model()

    model.summary(print_fn=lambda s: print(s, flush=True))

    output.mkdir(parents=True, exist_ok=True)
    cfg_path = output / f"run_from_epoch_{initial_epoch:04d}.json"
    cfg_path.write_text(json.dumps(vars(args), default=str, indent=2) + "\n",
                        encoding="utf-8")

    scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor="val_mean_iou", mode="max",
        factor=0.5, patience=4, min_lr=1e-6, verbose=1,
    )
    best_cb = keras.callbacks.ModelCheckpoint(
        str(output / "best.keras"),
        monitor="val_mean_iou", mode="max",
        save_best_only=True,
        initial_value_threshold=state.get("best_val_mean_iou"),
        verbose=1,
    )
    callbacks = [
        keras.callbacks.TerminateOnNaN(),
        VerifyFinite(),
        scheduler,
        RestoreScheduler(scheduler, state.get("scheduler", {})),
        best_cb,
        keras.callbacks.ModelCheckpoint(str(output / "last.keras")),
        keras.callbacks.CSVLogger(str(output / "history.csv"), append=args.resume),
        SaveState(output / "state.json", best_cb, scheduler),
    ]
    if args.smoke_test:
        callbacks.append(SmokeChecks())

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=initial_epoch + args.epochs,
        callbacks=callbacks,
        verbose=2,
        shuffle=False,  # shuffling handled inside dataset pipeline
    )
    print(
        "Final epoch:",
        {k: float(v[-1]) for k, v in history.history.items()},
        flush=True,
    )
    print(f"Checkpoints saved in {output}", flush=True)


if __name__ == "__main__":
    main()
