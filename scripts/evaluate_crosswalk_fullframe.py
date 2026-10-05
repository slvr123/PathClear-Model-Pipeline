"""Re-score a saved MicroSeg Crosswalk checkpoint with penalised metrics.

The Kaggle label contract ignores every unmarked pixel, so the training-time
weighted metrics can only see the marked crossing. A model that paints
Crosswalk across the whole frame is therefore not penalised. This script
reports the full-frame Crosswalk scores next to the legacy ignored-pixel
scores so the difference is visible instead of inferred.
"""
import argparse
import json
from pathlib import Path

import keras

from dataset_config import KAGGLE_CROSSWALK_ROOT
from microseg.dataset import BATCH_SIZE, make_crosswalk_dataset
from microseg.model import NUM_CLASSES
from microseg.taxonomy import CLASS_NAMES
from microseg.train import CrosswalkFullFrameMetrics, _pairs_from_manifest
from segmentation_metrics import make_metrics

_CROSSWALK_CLASS_ID = 3


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="best.keras")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = _arguments(argv)
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "crosswalk_split.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Crosswalk split manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest["dataset_root"])
    pairs = _pairs_from_manifest(manifest, root, "validation", source="real")
    if not pairs:
        raise ValueError("Held-out real Crosswalk validation split is empty.")
    checkpoint = run_dir / args.checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    model = keras.models.load_model(checkpoint, compile=False)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="mean_with_sample_weight"
        ),
        weighted_metrics=make_metrics(NUM_CLASSES),
    )
    dataset = make_crosswalk_dataset(
        training=False, batch_size=args.batch_size, root=root, pairs=pairs
    )

    legacy = model.evaluate(dataset, verbose=0, return_dict=True)
    logs = {}
    callback = CrosswalkFullFrameMetrics(dataset, class_id=_CROSSWALK_CLASS_ID)
    callback.set_model(model)
    callback.on_epoch_end(0, logs)

    report = {
        "checkpoint": str(checkpoint),
        "held_out_real_pairs": len(pairs),
        "class_names": list(CLASS_NAMES),
        "legacy_ignored_pixel_metrics": {
            key: float(value) for key, value in legacy.items()
        },
        "penalised_full_frame_metrics": {
            key: float(value) for key, value in logs.items()
        },
        "note": (
            "Legacy metrics ignore every unmarked pixel, so a Crosswalk flood "
            "is invisible to them. The full-frame metrics count a prediction "
            "on an unmarked pixel as a false positive."
        ),
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
