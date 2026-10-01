"""Compare saved histories and aggregate the old baseline's predictions to seven IDs.

No inference, training, checkpoint writes, or edits to prior reports occur.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathclear_taxonomy import CITYSCAPES_NAMES, CITYSCAPES_TO_PATHCLEAR, CLASS_NAMES

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "checkpoints/baseline-5epochs"
NEW = ROOT / "checkpoints/pathclear_10class/cityscapes_epochs2_11_2026-09-29"
OUT = ROOT / "output/comparisons/cityscapes_baseline_vs_adaptation_2026-09-29"


def metrics(cm):
    # Seven true classes and eight prediction columns. Column 7 represents
    # old predictions outside the taxonomy; these remain errors, never dropped.
    support = cm.sum(axis=1)
    predicted = cm[:, :7].sum(axis=0)
    correct = np.diag(cm[:, :7])
    union = support + predicted - correct
    return {"pixel_accuracy": float(correct.sum() / support.sum()),
            "mean_iou": float(np.mean(correct / union)),
            "macro_dice": float(np.mean(2 * correct / (support + predicted))),
            "pixels": int(support.sum()), "unsupported_prediction_pixels": int(cm[:, 7].sum()),
            "class_support": support.tolist(),
            "per_class_iou": dict(zip(CLASS_NAMES[:7], (correct / union).tolist()))}


def history(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    old_meta = json.loads((OLD / "evaluation/metrics.json").read_text())
    with (OLD / "best.keras").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == old_meta["sha256_before"]
    assert old_meta["images"] == 500 and old_meta["resolution"] == [512, 1024]
    with (OLD / "evaluation/confusion_matrix.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0][1:] == list(CITYSCAPES_NAMES)
    assert [row[0] for row in rows[1:]] == list(CITYSCAPES_NAMES)
    original = np.array([[int(value) for value in row[1:]] for row in rows[1:]], dtype=np.int64)
    assert original.shape == (19, 19) and original.sum() == old_meta["valid_pixels"]
    merged = np.zeros((7, 8), dtype=np.int64)
    for truth in range(19):
        new_truth = CITYSCAPES_TO_PATHCLEAR[truth]
        if new_truth == 255:
            continue
        for predicted in range(19):
            new_predicted = CITYSCAPES_TO_PATHCLEAR[predicted]
            merged[new_truth, 7 if new_predicted == 255 else new_predicted] += original[truth, predicted]
    current = np.load(NEW / "epoch_11_validation_confusion.npy")
    assert current.shape == (10, 10) and not np.any(current[7:, :]) and not np.any(current[:, 7:])
    np.testing.assert_array_equal(merged.sum(axis=1), current[:7, :7].sum(axis=1))
    new_cm = np.pad(current[:7, :7], ((0, 0), (0, 1)))
    old_scores, new_scores = metrics(merged), metrics(new_cm)
    new_result = json.loads((NEW / "result.json").read_text())
    np.testing.assert_allclose(new_scores["mean_iou"], new_result["best_validation"]["mean_iou"], atol=1e-6)
    with (NEW / "best.keras").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == new_result["checkpoint_sha256"]
    baseline_history = history(OLD / "history.csv")
    adaptation_history = history(NEW / "history.csv")
    result = {"old_best_epoch": 19, "new_best_adaptation_epoch": 11,
              "old_rescored_on_seven_classes": old_scores, "new_on_seven_classes": new_scores,
              "matching_ground_truth_support_verified": True,
              "method": "Map the old model's already-selected argmax class to the seven-class taxonomy. Ignore truth building/sky/terrain; count old predictions of those classes on valid truth as errors. No probabilities were pooled or new predictions generated.",
              "limitation": "This compares remapped hard predictions, not a probability-summing adapter. Same validation support was verified; validation has been used for model selection.",
              "baseline_history": baseline_history, "adaptation_history": adaptation_history}
    OUT.mkdir(parents=True)
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.save(OUT / "old_baseline_seven_class_confusion.npy", merged)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, rows, title in zip(axes, (baseline_history, adaptation_history),
                               ("Original model: 19 classes, full network training",
                                "Adaptation: 7 active classes, classifier-only training")):
        epochs = [int(row["epoch"]) + 1 for row in rows]
        ax.plot(epochs, [row["val_pixel_accuracy"] * 100 for row in rows], "o-", label="Validation pixel accuracy")
        ax.plot(epochs, [row["val_mean_iou"] * 100 for row in rows], "o-", label="Validation mean IoU")
        ax.set(title=title, xlabel="Epoch within this training stage", ylabel="Percent", ylim=(0, 100))
        ax.grid(alpha=.25)
        ax.legend(loc="lower right")
    fig.suptitle("Historical curves use different label sets and scored pixels; do not rank models from these raw scores")
    fig.tight_layout()
    fig.savefig(OUT / "historical_curves.png", dpi=150)
    plt.close(fig)
    print(json.dumps({key: value for key, value in result.items() if "history" not in key}, indent=2))


if __name__ == "__main__":
    main()
