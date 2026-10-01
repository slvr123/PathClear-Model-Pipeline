"""Plot recorded adaptation history without training or modifying checkpoints."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    with args.history.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) + 1 for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, label, scale in zip(axes, ("mean_iou", "loss", "pixel_accuracy"),
                                    ("Mean IoU (%)", "Loss", "Pixel accuracy (%)"), (100, 1, 100)):
        ax.plot(epochs, [float(row[key]) * scale for row in rows], "o-", label="Training", color="#1f77b4")
        ax.plot(epochs, [float(row["val_" + key]) * scale for row in rows], "o-", label="Validation", color="#d95f02")
        ax.set(xlabel="Adaptation epoch", ylabel=label, xticks=epochs)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("PathClear: seven active classes, frozen features | Cityscapes validation")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
