from dataset_loader import make_dataset
from fast_scnn import build_fast_scnn


# ============================================================
# BUILD DATASET
# ============================================================

print("\nBuilding Cityscapes training dataset...")

train_dataset = make_dataset(
    "train",
    training=False
)


# ============================================================
# BUILD FAST-SCNN
# ============================================================

print("\nBuilding Fast-SCNN...")

model = build_fast_scnn()


# ============================================================
# GET ONE REAL CITYSCAPES BATCH
# ============================================================

print("\nLoading one Cityscapes batch...")

for images, masks, weights in train_dataset.take(1):

    print("\n========== INPUT ==========")

    print("Images:")
    print(images.shape)

    print("\nMasks:")
    print(masks.shape)

    print("\nSample weights:")
    print(weights.shape)


    # ========================================================
    # FORWARD PASS
    # ========================================================

    print("\nRunning Fast-SCNN forward pass...")

    logits = model(
        images,
        training=False
    )


    # ========================================================
    # OUTPUT CHECK
    # ========================================================

    print("\n========== MODEL OUTPUT ==========")

    print("Logits shape:")
    print(logits.shape)

    print("\nLogits dtype:")
    print(logits.dtype)


    expected_shape = (
        images.shape[0],
        512,
        1024,
        19
    )

    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"Unexpected output shape: {logits.shape}. "
            f"Expected: {expected_shape}"
        )


    print(
        "\nCityscapes -> Fast-SCNN forward pass successful."
    )