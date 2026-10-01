import os

from dataset_config import CITYSCAPES_ROOT

import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================


# Cityscapes original resolution:
# 1024 x 2048
#
# We start with half resolution to make training more manageable.
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 1024

BATCH_SIZE = int(os.environ.get("PATHCLEAR_BATCH_SIZE", "2"))
if BATCH_SIZE < 1:
    raise ValueError("PATHCLEAR_BATCH_SIZE must be at least 1.")
NUM_CLASSES = 19

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# FIND IMAGE / MASK PAIRS
# ============================================================

def collect_pairs(split):
    image_root = CITYSCAPES_ROOT / "leftImg8bit" / split
    mask_root = CITYSCAPES_ROOT / "gtFine" / split

    image_paths = sorted(
        image_root.glob("*/*_leftImg8bit.png")
    )

    paired_images = []
    paired_masks = []
    missing_masks = []

    for image_path in image_paths:
        city = image_path.parent.name

        mask_name = image_path.name.replace(
            "_leftImg8bit.png",
            "_gtFine_labelTrainIds.png"
        )

        mask_path = mask_root / city / mask_name

        if mask_path.exists():
            paired_images.append(str(image_path))
            paired_masks.append(str(mask_path))
        else:
            missing_masks.append(mask_path)

    print(f"\n[{split.upper()}]")
    print(f"Images found:       {len(image_paths)}")
    print(f"Valid image/masks:  {len(paired_images)}")
    print(f"Missing masks:      {len(missing_masks)}")

    if len(paired_images) == 0:
        raise RuntimeError(
            f"No image/mask pairs found for split '{split}'. "
            f"Check CITYSCAPES_ROOT: {CITYSCAPES_ROOT}"
        )

    return paired_images, paired_masks


# ============================================================
# LOAD ONE IMAGE + MASK
# ============================================================

def load_pair(image_path, mask_path):

    # --------------------------------------------------------
    # RGB IMAGE
    # --------------------------------------------------------

    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_png(
        image_bytes,
        channels=3
    )

    # Bilinear interpolation is appropriate for RGB images.
    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method=tf.image.ResizeMethod.BILINEAR
    )

    # Convert RGB values:
    # 0-255  ->  0.0-1.0
    image = tf.cast(image, tf.float32) / 255.0


    # --------------------------------------------------------
    # SEGMENTATION MASK
    # --------------------------------------------------------

    mask_bytes = tf.io.read_file(mask_path)

    mask = tf.io.decode_png(
        mask_bytes,
        channels=1
    )

    # IMPORTANT:
    #
    # Segmentation masks MUST use nearest-neighbor resizing.
    #
    # Bilinear interpolation would create invalid values
    # between the class IDs.
    mask = tf.image.resize(
        mask,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    mask = tf.cast(mask, tf.int32)

    # Before:
    # (512, 1024, 1)
    #
    # After:
    # (512, 1024)
    mask = tf.squeeze(mask, axis=-1)


    # --------------------------------------------------------
    # HANDLE CITYSCAPES IGNORE / VOID PIXELS
    # --------------------------------------------------------
    #
    # Valid training classes:
    #
    # 0 - 18
    #
    # Cityscapes also uses:
    #
    # 255 = ignore / void
    #
    # We create a sample-weight mask so these pixels
    # do not contribute to the training loss.

    valid_pixels = tf.not_equal(mask, 255)

    sample_weight = tf.cast(
        valid_pixels,
        tf.float32
    )

    # Replace 255 with 0 inside the target tensor.
    #
    # This does NOT turn ignored pixels into road during training,
    # because their sample_weight is 0.
    mask = tf.where(
        valid_pixels,
        mask,
        tf.zeros_like(mask)
    )

    return image, mask, sample_weight


# ============================================================
# BUILD TENSORFLOW DATASET
# ============================================================

def make_dataset(split, training=False, batch_size=None):

    batch_size = BATCH_SIZE if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    image_paths, mask_paths = collect_pairs(split)

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, mask_paths)
    )

    # Shuffle only the training dataset.
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_pair,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(
        batch_size
    )

    # Prepare the next batch while the model is working
    # on the current batch.
    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


# ============================================================
# VISUALIZE ONE PROCESSED SAMPLE
# ============================================================

def visualize_sample(image, mask, weight):

    plt.figure(figsize=(14, 9))


    # --------------------------------------------------------
    # RGB IMAGE
    # --------------------------------------------------------

    plt.subplot(3, 1, 1)

    plt.imshow(image)

    plt.title(
        "Processed Cityscapes RGB Image"
    )

    plt.axis("off")


    # --------------------------------------------------------
    # 19-CLASS SEGMENTATION MASK
    # --------------------------------------------------------

    plt.subplot(3, 1, 2)

    plt.imshow(
        mask,
        vmin=0,
        vmax=NUM_CLASSES - 1
    )

    plt.title(
        "Processed 19-Class labelTrainIds Mask"
    )

    plt.axis("off")


    # --------------------------------------------------------
    # IGNORE / SAMPLE-WEIGHT MASK
    # --------------------------------------------------------

    plt.subplot(3, 1, 3)

    plt.imshow(
        weight,
        vmin=0,
        vmax=1
    )

    plt.title(
        "Training Weight Mask — Black = Ignored Pixels"
    )

    plt.axis("off")


    plt.tight_layout()

    plt.show()


# ============================================================
# TEST LOADER
# ============================================================

if __name__ == "__main__":

    print("\nBuilding Cityscapes datasets...")

    train_dataset = make_dataset(
        "train",
        training=True
    )

    val_dataset = make_dataset(
        "val",
        training=False
    )


    print("\nReading one training batch...")


    for images, masks, weights in train_dataset.take(1):

        print("\n========== DATASET TEST ==========")


        # ----------------------------------------------------
        # SHAPES
        # ----------------------------------------------------

        print("\nImage batch shape:")
        print(images.shape)

        print("\nMask batch shape:")
        print(masks.shape)

        print("\nWeight batch shape:")
        print(weights.shape)


        # ----------------------------------------------------
        # DATA TYPES
        # ----------------------------------------------------

        print("\nImage dtype:")
        print(images.dtype)

        print("\nMask dtype:")
        print(masks.dtype)


        # ----------------------------------------------------
        # IMAGE VALUE RANGE
        # ----------------------------------------------------

        print("\nImage value range:")

        print(
            float(tf.reduce_min(images)),
            "to",
            float(tf.reduce_max(images))
        )


        # ----------------------------------------------------
        # CLASSES FOUND
        # ----------------------------------------------------

        unique_classes = tf.unique(
            tf.reshape(masks, [-1])
        ).y

        unique_classes = tf.sort(
            unique_classes
        )

        print("\nClasses found in this batch:")

        print(
            unique_classes.numpy()
        )


        # ----------------------------------------------------
        # CHECK CLASS RANGE
        # ----------------------------------------------------

        min_class = int(
            tf.reduce_min(masks)
        )

        max_class = int(
            tf.reduce_max(masks)
        )

        print("\nClass ID range:")

        print(
            min_class,
            "to",
            max_class
        )


        # ----------------------------------------------------
        # COUNT IGNORED PIXELS
        # ----------------------------------------------------

        ignored_pixels = tf.reduce_sum(
            tf.cast(
                weights == 0,
                tf.int32
            )
        )

        print("\nIgnored pixels in batch:")

        print(
            int(ignored_pixels)
        )


        # ----------------------------------------------------
        # BASIC VALIDATION
        # ----------------------------------------------------

        assert images.shape[-1] == 3, (
            "Images should have 3 RGB channels."
        )

        assert min_class >= 0, (
            "Found invalid negative class ID."
        )

        assert max_class < NUM_CLASSES, (
            f"Found class ID outside 0-{NUM_CLASSES - 1}."
        )


        print("\nLoader test successful.")


        # ----------------------------------------------------
        # VISUAL CHECK
        # ----------------------------------------------------

        print(
            "\nOpening processed sample visualization..."
        )

        visualize_sample(
            images[0].numpy(),
            masks[0].numpy(),
            weights[0].numpy()
        )
