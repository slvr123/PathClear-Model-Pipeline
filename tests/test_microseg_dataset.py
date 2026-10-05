"""Synthetic checks for the 96 x 96 MicroSeg Cityscapes loader."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import tensorflow as tf

from microseg.dataset import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    attach_cityscapes_ignore_masks,
    collect_crosswalk_pairs,
    load_crosswalk_pair,
    load_pair,
    make_combined_dataset,
    make_crosswalk_dataset,
    make_dataset,
    prepare_targets,
)
from microseg.taxonomy import IGNORE_LABEL, NUM_CLASSES
import microseg.dataset as microseg_dataset
from segmentation_metrics import make_metrics


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_pair(self, mask_ids, index=0):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[..., 1] = 200
        image_path = self.root / f"image{index}.png"
        mask_path = self.root / f"mask{index}.png"
        Image.fromarray(image).save(image_path)
        Image.fromarray(np.asarray(mask_ids, dtype=np.uint8)).save(mask_path)
        return image_path, mask_path

    def test_prepare_targets_zero_weights_on_ignore(self):
        mask = tf.constant([[0, 3, IGNORE_LABEL]], dtype=tf.int32)
        targets, weights = prepare_targets(mask)
        np.testing.assert_array_equal(np.asarray(targets), [[0, 3, 0]])
        np.testing.assert_array_equal(np.asarray(weights), [[1.0, 1.0, 0.0]])

    def test_prepare_targets_rejects_out_of_range(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            prepare_targets(tf.constant([[NUM_CLASSES]], dtype=tf.int32))

    def test_prepare_targets_rejects_float(self):
        with self.assertRaises(TypeError):
            prepare_targets(tf.constant([[0.5]], dtype=tf.float32))

    def test_load_pair_shapes_dtypes_and_normalization(self):
        image_path, mask_path = self.write_pair([[0, 1], [10, 3]])
        image, targets, weights = load_pair(str(image_path), str(mask_path))
        self.assertEqual(tuple(image.shape), (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        self.assertEqual(tuple(targets.shape), (IMAGE_HEIGHT, IMAGE_WIDTH))
        self.assertEqual(tuple(weights.shape), (IMAGE_HEIGHT, IMAGE_WIDTH))
        self.assertEqual(image.dtype, tf.float32)
        self.assertGreaterEqual(float(tf.reduce_min(image)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(image)), 1.0)
        # Mask and image share the exact same geometry.
        self.assertEqual(targets.shape, image.shape[:2])
        self.assertEqual(weights.shape, image.shape[:2])

    def test_load_pair_applies_four_class_remap(self):
        # Cityscapes trainIds: road(0)->1, sidewalk(1)->0, sky(10)->ignore,
        # wall(3)->2. Upsampled with nearest neighbor, the corners are stable.
        image_path, mask_path = self.write_pair([[0, 1], [10, 3]])
        _, targets, weights = load_pair(str(image_path), str(mask_path))
        targets = np.asarray(targets)
        weights = np.asarray(weights)
        self.assertEqual(int(targets[0, 0]), 1)
        self.assertEqual(int(targets[0, -1]), 0)
        self.assertEqual(int(targets[-1, -1]), 2)
        # Sky is ignored: zero weight and a safe in-range placeholder target.
        self.assertEqual(float(weights[-1, 0]), 0.0)
        self.assertEqual(int(targets[-1, 0]), 0)
        # Supervised pixels always carry a valid class ID.
        supervised = weights > 0
        self.assertTrue(bool(supervised.any()))
        self.assertLess(int(targets[supervised].max()), NUM_CLASSES)

    def test_load_pair_applies_original_label_ids_remap(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        # Original Cityscapes labelIds: road 7, sidewalk 8, wall 12, void 0.
        image_path = self.root / "city_image.png"
        mask_path = self.root / "city_gtFine_labelIds.png"
        Image.fromarray(image).save(image_path)
        Image.fromarray(np.array([[7, 8], [0, 12]], dtype=np.uint8)).save(mask_path)

        _, targets, weights = load_pair(str(image_path), str(mask_path))

        targets = np.asarray(targets)
        weights = np.asarray(weights)
        self.assertEqual(int(targets[0, 0]), 1)
        self.assertEqual(int(targets[0, -1]), 0)
        self.assertEqual(int(targets[-1, -1]), 2)
        self.assertEqual(int(targets[-1, 0]), 0)
        self.assertEqual(float(weights[-1, 0]), 0.0)

    def test_load_pair_applies_reviewed_ignore_mask_without_changing_other_road(self):
        image_path = self.root / "city_image.png"
        mask_path = self.root / "city_gtFine_labelIds.png"
        ignore_path = self.root / "city_ignore.png"
        Image.fromarray(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)).save(
            image_path
        )
        Image.fromarray(
            np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 7, dtype=np.uint8)
        ).save(mask_path)
        ignore_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        ignore_mask[40:56, 40:56] = 1
        Image.fromarray(ignore_mask).save(ignore_path)

        _, targets, weights = load_pair(
            str(image_path),
            str(mask_path),
            ignore_mask_path=str(ignore_path),
        )

        self.assertTrue(bool(tf.reduce_all(tf.equal(targets[:40], 1))))
        self.assertTrue(bool(tf.reduce_all(tf.equal(weights[:40], 1.0))))
        self.assertTrue(bool(tf.reduce_all(tf.equal(targets[40:56, 40:56], 0))))
        self.assertTrue(bool(tf.reduce_all(tf.equal(weights[40:56, 40:56], 0.0))))
        self.assertEqual(int(targets[56, 56]), 1)
        self.assertEqual(float(weights[56, 56]), 1.0)

    def test_load_pair_rejects_misaligned_ignore_mask(self):
        image_path, mask_path = self.write_pair([[7, 7], [7, 7]])
        ignore_path = self.root / "misaligned_ignore.png"
        Image.fromarray(np.zeros((3, 2), dtype=np.uint8)).save(ignore_path)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            load_pair(
                str(image_path),
                str(mask_path),
                ignore_mask_path=str(ignore_path),
            )

    def test_load_pair_rejects_misaligned_semantic_mask(self):
        image_path = self.root / "image.png"
        mask_path = self.root / "mask.png"
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_path)
        Image.fromarray(np.full((3, 2), 7, dtype=np.uint8)).save(mask_path)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            load_pair(str(image_path), str(mask_path))

    def test_load_pair_rejects_non_binary_ignore_mask(self):
        image_path, mask_path = self.write_pair([[7, 7], [7, 7]])
        ignore_path = self.root / "invalid_ignore.png"
        Image.fromarray(np.full((2, 2), 2, dtype=np.uint8)).save(ignore_path)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            load_pair(
                str(image_path),
                str(mask_path),
                ignore_mask_path=str(ignore_path),
            )

    def test_make_dataset_preserves_ignore_mask_weight(self):
        image_path = self.root / "image.png"
        mask_path = self.root / "mask_labelIds.png"
        ignore_path = self.root / "ignore.png"
        Image.fromarray(
            np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        ).save(image_path)
        Image.fromarray(
            np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 7, dtype=np.uint8)
        ).save(mask_path)
        ignore = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        ignore[10:20, 10:20] = 255
        Image.fromarray(ignore).save(ignore_path)

        _, targets, weights = next(iter(make_dataset(
            "train",
            batch_size=1,
            cityscapes_pairs=[(
                str(image_path), str(mask_path), str(ignore_path)
            )],
        )))

        self.assertEqual(int(targets[0, 15, 15]), 0)
        self.assertEqual(float(weights[0, 15, 15]), 0.0)
        self.assertEqual(int(targets[0, 25, 25]), 1)
        self.assertEqual(float(weights[0, 25, 25]), 1.0)

    def test_weighted_loss_is_independent_of_ignored_target_values(self):
        import keras

        logits = tf.constant([[[[2.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 2.0]]]])
        weights = tf.constant([[[1.0, 0.0]]])
        first_targets = tf.constant([[[0, 0]]], dtype=tf.int32)
        second_targets = tf.constant([[[0, 3]]], dtype=tf.int32)
        loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="mean_with_sample_weight"
        )

        first = loss(first_targets, logits, sample_weight=weights)
        second = loss(second_targets, logits, sample_weight=weights)

        self.assertAlmostEqual(float(first), float(second), places=7)

    def test_weighted_metrics_are_independent_of_ignored_target_values(self):
        logits = tf.constant([[[[2.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 2.0]]]])
        weights = tf.constant([[[1.0, 0.0]]])
        first_targets = tf.constant([[[0, 0]]], dtype=tf.int32)
        second_targets = tf.constant([[[0, 3]]], dtype=tf.int32)

        first_results = []
        second_results = []
        for metric in make_metrics(NUM_CLASSES):
            metric.update_state(first_targets, logits, sample_weight=weights)
            first_results.append(float(metric.result()))
        for metric in make_metrics(NUM_CLASSES):
            metric.update_state(second_targets, logits, sample_weight=weights)
            second_results.append(float(metric.result()))

        np.testing.assert_allclose(first_results, second_results, rtol=0, atol=0)

    def test_attach_cityscapes_ignore_masks_allows_unreviewed_images(self):
        cityscapes_root = self.root / "cityscapes"
        image_path = (
            cityscapes_root / "leftImg8bit" / "train" / "aachen"
            / "aachen_000000_000019_leftImg8bit.png"
        )
        label_path = (
            cityscapes_root / "gtFine" / "train" / "aachen"
            / "aachen_000000_000019_gtFine_labelIds.png"
        )
        ignore_root = cityscapes_root / "ignore_masks"
        ignore_path = (
            ignore_root / "train" / "aachen"
            / "aachen_000000_000019_ignore.png"
        )
        ignore_path.parent.mkdir(parents=True)
        ignore_path.touch()

        with mock.patch.object(microseg_dataset, "CITYSCAPES_ROOT", cityscapes_root):
            pairs = attach_cityscapes_ignore_masks(
                [(str(image_path), str(label_path))], ignore_root
            )

        self.assertEqual(pairs, [(str(image_path), str(label_path), str(ignore_path))])

        ignore_path.unlink()
        with mock.patch.object(microseg_dataset, "CITYSCAPES_ROOT", cityscapes_root):
            pairs = attach_cityscapes_ignore_masks(
                [(str(image_path), str(label_path))], ignore_root
            )
        self.assertEqual(pairs, [(str(image_path), str(label_path), "")])

    def test_make_dataset_auto_detects_label_ids_in_tf_data(self):
        image_path = self.root / "city_image.png"
        mask_path = self.root / "city_gtFine_labelIds.png"
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_path)
        Image.fromarray(np.array([[7, 8], [0, 12]], dtype=np.uint8)).save(mask_path)

        with mock.patch.object(
            microseg_dataset,
            "collect_cityscapes_pairs",
            return_value=([str(image_path)], [str(mask_path)]),
        ):
            _, targets, weights = next(
                iter(make_dataset("train", batch_size=1))
            )

        self.assertTrue(bool(tf.reduce_any(tf.equal(targets, 1) & (weights > 0))))
        self.assertTrue(bool(tf.reduce_any(tf.equal(targets, 0) & (weights > 0))))
        self.assertTrue(bool(tf.reduce_any(tf.equal(targets, 2) & (weights > 0))))

    def test_collect_cityscapes_pairs_uses_label_ids_fallback(self):
        cityscapes_root = self.root / "cityscapes"
        image_path = (
            cityscapes_root / "leftImg8bit" / "train" / "aachen"
            / "aachen_000000_000019_leftImg8bit.png"
        )
        mask_path = (
            cityscapes_root / "gtFine" / "train" / "aachen"
            / "aachen_000000_000019_gtFine_labelIds.png"
        )
        image_path.parent.mkdir(parents=True)
        mask_path.parent.mkdir(parents=True)
        image_path.touch()
        mask_path.touch()

        with mock.patch.object(microseg_dataset, "CITYSCAPES_ROOT", cityscapes_root):
            images, masks = microseg_dataset.collect_cityscapes_pairs("train")

        self.assertEqual(images, [str(image_path)])
        self.assertEqual(masks, [str(mask_path)])

    def test_collect_cityscapes_pairs_prefers_train_ids_if_both_exist(self):
        cityscapes_root = self.root / "cityscapes"
        image_path = (
            cityscapes_root / "leftImg8bit" / "train" / "aachen"
            / "aachen_000000_000019_leftImg8bit.png"
        )
        mask_dir = cityscapes_root / "gtFine" / "train" / "aachen"
        train_ids_path = mask_dir / "aachen_000000_000019_gtFine_labelTrainIds.png"
        label_ids_path = mask_dir / "aachen_000000_000019_gtFine_labelIds.png"
        image_path.parent.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        image_path.touch()
        train_ids_path.touch()
        label_ids_path.touch()

        with mock.patch.object(microseg_dataset, "CITYSCAPES_ROOT", cityscapes_root):
            _, masks = microseg_dataset.collect_cityscapes_pairs("train")

        self.assertEqual(masks, [str(train_ids_path)])

    def test_make_dataset_batches(self):
        image_paths, mask_paths = [], []
        for index in range(2):
            image_path, mask_path = self.write_pair([[0, 1], [10, 3]], index)
            image_paths.append(str(image_path))
            mask_paths.append(str(mask_path))
        with mock.patch(
            "microseg.dataset.collect_cityscapes_pairs",
            return_value=(image_paths, mask_paths),
        ):
            dataset = make_dataset("train", training=False, batch_size=2)
            images, targets, weights = next(iter(dataset))
        self.assertEqual(tuple(images.shape), (2, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        self.assertEqual(tuple(targets.shape), (2, IMAGE_HEIGHT, IMAGE_WIDTH))
        self.assertEqual(tuple(weights.shape), (2, IMAGE_HEIGHT, IMAGE_WIDTH))

    def test_collect_crosswalk_pairs_supports_mirrored_directories(self):
        image_path = self.root / "real" / "rainy" / "images" / "frame_01.jpg"
        mask_path = self.root / "real" / "rainy" / "masks" / "frame_01.png"
        image_path.parent.mkdir(parents=True)
        mask_path.parent.mkdir(parents=True)
        image_path.touch()
        mask_path.touch()

        images, masks = collect_crosswalk_pairs(self.root)

        self.assertEqual(images, [str(image_path)])
        self.assertEqual(masks, [str(mask_path)])

    def test_collect_crosswalk_pairs_supports_role_suffix_directories(self):
        image_path = self.root / "synthetic_images" / "night" / "frame.jpeg"
        mask_path = self.root / "synthetic_masks" / "night" / "frame.jpg"
        image_path.parent.mkdir(parents=True)
        mask_path.parent.mkdir(parents=True)
        image_path.touch()
        mask_path.touch()

        images, masks = collect_crosswalk_pairs(self.root)

        self.assertEqual(images, [str(image_path)])
        self.assertEqual(masks, [str(mask_path)])

    def test_collect_crosswalk_pairs_rejects_unpaired_images(self):
        image_path = self.root / "real" / "images" / "frame.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.touch()

        with self.assertRaisesRegex(RuntimeError, "without masks"):
            collect_crosswalk_pairs(self.root)

    def test_collect_crosswalk_pairs_ignores_unlabeled_seed_images(self):
        seed_image = self.root / "Fine-tuning seed images" / "Crosswalk (1).jpg"
        image_path = self.root / "Synthetic dataset" / "sunny" / "images" / "img_1.jpg"
        mask_path = self.root / "Synthetic dataset" / "sunny" / "masks" / "img_1_mask.jpg"
        seed_image.parent.mkdir(parents=True)
        image_path.parent.mkdir(parents=True)
        mask_path.parent.mkdir(parents=True)
        seed_image.touch()
        image_path.touch()
        mask_path.touch()

        images, masks = collect_crosswalk_pairs(self.root)

        self.assertEqual(images, [str(image_path)])
        self.assertEqual(masks, [str(mask_path)])

    def test_collect_crosswalk_pairs_rejects_ambiguous_masks(self):
        image_path = self.root / "real" / "images" / "frame.jpg"
        masks = [
            self.root / "real" / "masks" / "frame.png",
            self.root / "real" / "annotations" / "frame.png",
        ]
        image_path.parent.mkdir(parents=True)
        image_path.touch()
        for mask_path in masks:
            mask_path.parent.mkdir(parents=True)
            mask_path.touch()

        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            collect_crosswalk_pairs(self.root)

    def test_load_crosswalk_pair_ignores_unmarked_mask_pixels(self):
        image = np.full((2, 2, 3), 100, dtype=np.uint8)
        mask = np.array([[0, 127], [128, 255]], dtype=np.uint8)
        image_path = self.root / "crosswalk.png"
        mask_path = self.root / "crosswalk_mask.png"
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)

        loaded_image, targets, weights = load_crosswalk_pair(
            str(image_path), str(mask_path)
        )

        self.assertEqual(tuple(loaded_image.shape), (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        self.assertEqual(tuple(targets.shape), (IMAGE_HEIGHT, IMAGE_WIDTH))
        self.assertEqual(tuple(weights.shape), (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Nearest-neighbor resize preserves the four source corner values.
        self.assertEqual(int(targets[0, 0]), 0)
        self.assertEqual(float(weights[0, 0]), 0.0)
        self.assertEqual(float(weights[0, -1]), 0.0)
        self.assertEqual(int(targets[-1, 0]), 3)
        self.assertEqual(float(weights[-1, 0]), 1.0)
        self.assertEqual(int(targets[-1, -1]), 3)
        self.assertEqual(float(weights[-1, -1]), 1.0)

    def test_make_crosswalk_dataset_batches_paired_files(self):
        image_path = self.root / "images" / "frame.png"
        mask_path = self.root / "masks" / "frame.png"
        image_path.parent.mkdir(parents=True)
        mask_path.parent.mkdir(parents=True)
        Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(image_path)
        mask_image = np.zeros((4, 4), dtype=np.uint8)
        mask_image[1:3, 1:3] = 255
        Image.fromarray(mask_image).save(mask_path)

        dataset = make_crosswalk_dataset(batch_size=1, root=self.root)
        images, targets, weights = next(iter(dataset))

        self.assertEqual(tuple(images.shape), (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        self.assertEqual(tuple(targets.shape), (1, IMAGE_HEIGHT, IMAGE_WIDTH))
        self.assertEqual(tuple(weights.shape), (1, IMAGE_HEIGHT, IMAGE_WIDTH))
        self.assertAlmostEqual(float(tf.reduce_min(images)), 128 / 255, places=6)
        self.assertTrue(bool(tf.reduce_any(tf.equal(targets, 3))))
        self.assertTrue(bool(tf.reduce_any(tf.equal(weights, 0.0))))

    def test_combined_validation_does_not_include_crosswalk_source(self):
        cityscapes_batch = tf.data.Dataset.from_tensors(
            (tf.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
             tf.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=tf.int32),
             tf.ones((1, IMAGE_HEIGHT, IMAGE_WIDTH)))
        )
        with mock.patch(
            "microseg.dataset.make_dataset", return_value=cityscapes_batch
        ), mock.patch("microseg.dataset.make_crosswalk_dataset") as make_crosswalk:
            result = make_combined_dataset("val", batch_size=1)

        self.assertIs(result, cityscapes_batch)
        make_crosswalk.assert_not_called()

    def test_combined_training_interleaves_crosswalk_batches(self):
        cityscapes_batch = tf.data.Dataset.from_tensors(
            (tf.fill((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), 0.2),
             tf.ones((1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=tf.int32),
             tf.ones((1, IMAGE_HEIGHT, IMAGE_WIDTH)))
        )
        crosswalk_batch = tf.data.Dataset.from_tensors(
            (tf.fill((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), 0.8),
             tf.fill(
                 (1, IMAGE_HEIGHT, IMAGE_WIDTH), tf.constant(3, dtype=tf.int32)
             ),
             tf.ones((1, IMAGE_HEIGHT, IMAGE_WIDTH)))
        )
        with mock.patch(
            "microseg.dataset.make_dataset", return_value=cityscapes_batch
        ), mock.patch(
            "microseg.dataset.make_crosswalk_dataset", return_value=crosswalk_batch
        ) as make_crosswalk:
            combined = make_combined_dataset(
                "train", training=True, batch_size=1, crosswalk_weight=0.5
            )
            batches = list(combined.as_numpy_iterator())

        make_crosswalk.assert_called_once()
        self.assertEqual(len(batches), 2)
        self.assertEqual({int(batch[1][0, 0, 0]) for batch in batches}, {1, 3})

    def test_combined_training_balances_real_and_synthetic_crosswalk_sources(self):
        cityscapes_batch = tf.data.Dataset.from_tensors(
            (tf.fill((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), 0.5),
             tf.ones((1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=tf.int32),
             tf.ones((1, IMAGE_HEIGHT, IMAGE_WIDTH)))
        )

        def crosswalk_batch(training, batch_size, root, pairs, negative_weight=0.0):
            self.assertEqual(negative_weight, 0.0)
            pixel_value = 0.25 if pairs == [("real", "real_mask")] else 0.75
            return tf.data.Dataset.from_tensors(
                (tf.fill((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), pixel_value),
                 tf.fill((1, IMAGE_HEIGHT, IMAGE_WIDTH), 3),
                 tf.ones((1, IMAGE_HEIGHT, IMAGE_WIDTH)))
            )

        tf.keras.utils.set_random_seed(21)
        with mock.patch(
            "microseg.dataset.make_dataset", return_value=cityscapes_batch
        ), mock.patch(
            "microseg.dataset.make_crosswalk_dataset",
            side_effect=crosswalk_batch,
        ):
            combined = make_combined_dataset(
                "train",
                training=True,
                batch_size=1,
                crosswalk_weight=0.99,
                crosswalk_real_weight=0.5,
                crosswalk_real_pairs=[("real", "real_mask")],
                crosswalk_synthetic_pairs=[("synthetic", "synthetic_mask")],
                cityscapes_pairs=[("city", "city_mask")],
            )
            pixels = [
                round(float(tf.reduce_mean(images)), 2)
                for images, _, _ in combined.take(201)
            ]

        real_count = pixels.count(0.25)
        synthetic_count = pixels.count(0.75)
        self.assertGreater(real_count, 0)
        self.assertGreater(synthetic_count, 0)
        self.assertGreaterEqual(
            real_count / (real_count + synthetic_count), 0.35
        )
        self.assertLessEqual(
            real_count / (real_count + synthetic_count), 0.65
        )


class CrosswalkNegativeSupervisionTests(unittest.TestCase):
    """Unmarked Crosswalk pixels must be able to push the Crosswalk logit down."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_masked_pair(self):
        image_path = self.root / "frame.png"
        mask_path = self.root / "frame_mask.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_path)
        Image.fromarray(
            np.array([[0, 0], [255, 255]], dtype=np.uint8)
        ).save(mask_path)
        return image_path, mask_path

    def test_default_contract_keeps_unmarked_pixels_ignored(self):
        image_path, mask_path = self.write_masked_pair()
        _, targets, weights = load_crosswalk_pair(str(image_path), str(mask_path))

        # Unmarked pixel: safe placeholder target and zero weight.
        self.assertEqual(int(targets[0, 0]), 0)
        self.assertEqual(float(weights[0, 0]), 0.0)
        # Marked pixel: Crosswalk at full weight.
        self.assertEqual(int(targets[-1, 0]), 3)
        self.assertEqual(float(weights[-1, 0]), 1.0)

    def test_negative_weight_turns_unmarked_pixels_into_weak_evidence(self):
        image_path, mask_path = self.write_masked_pair()
        _, targets, weights = load_crosswalk_pair(
            str(image_path), str(mask_path), negative_weight=0.25
        )

        negative_id = microseg_dataset.NEGATIVE_CLASS_ID
        self.assertEqual(int(targets[0, 0]), negative_id)
        self.assertAlmostEqual(float(weights[0, 0]), 0.25, places=6)
        # Marked pixels keep full weight.
        self.assertEqual(int(targets[-1, -1]), 3)
        self.assertAlmostEqual(float(weights[-1, -1]), 1.0, places=6)
        # Nothing is unsupervised any more, so a Crosswalk flood cannot hide.
        self.assertFalse(bool(tf.reduce_any(weights == 0.0)))

    def test_dataset_propagates_the_negative_weight(self):
        image_path = self.root / "images" / "frame.png"
        mask_path = self.root / "masks" / "frame.png"
        image_path.parent.mkdir(parents=True)
        mask_path.parent.mkdir(parents=True)
        Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(image_path)
        mask_image = np.zeros((4, 4), dtype=np.uint8)
        mask_image[1:3, 1:3] = 255
        Image.fromarray(mask_image).save(mask_path)

        dataset = make_crosswalk_dataset(
            batch_size=1, root=self.root, negative_weight=0.5
        )
        _, targets, weights = next(iter(dataset))

        self.assertAlmostEqual(float(tf.reduce_min(weights)), 0.5, places=6)
        self.assertAlmostEqual(float(tf.reduce_max(weights)), 1.0, places=6)
        self.assertTrue(bool(tf.reduce_any(tf.equal(targets, 3))))

    def test_out_of_range_negative_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            make_crosswalk_dataset(batch_size=1, root=self.root, negative_weight=1.5)

    def test_prepare_targets_scales_supervision(self):
        targets, weights = prepare_targets(
            tf.constant([[0, 3, IGNORE_LABEL]], dtype=tf.int32),
            tf.constant([[1.0, 1.0, 0.25]], dtype=tf.float32),
        )

        np.testing.assert_array_equal(np.asarray(targets), [[0, 3, 0]])
        # Ignored pixels stay at zero weight even when a weight map is given.
        np.testing.assert_allclose(np.asarray(weights), [[1.0, 1.0, 0.0]])

    def test_prepare_targets_defaults_to_full_weight(self):
        _, weights = prepare_targets(tf.constant([[0, IGNORE_LABEL]], dtype=tf.int32))

        np.testing.assert_allclose(np.asarray(weights), [[1.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
