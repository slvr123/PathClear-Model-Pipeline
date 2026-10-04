"""Tests for Cityscapes ignore-mask annotation helpers."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from scripts.annotate_cityscapes_ignore import (
    ignore_mask_path,
    load_candidate_pairs,
    load_review_state,
    rasterize_polygons,
    review_state_path,
    save_review_state,
    save_reviewed_mask,
)


class CityscapesIgnoreAnnotationTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.cityscapes_root = self.root / "cityscapes"

    def tearDown(self):
        self._temporary.cleanup()

    def test_ignore_mask_path_mirrors_split_and_city(self):
        image = (
            self.cityscapes_root / "leftImg8bit" / "train" / "aachen"
            / "aachen_000000_000019_leftImg8bit.png"
        )
        ignore_root = self.cityscapes_root / "ignore_masks"
        with mock.patch(
            "scripts.annotate_cityscapes_ignore.CITYSCAPES_ROOT",
            self.cityscapes_root,
        ):
            result = ignore_mask_path(image, ignore_root)

        self.assertEqual(
            result,
            ignore_root / "train" / "aachen"
            / "aachen_000000_000019_ignore.png",
        )

    def test_ignore_mask_path_rejects_image_outside_cityscapes(self):
        with mock.patch(
            "scripts.annotate_cityscapes_ignore.CITYSCAPES_ROOT",
            self.cityscapes_root,
        ):
            with self.assertRaisesRegex(ValueError, "must be under"):
                ignore_mask_path(self.root / "outside.png", self.root / "masks")

    def test_rasterize_polygons_produces_binary_source_size_mask(self):
        mask = np.asarray(rasterize_polygons(
            (8, 6),
            [[(1, 1), (5, 1), (5, 4), (1, 4)]],
        ))

        self.assertEqual(mask.shape, (6, 8))
        self.assertEqual(set(np.unique(mask)), {0, 255})
        self.assertEqual(int(mask[2, 3]), 255)
        self.assertEqual(int(mask[0, 0]), 0)

    def test_rasterize_polygons_rejects_too_few_points(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            rasterize_polygons((8, 6), [[(1, 1), (5, 1)]])

    def test_save_reviewed_mask_writes_binary_png(self):
        path = self.root / "train" / "aachen" / "frame_ignore.png"
        mask = np.zeros((4, 7), dtype=np.uint8)
        mask[1:3, 2:5] = 255

        save_reviewed_mask(path, mask)

        with Image.open(path) as saved:
            array = np.asarray(saved)
            self.assertEqual(saved.mode, "L")
            self.assertEqual(saved.size, (7, 4))
            np.testing.assert_array_equal(array, mask)
        self.assertFalse(path.with_name("frame_ignore.pending.png").exists())

    def test_save_reviewed_mask_records_empty_review(self):
        path = self.root / "train" / "aachen" / "empty_ignore.png"
        save_reviewed_mask(path, np.zeros((3, 5), dtype=np.uint8))

        with Image.open(path) as saved:
            np.testing.assert_array_equal(
                np.asarray(saved), np.zeros((3, 5), dtype=np.uint8)
            )

    def test_save_reviewed_mask_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "only 0 or 255"):
            save_reviewed_mask(
                self.root / "invalid.png",
                np.array([[0, 1], [0, 255]], dtype=np.uint8),
            )

    def test_candidate_queue_preserves_score_order_and_excludes_rejected(self):
        report = self.root / "candidates.json"
        report.write_text(json.dumps({
                "schema_version": 1,
                "dataset_root": str(self.cityscapes_root),
                "split": "train",
                "candidates": [
                    {"image": "leftImg8bit/train/b/image.png", "score": 0.9},
                    {
                        "image": "leftImg8bit/train/a/image.png",
                        "score": 0.2,
                        "candidate": False,
                    },
                ],
            }), encoding="utf-8")
        pairs = [
            (
                str(self.cityscapes_root / "leftImg8bit" / "train" / "b" / "image.png"),
                "b_mask",
            ),
            (
                str(self.cityscapes_root / "leftImg8bit" / "train" / "a" / "image.png"),
                "a_mask",
            ),
        ]
        with mock.patch(
            "scripts.annotate_cityscapes_ignore.CITYSCAPES_ROOT",
            self.cityscapes_root,
        ):
            selected, scores = load_candidate_pairs(
                report, "train", self.cityscapes_root, pairs
            )

        self.assertEqual(selected, [pairs[0]])
        self.assertEqual(scores["leftImg8bit/train/b/image.png"], 0.9)

    def test_candidate_queue_can_sample_rejected_images_deterministically(self):
        report = self.root / "candidates.json"
        report.write_text(json.dumps({
            "schema_version": 1,
            "dataset_root": str(self.cityscapes_root),
            "split": "train",
            "candidates": [
                {"image": "leftImg8bit/train/candidate.png", "score": 0.9},
            ],
            "scored_images": [
                {
                    "image": "leftImg8bit/train/candidate.png",
                    "score": 0.9,
                    "candidate": True,
                },
                {
                    "image": "leftImg8bit/train/rejected.png",
                    "score": 0.01,
                    "candidate": False,
                },
            ],
        }), encoding="utf-8")
        pairs = [
            (
                str(
                    self.cityscapes_root / "leftImg8bit" / "train"
                    / "candidate.png"
                ),
                "candidate_mask",
            ),
            (
                str(
                    self.cityscapes_root / "leftImg8bit" / "train"
                    / "rejected.png"
                ),
                "rejected_mask",
            ),
        ]

        with mock.patch(
            "scripts.annotate_cityscapes_ignore.CITYSCAPES_ROOT",
            self.cityscapes_root,
        ):
            selected, _ = load_candidate_pairs(
                report,
                "train",
                self.cityscapes_root,
                pairs,
                rejected_sample_size=1,
                seed=42,
            )

        self.assertEqual(selected, [(pairs[1][0], "rejected_mask")])

    def test_candidate_review_progress_is_saved_and_bound_to_report(self):
        report = self.root / "train.json"
        report.write_text(
            json.dumps({
                "schema_version": 1,
                "dataset_root": str(self.cityscapes_root),
                "split": "train",
                "candidates": [],
            }),
            encoding="utf-8",
        )
        state_path, reviews = load_review_state(
            report, "train", self.cityscapes_root
        )
        reviews["leftImg8bit/train/aachen/frame.png"] = {
            "status": "rejected"
        }
        save_review_state(
            report, "train", self.cityscapes_root, state_path, reviews
        )
        loaded_path, loaded_reviews = load_review_state(
            report, "train", self.cityscapes_root
        )
        saved_state = json.loads(loaded_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded_path, review_state_path(report))
        self.assertEqual(loaded_reviews, reviews)
        self.assertEqual(saved_state["reviewed_count"], 1)
        self.assertEqual(saved_state["rejected_count"], 1)
        report.write_text(report.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_review_state(report, "train", self.cityscapes_root)


if __name__ == "__main__":
    unittest.main()
