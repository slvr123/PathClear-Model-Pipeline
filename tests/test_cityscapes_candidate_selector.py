"""Tests for non-authoritative Cityscapes candidate ranking."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.select_cityscapes_candidates import (
    score_cityscapes_pair,
    select_candidates,
)


class CityscapesCandidateSelectorTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def write_scene(self, name, crossing, stripe_value=230, single_band=False):
        height, width = 96, 128
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = (35, 40, 45) if stripe_value < 155 else (90, 110, 130)
        labels = np.full((height, width), 10, dtype=np.uint8)
        labels[32:, :] = 0
        if crossing:
            rows = (60,) if single_band else (48, 54, 60, 66, 72, 78)
            for row in rows:
                image[row:row + 3, 24:104] = stripe_value
        image_path = self.root / f"{name}_leftImg8bit.png"
        mask_path = self.root / f"{name}_gtFine_labelTrainIds.png"
        Image.fromarray(image).save(image_path)
        Image.fromarray(labels).save(mask_path)
        return image_path, mask_path

    def test_repeated_bright_road_bands_raise_candidate_score(self):
        crossing = self.write_scene("crossing", True)
        ordinary_road = self.write_scene("ordinary", False)

        crossing_score = score_cityscapes_pair(
            str(crossing[0]), str(crossing[1])
        )["score"]
        ordinary_score = score_cityscapes_pair(
            str(ordinary_road[0]), str(ordinary_road[1])
        )["score"]

        self.assertGreater(crossing_score, ordinary_score)
        self.assertGreater(crossing_score, 0.0)

    def test_selector_retains_scored_images_and_only_marks_candidates(self):
        crossing = self.write_scene("crossing", True)
        ordinary_road = self.write_scene("ordinary", False)

        records = select_candidates(
            [tuple(map(str, crossing)), tuple(map(str, ordinary_road))],
            self.root,
            score_threshold=0.08,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(
            {record["image"] for record in records if record["candidate"]},
            {"crossing_leftImg8bit.png"},
        )
        self.assertTrue(all(record["review_status"] == "pending" for record in records))

    def test_adaptive_luminance_finds_dim_repeated_crosswalk_bands(self):
        dim_crossing = self.write_scene("dim_crossing", True, stripe_value=90)
        score = score_cityscapes_pair(*map(str, dim_crossing))["score"]
        self.assertGreaterEqual(score, 0.08)

    def test_single_broad_road_marking_is_not_a_transverse_crossing_candidate(self):
        single_band = self.write_scene(
            "single_band", True, stripe_value=230, single_band=True
        )
        score = score_cityscapes_pair(*map(str, single_band))["score"]
        self.assertLess(score, 0.08)


if __name__ == "__main__":
    unittest.main()
