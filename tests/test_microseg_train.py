"""Tests for MicroSeg's reproducible, leakage-safe Crosswalk split."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

from microseg.train import (
    _read_cityscapes_manifest,
    create_crosswalk_split_manifest,
    parse_args,
)


class CrosswalkSplitTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def write_pair(self, relative_image):
        image_path = self.root / relative_image
        mask_path = (
            image_path.parent.parent / "masks" /
            f"{image_path.stem}_mask.jpg"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        Image.fromarray(mask).save(mask_path)

    def make_dataset(self):
        for video in ("clip_a", "clip_b", "clip_c"):
            for frame in (10, 20):
                self.write_pair(
                    Path("Real-world dataset") / "sunny" / "images" /
                    f"20260101_120000_{video}.mp4_{frame:09d} (Custom).jpg"
                )
        self.write_pair(
            Path("Synthetic dataset") / "sunny" / "images" / "img_sunny_1.jpg"
        )

    def test_split_keeps_video_frames_together_and_synthetic_in_training(self):
        self.make_dataset()
        manifest = create_crosswalk_split_manifest(self.root, seed=42)

        real_splits = {}
        for item in manifest["items"]:
            if item["source"] == "real":
                real_splits.setdefault(item["group"], set()).add(item["split"])
            else:
                self.assertEqual(item["split"], "train")

        self.assertEqual(len(real_splits), 3)
        self.assertTrue(all(len(splits) == 1 for splits in real_splits.values()))
        self.assertEqual(
            sum(item["split"] == "validation" for item in manifest["items"]), 2
        )
        self.assertEqual(
            sum(item["source"] == "synthetic" for item in manifest["items"]), 1
        )

    def test_split_is_deterministic_for_same_seed(self):
        self.make_dataset()

        first = create_crosswalk_split_manifest(self.root, seed=91)
        second = create_crosswalk_split_manifest(self.root, seed=91)

        self.assertEqual(first, second)

    def test_cli_accepts_an_explicit_crosswalk_sampling_weight(self):
        args = parse_args(["--crosswalk-weight", "0.65"])

        self.assertEqual(args.crosswalk_weight, 0.65)

    def test_cityscapes_manifest_resolves_relative_pair_paths(self):
        manifest_path = self.root / "cityscapes.json"
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "dataset_root": str(self.root),
            "splits": {
                split: [{
                    "image": f"leftImg8bit/{split}/city/image.png",
                    "mask": f"gtFine/{split}/city/mask.png",
                }]
                for split in ("train", "val")
            },
        }), encoding="utf-8")

        pairs = _read_cityscapes_manifest(manifest_path, self.root)

        self.assertEqual(
            pairs["train"],
            [(
                str(self.root / "leftImg8bit" / "train" / "city" / "image.png"),
                str(self.root / "gtFine" / "train" / "city" / "mask.png"),
            )],
        )

    def test_cityscapes_manifest_resolves_optional_ignore_mask_path(self):
        manifest_path = self.root / "cityscapes.json"
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "dataset_root": str(self.root),
            "splits": {
                split: [{
                    "image": f"leftImg8bit/{split}/city/image.png",
                    "mask": f"gtFine/{split}/city/mask.png",
                    "ignore_mask": f"ignore_masks/{split}/city/image_ignore.png",
                }]
                for split in ("train", "val")
            },
        }), encoding="utf-8")

        pairs = _read_cityscapes_manifest(manifest_path, self.root)

        self.assertEqual(
            pairs["val"],
            [(
                str(self.root / "leftImg8bit" / "val" / "city" / "image.png"),
                str(self.root / "gtFine" / "val" / "city" / "mask.png"),
                str(
                    self.root / "ignore_masks" / "val" / "city"
                    / "image_ignore.png"
                ),
            )],
        )

    def test_cityscapes_manifest_rejects_ignore_mask_parent_traversal(self):
        manifest_path = self.root / "cityscapes.json"
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "dataset_root": str(self.root),
            "splits": {
                "train": [{
                    "image": "image.png",
                    "mask": "mask.png",
                    "ignore_mask": "../outside.png",
                }],
                "val": [{"image": "image.png", "mask": "mask.png"}],
            },
        }), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "relative to"):
            _read_cityscapes_manifest(manifest_path, self.root)

    def test_cityscapes_manifest_rejects_parent_traversal(self):
        manifest_path = self.root / "cityscapes.json"
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "dataset_root": str(self.root),
            "splits": {
                "train": [{"image": "../outside.png", "mask": "mask.png"}],
                "val": [{"image": "image.png", "mask": "mask.png"}],
            },
        }), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "relative to"):
            _read_cityscapes_manifest(manifest_path, self.root)


if __name__ == "__main__":
    unittest.main()
