"""Contract regressions for labels, independent detection, and scoring mistakes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from config import CLASS_NAMES, ROOT, assert_branch, output_directory
from dataset import centroid_target, inspect_manifest, load_arrays, preprocess, validated_records
from evaluate import match_centroids, scores
from inference import crop_around_centroid, decode_grid, predict


class HandoffTests(unittest.TestCase):
    def test_writes_refused_on_main_without_switching(self):
        with patch("config.subprocess.check_output", return_value="main\n"):
            with self.assertRaisesRegex(RuntimeError, "Refusing writes"):
                assert_branch()

    def test_unrelated_changes_refused(self):
        with patch("config.subprocess.check_output", side_effect=["tinyml-fomo\n", "fast_scnn.py\n", ""]):
            with self.assertRaisesRegex(RuntimeError, "Unexpected changes"):
                assert_branch()

    def test_output_cannot_escape_fomo_artifacts(self):
        with self.assertRaisesRegex(ValueError, "Output must be inside"):
            output_directory(ROOT.parent.parent / "outputs")

    def test_frozen_checkpoint_rejects_generated_outputs(self):
        with patch("config.assert_branch"):
            with self.assertRaisesRegex(ValueError, "Output must be inside"):
                output_directory(ROOT / "checkpoints" / "cityscapes_v2")

    def test_existing_experiment_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            marker = Path(directory) / "preserved.txt"
            marker.write_text("existing experiment", encoding="utf-8")
            with patch("config.assert_branch"):
                with self.assertRaises(FileExistsError):
                    output_directory(directory)
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing experiment")

    def test_center_mapping_and_public_id(self):
        grid = np.zeros((20, 20, 3), np.float32)
        grid[:, :, 0] = 1
        grid[5, 10] = [0.1, 0.2, 0.7]
        result = decode_grid(grid, 640, 480)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["coarse_class_id"], 1)
        self.assertEqual(result[0]["class_name"], "vehicle")
        self.assertEqual(result[0]["centroid_x"], 336)
        self.assertEqual(result[0]["centroid_y"], 132)
        crop = crop_around_centroid(np.full((480, 640, 3), 255, np.uint8), 336, 132)
        self.assertEqual(crop.shape, (96, 96, 3))
        self.assertTrue(np.all(crop == 1))

    def test_edge_crop_preserves_center_and_normalization(self):
        frame = np.full((20, 30, 3), 255, np.uint8)
        crop = crop_around_centroid(frame, 0, 0)
        self.assertEqual(crop.dtype, np.float32)
        self.assertTrue(np.all(crop[:48] == 0))
        self.assertTrue(np.all(crop[48:68, 48:78] == 1))
        self.assertTrue(np.all(crop[:, :48] == 0))
        for cx, cy in ((30, 20), (15, 10), (30, 0), (0, 20)):
            self.assertEqual(crop_around_centroid(frame, cx, cy).shape, (96, 96, 3))
        with self.assertRaises(ValueError):
            crop_around_centroid(frame, float("nan"), 0)

    def test_connected_cells_merge_and_top_three_sort(self):
        grid = np.zeros((20, 20, 3), np.float32)
        grid[:, :, 0] = 1
        for r, c, p in ((1, 1, .8), (1, 2, .9), (5, 5, .7), (9, 9, .65), (14, 14, .95)):
            grid[r, c] = [1-p, p, 0]
        all_results = decode_grid(grid, 160, 160, max_detections=0, decoder="connected")
        self.assertEqual(len(all_results), 4)
        self.assertEqual(len(decode_grid(grid, 160, 160, max_detections=3, decoder="connected")), 3)
        self.assertAlmostEqual(all_results[0]["confidence"], .95)
        merged = next(r for r in all_results if r["centroid_y"] == 12)
        self.assertEqual(merged["centroid_x"], 16)

    def test_nms_keeps_separated_peaks_inside_one_connected_chain(self):
        grid = np.zeros((20, 20, 3), np.float32)
        grid[:, :, 0] = 1
        for col, confidence in ((3, .9), (4, .61), (5, .62), (6, .95)):
            grid[5, col] = [1-confidence, confidence, 0]
        self.assertEqual(len(decode_grid(grid, 160, 160, decoder="connected")), 1)
        result = decode_grid(grid, 160, 160)
        self.assertEqual([d["centroid_x"] for d in result], [52, 28])
        counts, _ = match_centroids(result, [{"class_id": 0, "bbox": [24, 40, 8, 8]},
                                            {"class_id": 0, "bbox": [48, 40, 8, 8]}], 160, 160)
        self.assertEqual(counts[0]["tp"], 2)

    def test_nms_suppresses_duplicates_but_not_a_different_class(self):
        grid = np.zeros((20, 20, 3), np.float32)
        grid[:, :, 0] = 1
        grid[0, 0] = [.1, .9, 0]
        grid[0, 1] = [.2, .8, 0]
        grid[1, 0] = [.1, 0, .9]
        result = decode_grid(grid, 160, 160)
        self.assertEqual([d['coarse_class_id'] for d in result], [0, 1])
        self.assertEqual(len(decode_grid(grid, 160, 160, decoder='cells')), 3)

    def test_inference_keeps_more_than_three_and_never_creates_crops(self):
        grid = np.zeros((20, 20, 3), np.float32)
        grid[:, :, 0] = 1
        for row in (1, 5, 9, 13, 17):
            grid[row, 2] = [.1, .9, 0]
        def model(batch, training=False):
            self.assertEqual(batch.shape, (1, 160, 160, 3))
            return grid[None]
        with patch('inference.crop_around_centroid', side_effect=AssertionError('No crop handoff')):
            detections, _ = predict(model, np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(len(detections), 5)

    def test_decoder_rejects_invalid_inputs(self):
        grid = np.zeros((20, 20, 3), np.float32)
        grid[:, :, 0] = 1
        for kwargs in ({'decoder':'unknown'}, {'max_detections':-1}, {'threshold':float('nan')}):
            with self.assertRaises(ValueError):
                decode_grid(grid, 160, 160, **kwargs)
        with self.assertRaises(ValueError):
            decode_grid(grid, float('nan'), 160)

    def test_no_object_when_background_wins(self):
        grid = np.zeros((20, 20, 3), np.float32)
        grid[:, :, 0] = .7
        grid[:, :, 1] = .3
        self.assertEqual(decode_grid(grid, 160, 160, threshold=.2), [])

    def test_preprocess_keeps_rgb_order(self):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :, 0] = 255
        result = preprocess(frame)
        self.assertEqual(result.shape, (160, 160, 3))
        self.assertTrue(np.all(result[:, :, 0] == 1))
        self.assertTrue(np.all(result[:, :, 1:] == 0))

    def test_one_to_one_scoring_wrong_class_and_duplicate(self):
        annotations = [{"class_id": 0, "bbox": [40, 40, 16, 16]}]
        detections = [{"coarse_class_id": 0, "confidence": .9, "centroid_x": 48, "centroid_y": 48},
                      {"coarse_class_id": 0, "confidence": .8, "centroid_x": 49, "centroid_y": 48},
                      {"coarse_class_id": 1, "confidence": .9, "centroid_x": 48, "centroid_y": 48}]
        counts, distances = match_centroids(detections, annotations, 160, 160)
        self.assertEqual(dict(counts[0]), {"tp": 1, "fp": 1, "fn": 0})
        self.assertEqual(counts[1]["fp"], 1)
        self.assertEqual(distances, [0])
        self.assertEqual(scores(counts[0])["precision"], .5)
        self.assertIsNone(scores({"tp":0,"fp":0,"fn":0})["recall"])

    def test_matching_radius_uses_detector_pixels(self):
        annotations = [{"class_id": 0, "bbox": [160, 160, 64, 64]}]
        detections = [{"coarse_class_id": 0, "confidence": .9, "centroid_x": 220, "centroid_y": 192}]
        counts, distances = match_centroids(detections, annotations, 640, 640, radius=8)
        self.assertEqual(counts[0]["tp"], 1)
        self.assertEqual(distances, [7])
        counts, _ = match_centroids(detections, annotations, 640, 640, radius=6)
        self.assertEqual(counts[0]["fn"], 1)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        assert_branch()
        base = ROOT / "artifacts"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="unit-", dir=base)
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        Image.fromarray(np.full((160, 160, 3), 10, np.uint8)).save(self.directory / "a.png")
        Image.fromarray(np.full((160, 160, 3), 20, np.uint8)).save(self.directory / "b.png")

    def inspect(self, records):
        manifest = self.directory / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "class_names": list(CLASS_NAMES),
                                        "synthetic": False, "reviewed": True, "images": records}), encoding="utf-8")
        return inspect_manifest(manifest)

    def record(self, **overrides):
        return dict({"id": "a", "group": "route-a", "split": "train", "file": "a.png",
                     "annotations": [{"class_id": 0, "bbox": [20, 20, 16, 16]}]}, **overrides)

    def test_valid_box_and_background_channel(self):
        _, records, report = self.inspect([self.record()])
        self.assertEqual(report["errors"], [])
        target = centroid_target(records[0]["annotations"], 160, 160)
        self.assertEqual(target[3, 3], 1)
        self.assertEqual(np.count_nonzero(target), 1)

    def test_same_class_collision_retained_as_one_positive_cell(self):
        annotations = [{"class_id": 0, "bbox": [20, 20, 16, 16]},
                       {"class_id": 0, "bbox": [21, 21, 16, 16]}]
        _, _, report = self.inspect([self.record(annotations=annotations)])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["grid_diagnostics"]["extra_centers_in_shared_cells"], 1)
        target=centroid_target(annotations,160,160)
        self.assertEqual(target[3,3],1)

    def test_mixed_collision_ignored_without_losing_evaluation_boxes(self):
        annotations=[{"class_id":0,"bbox":[20,20,16,16]}, {"class_id":1,"bbox":[21,21,16,16]},
                     {"class_id":1,"bbox":[22,22,16,16]}]
        _, records, report=self.inspect([self.record(annotations=annotations)])
        self.assertEqual(report["errors"],[])
        self.assertEqual(len(records[0]["annotations"]),3)
        self.assertEqual(centroid_target(annotations,160,160)[3,3],-1)

    def test_positive_center_overrides_void_cell_but_not_class_conflict(self):
        a=[{"class_id":0,"bbox":[20,20,16,16]}]
        target=centroid_target(a,160,160,[[3,3],[0,0]])
        self.assertEqual(target[3,3],1)
        self.assertEqual(target[0,0],-1)

    def test_training_mask_can_change_without_changing_evaluation_mask(self):
        _, records, report = self.inspect([self.record(ignore_cells=[[0,0]], target_ignore_cells=[])])
        self.assertEqual(report['errors'], [])
        _, targets = load_arrays(records)
        self.assertEqual(targets[0,0,0], 0)
        self.assertEqual(records[0]['ignore_cells'], [[0,0]])
        detection = dict(coarse_class_id=0, confidence=.9, centroid_x=4., centroid_y=4.)
        counts, _ = match_centroids([detection], records[0]['annotations'], 160, 160,
                                    ignore_cells=records[0]['ignore_cells'])
        self.assertEqual(counts[0]['ignored'], 1)

    def test_cross_split_session_leakage(self):
        _, _, report = self.inspect([self.record(), self.record(id="b", file="b.png", split="test")])
        self.assertTrue(any("occurs across splits" in e for e in report["errors"]))

    def test_duplicate_pixels_across_splits(self):
        Image.open(self.directory / "a.png").save(self.directory / "b.png")
        _, _, report = self.inspect([self.record(), self.record(id="b", file="b.png", group="other", split="test")])
        self.assertTrue(any("duplicate image content" in e for e in report["errors"]))

    def test_missing_and_broken_images_and_bad_boxes(self):
        (self.directory / "broken.png").write_text("not an image", encoding="utf-8")
        records = [self.record(file="missing.png"), self.record(id="broken", file="broken.png"),
                   self.record(id="bad", annotations=[{"class_id": True, "bbox": [0, 0, 1, 1]},
                                                       {"class_id": 0, "bbox": [159, 0, 2, 2]},
                                                       {"class_id": 2, "bbox": [0, 0, float("nan"), 2]}])]
        _, _, report = self.inspect(records)
        self.assertEqual(report["missing_images"], 1)
        self.assertEqual(report["broken_images"], 1)
        self.assertEqual(report["invalid_annotations"], 3)

    def test_training_blocks_missing_splits(self):
        self.inspect([self.record()])
        with self.assertRaisesRegex(ValueError, "Missing validation split"):
            validated_records(self.directory / "manifest.json")


if __name__ == "__main__":
    unittest.main()
