"""Tests for microseg_taxonomy.py.

Run with: bash run_wsl.sh tests/test_microseg_taxonomy.py
Or:        pytest tests/test_microseg_taxonomy.py
"""
import numpy as np
import pytest

from microseg_taxonomy import (
    CLASS_NAMES, CLASS_IDS, NUM_CLASSES, IGNORE_LABEL,
    CITYSCAPES_TO_MICROSEG, COLORS,
    cityscapes_lookup, remap_cityscapes, colorize,
)


class TestConstants:
    def test_num_classes_matches_class_names(self):
        assert NUM_CLASSES == len(CLASS_NAMES)

    def test_num_classes_is_five(self):
        # Hard-assert the target class count for ESP32-CAM deployment.
        assert NUM_CLASSES == 5

    def test_class_ids_are_zero_indexed(self):
        for i, name in enumerate(CLASS_NAMES):
            assert CLASS_IDS[name] == i

    def test_ignore_label_is_255(self):
        assert IGNORE_LABEL == 255

    def test_colors_shape(self):
        assert COLORS.shape == (NUM_CLASSES, 3)
        assert COLORS.dtype == np.uint8

    def test_expected_class_names(self):
        assert CLASS_NAMES == (
            "Background", "Walkable", "Non-walkable", "Boundary", "Crosswalk"
        )


class TestCityscapesMapping:
    def test_all_19_cityscapes_train_ids_are_mapped(self):
        """Every Cityscapes trainId 0-18 must be explicitly mapped."""
        for train_id in range(19):
            assert train_id in CITYSCAPES_TO_MICROSEG, (
                f"Cityscapes trainId {train_id} is missing from CITYSCAPES_TO_MICROSEG"
            )

    def test_road_maps_to_non_walkable(self):
        assert CITYSCAPES_TO_MICROSEG[0] == 2  # road → Non-walkable

    def test_sidewalk_maps_to_walkable(self):
        assert CITYSCAPES_TO_MICROSEG[1] == 1  # sidewalk → Walkable

    def test_wall_and_fence_map_to_boundary(self):
        assert CITYSCAPES_TO_MICROSEG[3] == 3  # wall → Boundary
        assert CITYSCAPES_TO_MICROSEG[4] == 3  # fence → Boundary

    def test_person_and_vehicles_are_ignored(self):
        person_and_vehicle_ids = [11, 12, 13, 14, 15, 16, 17, 18]
        for tid in person_and_vehicle_ids:
            assert CITYSCAPES_TO_MICROSEG[tid] == IGNORE_LABEL, (
                f"trainId {tid} should map to IGNORE_LABEL"
            )

    def test_crosswalk_has_no_cityscapes_source(self):
        """Crosswalk (ID 4) must NOT appear in any Cityscapes mapping value."""
        for src, dst in CITYSCAPES_TO_MICROSEG.items():
            assert dst != 4, (
                f"Cityscapes trainId {src} unexpectedly maps to Crosswalk (4). "
                "Crosswalk comes exclusively from Kaggle and local data."
            )

    def test_all_mapped_values_are_valid_ids_or_ignore(self):
        valid = set(range(NUM_CLASSES)) | {IGNORE_LABEL}
        for src, dst in CITYSCAPES_TO_MICROSEG.items():
            assert dst in valid, f"trainId {src} → {dst} is not a valid MicroSeg ID"


class TestCityscapesLookup:
    def setup_method(self):
        self.lut = cityscapes_lookup()

    def test_lut_length(self):
        assert len(self.lut) == 256

    def test_lut_dtype(self):
        assert self.lut.dtype == np.uint8

    def test_lut_road_entry(self):
        assert self.lut[0] == 2   # road → Non-walkable

    def test_lut_sidewalk_entry(self):
        assert self.lut[1] == 1   # sidewalk → Walkable

    def test_lut_boundary_entries(self):
        assert self.lut[3] == 3   # wall → Boundary
        assert self.lut[4] == 3   # fence → Boundary

    def test_lut_unmapped_entries_are_ignore(self):
        # trainId 19+ are not valid Cityscapes trainIds — must be Ignore.
        for i in range(19, 256):
            assert self.lut[i] == IGNORE_LABEL, f"LUT[{i}] should be IGNORE_LABEL"

    def test_lut_is_deterministic(self):
        lut2 = cityscapes_lookup()
        np.testing.assert_array_equal(self.lut, lut2)


class TestRemapCityscapes:
    def test_basic_remap(self):
        mask = np.array([[0, 1, 3, 4]], dtype=np.int32)
        result = remap_cityscapes(mask)
        assert result[0, 0] == 2   # road → Non-walkable
        assert result[0, 1] == 1   # sidewalk → Walkable
        assert result[0, 2] == 3   # wall → Boundary
        assert result[0, 3] == 3   # fence → Boundary

    def test_remap_rejects_float_input(self):
        with pytest.raises(TypeError):
            remap_cityscapes(np.array([[0.5, 1.0]]))

    def test_remap_255_stays_ignore(self):
        mask = np.array([[255]], dtype=np.int32)
        result = remap_cityscapes(mask)
        assert result[0, 0] == IGNORE_LABEL


class TestColorize:
    def test_output_shape(self):
        mask = np.zeros((96, 96), dtype=np.int32)
        rgb = colorize(mask)
        assert rgb.shape == (96, 96, 3)
        assert rgb.dtype == np.uint8

    def test_background_color(self):
        mask = np.full((1, 1), 0, dtype=np.int32)
        rgb = colorize(mask)
        np.testing.assert_array_equal(rgb[0, 0], [64, 64, 64])

    def test_walkable_color(self):
        mask = np.full((1, 1), 1, dtype=np.int32)
        rgb = colorize(mask)
        np.testing.assert_array_equal(rgb[0, 0], [144, 238, 144])

    def test_non_walkable_color(self):
        mask = np.full((1, 1), 2, dtype=np.int32)
        rgb = colorize(mask)
        np.testing.assert_array_equal(rgb[0, 0], [220, 50, 50])

    def test_ignore_pixels_are_black(self):
        mask = np.full((1, 1), 255, dtype=np.int32)
        rgb = colorize(mask)
        np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])

    def test_out_of_range_pixels_are_black(self):
        mask = np.full((1, 1), 200, dtype=np.int32)
        rgb = colorize(mask)
        np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])
