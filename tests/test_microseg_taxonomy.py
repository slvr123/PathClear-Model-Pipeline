"""Exact checks for the four-class MicroSeg label mapping."""
import unittest

import numpy as np

from microseg.taxonomy import (
    CLASS_IDS,
    CLASS_NAMES,
    COLORS,
    IGNORE_LABEL,
    NUM_CLASSES,
    cityscapes_label_ids_lookup,
    colorize,
    microseg_lookup,
    remap_cityscapes_ids,
    remap_cityscapes_label_ids,
    remap_pathclear_ids,
)


class TaxonomyTests(unittest.TestCase):
    def test_four_classes_and_ignore(self):
        self.assertEqual(NUM_CLASSES, 4)
        self.assertEqual(IGNORE_LABEL, 255)
        self.assertEqual(CLASS_NAMES, ("Walkable", "Non-walkable", "Boundary", "Crosswalk"))
        # Zero-based, in declaration order.
        self.assertEqual(CLASS_IDS, {
            "Walkable": 0, "Non-walkable": 1, "Boundary": 2, "Crosswalk": 3,
        })
        self.assertEqual(COLORS.shape, (NUM_CLASSES, 3))

    def test_every_pathclear_class_and_ignore(self):
        # Independent expectation: Sidewalk->0, Road->1, Boundary->2,
        # Crosswalk->3; Person, Vehicle, Obstacle, Vegetation, Stairs and
        # Uneven Pavement stay ignored.
        original = np.array(list(range(10)) + [255], dtype=np.uint8)
        before = original.copy()
        mapped = remap_pathclear_ids(original)
        np.testing.assert_array_equal(
            mapped,
            np.array([1, 0, 255, 255, 2, 255, 255, 255, 3, 255, 255], dtype=np.uint8),
        )
        # The source mask is not mutated in place.
        np.testing.assert_array_equal(original, before)

    def test_lookup_defaults_to_ignore(self):
        lookup = microseg_lookup()
        self.assertEqual(lookup.shape, (256,))
        self.assertEqual(lookup[255], IGNORE_LABEL)
        for index in range(10):
            if index in (0, 1, 4, 8):
                self.assertLess(int(lookup[index]), NUM_CLASSES)
            else:
                self.assertEqual(int(lookup[index]), IGNORE_LABEL)

    def test_out_of_range_values_become_ignore(self):
        mapped = remap_pathclear_ids(np.array([[-5, 300, 8]], dtype=np.int32))
        np.testing.assert_array_equal(mapped, np.array([[255, 255, 3]], dtype=np.uint8))

    def test_non_integer_mask_rejected(self):
        for bad in (np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=bool)):
            with self.assertRaises(TypeError):
                remap_pathclear_ids(bad)

    def test_cityscapes_chain(self):
        # road(0)->1, sidewalk(1)->0, building(2)->2 (Boundary via trainId chain),
        # wall(3)->2, fence(4)->2, sky(10)->1 (Non-walkable via trainId chain),
        # person(11)->ignored, void 255->ignored.
        source = np.array([0, 1, 2, 3, 4, 10, 11, 255], dtype=np.uint8)
        np.testing.assert_array_equal(
            remap_cityscapes_ids(source),
            np.array([1, 0, 255, 2, 2, 255, 255, 255], dtype=np.uint8),
        )

    def test_original_cityscapes_label_ids_mapping(self):
        # Original labelIds:
        #   road(7)->1, sidewalk(8)->0,
        #   building(11)->2 (Boundary, shorelining anchor),
        #   wall(12)->2, fence(13)->2,
        #   sky(23)->1 (Non-walkable, stops sky->Walkable predictions).
        source = np.array(
            [0, 7, 8, 11, 12, 13, 23, 24, 26, 255], dtype=np.uint8
        )
        np.testing.assert_array_equal(
            remap_cityscapes_label_ids(source),
            np.array([255, 1, 0, 2, 2, 2, 1, 255, 255, 255], dtype=np.uint8),
        )

    def test_original_label_ids_lookup_ignores_unmapped_ids(self):
        lookup = cityscapes_label_ids_lookup()
        self.assertEqual(lookup.shape, (256,))
        for source, target in {7: 1, 8: 0, 11: 2, 12: 2, 13: 2, 23: 1}.items():
            self.assertEqual(int(lookup[source]), target)
        for source in (0, 24, 26, 255):
            self.assertEqual(int(lookup[source]), IGNORE_LABEL)


    def test_colorize_keeps_ignore_black(self):
        rgb = colorize(np.array([[0, 1, 2, 3, 255]], dtype=np.uint8))
        self.assertEqual(rgb.shape, (1, 5, 3))
        np.testing.assert_array_equal(rgb[0, -1], [0, 0, 0])
        for index in range(NUM_CLASSES):
            np.testing.assert_array_equal(rgb[0, index], COLORS[index])


if __name__ == "__main__":
    unittest.main()
