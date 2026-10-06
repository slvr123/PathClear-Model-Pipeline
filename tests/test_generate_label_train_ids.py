import unittest

import numpy as np

from scripts.generate_label_train_ids import (
    _LABEL_TO_TRAIN,
    label_to_train_lookup,
)


class LabelToTrainTests(unittest.TestCase):
    def test_official_mapping_spot_checks(self):
        lookup = label_to_train_lookup()
        self.assertEqual(
            [int(lookup[i]) for i in (7, 8, 11, 12, 13, 23, 24, 33)],
            [0, 1, 2, 3, 4, 10, 11, 18],
        )

    def test_every_train_id_reachable_and_rest_void(self):
        lookup = label_to_train_lookup()
        self.assertEqual(set(int(v) for v in lookup), set(range(19)) | {255})
        for source, target in _LABEL_TO_TRAIN.items():
            self.assertEqual(int(lookup[source]), target)

    def test_void_and_unlabeled_stay_255(self):
        lookup = label_to_train_lookup()
        for source in (0, 1, 2, 3, 4, 5, 6, 9, 10, 29, 30, 255):
            self.assertEqual(int(lookup[source]), 255)


if __name__ == "__main__":
    unittest.main()
