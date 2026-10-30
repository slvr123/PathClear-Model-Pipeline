"""Regressions for distinct training grids and fixed input-space evaluation."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import ROOT, assert_branch, contract
from dataset import centroid_target, load_arrays
from evaluate import match_centroids
from inference import decode_grid
from prepare_cityscapes import center_ignore_cells


class GridResolutionTests(unittest.TestCase):
    def test_finer_grid_resolves_a_previous_same_cell_collision(self):
        annotations = [{'class_id':0,'bbox':[0,0,4,4]}, {'class_id':0,'bbox':[4,0,4,4]}]
        coarse = centroid_target(annotations,160,160)
        fine = centroid_target(annotations,160,160,grid_size=40)
        self.assertEqual(np.count_nonzero(coarse>0),1)
        self.assertEqual(np.count_nonzero(fine>0),2)

    def test_unknown_masks_are_resampled_from_source_not_upsampled(self):
        mask = np.full((160,160),7,np.uint16)
        mask[2,2] = 0
        self.assertEqual(center_ignore_cells(mask,20),[])
        self.assertEqual(center_ignore_cells(mask,40),[[0,0]])
        mask[158,158] = 26
        self.assertIn([39,39],center_ignore_cells(mask,40))

    def test_loader_refuses_a_grid_without_matching_targets(self):
        with self.assertRaisesRegex(ValueError,'Target grid differs'):
            load_arrays([{'target_grid_size':20}],grid_size=40)

    def test_nms_uses_eight_input_pixels_at_both_resolutions(self):
        for g in (20,40):
            grid = np.zeros((g,g,3),np.float32)
            grid[:,:,0] = 1
            offset = g//20
            for col,p in ((0,.95),(offset,.8),(offset+1,.7)):
                grid[0,col] = [1-p,p,0]
            result = decode_grid(grid,640,480)
            self.assertEqual(len(result),2)
            self.assertAlmostEqual(result[0]['centroid_x'],.5/g*640)
            self.assertAlmostEqual(result[1]['centroid_x'],(offset+1.5)/g*640)

    def test_fine_detection_uses_original_coarse_evaluation_mask(self):
        grid = np.zeros((40,40,3),np.float32)
        grid[:,:,0] = 1
        grid[39,39] = [.1,.9,0]
        detection = decode_grid(grid,160,160)
        counts,_ = match_centroids(detection,[],160,160,ignore_cells=[[19,19]])
        self.assertEqual(counts[0]['ignored'],1)

    def test_fine_network_forward_save_and_load(self):
        assert_branch()
        os.environ['KERAS_HOME'] = str(ROOT/'artifacts'/'keras-cache')
        from model import build_model, load_model
        model = build_model('random',grid_size=40)
        sample = np.zeros((1,160,160,3),np.float32)
        expected = np.asarray(model(sample,training=False))
        self.assertEqual(expected.shape,(1,40,40,3))
        np.testing.assert_allclose(expected.sum(-1),1,atol=1e-6)
        with tempfile.TemporaryDirectory(prefix='unit-grid-',dir=ROOT/'artifacts') as directory:
            path = Path(directory)/'model.keras'
            model.save(path)
            path.with_suffix('.json').write_text(json.dumps({'contract':contract(40)}))
            restored,_ = load_model(path)
            np.testing.assert_allclose(restored(sample,training=False),expected,atol=1e-6)


if __name__ == '__main__':
    unittest.main()
