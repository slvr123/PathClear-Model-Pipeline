"""Cityscapes encoding and evaluation regressions using small hand-checked masks."""
import unittest
import numpy as np
from prepare_cityscapes import extract_instances, center_ignore_cells
from dataset import centroid_target
from train import weights_for
from evaluate import match_centroids

class CityscapesTests(unittest.TestCase):
    def test_thin_unknown_border_does_not_remove_known_center_supervision(self):
        mask = np.full((160, 160), 7, np.uint16)
        mask[:, 0] = 0
        _, original, _ = extract_instances(mask)
        self.assertEqual(len(original), 20)
        self.assertEqual(center_ignore_cells(mask), [])

    def test_center_mask_keeps_real_unknown_groups_and_void_ignored(self):
        mask = np.full((160, 160), 7, np.uint16)
        mask[4, 4] = 26
        mask[12, 12] = 1
        mask[20, 20] = 29000
        mask[28, 28] = 26000
        self.assertEqual(center_ignore_cells(mask), [[0, 0], [1, 1], [2, 2]])

    def test_instance_zero_is_valid_and_box_is_inclusive(self):
        mask=np.full((20,20),7,np.uint16)
        mask[2:7,3:11]=24000
        annotations,ignored,_=extract_instances(mask)
        self.assertEqual(len(annotations),1)
        self.assertEqual(annotations[0]['bbox'],[3,2,8,5])
        self.assertEqual(annotations[0]['source_instance_number'],0)
        self.assertEqual(annotations[0]['class_id'],0)
        self.assertEqual(ignored,[])

    def test_all_requested_classes_map_and_excluded_classes_do_not(self):
        mask=np.full((20,20),7,np.uint16)
        ids=[24000,25001,26002,27003,28004,31005,32006,33007,29008,30009]
        for i,value in enumerate(ids):
            mask[2,i+1]=value
        annotations,_,stats=extract_instances(mask)
        self.assertEqual([a['class_id'] for a in annotations],[0,0,1,1,1,1,1,1])
        self.assertEqual(stats['unrelated_individual_instances'],2)

    def test_groups_have_no_invented_boxes(self):
        mask=np.full((20,20),7,np.uint16)
        mask[5:8,5:8]=26
        annotations,ignored,stats=extract_instances(mask)
        self.assertEqual(annotations,[])
        self.assertIn([6,6],ignored)
        self.assertEqual(stats['group_pixels'],9)

    def test_original_label_ids_are_not_train_ids(self):
        mask=np.full((20,20),7,np.uint16)
        mask[0:2,0:2]=11000
        annotations,_,_=extract_instances(mask)
        self.assertEqual(annotations,[])

    def test_ignored_cells_have_zero_loss_weight(self):
        targets=np.zeros((2,20,20),np.int32)
        targets[:,0,0]=-1
        targets[:,3,3]=1
        weights=weights_for(targets)
        self.assertTrue(np.all(weights[:,0,0]==0))
        self.assertTrue(np.all(weights[:,3,3]>weights[:,1,1]))
        np.testing.assert_allclose(weights.mean((1,2)),1,atol=1e-6)

    def test_unknown_group_prediction_is_ignored_after_matching_known_objects(self):
        detection={'coarse_class_id':0,'confidence':.9,'centroid_x':28.,'centroid_y':28.}
        counts,_=match_centroids([detection],[],160,160,ignore_cells=[[3,3]])
        self.assertEqual(counts[0]['fp'],0)
        self.assertEqual(counts[0]['ignored'],1)
        counts,_=match_centroids([detection],[{'class_id':0,'bbox':[20,20,16,16]}],160,160,ignore_cells=[[3,3]])
        self.assertEqual(counts[0]['tp'],1)
        self.assertEqual(counts[0]['ignored'],0)

if __name__=='__main__':
    unittest.main()
