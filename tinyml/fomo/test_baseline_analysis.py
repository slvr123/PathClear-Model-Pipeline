"""Check error attribution, size boundaries, and controlled loss weighting."""
import unittest
import numpy as np
from analyze_baseline_v1 import assignments, measure, size_bin, review_page
from html.parser import HTMLParser
from train import weights_for
from inference import decoding_policy


class BaselineAnalysisTests(unittest.TestCase):
    def test_gallery_urls_resolve_from_review_folder_for_both_path_styles(self):
        class Images(HTMLParser):
            def __init__(self):
                super().__init__()
                self.sources=[]
            def handle_starttag(self,tag,attrs):
                if tag=='img': self.sources.append(dict(attrs)['src'])
        parser=Images()
        parser.feed(review_page([
            dict(kind='tp',file=r'review\tp\person example.jpg'),
            dict(kind='fn',file='review/fn/vehicle.jpg')]))
        self.assertEqual(parser.sources,['tp/person%20example.jpg','fn/vehicle.jpg'])

    def test_policy_metadata_reports_suppression_only_when_used(self):
        self.assertTrue(decoding_policy()['nms_enabled'])
        self.assertFalse(decoding_policy('cells')['nms_enabled'])
        self.assertEqual(decoding_policy('cells')['nms_radius_input_pixels'],0)

    def test_one_detection_cannot_recover_two_objects_across_size_bins(self):
        r=dict(id='fixture',width=160,height=160,annotations=[
            dict(class_id=0,bbox=[11,11,2,2]), dict(class_id=0,bbox=[9,9,8,8])])
        grid=np.zeros((20,20,3),np.float32); grid[:,:,0]=1
        grid[1,1]=[.1,.9,0]
        result,events=measure([r],grid[None],.6,True)
        self.assertEqual(result['micro']['tp'],1)
        self.assertEqual(result['size_recall']['2to4']['matched'],1)
        self.assertEqual(result['size_recall']['8to16']['matched'],0)
        self.assertEqual(sum(e['kind']=='fn' for e in events),1)

    def test_unknown_unmatched_prediction_is_separate_from_false_positive(self):
        r=dict(width=160,height=160,annotations=[],ignore_cells=[[1,1]])
        d=dict(centroid_x=12,centroid_y=12,coarse_class_id=0,class_name='person',confidence=.9)
        used,events=assignments(r,[d])
        self.assertFalse(used); self.assertEqual(events[0]['kind'],'ignored')

    def test_nms_ablation_recovers_two_adjacent_objects_without_changing_matching(self):
        r=dict(id='crowd',width=160,height=160,annotations=[
            dict(class_id=0,bbox=[11,11,2,2]),dict(class_id=0,bbox=[19,11,2,2])])
        grid=np.zeros((20,20,3),np.float32);grid[:,:,0]=1
        grid[1,1]=[.1,.9,0];grid[1,2]=[.15,.85,0]
        original,_=measure([r],grid[None],.8)
        ablated,_=measure([r],grid[None],.8,decoder='cells')
        self.assertEqual(original['micro']['tp'],1)
        self.assertEqual(ablated['micro']['tp'],2)
        self.assertEqual(ablated['micro']['fp'],0)

    def test_exact_size_boundaries(self):
        self.assertEqual([size_bin(s) for s in (1.99,2,4,8,16)],['lt2','2to4','4to8','8to16','ge16'])

    def test_weights_preserve_zero_ignore_and_selected_ratio(self):
        target=np.array([[[-1,0],[1,2]]])
        for value in (10,20,40):
            weight=weights_for(target,value)
            self.assertEqual(weight[0,0,0],0)
            self.assertAlmostEqual(float(weight.mean()),1,places=6)
            self.assertAlmostEqual(float(weight[0,1,0]/weight[0,0,1]),value,places=5)
        np.testing.assert_array_equal(weights_for(target),weights_for(target,20))

    def test_invalid_weights_rejected(self):
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): weights_for(np.zeros((1,2,2)),value)


if __name__=='__main__': unittest.main()
