from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from rulers_v2 import ruler_vector, signed_gain, metrics, paired_cluster_scores

SETTING = {'drop':'relative','floor':False,'headroom':'none','rho':.5,'parse_fail':'wrong'}

class ProtocolV2Tests(unittest.TestCase):
    def test_reference_card_relative_denominator(self):
        scores=np.array([[.4]*4,[.8]*4]);floors=np.array([.2]*4)
        np.testing.assert_allclose(ruler_vector(scores,floors,SETTING)[0],.5)
        np.testing.assert_allclose(ruler_vector(scores,floors,{**SETTING,'floor':True})[0],2/3)
        np.testing.assert_allclose(ruler_vector(scores,floors,SETTING,'small')[0],1)

    def test_headroom_flat_class_cannot_count_as_strict_reversal(self):
        classes=[{'vector':[0,0,0,1]} for _ in range(8)]+[{'vector':[1,1,1,1]}]
        result=metrics(classes,[0,1,2,3])
        self.assertAlmostEqual(result['F'],1/12)
        self.assertEqual(result['strict_pair_flip_rate'],0)
        self.assertEqual(result['F_without_full_ties'],0)
        self.assertEqual(result['full_ties'],1)

    def test_guarded_axis_not_silently_used_in_full_axis_primary(self):
        result=metrics([{'vector':[1,0,np.nan,0]}],[0,1,2,3])
        self.assertEqual(result['usable_classes'],0)
        self.assertIsNone(result['F'])

    def test_floor_signed_gain_keeps_post_subfloor_and_negative_transfer(self):
        result=signed_gain([.6,.6,.6,.6],[.65,.1,.6,.58],[.2]*4)
        self.assertTrue(result['defined'])
        self.assertAlmostEqual(result['G_floor'],(.05-.5-.02)/.8)
        self.assertLess(result['signed_floor'][1],0)

    def test_invalid_floor_does_not_delete_the_axis(self):
        result=signed_gain([.6]*4,[.7]*4,[0,1,0,0])
        self.assertIsNone(result['G_floor'])
        self.assertAlmostEqual(result['G_raw'],.4)

    def test_image_cluster_weight_is_shared_across_axes_and_models(self):
        rows=[]
        for axis in ['a','b']:
            for i,bit in enumerate([0,1,0,1]):
                rows.append({'axis':axis,'stratum':axis,'unit':str(i),'image_group':str(i),'correct_bit':bit})
        records=[rows,[{**r,'correct_bit':1-r['correct_bit']} for r in rows]]
        samples=paired_cluster_scores(records,['a','b'],100,20261016)
        np.testing.assert_allclose(samples[:,:,0],samples[:,:,1])
        np.testing.assert_allclose(samples[:,0,:]+samples[:,1,:],1)

    def test_more_than_two_edges_requires_registered_weight_definition(self):
        with self.assertRaises(ValueError):
            ruler_vector(np.full((4,4),.5),np.zeros(4),SETTING)

if __name__=='__main__':
    unittest.main()
