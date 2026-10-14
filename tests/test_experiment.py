import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from experiment_common import parse_answer, score_generation, units, ROOT
from analyze_rulers import ruler_vector, canonical, grid, planted, scan


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.setting={'drop':'absolute','floor':False,'headroom':'none','rho':.5,'parse_fail':'wrong'}

    def test_parser_does_not_extract_letters_from_prose(self):
        item={'answer_type':'option_letter','options':['A','B','C']}
        self.assertIsNone(parse_answer('A cat is visible.',item))
        self.assertIsNone(parse_answer('D',item))
        self.assertEqual(parse_answer('(b)',item),'B')

    def test_grounding_parser_checks_original_pixel_bounds(self):
        item={'answer_type':'bbox_xyxy_pixels','width':640,'height':480}
        self.assertIsNone(parse_answer('[0, 0, 1000, 1000]',item))
        self.assertIsNone(parse_answer('[10, 10, 5, 5]',item))
        self.assertIsNone(parse_answer('[false,0,4,5]',item))
        self.assertEqual(parse_answer('[0,0,640,480]',item),[0,0,640,480])

    def test_own_declared_parse_failure_scores_zero(self):
        item={'axis':'pope','answer_type':'yes_no','answer_key':'yes'}
        scored=score_generation('maybe',item)
        self.assertEqual((scored['correct_bit'],scored['parse_fail_flag']),(0,1))

    def test_five_paraphrases_are_one_scoring_unit(self):
        manifest=[{'axis':'consistency','dataset':'mmbench','bootstrap_unit':'q1','item_id':str(i),'image_group':'img'} for i in range(5)]
        ledger=[{'axis':'consistency','item_id':str(i),'correct_bit':int(i<4),'parse_fail_flag':0} for i in range(5)]
        result=units(manifest,ledger)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['correct_bit'],0)
        with self.assertRaises(ValueError):
            units(manifest,ledger[:-1])

    def test_floor_amplification_boundary_is_recovered(self):
        result=planted()
        self.assertAlmostEqual(result['flip_ratio_boundary'][0],.5102040816326531)
        self.assertTrue(result['heterogeneous_flip_recovered'])

    def test_subfloor_does_not_clip_a_negative_capability_to_zero(self):
        scores=np.array([[.5,.6,.6,.6],[.7,.8,.8,.8]])
        vector,reasons=ruler_vector(scores,np.full(4,.5),{**self.setting,'floor':True})
        self.assertTrue(np.isnan(vector[0]))
        self.assertIn('sub_floor',reasons[0])
        self.assertTrue(np.isfinite(vector[1:]).all())

    def test_zero_and_negative_headroom_are_guarded(self):
        scores=np.array([[.6,.8,.6,.6],[.6,.7,.9,.9]])
        vector,reasons=ruler_vector(scores,np.full(4,.1),{**self.setting,'headroom':'raw_family'})
        self.assertTrue(np.isnan(vector[:2]).all())
        self.assertIn('zero_denominator',reasons[0])
        self.assertIn('headroom_sign',reasons[1])

    def test_floor_cancels_algebraically_under_headroom_on_same_domain(self):
        scores=np.array([[.6,.65,.7,.75],[.8,.9,.9,.9]])
        floors=np.array([.5,.02,.25,.35])
        a,_=ruler_vector(scores,floors,{**self.setting,'floor':True,'headroom':'raw_family'})
        b,_=ruler_vector(scores,floors,{**self.setting,'headroom':'corrected_family'})
        np.testing.assert_allclose(a,b)
        self.assertEqual(canonical({**self.setting,'floor':True,'headroom':'raw_family'},1),
                         canonical({**self.setting,'headroom':'corrected_family'},1))

    def test_equivalence_preserves_guard_domain(self):
        self.assertNotEqual(canonical({**self.setting,'headroom':'raw_family'},1),
                            canonical({**self.setting,'headroom':'corrected_family'},1))

    def test_owa_is_irrelevant_for_single_edge(self):
        self.assertEqual(canonical({**self.setting,'rho':0},1),canonical({**self.setting,'rho':1},1))
        self.assertNotEqual(canonical({**self.setting,'rho':0},2),canonical({**self.setting,'rho':1},2))

    def test_full_grid_and_all_tied_case(self):
        protocol=json.loads((ROOT/'configs/experiment.json').read_text())
        self.assertEqual(len(grid(protocol)),120)
        scores=np.full((2,4),.75)
        result=scan({'wrong':scores,'exclude':scores},{'wrong':np.full(4,.2),'exclude':np.full(4,.2)},protocol)
        self.assertTrue(result['all_tied'])
        self.assertIsNone(result['F'])

    def test_negative_cross_axis_gain_is_retained(self):
        pre=np.full(4,.6)
        post=np.array([.65,.59,.60,.58])
        gains,_=ruler_vector(np.array([pre,post]),np.full(4,.1),self.setting)
        self.assertAlmostEqual(float(gains.sum()),.02)
        self.assertLess(gains[1],0)
        self.assertLess(gains[3],0)


if __name__=='__main__':
    unittest.main()
