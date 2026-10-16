from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from independent_manifest_v2 import assign_roles, select_axis, expand_item
from freeze_independent_candidates import HashTree
from scoring_v2 import score_generation_v2


class DataRoleTests(unittest.TestCase):
    def test_multi_image_chain_is_one_role_and_cluster(self):
        rows=[{'image_pixel_sha256':['a','b']},{'image_pixel_sha256':['c','b']},
              {'image_pixel_sha256':['c']},{'image_pixel_sha256':['unrelated']}]
        result=assign_roles(rows)
        self.assertEqual(len({r['image_group'] for r in result[:3]}),1)
        self.assertEqual(len({r['role'] for r in result[:3]}),1)
        self.assertNotEqual(result[0]['image_group'],result[3]['image_group'])

    def test_pope_selection_preserves_exact_yes_no_balance_per_stratum(self):
        rows=[{'axis':'pope','dataset':'new-pope','subset':subset,'answer_key':answer,
               'item_id':f'{subset}:{answer}:{i}'} for subset in ('random','popular','adversarial')
              for answer in ('yes','no') for i in range(50)]
        selected=select_axis(rows,120)
        for subset in ('random','popular','adversarial'):
            for answer in ('yes','no'):
                self.assertEqual(sum(r['subset']==subset and r['answer_key']==answer for r in selected),20)

    def test_multi_image_item_does_not_expose_unsafe_single_image_alias(self):
        item={'item_id':'x','role':'confirm','axis':'mmstar','question':'Q <image 1> <image 2>',
              'options':{chr(65+i):str(i) for i in range(9)},'image_paths':['one','two']}
        result=expand_item(item)[0]
        self.assertNotIn('image_path',result)
        self.assertEqual(len(result['options']),9)
        self.assertFalse(result['independent_confirmation'])
        self.assertTrue(result['candidate_for_confirmation'])

    def test_hash_tree_search_matches_exhaustive_hamming_search(self):
        values=[0,1,3,7,15,31,255,65535,2**63]
        tree=HashTree()
        for value in values: tree.add(value,str(value))
        expected={str(value) for value in values if (value^3).bit_count()<=4}
        self.assertEqual({identity for identity,_ in tree.near(3,4)},expected)

    def test_ninth_choice_is_scored_and_prose_is_rejected(self):
        item={'answer_type':'option_letter_v2','options':dict.fromkeys('ABCDEFGHI'),'answer_key':'I'}
        self.assertEqual(score_generation_v2('I',item)['correct_bit'],1)
        self.assertEqual(score_generation_v2('The answer is I',item)['parse_fail_flag'],1)

    def test_official_mmmu_open_numeric_normalization(self):
        item={'answer_type':'mmmu_open','answer_key':'1234.57'}
        self.assertEqual(score_generation_v2('The answer is 1,234.567.',item)['correct_bit'],1)
        self.assertEqual(score_generation_v2('',item)['parse_fail_flag'],1)


if __name__=='__main__': unittest.main()
