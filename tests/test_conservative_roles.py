from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from freeze_clean_candidates_v3 import conservative_components, repair_role_quotas


class ConservativeRoleTests(unittest.TestCase):
    def test_near_chain_reaching_history_removes_whole_question_component(self):
        historic = [{'pixel_sha256': 'old', 'dhash64': '0000000000000000'}]
        assets = [{'pixel_sha256': 'a', 'dhash64': '000000000000000f'},
                  {'pixel_sha256': 'b', 'dhash64': '00000000000000ff'},
                  {'pixel_sha256': 'c', 'dhash64': 'ffffffffffffffff'}]
        rows = [{'item_id': 'x', 'image_pixel_sha256': ['b', 'c']}]
        assigned, excluded, _, _ = conservative_components(rows, assets, historic)
        self.assertFalse(assigned)
        self.assertEqual(excluded[0]['item_id'], 'x')

    def test_near_images_receive_same_role_without_declaring_human_review(self):
        assets = [{'pixel_sha256': 'a', 'dhash64': '000000000000000f'},
                  {'pixel_sha256': 'b', 'dhash64': '00000000000000ff'}]
        rows = [{'item_id': key, 'image_pixel_sha256': [key]} for key in ('a', 'b')]
        assigned, excluded, edges, _ = conservative_components(rows, assets, [])
        self.assertFalse(excluded)
        self.assertEqual(len(edges), 1)
        self.assertEqual(assigned[0]['image_group'], assigned[1]['image_group'])
        self.assertEqual(assigned[0]['role'], assigned[1]['role'])

    def test_role_repair_preserves_whole_components_and_both_sample_targets(self):
        rows = [{'axis':'consistency','dataset':'ai2d','subset':'test','role':role,
                 'image_group':str(group),'item_id':str(group)+':'+str(question)}
                for group,role in enumerate(['calibration','audit','confirm','confirm','confirm'])
                for question in range(2)]
        assigned,moves = repair_role_quotas(rows,{'calibration':2,'audit':4,'confirm':4})
        self.assertTrue(moves)
        for group in range(5):
            self.assertEqual(len({r['role'] for r in assigned if r['image_group']==str(group)}),1)
        for role in ('audit','confirm'):
            self.assertGreaterEqual(sum(r['role']==role for r in assigned),4)

    def test_insufficient_original_units_fail_instead_of_lowering_target(self):
        rows = [{'axis':'consistency','dataset':'ai2d','role':'confirm','image_group':str(i)} for i in range(6)]
        with self.assertRaises(ValueError):
            repair_role_quotas(rows,{'audit':4,'confirm':4})


if __name__ == '__main__':
    unittest.main()
