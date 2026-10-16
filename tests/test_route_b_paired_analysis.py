"""Paired bootstrap must preserve identical-model zero differences and constants."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_route_b_staged as analysis


class PairedAnalysisTest(unittest.TestCase):
    def test_perfect_pairing_and_constant_effects(self):
        manifest = []
        for axis in analysis.AXES:
            for unit in range(3):
                for paraphrase in range(5 if axis == 'consistency' else 1):
                    manifest.append({'item_id':f'{axis}-{unit}-{paraphrase}', 'axis':axis,
                                     'dataset':'synthetic', 'bootstrap_unit':f'{axis}-{unit}',
                                     'image_group':f'image-{unit}'})
        def rows(path):
            if Path(path).name == 'synthetic-manifest.jsonl':
                return manifest
            correct = int(not Path(path).parent.name.startswith('N-'))
            return [{'item_id':r['item_id'],'axis':r['axis'],'correct_bit':correct,'parse_fail_flag':0} for r in manifest]
        def metadata(path):
            if Path(path).name == 'protocol.json':
                return {'eval':{'formal_manifest':'synthetic-manifest.jsonl'}}
            return {'rows':16900,'ledger_sha256':'synthetic'}
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(analysis,'OUT',Path(folder)), patch.object(analysis,'read_rows',rows), \
                 patch.object(analysis,'read_json',metadata), patch.object(analysis,'sha',lambda p:'synthetic'), \
                 patch.object(analysis,'verify',lambda phase:None):
                analysis.analyze(200)
            import json
            result = json.loads((Path(folder)/'formal-paired-analysis.json').read_text(encoding='utf-8'))
        for axis in analysis.AXES:
            c = result['axes'][axis]['paired_contrasts']
            for name in ('A1-U','A2-U'):
                self.assertEqual(c[name],{'difference':0.0,'ci95':[0.0,0.0]})
            for name in ('A1-N','A2-N','U-N'):
                self.assertEqual(c[name],{'difference':1.0,'ci95':[1.0,1.0]})


if __name__ == '__main__':
    unittest.main()
