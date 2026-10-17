from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from processor_pixel_contract_csv import score


class CSVContractTests(unittest.TestCase):
    def item(self):return {'axis':'grounding','width':640,'height':480,'processor_width':320,'processor_height':120,'answer_key':[64,96,320,384]}
    def test_csv_matches_explicit_json_values_without_rewriting_raw(self):
        raw='32,24,160,96';a=score(raw,self.item());b=score('[32,24,160,96]',self.item())
        self.assertEqual(a['raw_generation'],raw)
        for key in ('parsed_answer','parsed_processor_pixels','correct_bit','iou'):self.assertEqual(a[key],b[key])
    def test_no_prose_extra_values_nonfinite_or_unit_fallback(self):
        for raw in ('box:32,24,160,96','32,24,160,96,1','NaN,24,160,96','100,200,500,800'):
            self.assertEqual(score(raw,self.item())['parse_fail_flag'],1)


if __name__=='__main__':unittest.main()
