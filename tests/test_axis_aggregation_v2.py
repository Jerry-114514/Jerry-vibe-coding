from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from axis_aggregation_v2 import summarize_units


class AxisAggregationTests(unittest.TestCase):
    def test_datasets_keep_equal_weight_when_one_has_more_test_splits(self):
        rows=[{'axis':'grounding','stratum':s,'correct_bit':value}
              for s,value in [('refcoco/testA',1),('refcoco/testB',1),('refcocog/test',0)]]
        self.assertAlmostEqual(summarize_units(rows)['grounding'],.5)

    def test_subset_size_does_not_change_dataset_weight(self):
        rows=[{'axis':'x','stratum':'one/a','correct_bit':1}]*100+[
              {'axis':'x','stratum':'one/b','correct_bit':0},{'axis':'x','stratum':'two/a','correct_bit':0}]
        self.assertAlmostEqual(summarize_units(rows)['x'],.25)


if __name__=='__main__': unittest.main()
