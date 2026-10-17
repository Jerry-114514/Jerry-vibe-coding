"""Schema-only repair: accept exact four-number CSV as requested by prompt."""
import re
from processor_pixel_contract import dimensions,item_with_contract,assert_runtime_grid,cpu_7b_inputs
from processor_pixel_contract import score as json_score

NUMBER=r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?'
FOUR=re.compile(r'\s*'+NUMBER+r'(?:\s*,\s*'+NUMBER+r'){3}\s*')


def score(text,item):
    if item['axis']=='grounding' and FOUR.fullmatch(text):
        result=json_score('['+text.strip()+']',item)
        result['raw_generation']=text
        result['output_schema']='strict_four_number_csv'
        return result
    result=json_score(text,item)
    if item['axis']=='grounding':result['output_schema']='json_bbox_or_invalid'
    return result
