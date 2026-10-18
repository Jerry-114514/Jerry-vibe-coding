import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from repair_9b_codec import training_example,combined_loss


def grounding():
    return {'axis':'grounding','width':640,'height':480,'answer_key':[64,96,256,288],
        'question':'Locate object in original-image pixel coordinates. The image is 640 pixels wide and 480 pixels high.'}


def test_historical_pixel_dual_ce_is_unchanged():
    row=grounding();teacher={'raw_generation':'[100,200,400,600]'}
    example=training_example(row,teacher,'P-D')
    assert example['question']==row['question']
    assert example['gt_target']==json.dumps(row['answer_key'])
    assert example['teacher_target']=='[64.0, 96.0, 256.0, 288.0]'


def test_normalized_gt_and_teacher_share_geometry_without_gt_fitting():
    result=training_example(grounding(),{'raw_generation':'[100,200,400,600]'},'N-D')
    assert json.loads(result['gt_target'])==[100,200,400,600]
    assert json.loads(result['teacher_target'])==[100,200,400,600]
    assert 'normalized coordinates from 0 to 1000' in result['question']


def test_gt_only_maintains_full_scale():
    assert combined_loss(4.,100.,'P-G')==4.
    assert combined_loss(4.,100.,'N-G')==4.
    assert combined_loss(4.,100.,'P-D')==52.


def test_non_grounding_targets_are_not_canonicalized_or_filtered():
    row={'axis':'pope','answer_key':'yes','question':'Answer yes or no'}
    result=training_example(row,{'raw_generation':'Yes.'},'N-D')
    assert result['question']==row['question']
    assert result['gt_target']=='yes' and result['teacher_target']=='Yes.'


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):training_example({}, {}, 'pick-best')
