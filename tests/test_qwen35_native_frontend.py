"""Test geometry and explicit schema without running a model or using outputs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen35_native_frontend import native_prompt, parse_native_box, score_native_frontend


def item():
    return {'axis': 'grounding', 'item_id': 'geometry-fixture', 'width': 640, 'height': 480,
            'processor_width': 704, 'processor_height': 512,
            'answer_key': [64, 96, 256, 288],
            'question': 'Return the bounding box in original-image pixel coordinates. '
                        'The image is 640 pixels wide and 480 pixels high.'}


def test_two_stage_mapping_is_resolution_independent():
    row = item()
    expected = [64, 96, 256, 288]
    for width, height in [(704, 512), (448, 672), (1000, 1000)]:
        result = score_native_frontend('[100, 200, 400, 600]',
                                       {**row, 'processor_width': width, 'processor_height': height})
        assert result['parsed_answer'] == pytest.approx(expected)
        assert result['parsed_processor_pixels'] == pytest.approx([.1*width, .2*height, .4*width, .6*height])
        assert result['iou'] == pytest.approx(1.) and result['correct_bit'] == 1


def test_strict_schemas_share_identical_values():
    for text in ['[100,200,400,600]', '100, 200, 400, 600', '{"bbox_2d":[100,200,400,600]}']:
        assert parse_native_box(text) == [100, 200, 400, 600]


@pytest.mark.parametrize('text', ['[0,0,1001,900]', '[0,0,0,100]', '[true,0,100,100]',
                                  '[0,0,NaN,100]', 'box: [0,0,100,100]', '0,0,100,100,200'])
def test_invalid_units_or_ambiguous_schema_are_not_rescued(text):
    result = score_native_frontend(text, item())
    assert result['parse_fail_flag'] == 1 and result['correct_bit'] == 0
    assert result['parsed_answer'] is None and result['parsed_processor_pixels'] is None


def test_ground_truth_cannot_change_parser_or_unit():
    text = '[100,200,400,600]'
    a = score_native_frontend(text, item())
    b = score_native_frontend(text, {**item(), 'answer_key': [0, 0, 10, 10]})
    assert a['parsed_native1000'] == b['parsed_native1000']
    assert a['parsed_processor_pixels'] == b['parsed_processor_pixels']
    assert a['parsed_answer'] == b['parsed_answer']
    assert a['correct_bit'] == 1 and b['correct_bit'] == 0


def test_prompt_amendment_preserves_identity_labels_and_original():
    row = item()
    new = native_prompt(row)
    assert 'normalized coordinates from 0 to 1000' in new['question']
    assert 'original-image pixel coordinates' in row['question']
    assert {k: v for k, v in new.items() if k != 'question'} == {k: v for k, v in row.items() if k != 'question'}
    with pytest.raises(ValueError):
        native_prompt({**row, 'question': 'Unregistered prompt'})
