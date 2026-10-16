"""Explicit coordinate contracts; never infer units from predicted values."""
import json
from experiment_common import parse_answer, score_generation


def score_normalized_grounding(text, item):
    if item['axis'] != 'grounding':
        raise ValueError('Normalized box contract applies only to grounding')
    parsed = parse_answer(text, {**item, 'width': 1000, 'height': 1000})
    if parsed is None:
        return {'raw_generation': text, 'parsed_answer': None, 'correct_bit': 0,
                'parse_fail_flag': 1, 'iou': None, 'coordinate_contract': 'normalized_1000'}
    pixels = [parsed[0]*item['width']/1000, parsed[1]*item['height']/1000,
              parsed[2]*item['width']/1000, parsed[3]*item['height']/1000]
    return {**score_generation(json.dumps(pixels), item), 'raw_generation': text,
            'normalized_box': parsed, 'coordinate_contract': 'normalized_1000'}
