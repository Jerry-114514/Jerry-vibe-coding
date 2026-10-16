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


def score_processed_grounding(text, item, processed_width, processed_height):
    if item['axis'] != 'grounding' or min(processed_width,processed_height) <= 0:
        raise ValueError('Processed pixel contract requires grounding and positive image dimensions')
    parsed = parse_answer(text,{**item,'width':processed_width,'height':processed_height})
    if parsed is None:
        return {'raw_generation':text,'parsed_answer':None,'correct_bit':0,'parse_fail_flag':1,
                'iou':None,'coordinate_contract':'processed_pixels'}
    pixels=[parsed[0]*item['width']/processed_width,parsed[1]*item['height']/processed_height,
            parsed[2]*item['width']/processed_width,parsed[3]*item['height']/processed_height]
    return {**score_generation(json.dumps(pixels),item),'raw_generation':text,
            'processed_box':parsed,'coordinate_contract':'processed_pixels'}
