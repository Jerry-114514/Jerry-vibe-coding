"""Explicit multi-source scoring; historical v1 parser remains unchanged."""
import math
import random
import re

from coordinate_contracts import score_normalized_grounding
from experiment_common import score_generation

# Upstream module seeds Python's global RNG at import. Preserve caller state.
_random_state = random.getstate()
try:
    from vendor.mmmu_eval_utils import parse_open_response, eval_open
finally:
    random.setstate(_random_state)


def score_generation_v2(text, item, coordinate_contract='pixels'):
    kind = item['answer_type']
    if kind in ('bbox', 'bbox_xyxy_pixels'):
        if coordinate_contract == 'normalized_1000':
            return score_normalized_grounding(text, item)
        if coordinate_contract != 'pixels':
            raise ValueError('Unregistered coordinate contract')
        return score_generation(text, item)
    if kind == 'yes_no':
        return score_generation(text, item)
    if kind in ('option_letter', 'option_letter_v2'):
        value = text.strip()
        value = re.sub(r'^```(?:json)?\s*|\s*```$', '', value).strip()
        match = re.fullmatch(r'(?:\(([A-Z])\)|([A-Z]))[.!]?\s*', value, re.I)
        parsed = (match[1] or match[2]).upper() if match else None
        if parsed not in item['options']:
            parsed = None
        return {'raw_generation': text, 'parsed_answer': parsed,
                'correct_bit': int(parsed is not None and parsed == item['answer_key']),
                'parse_fail_flag': int(parsed is None), 'iou': None}
    if kind == 'mmmu_open':
        parsed = parse_open_response(text) if text.strip() else []
        parsed = [value for value in parsed if not isinstance(value, float) or math.isfinite(value)]
        # Upstream returns a set-derived list; serialize in a stable order.
        parsed = sorted(parsed, key=lambda value: (type(value).__name__, str(value)))
        return {'raw_generation': text, 'parsed_answer': parsed or None,
                'correct_bit': int(bool(parsed) and eval_open(item['answer_key'], parsed)),
                'parse_fail_flag': int(not parsed), 'iou': None,
                'parser': 'MMMU official open-answer parser and evaluator, vendored content lock'}
    raise ValueError('Unknown protocol-v2 answer type')
