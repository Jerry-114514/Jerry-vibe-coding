"""Independent proposed 9B frontend; never replace the failed v3 score.

No model loading, network calls or generation. Fixed normalized coordinates
become processor pixels, then original pixels under the shared measurement.
"""
from experiment_common import parse_answer
from processor_pixel_contract_csv import FOUR
from scoring import box_iou


def native_prompt(item):
    """Amend only a recognized grounding prompt; retain labels and IDs."""
    row = dict(item)
    if row['axis'] != 'grounding':
        return row
    suffix = (f'in original-image pixel coordinates. The image is {row["width"]} pixels wide '
              f'and {row["height"]} pixels high.')
    if not row['question'].endswith(suffix):
        raise ValueError('Unknown source prompt; refuse heuristic text replacement')
    row['question'] = row['question'][:-len(suffix)] + (
        'in normalized coordinates from 0 to 1000 relative to the supplied image. '
        'The full image spans [0, 0, 1000, 1000]. '
        'Return exactly one JSON array [x_min, y_min, x_max, y_max].')
    return row


def parse_native_box(text):
    # The two explicit syntaxes are fixed for every arm. No prose extraction,
    # output-magnitude heuristic or per-arm/GT-dependent unit inference.
    payload = '[' + text.strip() + ']' if FOUR.fullmatch(text) else text
    return parse_answer(payload, {'answer_type': 'bbox', 'width': 1000, 'height': 1000})


def score_native_frontend(text, item):
    assert item['axis'] == 'grounding'
    width, height = item['processor_width'], item['processor_height']
    assert min(width, height, item['width'], item['height']) > 0
    native = parse_native_box(text)
    processor = [v*(width if i % 2 == 0 else height)/1000 for i, v in enumerate(native)] if native else None
    original = [v*(item['width']/width if i % 2 == 0 else item['height']/height)
                for i, v in enumerate(processor)] if processor else None
    iou = box_iou(original, item['answer_key']) if original else None
    return {'raw_generation': text, 'parsed_native1000': native,
            'parsed_processor_pixels': processor, 'parsed_answer': original,
            'coordinate_system': 'fixed_native1000_to_processor_pixels_to_original_pixels',
            'parse_fail_flag': int(native is None), 'iou': iou,
            'correct_bit': int(iou >= .5) if iou is not None else 0}
