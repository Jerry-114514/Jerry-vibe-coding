"""Independent coordinate x supervision intervention; legacy pixel bytes preserved."""
import json
import math

from qwen35_native_frontend import native_prompt

CONDITIONS = ('P-D', 'P-G', 'N-D', 'N-G')


def training_example(row, teacher, condition):
    from preflight_9b_full import answer_text
    if condition not in CONDITIONS:
        raise ValueError(condition)
    prompt = native_prompt(row) if condition.startswith('N-') else dict(row)
    gt, taught = answer_text(row), answer_text(row, teacher)
    if condition.startswith('N-') and row['axis'] == 'grounding':
        def normalize(text):
            box = json.loads(text)
            values = [v/(row['width'] if i % 2 == 0 else row['height'])*1000 for i,v in enumerate(box)]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Nonfinite coordinate')
            x0,y0,x1,y1 = values
            if not 0<=x0<x1<=1000 or not 0<=y0<y1<=1000:
                raise ValueError('Invalid normalized training target')
            return json.dumps(values)
        gt, taught = normalize(gt), normalize(taught)
    return {**prompt, 'gt_target':gt, 'teacher_target':taught,
            'repair_condition':condition,'teacher_loss_enabled':condition.endswith('-D')}


def combined_loss(gt, teacher, condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    return .5*gt+.5*teacher if condition.endswith('-D') else gt
