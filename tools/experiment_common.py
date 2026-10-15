"""Portable, frozen experiment IO and scoring primitives (B--E)."""
import hashlib
import json
import math
from pathlib import Path
import re
from scoring import box_iou, consistency_bit

ROOT = Path(__file__).resolve().parents[1]
AXES = ['pope', 'grounding', 'consistency', 'mmstar']
MODELS = {
    'qwen3b': ('Qwen/Qwen2.5-VL-3B-Instruct', 'qwen'),
    'qwen7b': ('Qwen/Qwen2.5-VL-7B-Instruct', 'qwen'),
    'intern1b': ('OpenGVLab/InternVL3-1B', 'intern'),
    'intern2b': ('OpenGVLab/InternVL3-2B', 'intern'),
    'intern8b': ('OpenGVLab/InternVL3-8B', 'intern'),
    'llava05b': ('llava-hf/llava-onevision-qwen2-0.5b-ov-hf', 'llava'),
    'llava7b': ('llava-hf/llava-onevision-qwen2-7b-ov-hf', 'llava'),
}
LADDERS = {'qwen': ['qwen3b', 'qwen7b'], 'intern': ['intern1b', 'intern2b', 'intern8b'],
           'llava': ['llava05b', 'llava7b']}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def parse_answer(text, item):
    """Strict, checkpoint-independent parser; no guessing or unit conversions."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text).strip()
    if item['answer_type'] == 'yes_no':
        match = re.fullmatch(r'(yes|no)[.!]?\s*', text, re.I)
        return match[1].lower() if match else None
    if item['answer_type'] == 'option_letter':
        match = re.fullmatch(r'(?:\(([A-D])\)|([A-D]))[.!]?\s*', text, re.I)
        value = (match[1] or match[2]).upper() if match else None
        return value if value in item['options'] else None
    try:
        box = json.loads(text)
        # Explicit JSON schemas only; accept common native bbox_2d wrappers,
        # never guess coordinates from prose, resize them or select among objects.
        if isinstance(box,list) and len(box)==1 and isinstance(box[0],dict):
            box=box[0]
        if isinstance(box, dict):
            if set(box)=={'bbox'}:
                box=box['bbox']
            elif 'bbox_2d' in box and set(box)<={'bbox_2d','label'}:
                box=box['bbox_2d']
            else:
                return None
        if not isinstance(box, list) or len(box) != 4:
            return None
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            return None
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= item['width'] and 0 <= y0 < y1 <= item['height']):
            return None
        return box
    except (ValueError, TypeError):
        return None


def score_generation(text, item):
    parsed = parse_answer(text, item)
    iou = box_iou(parsed, item['answer_key']) if parsed is not None and item['axis'] == 'grounding' else None
    bit = int(iou >= .5) if iou is not None else int(parsed is not None and parsed == item['answer_key'])
    return {'raw_generation': text, 'parsed_answer': parsed, 'correct_bit': bit,
            'parse_fail_flag': int(parsed is None), 'iou': iou}


def units(manifest, ledger):
    """Require exact coverage; retain the five correlated paraphrases as one unit."""
    by_id = {}
    for row in ledger:
        if row['item_id'] in by_id:
            raise ValueError('Duplicate ledger item: ' + row['item_id'])
        by_id[row['item_id']] = row
    if set(by_id) != {x['item_id'] for x in manifest}:
        raise ValueError('Ledger coverage differs from frozen manifest')
    grouped = {}
    for item in manifest:
        row = by_id[item['item_id']]
        if row['axis'] != item['axis']:
            raise ValueError('Axis mismatch')
        grouped.setdefault((item['axis'], item['dataset'], item['bootstrap_unit']), []).append((item, row))
    output = []
    for (axis, dataset, unit), group in sorted(grouped.items()):
        bits = [row['correct_bit'] for _, row in group]
        subset = group[0][0].get('subset', '')
        output.append({'axis': axis, 'dataset': dataset, 'stratum': dataset + '/' + subset, 'unit': unit,
                       'image_group': group[0][0]['image_group'],
                       'correct_bit': consistency_bit(bits) if axis == 'consistency' else bits[0],
                       'parse_fail_flag': int(any(row['parse_fail_flag'] for _, row in group))})
    return output
