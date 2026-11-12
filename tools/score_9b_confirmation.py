"""CPU scoring for frozen confirmation prompts and raw generation ledgers."""
import argparse
import json
from collections import Counter
from pathlib import Path

from experiment_common import parse_answer, read_rows, score_generation, units, write_json, write_rows
from scoring import box_iou

EOS_IDS = {248046, 248044}
MAX_NEW_TOKENS = 128


def score_row(item, prediction):
    token_ids = prediction['generated_token_ids']
    if len(token_ids) > MAX_NEW_TOKENS or any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError(f"Invalid token ledger for {item['item_id']}")
    if EOS_IDS.intersection(token_ids[:-1]):
        raise ValueError(f"Tokens after EOS for {item['item_id']}")
    eos = bool(token_ids and token_ids[-1] in EOS_IDS)
    finish = 'eos' if eos else 'length' if len(token_ids) == MAX_NEW_TOKENS else 'incomplete'
    if prediction.get('finish_reason', finish) != finish or finish == 'incomplete':
        raise ValueError(f"Unverified generation termination for {item['item_id']}")
    text = prediction['raw_generation']
    if item['axis'] != 'grounding':
        scored = score_generation(text, item)
    else:
        contract = item['coordinate_system']
        width, height = item['width'], item['height']
        if min(width, height) <= 0 or contract not in ('original_pixels', 'native_1000'):
            raise ValueError('Invalid grounding manifest geometry or coordinate system')
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
        limits = {'width': width, 'height': height} if contract == 'original_pixels' else {
            'width': 1000, 'height': 1000}
        box = parse_answer(text, {**item, **limits}) if isinstance(payload, list) and len(payload) == 4 else None
        if box is not None and contract == 'native_1000':
            box = [v * (height if i % 2 else width) / 1000 for i, v in enumerate(box)]
        iou = box_iou(box, item['answer_key']) if box is not None else None
        scored = {'raw_generation': text, 'parsed_answer': box, 'iou': iou,
                  'parse_fail_flag': int(box is None), 'correct_bit': int(iou is not None and iou >= .5)}
    if not eos:
        scored['correct_bit'] = 0
    return {**scored, 'item_id': item['item_id'], 'axis': item['axis'],
            'finish_reason': finish, 'generated_token_ids': token_ids}


def score_manifest(manifest, predictions):
    lookup = {row['item_id']: row for row in predictions}
    identifiers = {row['item_id'] for row in manifest}
    if len(identifiers) != len(manifest) or len(lookup) != len(predictions) or identifiers != lookup.keys():
        raise ValueError('Duplicate, missing or extra prediction/manifest rows')
    scored = [score_row(item, lookup[item['item_id']]) for item in manifest]
    # The pixel replay is a separate endpoint for scoring; a grounding original
    # must not collapse the two prompts into a single statistical unit.
    results = {}
    for contract in ('native_1000', 'original_pixels'):
        pairs = [(item, row) for item, row in zip(manifest, scored)
                 if item.get('coordinate_system', 'native_1000') == contract]
        if not pairs:
            continue
        prompts, ledger = zip(*pairs)
        by_axis = Counter()
        correct = Counter()
        for unit in units(prompts, ledger):
            axis = unit['axis']
            by_axis[axis] += 1
            correct[axis] += unit['correct_bit']
        results[contract] = {
            'prompt_rows': len(ledger), 'units': dict(by_axis),
            'accuracy': {axis: correct[axis] / n for axis, n in by_axis.items()},
        }
    return scored, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('raw_ledger', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    scored, results = score_manifest(read_rows(args.manifest), read_rows(args.raw_ledger))
    write_rows(args.output / 'scored.jsonl', scored)
    write_json(args.output / 'scores.json', results)
