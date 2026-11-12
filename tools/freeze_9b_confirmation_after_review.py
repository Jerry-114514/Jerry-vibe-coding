"""Validate independent localization reviews and emit the confirmation manifests."""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

from experiment_common import parse_answer, read_json, read_rows, sha, write_json, write_rows
from scoring import box_iou

IOU_AGREEMENT = 0.5
QUOTA = 48
AXIS_PROMPTS = {'pope': 120, 'consistency': 600, 'mmstar': 120, 'grounding': 240}


def read_review(path, queue):
    record = read_json(path)
    reviewer = record.get('reviewer')
    answers = record.get('answers')
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(answers, dict):
        raise ValueError(f'{path}: expected a reviewer code and answers')
    if answers.keys() != queue.keys():
        raise ValueError(f'{path}: missing {len(queue.keys() - answers.keys())} and '
                         f'extra {len(answers.keys() - queue.keys())} review IDs')
    for key, answer in answers.items():
        if not isinstance(answer, dict) or set(answer) != {'verdict', 'note', 'box'}:
            raise ValueError(f'{path}: invalid review schema for {key}')
        verdict, note, box = answer['verdict'], answer['note'], answer['box']
        if verdict not in ('accept', 'ambiguous', 'unusable') or not isinstance(note, str):
            raise ValueError(f'{path}: invalid verdict or note for {key}')
        if verdict == 'accept':
            item = queue[key]['source']
            if not isinstance(box, list) or parse_answer(str(box), item) is None:
                raise ValueError(f'{path}: invalid original-pixel box for {key}')
        elif box is not None or not note.strip():
            raise ValueError(f'{path}: reason required for {key}')
    return reviewer.strip().casefold(), answers


def review_decisions(queue, left, right):
    decisions = []
    selected = defaultdict(list)
    for key, row in sorted(queue.items(), key=lambda pair: (pair[1]['stratum'], pair[1]['order'])):
        first, second = left[key], right[key]
        stratum = row['stratum']
        ious = None
        if first['verdict'] != 'accept' or second['verdict'] != 'accept':
            status = 'reviewer_rejected'
        else:
            gt = row['source']['answer_key']
            ious = [box_iou(first['box'], second['box']), box_iou(first['box'], gt), box_iou(second['box'], gt)]
            if min(ious) < IOU_AGREEMENT:
                status = 'review_or_source_box_disagreement'
            elif len(selected[stratum]) == QUOTA:
                status = 'eligible_reserve'
            else:
                selected[stratum].append(row)
                status = 'selected'
        decisions.append({'review_id': key, 'stratum': stratum, 'order': row['order'],
                          'status': status, 'reviewer_verdicts': [first['verdict'], second['verdict']],
                          'mutual_and_source_ious': ious})
    return selected, decisions


def prompt(row, contract):
    width, height = row['width'], row['height']
    # Both arms receive the same source dimensions, expression, and output schema.
    limits = f'0..{width} along x and 0..{height} along y, in original pixels' if contract == 'original_pixels' \
        else '0..1000 along x and 0..1000 along y, normalized to the original image'
    return (f'Locate the object referred to by: {row["referring_expression"]}. '
            f'The original image is {width} pixels wide and {height} pixels high. '
            f'Use coordinates {limits}. Return exactly one JSON array '
            '[x_min, y_min, x_max, y_max].')


def freeze(run, first_path, second_path, output):
    prep_path = run / 'preparation.json'
    prep = read_json(prep_path)
    for name, expected in prep['sha256'].items():
        if sha(run / name) != expected:
            raise ValueError(f'Candidate input changed: {name}')
    rows = read_rows(run / 'private/grounding-queue.jsonl')
    queue = {row['review_id']: row for row in rows}
    if len(rows) != len(queue) or len({row['source']['image_group'] for row in rows}) != len(rows):
        raise ValueError('Duplicate review ID or image component')
    for stratum, counts in prep['grounding_by_stratum'].items():
        if sorted(row['order'] for row in rows if row['stratum'] == stratum) != list(
                range(1, counts['main'] + counts['reserve'] + 1)):
            raise ValueError(f'Review order changed: {stratum}')
    for key, row in queue.items():
        if sha(run / 'review/images' / f'{key}.jpg') != row['asset']['sha256']:
            raise ValueError(f'Blind review image changed: {key}')
    first, left = read_review(first_path, queue)
    second, right = read_review(second_path, queue)
    if first == second or left == right:
        raise ValueError('Separate reviewer codes and independently recorded annotations are required')
    selected, decisions = review_decisions(queue, left, right)
    quotas = {key: len(selected.get(key, [])) for key in prep['grounding_by_stratum']}
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    output.mkdir(parents=True)
    write_rows(output / 'adjudication.jsonl', decisions)
    if len(quotas) != 5 or set(quotas.values()) != {QUOTA}:
        write_json(output / 'review-status.json', {'status': 'insufficient_reviewed_examples', 'quotas': quotas})
        raise ValueError(f'Insufficient reviewed examples within the registered strata: {quotas}')
    items = read_rows(run / 'private/secondary-provisional.jsonl')
    for rows in selected.values():
        for entry in rows:
            row = entry['source']
            for contract in ('native_1000', 'original_pixels'):
                items.append({**row, 'item_id': f"{row['item_id']}:{contract}",
                              'question': prompt(row, contract), 'coordinate_system': contract,
                              'review_id': entry['review_id']})
    main = [item for item in items if item.get('coordinate_system') != 'original_pixels']
    if Counter(row['axis'] for row in main) != AXIS_PROMPTS or len({row['item_id'] for row in items}) != 1320:
        raise ValueError('Confirmation prompt quotas or IDs differ from the protocol')
    token_file = run / 'private/evaluation-token-verification.jsonl'
    token_report = read_json(run / 'tokenization-verification.json')
    if sha(token_file) != token_report['evaluation_token_rows_sha256']:
        raise ValueError('Evaluation token or image lock changed')
    token_rows = read_rows(token_file)
    tokens = {row['item_id']: row for row in token_rows}
    if len(tokens) != len(token_rows):
        raise ValueError('Evaluation token lock contains duplicate IDs')
    for row in items:
        lock = tokens[row['item_id']]
        if (lock['image_path'] != row['image_path']
                or lock['prompt_sha256'] != hashlib.sha256(row['question'].encode('utf-8')).hexdigest()):
            raise ValueError(f"Prompt or image path differs from token lock: {row['item_id']}")
        for field in ('image_sha256', 'image_grid_thw', 'prompt_tokens', 'prompt_ids_sha256'):
            row[field] = lock[field]
    write_rows(output / 'evaluation-manifest.jsonl', items)
    write_json(output / 'review-status.json', {
        'status': 'reviewed_manifest_frozen', 'scope': 'Evaluation manifest only. GPU run authorization is separate.',
        'grounding_strata': quotas, 'prompt_rows': len(items),
        'decisions': dict(Counter(row['status'] for row in decisions)),
        'preparation_sha256': sha(prep_path),
        'review_sha256': [sha(first_path), sha(second_path)],
        'script_sha256': sha(Path(__file__)),
        'adjudication_sha256': sha(output / 'adjudication.jsonl'),
        'model_revision': token_report['model_revision'],
        'evaluation_token_lock_sha256': sha(token_file),
        'evaluation_manifest_sha256': sha(output / 'evaluation-manifest.jsonl'),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'first_review', 'second_review', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    freeze(args.run, args.first_review, args.second_review, args.output)
