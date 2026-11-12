"""Build the pre-tokenization paired GT-only training plan without loading a model."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random

from experiment_common import AXES, read_rows, sha, write_json, write_rows
from freeze_9b_confirmation_after_review import prompt
from prepare_9b_confirmation_review import rank

SEEDS = range(2026111101, 2026111109)
POOL_PER_AXIS = 150
UPDATES_PER_AXIS = 250


def training_record(row, contract):
    target = row['answer_key']
    if row['axis'] != 'grounding':
        return {**row, 'target': target}
    prefix = 'Locate the object referred to by: '
    marker = '. Return only its bounding box as '
    question = row['question']
    if not question.startswith(prefix) or marker not in question:
        raise ValueError(f"Unknown source prompt: {row['item_id']}")
    row = {**row, 'referring_expression': question[len(prefix):].split(marker, 1)[0]}
    width, height = row['width'], row['height']
    if not 0 <= target[0] < target[2] <= width or not 0 <= target[1] < target[3] <= height:
        raise ValueError(f"Invalid source box: {row['item_id']}")
    if contract == 'native_1000':
        target = [value * 1000 / (height if index % 2 else width) for index, value in enumerate(target)]
    target = [round(value, 2) for value in target]
    if target[0] >= target[2] or target[1] >= target[3]:
        raise ValueError(f"Degenerate rounded box: {row['item_id']}")
    return {**row, 'question': prompt(row, contract), 'target': json.dumps(target, separators=(',', ':'))}


def prepare(eligible, restore_map, output):
    rows = read_rows(eligible)
    with gzip.open(restore_map, 'rt', encoding='utf-8') as stream:
        assets = {asset['path']: asset for asset in json.load(stream)['files']}
    chosen = {}
    for axis in AXES:
        ordered = sorted((row for row in rows if row['axis'] == axis), key=lambda row: rank(row['item_id']))
        chosen[axis] = ordered[:POOL_PER_AXIS]
        if len(chosen[axis]) != POOL_PER_AXIS:
            raise ValueError(f'Insufficient training rows for {axis}')
    pool = [row for axis in AXES for row in chosen[axis]]
    if len({row['item_id'] for row in pool}) != len(pool):
        raise ValueError('Duplicate training item')
    for row in pool:
        row['image_sha256'] = assets[row['image_path']]['sha256']
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for arm, contract in (('P-G', 'original_pixels'), ('N-G', 'native_1000')):
        file = output / f'{arm}.jsonl'
        write_rows(file, [training_record(row, contract) for row in pool])
        files.append(file)
    first_arms = ['P-G', 'N-G'] * 4
    random.Random(20261112).shuffle(first_arms)
    for seed, first in zip(SEEDS, first_arms):
        rng = random.Random(seed)
        schedule = []
        for axis in AXES:
            ids = [row['item_id'] for row in chosen[axis]]
            one = rng.sample(ids, len(ids))
            two = rng.sample(ids, UPDATES_PER_AXIS - len(ids))
            schedule.extend({'axis': axis, 'item_id': key} for key in one + two)
        rng.shuffle(schedule)
        file = output / f'seed-{seed}.json'
        write_json(file, {'seed': seed, 'first_arm': first, 'updates': schedule})
        files.append(file)
    write_json(output / 'training-plan.json', {
        'status': 'pre_tokenization_plan',
        'source_pool_sha256': sha(eligible), 'restore_map_sha256': sha(restore_map),
        'pool_by_axis': dict(Counter(row['axis'] for row in pool)),
        'real_updates_per_axis_per_arm': UPDATES_PER_AXIS, 'seeds': list(SEEDS),
        'sha256': {file.name: sha(file) for file in files},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('eligible', 'restore_map', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.eligible, args.restore_map, args.output)
