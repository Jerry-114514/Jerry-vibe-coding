"""Verify frozen Qwen token IDs and image grids locally without loading weights."""
import argparse
import ast
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import statistics
import struct

from jinja2 import Environment
from PIL import Image
from tokenizers import Tokenizer, __version__ as tokenizer_version

from experiment_common import ROOT, read_json, read_rows, sha, write_json, write_rows
from freeze_9b_confirmation_after_review import prompt


class OfflineProcessor:
    def __init__(self, metadata, resize_source):
        tree = ast.parse(resize_source.read_text(encoding='utf-8'))
        resize = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'smart_resize')
        namespace = {'math': math}
        exec(compile(ast.Module(body=[resize], type_ignores=[]), str(resize_source), 'exec'), namespace)
        self.resize = namespace['smart_resize']
        self.template = Environment(trim_blocks=True, lstrip_blocks=True).from_string(
            (metadata / 'chat_template.jinja').read_text(encoding='utf-8'))
        self.tokenizer = Tokenizer.from_file(str(metadata / 'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        config = read_json(metadata / 'tokenizer_config.json')
        self.eos = self.tokenizer.token_to_id(config['eos_token'])
        self.image_token = config['extra_special_tokens']['image_token']
        self.image_id = self.tokenizer.token_to_id(self.image_token)
        preprocessor = read_json(metadata / 'preprocessor_config.json')
        self.patch, self.merge = preprocessor['patch_size'], preprocessor['merge_size']
        if (self.eos, self.patch, self.merge) != (248046, 16, 2):
            raise ValueError('Unexpected frozen tokenizer or processor contract')

    def inputs(self, question, width, height):
        if min(width, height) < 32:
            scale = 32 / min(width, height)
            width, height = round(width * scale), round(height * scale)
        factor = self.patch * self.merge
        height, width = self.resize(height, width, factor=factor,
                                    min_pixels=256 * 32**2, max_pixels=1280 * 32**2)
        grid = [1, height // self.patch, width // self.patch]
        visual = math.prod(grid) // self.merge**2
        rendered = self.template.render(messages=[{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': question}]}],
            add_generation_prompt=True, enable_thinking=False)
        if rendered.count(self.image_token) != 1:
            raise ValueError('Expected exactly one image token placeholder')
        ids = self.tokenizer.encode(rendered.replace(self.image_token, self.image_token * visual),
                                    add_special_tokens=False).ids
        if ids.count(self.image_id) != visual or visual > 1280:
            raise ValueError('Expanded vision token count disagrees with image grid')
        return grid, ids

    def target(self, text):
        return self.tokenizer.encode(str(text), add_special_tokens=False).ids + [self.eos]


def digest(ids):
    return hashlib.sha256(struct.pack(f'<{len(ids)}q', *ids)).hexdigest()


def describe(values):
    return {'count': len(values), 'min': min(values), 'max': max(values), 'mean': statistics.mean(values)}


def verify(run, historical, metadata, source, historical_lock):
    lock = read_json(historical_lock)
    names = ('chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json')
    for name in names:
        if sha(metadata / name) != lock['small_files_sha256'][name]:
            raise ValueError(f'Unexpected Qwen metadata hash: {name}')
    processor = OfflineProcessor(metadata, source)
    old_rows = 0
    for path in sorted(historical.glob('[PN]-G-seed*.jsonl')):
        for row in read_rows(path):
            grid, ids = processor.inputs(row['question'], row['width'], row['height'])
            if (grid != row['processor_grid_thw'] or len(ids) != row['prompt_tokens']
                    or digest(ids) != row['prompt_ids_sha256']
                    or processor.target(row['gt_target']) != row['gt_target_ids']):
                raise ValueError(f"Historical token fixture differs: {path.name}/{row['item_id']}")
            old_rows += 1
    if old_rows != 2400:
        raise ValueError(f'Expected 2400 archived GT-only fixtures, observed {old_rows}')

    train = run / 'private' / 'training-plan'
    records, lengths = [], defaultdict(lambda: defaultdict(list))
    rows = {arm: read_rows(train / f'{arm}.jsonl') for arm in ('P-G', 'N-G')}
    if any(len(arm) != 600 for arm in rows.values()):
        raise ValueError('Unexpected training pool length')
    for pair in zip(*rows.values()):
        pair_records = []
        for arm, row in zip(rows, pair):
            grid, ids = processor.inputs(row['question'], row['width'], row['height'])
            target = processor.target(row['target'])
            total = len(ids) + len(target)
            if total > 2048 or len(target) > 128:
                raise ValueError(f"Training sequence exceeds cap: {arm}/{row['item_id']}")
            key = f"{arm}/{row['axis']}"
            for metric, n in (('prompt', len(ids)), ('target_including_eos', len(target)), ('total', total)):
                lengths[key][metric].append(n)
            pair_records.append({'item_id': row['item_id'], 'image_grid_thw': grid,
                                 'prompt_ids_sha256': digest(ids), 'target_ids': target,
                                 'prompt_tokens': len(ids), 'target_tokens': len(target),
                                 'visual_tokens': math.prod(grid) // processor.merge**2})
        left, right = pair_records
        if left['item_id'] != right['item_id'] or left['image_grid_thw'] != right['image_grid_thw']:
            raise ValueError('Paired training image grids or sample IDs differ')
        if pair[0]['axis'] != 'grounding' and left != right:
            raise ValueError('Paired non-grounding input/target tokens differ')
        records.extend({**row, 'arm': arm} for arm, row in zip(rows, pair_records))

    candidates = read_rows(run / 'private/secondary-provisional.jsonl')
    for entry in read_rows(run / 'private/grounding-queue.jsonl'):
        row = entry['source']
        for contract in ('native_1000', 'original_pixels'):
            candidates.append({**row, 'item_id': f"{row['item_id']}:{contract}",
                               'question': prompt(row, contract), 'coordinate_system': contract})
    eval_lengths = defaultdict(list)
    eval_tokens = []
    for row in candidates:
        image_path = ROOT / row['image_path']
        with Image.open(image_path) as image:
            width, height = image.size
        grid, ids = processor.inputs(row['question'], width, height)
        if len(ids) > 2048:
            raise ValueError(f"Evaluation prompt exceeds cap: {row['item_id']}")
        eval_lengths[f"{row['axis']}/{row.get('coordinate_system', 'default')}"].append(len(ids))
        eval_tokens.append({'item_id': row['item_id'], 'image_path': row['image_path'],
                            'image_sha256': sha(image_path), 'image_grid_thw': grid,
                            'prompt_sha256': hashlib.sha256(row['question'].encode('utf-8')).hexdigest(),
                            'prompt_tokens': len(ids), 'prompt_ids_sha256': digest(ids)})
    token_path = run / 'private' / 'training-token-verification.jsonl'
    eval_token_path = run / 'private' / 'evaluation-token-verification.jsonl'
    write_rows(token_path, records)
    write_rows(eval_token_path, eval_tokens)
    write_json(run / 'tokenization-verification.json', {
        'scope': 'CPU tokenizer IDs and image grids. No tensor preprocessing or model forward.',
        'model_revision': lock['model_revision'], 'tokenizers_version': tokenizer_version,
        'resize_source_sha256': sha(source), 'historical_gt_rows_exact': old_rows,
        'training_rows': len(records), 'training_token_rows_sha256': sha(token_path),
        'training': {key: {metric: describe(values) for metric, values in metrics.items()}
                     for key, metrics in sorted(lengths.items())},
        'provisional_evaluation_prompts': len(candidates),
        'evaluation_token_rows_sha256': sha(eval_token_path),
        'evaluation_prompt_lengths': {key: describe(values) for key, values in sorted(eval_lengths.items())},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'historical', 'metadata', 'resize_source', 'historical_lock'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    verify(args.run, args.historical, args.metadata, args.resize_source, args.historical_lock)
