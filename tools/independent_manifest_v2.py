"""Global exact-image components and outcome-independent role selection."""
from collections import defaultdict
import hashlib

from prepare_phase_a import TEMPLATES

SEED = '20261016:independent-manifests-v2:'


def rank(value):
    return hashlib.sha256((SEED + str(value)).encode()).hexdigest()


class ImageComponents:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def assign_roles(candidates):
    components = ImageComponents()
    for row in candidates:
        pixels = row['image_pixel_sha256']
        if not pixels:
            raise ValueError('Cannot assign an image-free candidate')
        for pixel in pixels:
            components.union(pixels[0], pixel)
    result = []
    for row in candidates:
        component = components.find(row['image_pixel_sha256'][0])
        bucket = int(rank('component:' + component)[:12], 16) % 10000
        role = 'calibration' if bucket < 800 else 'audit' if bucket < 5400 else 'confirm'
        result.append({**row, 'role': role, 'image_group': 'rgb-component:' + component})
    return result


def select_axis(rows, target):
    strata = defaultdict(list)
    for row in rows:
        strata[(row['dataset'], row.get('subset', row.get('subject', '')))].append(row)
    selected = []
    for index, stratum in enumerate(sorted(strata)):
        count = target // len(strata) + int(index < target % len(strata))
        pool = strata[stratum]
        if len(pool) < count:
            raise ValueError('Insufficient original units in stratum ' + repr(stratum) + ': ' + str(len(pool)) + ' < ' + str(count))
        if pool[0]['axis'] == 'pope':
            # Targets are divisible by six: equal strata and yes/no labels.
            if count % 2:
                raise ValueError('POPE-style stratum must permit an exactly balanced label quota')
            for answer in ('yes', 'no'):
                eligible = [row for row in pool if row['answer_key'] == answer]
                if len(eligible) < count//2:
                    raise ValueError('Insufficient balanced POPE-style labels')
                selected.extend(sorted(eligible, key=lambda row: rank(row['item_id']))[:count//2])
        else:
            selected.extend(sorted(pool, key=lambda row: rank(row['item_id']))[:count])
    if len(selected) != target:
        raise ValueError('Unit selection did not meet exact target')
    return selected


def expand_item(row):
    result = {**row, 'bootstrap_unit': row['item_id'], 'base_item_id': row['item_id'],
              'subset': row.get('subset', row.get('subject', '')),
              'independent_confirmation': False, 'candidate_for_confirmation': row['role'] == 'confirm'}
    # One-image compatibility is explicit; multi-image items never expose a
    # singular alias that would allow an old backend to silently drop images.
    if len(row['image_paths']) == 1:
        result['image_path'] = row['image_paths'][0]
    if row['axis'] == 'grounding' or row['axis'] == 'pope':
        return [result]
    if row['options']:
        options = '\n'.join('(' + letter + ') ' + value for letter, value in row['options'].items())
        suffix = '\nChoices:\n' + options + '\nAnswer with exactly one option letter and no explanation.'
    else:
        suffix = '\nGive only the final answer, as briefly as possible, without explanation.'
    if row['axis'] == 'consistency':
        return [{**result, 'item_id': row['item_id'] + ':p' + str(index), 'paraphrase_id': index,
                 'question': template.format(question=row['question']) + suffix}
                for index, template in enumerate(TEMPLATES, 1)]
    return [{**result, 'question': row['question'] + suffix}]
