"""Prepare a reproducible, pre-annotation queue for the 9B confirmation study."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from audit_confirmation_candidate_exposure import audit
from experiment_common import sha, write_json, write_rows


def rank(value):
    return hashlib.sha256(f'9b-confirmation-20261112:{value}'.encode()).hexdigest()


def grounding_queue(by_stratum, size=60):
    """Match quota slots to distinct components across all five strata."""
    occupied = {}
    options = {}
    for stratum, rows in by_stratum.items():
        options[stratum] = {}
        for row in sorted(rows, key=lambda row: rank(row['item_id'])):
            options[stratum].setdefault(row['image_group'], row)

    def assign(slot, visited):
        for component in options[slot[0]]:
            if component in visited:
                continue
            visited.add(component)
            previous = occupied.get(component)
            if previous is None or assign(previous, visited):
                occupied[component] = slot
                return True
        return False

    for index in range(size):
        for stratum in sorted(options):
            if not assign((stratum, index), set()):
                raise ValueError(f'No distinct-component assignment for {stratum}, slot {index + 1}')
    selected = defaultdict(list)
    for component, (stratum, _) in occupied.items():
        selected[stratum].append(options[stratum][component])
    return {key: sorted(rows, key=lambda row: rank(row['item_id']))
            for key, rows in sorted(selected.items())}


def prepare(cache, output):
    decisions = []
    exposure = audit(cache, candidate_export=decisions)
    if exposure['missing_expected_compact_archives'] or len(exposure['available_compact_ledger_checks']) < 15:
        raise ValueError('Historical exposure audit inputs incomplete')
    grouped = defaultdict(list)
    for decision in decisions:
        if decision['exclusion'] is None:
            grouped[decision['axis']].append(decision['rows'])
    with gzip.open(cache / 'map.json.gz', 'rt', encoding='utf-8') as source:
        restore = {entry['path']: entry for entry in json.load(source)['files']}
    by_stratum = defaultdict(list)
    for (row,) in grouped['grounding']:
        by_stratum[f"{row['dataset']}/{row['subset']}"].append(row)
    queue = grounding_queue(by_stratum)
    public, private = [], []
    for stratum, rows in queue.items():
        for index, row in enumerate(rows):
            key = rank(row['item_id'])[:20]
            asset = restore[row['image_paths'][0]]
            public.append({'review_id': key, 'image': f'images/{key}.jpg',
                           'width': row['width'], 'height': row['height'],
                           'referring_expression': row['referring_expression']})
            private.append({'review_id': key, 'stratum': stratum, 'order': index + 1,
                            'asset': asset, 'source': row})
    if len({row['review_id'] for row in public}) != 300:
        raise ValueError('Expected 300 unique review IDs')
    assert len({row['source']['image_group'] for row in private}) == 300
    public.sort(key=lambda row: rank(row['review_id']))
    secondary = []
    for axis in ('pope', 'consistency', 'mmstar'):
        by_stratum = defaultdict(list)
        for group in grouped[axis]:
            row = group[0]
            by_stratum[f"{row['dataset']}/{row['subset']}"].append(group)
        if axis == 'mmstar':
            quota = exposure['feasibility'][axis]['unfrozen_capped_quota_alternative']['counts']
        else:
            quota = dict.fromkeys(by_stratum, 120 // len(by_stratum))
        assert sum(quota.values()) == 120
        for key, target in sorted(quota.items()):
            if len(by_stratum[key]) < target:
                raise ValueError(f'Insufficient examples in {axis}/{key}')
            groups = sorted(by_stratum[key], key=lambda group: rank(group[0]['bootstrap_unit']))
            secondary.extend(row for group in groups[:target] for row in group)
    outputs = {
        'private/exposure-decisions.jsonl': [
            {k: v for k, v in item.items() if k != 'rows'} for item in decisions],
        'private/grounding-queue.jsonl': private,
        'private/secondary-provisional.jsonl': secondary,
        'review/items.jsonl': public,
    }
    for name, rows in outputs.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_rows(path, rows)
    proof = {
        'status': 'provisional_pending_images_and_human_review',
        'candidate_counts': dict(Counter(row['axis'] for row in decisions)),
        'historical_ledgers_checked': len(exposure['available_compact_ledger_checks']),
        'exclusions': exposure['exclusion_counts'],
        'grounding_by_stratum': {key: {'main': 48, 'reserve': len(rows)-48} for key, rows in queue.items()},
        'unique_grounding_image_components': len(private),
        'secondary_prompt_counts': dict(Counter(row['axis'] for row in secondary)),
        'sha256': {name: sha(output / name) for name in outputs},
    }
    write_json(output / 'preparation.json', proof)
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.cache, args.output)
