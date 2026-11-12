"""Calculate the registered eight-seed 9B confirmation contrasts."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from experiment_common import read_json, read_rows, sha, units, write_json
from prepare_9b_confirmation_training import SEEDS


def sign_flip(numerators):
    numerators = [int(value) for value in numerators]
    observed = abs(sum(numerators))
    return sum(abs(sum(s * d for s, d in zip(signs, numerators))) >= observed
               for signs in product((-1, 1), repeat=len(numerators))) / 2 ** len(numerators)


def holm(values):
    adjusted, previous = {}, 0
    for index, (key, p) in enumerate(sorted(values.items(), key=lambda pair: pair[1])):
        previous = min(1.0, max(previous, (len(values) - index) * p))
        adjusted[key] = previous
    return adjusted


def paired_intervals(deltas, strata, draws=20000):
    rng = np.random.default_rng(2026111201)
    separate = [deltas[:, strata == s] for s in sorted(set(strata))]
    values = {name: [] for name in ('seed_only', 'images_only', 'joint')}
    seeds, images = deltas.shape
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        resampled_seeds = rng.integers(seeds, size=(count, seeds))
        image_totals = np.zeros((count, seeds))
        for group in separate:
            picks = rng.integers(group.shape[1], size=(count, group.shape[1]))
            image_totals += group[:, picks].sum(axis=2).T
        values['images_only'].extend(image_totals.mean(axis=1) / images)
        values['joint'].extend(np.take_along_axis(image_totals, resampled_seeds, axis=1).mean(axis=1) / images)
        values['seed_only'].extend(deltas.mean(axis=1)[resampled_seeds].mean(axis=1))
    return {key: (100 * np.quantile(samples, [.025, .975])).tolist() for key, samples in values.items()}


def report(manifest_path, scores, output):
    freeze = read_json(manifest_path.parent / 'review-status.json')
    if freeze['status'] != 'reviewed_manifest_frozen' or freeze['evaluation_manifest_sha256'] != sha(manifest_path):
        raise ValueError('Expected a reviewed and frozen evaluation manifest')
    manifest = read_rows(manifest_path)
    jobs = ['BASE'] + [f'{arm}-seed{seed}' for seed in SEEDS for arm in ('P-G', 'N-G')]
    groups, endpoint_scores, hashes = {}, {}, {}
    for name in jobs:
        path = scores / name / 'scored.jsonl'
        ledger = read_rows(path)
        by_id = {row['item_id']: row for row in ledger}
        if len(by_id) != len(ledger) or by_id.keys() != {row['item_id'] for row in manifest}:
            raise ValueError(f'Incomplete or duplicate endpoint: {name}')
        if any(row['correct_bit'] not in (0, 1) for row in ledger):
            raise ValueError(f'Non-binary correctness: {name}')
        hashes[name] = sha(path)
        groups[name] = {}
        for contract in ('native_1000', 'original_pixels'):
            selected = [row for row in manifest if row.get('coordinate_system', 'native_1000') == contract]
            for row in units(selected, [by_id[item['item_id']] for item in selected]):
                groups[name].setdefault((contract, row['axis']), []).append(row)
        endpoint_scores[name] = {f'{contract}/{axis}': float(np.mean([row['correct_bit'] for row in rows]))
                                 for (contract, axis), rows in groups[name].items()}
    comparisons, secondary, grounding = {}, {}, {}
    for key, base in groups['BASE'].items():
        contract, axis = key
        label = f'{contract}/{axis}'
        data = np.array([[row['correct_bit'] for row in groups[job][key]] for job in jobs[1:]], dtype=int)
        deltas = data[1::2] - data[::2]
        row = {'per_seed_pp': (100 * deltas.mean(axis=1)).tolist(), 'mean_pp': float(100 * deltas.mean()),
               'p_two_sided': sign_flip(deltas.sum(axis=1)), 'base_accuracy': endpoint_scores['BASE'][label]}
        for arm, offset in (('P-G', 0), ('N-G', 1)):
            row[f'{arm}_minus_base_per_seed_pp'] = (100 * (data[offset::2].mean(axis=1) - row['base_accuracy'])).tolist()
        if axis == 'grounding':
            if len({example['image_group'] for example in base}) != len(base) or len(base) != 240:
                raise ValueError('Grounding bootstrap requires 240 distinct registered components')
            row['intervals_95_pp'] = paired_intervals(deltas, np.array([example['stratum'] for example in base]))
            grounding[contract] = (deltas, [example['unit'] for example in base],
                                   np.array([example['stratum'] for example in base]))
        else:
            secondary[label] = row['p_two_sided']
        comparisons[label] = row
    for label, p in holm(secondary).items():
        comparisons[label]['p_holm'] = p
    native, native_ids, strata = grounding['native_1000']
    pixel, pixel_ids, _ = grounding['original_pixels']
    if native_ids != pixel_ids:
        raise ValueError('Cross-prompt grounding units are misaligned')
    interaction = native - pixel
    primary = comparisons['native_1000/grounding']
    write_json(output, {'manifest_sha256': sha(manifest_path), 'ledger_sha256': hashes, 'seeds': list(SEEDS),
                        'endpoint_accuracies': endpoint_scores, 'contrasts': comparisons,
                        'grounding_prompt_interaction': {
                            'per_seed_pp': (100 * interaction.mean(axis=1)).tolist(),
                            'mean_pp': float(100 * interaction.mean()),
                            'intervals_95_pp': paired_intervals(interaction, strata)},
                        'primary_direction_supported': primary['p_two_sided'] < .05 and primary['mean_pp'] >= 5,
                        'primary_interval_lower_at_least_5pp': primary['intervals_95_pp']['joint'][0] >= 5})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'scores', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report(args.manifest, args.scores, args.output)
