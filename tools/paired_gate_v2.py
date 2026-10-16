"""Shared-outer-image bootstrap for scale and nested-noise endpoints.

Historical mode validates the new estimator; it never authorizes formal training.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from analyze_rulers import finite_json
from experiment_common import AXES, LADDERS, ROOT, read_json, read_rows, sha, units, write_json
from reanalyze_v2 import verified_ledger
from rulers_v2 import metrics, scan


class JointImageBootstrap:
    """Stratify by image's joint stratum-membership pattern across all axes.

    Every model/axis uses the same image count. Profiles preserve all stratum
    cluster counts, including clusters shared by multiple axes or subsets.
    Correctness/parse-pattern compression is an exact multinomial aggregation.
    """
    def __init__(self, records):
        self.records = records
        self.nrecords = len(records)
        ids = [(r['axis'], r['stratum'], r['unit'], r['image_group']) for r in records[0]]
        if any([(r['axis'], r['stratum'], r['unit'], r['image_group']) for r in rows] != ids for rows in records):
            raise ValueError('Models and baseline must share exact scoring-unit identities')
        self.strata = sorted({(r['axis'], r['stratum']) for r in records[0]})
        stratum_index = {key: i for i, key in enumerate(self.strata)}
        groups = sorted({r['image_group'] for r in records[0]})
        group_index = {key: i for i, key in enumerate(groups)}
        width = 2 + 2*self.nrecords
        features = np.zeros((len(groups), len(self.strata), width))
        for i, row in enumerate(records[0]):
            bits = np.array([rows[i]['correct_bit'] for rows in records])
            valid = not any(rows[i]['parse_fail_flag'] for rows in records[1:])
            feature = features[group_index[row['image_group']], stratum_index[(row['axis'], row['stratum'])]]
            feature[0] += 1
            feature[1] += valid
            feature[2:2+self.nrecords] += bits
            feature[2+self.nrecords:] += bits*valid
        profiles = defaultdict(list)
        for i, row in enumerate(features):
            profiles[tuple(row[:, 0] > 0)].append(i)
        self.blocks = []
        flat = features.reshape(len(groups), -1)
        for profile in sorted(profiles):
            values, counts = np.unique(flat[profiles[profile]], axis=0, return_counts=True)
            self.blocks.append((values, counts))
        self.nclusters = len(groups)

    def totals(self, counts=None):
        counts = [count for _, count in self.blocks] if counts is None else counts
        return sum(count@value for (value, _), count in zip(self.blocks, counts))

    def draw_counts(self, rng, parent=None, size=None):
        parent = [counts for _, counts in self.blocks] if parent is None else parent
        return [rng.multinomial(int(original.sum()), counts/counts.sum(), size=size)
                for (_, original), counts in zip(self.blocks, parent)]

    def scores(self, totals):
        values = np.atleast_2d(totals).reshape(-1, len(self.strata), 2+2*self.nrecords)
        output, floor = {}, {}
        for mode, denominator, start in [('wrong', 0, 2), ('exclude', 1, 2+self.nrecords)]:
            with np.errstate(divide='ignore', invalid='ignore'):
                means = values[:, :, start:start+self.nrecords]/values[:, :, denominator:denominator+1]
            by_axis = np.stack([means[:, [i for i, s in enumerate(self.strata) if s[0] == axis]].mean(axis=1)
                                for axis in AXES], axis=-1)
            floor[mode] = by_axis[:, 0]
            output[mode] = by_axis[:, 1:]
        return output, floor


def interval(values, alpha=.05):
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    return np.quantile(finite, [alpha/2, 1-alpha/2]).tolist() if len(finite) else [None, None]


def endpoints(engine, totals, inner_counts, protocol, axis_indices):
    scores, floors = engine.scores(totals)
    scale = scan({m: v[0] for m, v in scores.items()}, {m: v[0] for m, v in floors.items()},
                 protocol, axis_indices=axis_indices)
    inner_totals = engine.totals(inner_counts)
    inner_scores, _ = engine.scores(inner_totals)
    # Raw absolute equal-edge reference is the endpoint gap / edge count.
    reference = (inner_scores['wrong'][:, -1] - inner_scores['wrong'][:, 0])/(engine.nrecords-2)
    noise = metrics([{'vector': row} for row in reference], axis_indices, protocol['tie_epsilon'])
    eligible = scale['usable_classes']/scale['structural_classes'] >= .8
    return {'F_scale': scale['F'], 'F_noise': noise['F'],
            'F_difference': scale['F'] - noise['F'] if eligible and scale['F'] is not None and noise['F'] is not None else None,
            'strict_scale': scale['strict_pair_flip_rate'], 'strict_noise': noise['strict_pair_flip_rate'],
            'strict_difference': scale['strict_pair_flip_rate'] - noise['strict_pair_flip_rate']
                if eligible and scale['strict_pair_flip_rate'] is not None and noise['strict_pair_flip_rate'] is not None else None,
            'valid_structural_fraction': scale['usable_classes']/scale['structural_classes'],
            'classes': scale['structural_classes'], 'usable_classes': scale['usable_classes'],
            'full_ties': scale['full_ties'], 'partial_ties': scale['partial_ties']}


def analyze_family(records, protocol, outer=1000, inner=128, seed=2026101607, axes=None, progress=None):
    engine = JointImageBootstrap(records)
    axes = list(range(4)) if axes is None else axes
    rng = np.random.default_rng(seed)
    point = endpoints(engine, engine.totals(), engine.draw_counts(rng, size=inner), protocol, axes)
    draws = []
    for b in range(outer):
        counts = engine.draw_counts(rng)
        draws.append(endpoints(engine, engine.totals(counts), engine.draw_counts(rng, counts, size=inner), protocol, axes))
        if progress and (b+1) % 100 == 0:
            progress(b+1)
    summary = {'point': point, 'n_outer': outer, 'n_inner': inner, 'seed': seed,
               'image_clusters': engine.nclusters, 'joint_membership_profiles': len(engine.blocks),
               'axis_indices': axes, 'paired': True,
               'CI_95': {key: interval([d[key] for d in draws]) for key in
                         ('F_scale', 'F_noise', 'F_difference', 'strict_scale', 'strict_noise', 'strict_difference')},
               'valid_difference_fraction': sum(d['F_difference'] is not None for d in draws)/outer,
               'monte_carlo_stability': {
                   'first_half_difference_CI': interval([d['F_difference'] for d in draws[:outer//2]]),
                   'second_half_difference_CI': interval([d['F_difference'] for d in draws[outer//2:]])}}
    interval_f = summary['CI_95']['F_difference']
    interval_s = summary['CI_95']['strict_difference']
    candidate = ('go' if summary['valid_difference_fraction'] >= .95 and interval_f[0] is not None
                 and interval_f[0] > .05 and interval_s[0] is not None and interval_s[0] > 0
                 and point['strict_scale'] >= .05 else 'indeterminate')
    summary['statistical_candidate_decision'] = candidate
    return finite_json(summary), finite_json(draws)


def main(source, output, n_outer, n_inner):
    output.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'worker_script': Path(__file__).name, 'pid': os.getpid(),
             'source_scope': 'historical_exploratory', 'formal_training_allowed': False,
             'n_outer': n_outer, 'n_inner': n_inner, 'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'status.json', state)
    manifest = read_rows(ROOT / 'data/manifest.jsonl')
    protocol = read_json(ROOT / 'configs/experiment.json')
    design = {'schema': 'arp.paired-scale-noise.development.v2', 'n_outer': n_outer, 'n_inner': n_inner,
              'outer_sampling': 'Image clusters stratified by joint cross-axis stratum membership profiles',
              'shared_weights': 'All models and baseline; wrong and common-valid exclude; all axes',
              'nested_sampling': 'Inner image-cluster resampling from the SAME outer empirical counts',
              'noise_reference': 'raw absolute, parse_fail=wrong, equal adjacent-edge mean, fixed tie epsilon',
              'minimum_valid_structural_fraction': .8, 'minimum_defined_bootstrap_fraction': .95,
              'minimum_excess_F': .05, 'minimum_point_strict_pair_flip_rate': .05,
              'go_requires': 'paired F difference lower CI > .05 AND paired strict difference lower CI > 0',
              'source_manifest_sha256': sha(ROOT / 'data/manifest.jsonl'),
              'preregistration_scope': 'Estimator engineering protocol; historical data are already observed, not a confirmatory preregistration',
              'training_allowed': False}
    write_json(output / 'design.json', design)
    baseline = units(manifest, read_rows(ROOT / 'data/baselines/per_item.jsonl'))
    results = {}
    try:
        for family, keys in LADDERS.items():
            records = [baseline] + [verified_ledger(source / 'ledger', key, manifest) for key in keys]
            for suffix, axes in [('all_axes', [0,1,2,3]), ('without_grounding', [0,2,3])]:
                name = family + '/' + suffix

                def progress(done):
                    state.update(current_analysis=name, completed_outer_draws=done,
                                 updated_utc=datetime.now(timezone.utc).isoformat())
                    write_json(output / 'status.json', state)
                    print(name, done, '/', n_outer, flush=True)

                summary, draws = analyze_family(records, protocol, n_outer, n_inner,
                                                seed=2026101607, axes=axes, progress=progress)
                summary['measurement'] = {axis: {
                    'units': sum(r['axis'] == axis for r in baseline),
                    'image_clusters': len({r['image_group'] for r in baseline if r['axis'] == axis}),
                    'model_parse_rates': {key: 1-sum(r['parse_fail_flag'] for r in rows if r['axis'] == axis)/sum(r['axis'] == axis for r in rows)
                                          for key, rows in zip(keys, records[1:])}}
                    for axis in AXES}
                summary.update(formal=False, training_allowed=False,
                               reason='Historical data and unreviewed measurement validity; statistical candidate is not training permission')
                results[name] = summary
                write_json(output / (family + '-' + suffix + '.json'), {'summary': summary, 'paired_draws': draws})
                write_json(output / 'summary.json', results)
        write_json(output / 'gate.json', {'decision': 'indeterminate', 'formal': False,
                   'training_allowed': False, 'families': {key: value['statistical_candidate_decision'] for key, value in results.items()},
                   'reason': 'Paired estimator validated on historical data only; new audit and measurement acceptance remain required'})
        state.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / 'complete.json', {'status': 'complete', 'analyses': len(results),
                   'n_outer': n_outer, 'n_inner': n_inner, 'summary_sha256': sha(output / 'summary.json'), 'formal_training_allowed': False})
    except Exception as error:
        state.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        write_json(output / 'status.json', state)
    if state['status'] == 'complete' and os.environ.get('GH_TOKEN'):
        from publish_artifacts import publish
        publish(output, 'arp-protocol-v2-20261016', token=os.environ['GH_TOKEN'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'runs/formal-20261014')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/protocol-v2/paired-gate-development')
    parser.add_argument('--n-outer', type=int, default=1000)
    parser.add_argument('--n-inner', type=int, default=128)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.n_outer < 2 or args.n_inner < 2:
        parser.error('At least two outer and inner draws are required')
    if args.detach:
        token = sys.stdin.readline().strip()
        if not token:
            raise SystemExit('Completion-backup credential required on stdin')
        sessions = subprocess.run(['screen', '-ls'], capture_output=True, text=True).stdout
        if '.arp-v2-paired-gate' in sessions:
            raise SystemExit('Paired-gate worker already exists')
        args.output.mkdir(parents=True, exist_ok=True)
        command = ('source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; '
                   'exec /root/autodl-tmp/arp-venv/bin/python -u tools/paired_gate_v2.py >> ' +
                   str(args.output / 'console.log') + ' 2>&1')
        if args.source != ROOT / 'runs/formal-20261014' or args.output != ROOT / 'runs/protocol-v2/paired-gate-development' or args.n_outer != 1000 or args.n_inner != 128:
            raise SystemExit('Detached run must use the full frozen default protocol')
        subprocess.run(['screen', '-dmS', 'arp-v2-paired-gate', 'bash', '-c', command],
                       env={**os.environ, 'GH_TOKEN': token, 'OPENBLAS_NUM_THREADS': '1',
                            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, check=True)
        print('Started paired scale/noise development analysis: 1000 outer x 128 inner draws', flush=True)
    else:
        main(args.source, args.output, args.n_outer, args.n_inner)
