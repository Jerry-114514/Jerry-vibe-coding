"""Add user-requested multiplicity-adjusted paired intervals without retraining.

Reporting amendment made after partial development results existed; exploratory.
Bonferroni covers the fixed family of three contrasts x four axes (12 intervals).
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from experiment_common import ROOT, AXES, read_json, sha, write_json
from route_b_staged import OUT, SEEDS
from route_b_completion_reports import read_complete_evaluations

REPORT = OUT / 'completion-reports/multiplicity'
CONTRASTS = [('A1', 'A2'), ('A1', 'U'), ('A1', 'N')]
REPLICATES = 10000


def register():
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / 'reporting-amendment.json'
    if path.exists():
        result = read_json(path)
        assert result['source_sha256'] == sha(Path(__file__))
        return result
    result = {'timestamp': datetime.now(timezone.utc).isoformat(),
              'source_sha256': sha(Path(__file__)), 'family_size': 12,
              'contrasts': ['A1-A2', 'A1-U', 'A1-N'], 'axes': AXES,
              'bootstrap_replicates': REPLICATES, 'random_seed': 2026101702,
              'method': 'Percentile paired seed x source-image cluster bootstrap; Bonferroni alpha=.05/12',
              'registration_timing': 'User-requested reporting amendment after partial formal development results; not a confirmatory preregistration',
              'does_not_change_training_or_evaluation': True,
              'coverage_caveat': 'Nominal approximate bootstrap intervals; five training seeds limit precision'}
    write_json(path, result)
    return result


def intervals(draws, family_size=12):
    alpha = .05 / family_size
    return {'pointwise_ci95': [float(v) for v in np.quantile(draws, [.025, .975])],
            'bonferroni_familywise_ci95': [float(v) for v in np.quantile(draws, [alpha / 2, 1 - alpha / 2])]}


def analyze():
    plan = register()
    complete = read_complete_evaluations()
    assert len(complete) == 16, 'Wait for all16 complete model ledgers'
    from verify_route_b_stage_outputs import verify
    verify('formal')
    keys = [('N', 42)] + [(arm, seed) for seed in SEEDS for arm in ('A1', 'A2', 'U')]
    reference = complete[('N', 42)][1]
    assert all(set(rows) == set(reference) for _, rows in complete.values())
    rng = np.random.default_rng(plan['random_seed'])
    seed_indices = rng.integers(0, len(SEEDS), size=(REPLICATES, len(SEEDS)))
    output = {'status': 'all16_evaluations_complete', 'scope': 'Independent routeB, exploratory development benchmark',
              'amendment_sha256': sha(REPORT / 'reporting-amendment.json'), 'family_size': 12,
              'bootstrap_replicates': REPLICATES, 'matched_seeds': SEEDS,
              'scores': {f'{arm}-seed{seed}': complete[(arm, seed)][0]['axis_scores'] for arm, seed in keys},
              'ledger_sha256': {f'{arm}-seed{seed}': complete[(arm, seed)][0]['ledger_sha256'] for arm, seed in keys},
              'paired_contrasts': {}, 'nominal_familywise_confidence': .95,
              'correction': 'Bonferroni over3 contrasts x4 axes; no additional macro tests in this family',
              'caveat': plan['coverage_caveat']}
    for axis in AXES:
        axis_keys = sorted(k for k in reference if k[0] == axis)
        groups = sorted({reference[k]['image_group'] for k in axis_keys})
        positions = {g: i for i, g in enumerate(groups)}
        sums = np.zeros((len(groups), len(keys)))
        counts = np.zeros(len(groups))
        for key in axis_keys:
            group = positions[reference[key]['image_group']]
            counts[group] += 1
            for j, model in enumerate(keys):
                unit = complete[model][1][key]
                assert unit['image_group'] == reference[key]['image_group']
                sums[group, j] += unit['correct_bit']
        point = sums.sum(0) / counts.sum()
        samples = np.empty((REPLICATES, len(keys)))
        for start in range(0, REPLICATES, 40):
            size = min(40, REPLICATES - start)
            weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=size)
            samples[start:start + size] = (weights @ sums) / (weights @ counts)[:, None]
        arm_points, arm_draws = {'N': point[0]}, {'N': samples[:, 0]}
        for i, arm in enumerate(('A1', 'A2', 'U')):
            columns = 1 + 3 * seed_indices + i
            arm_draws[arm] = samples[np.arange(REPLICATES)[:, None], columns].mean(1)
            arm_points[arm] = point[1 + i::3].mean()
        for left, right in CONTRASTS:
            name = left + '-' + right
            result = {'difference': float(arm_points[left] - arm_points[right]),
                      **intervals(arm_draws[left] - arm_draws[right]),
                      'image_clusters': len(groups), 'original_units': len(axis_keys)}
            output['paired_contrasts'].setdefault(name, {})[axis] = result
    write_json(REPORT / 'multiplicity-results.json', output)
    lines = ['# 独立7B：16模型四轴结果与多重比较校正', '',
             '全部16模型、每模型16,900提示已完成独立验收。区间配对重采样5个训练seed与来源图像簇；一致性五改写先合并为原题。', '',
             '| 模型 | pope | grounding | consistency | mmstar |', '|---|---:|---:|---:|---:|']
    for name, scores in output['scores'].items():
        lines.append('| ' + name + ' | ' + ' | '.join(f'{scores[a]:.5f}' for a in AXES) + ' |')
    lines += ['', '| 对比 | 轴 | 差值 | 逐项95%区间 | Bonferroni校正区间 |', '|---|---|---:|---|---|']
    for name, values in output['paired_contrasts'].items():
        for axis, result in values.items():
            lo, hi = result['pointwise_ci95']
            adj_lo, adj_hi = result['bonferroni_familywise_ci95']
            lines.append(f'| {name} | {axis} | {result["difference"]:.5f} | [{lo:.5f}, {hi:.5f}] | [{adj_lo:.5f}, {adj_hi:.5f}] |')
    lines += ['', '校正族固定为3组对比×4轴，共12项；每个校正区间的名义覆盖率为99.5833%，以Bonferroni方式控制该族名义95%覆盖。使用10,000次bootstrap，有限seed及尾部分位数有不确定性。',
              '', '这是部分开发结果出现后按用户要求增加的报告校正，不是事前确认性检验；不改变原始评测或训练预算，不作ARP阶段C放行证据。']
    (REPORT / 'multiplicity-report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output


def watch():
    import fcntl
    register()
    with (REPORT / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 40 * 3600
        while time.monotonic() < deadline:
            parent_cmd = Path('/proc/7177/cmdline')
            parent_alive = parent_cmd.exists() and b'run_route_b_storage_safe.py' in parent_cmd.read_bytes()
            ready = ((OUT / 'formal-independent-verification.json').exists()
                     and (OUT / 'completion-reports/final-report-delivered.json').exists()
                     and read_json(OUT / 'status.json').get('phase') == 'all_four_stages_complete')
            if ready and not parent_alive:
                analyze()
                from publish_artifacts import publish
                publish(REPORT, 'arp-protocol-v2-20261016', token=os.environ['GH_TOKEN'])
                write_json(REPORT / 'delivered.json', {'status': 'uploaded', 'timestamp': datetime.now(timezone.utc).isoformat()})
                return
            if (OUT / 'failure.json').exists():
                raise RuntimeError('7B pipeline failure requires inspection; no partial final report')
            time.sleep(30)
        raise RuntimeError('Reporting deadline reached; incomplete models must not be omitted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'analyze', 'watch'])
    args = parser.parse_args()
    if args.action == 'register':
        print(json.dumps(register()), flush=True)
    elif args.action == 'watch':
        watch()
    else:
        print(json.dumps({'status': analyze()['status']}), flush=True)
