"""All16 local9B evals: paired seed/image bootstrap, fixed12-comparison family."""
from datetime import datetime, timezone
import json

import numpy as np

from experiment_common import ROOT, AXES, read_json, read_rows, sha, write_json
from route_b_9b_loop import OUT, SEEDS
from report_route_b_multiplicity import intervals
from verify_9b_student_outputs import verify_formal, verify_evaluation


def training_report():
    results, curves = [], {}
    for seed in SEEDS:
        for arm in ('A1', 'A2', 'U'):
            folder = OUT / 'formal' / f'{arm}-seed{seed}'
            marker = read_json(folder / 'complete.json')
            ledger = read_rows(folder / 'loss.jsonl')
            assert len(ledger) == marker['actual_updates'] == 1000
            assert sha(folder / 'loss.jsonl') == marker['loss_sha256']
            seconds = sum(r['seconds'] for r in ledger)
            results.append({'arm': arm, 'seed': seed, 'updates': 1000, 'seconds_per_step': seconds / 1000,
                            'step_seconds': seconds, 'reserved_peak_bytes': max(r['reserved_peak_bytes'] for r in ledger),
                            'first_loss': ledger[0]['loss'], 'last_loss': ledger[-1]['loss'],
                            'loss_sha256': marker['loss_sha256'], 'adapter_sha256': marker['adapter_sha256']})
            curves[(arm, seed)] = [r['loss'] for r in ledger]
    folder = OUT / 'formal-report'
    folder.mkdir(exist_ok=True)
    data = {'status': '15_trainings_complete', 'updates': 15000, 'runs': results,
            'step_seconds_total': sum(r['step_seconds'] for r in results),
            'scope': 'Step processing time excludes loading/checkpoint/backup; fixed last-step endpoints'}
    write_json(folder / 'training-results.json', data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, arm in zip(axes, ('A1', 'A2', 'U')):
        for seed in SEEDS:
            values = np.convolve(curves[(arm, seed)], np.ones(20) / 20, mode='valid')
            axis.plot(np.arange(20, 1001), np.maximum(values, 1e-8), label=str(seed), linewidth=1)
        axis.set(title=arm, xlabel='Actual optimizer update', ylabel='Dual CE (20-step mean)', yscale='log')
        axis.legend(title='Seed', fontsize=7)
        axis.grid(alpha=.2)
    figure.savefig(folder / 'loss-curves.png', dpi=180)
    plt.close(figure)
    lines = ['# 独立9B+122B：15次训练完成', '', '| 臂 | seed | 秒/步 | 训练步骤分钟 | reserved GB | 最后损失 |', '|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["arm"]} | {r["seed"]} | {r["seconds_per_step"]:.3f} | {r["step_seconds"]/60:.2f} | {r["reserved_peak_bytes"]/1e9:.3f} | {r["last_loss"]:.6g} |')
    lines += ['', '![损失曲线](loss-curves.png)', '', '耗时不含装载、保存、上传和等待。训练完成不是完整评测完成；全部固定1,000步，未按分数追加预算。']
    (folder / 'training-report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return data


def analyze(replicates=10000):
    _, complete = verify_formal(return_evaluations=True)
    models = [('N', 42)] + [(a, s) for s in SEEDS for a in ('A1', 'A2', 'U')]
    reference = complete[('N', 42)][1]
    assert all(set(records) == set(reference) for _, records in complete.values())
    rng = np.random.default_rng(2026101703)
    seed_indices = rng.integers(0, len(SEEDS), size=(replicates, len(SEEDS)))
    output = {'status': 'all16_verified', 'timestamp': datetime.now(timezone.utc).isoformat(),
              'scope': 'Independent9B+122B extension; development benchmark, not ARP confirmation',
              'bootstrap_replicates': replicates, 'bootstrap_seed': 2026101703,
              'matched_seeds': SEEDS, 'family_size': 12, 'correction': 'Bonferroni nominal95% percentile bootstrap intervals',
              'caveat': 'Approximate bootstrap coverage; only5 training seeds; no capacity-only causal attribution versus7B',
              'scores': {f'{a}-seed{s}': complete[(a, s)][0]['axis_scores'] for a, s in models},
              'ledger_sha256': {f'{a}-seed{s}': complete[(a, s)][0]['ledger_sha256'] for a, s in models},
              'paired_contrasts': {name: {} for name in ('A1-A2', 'A1-U', 'A1-N')}}
    for axis in AXES:
        keys = sorted(k for k in reference if k[0] == axis)
        groups = sorted({reference[k]['image_group'] for k in keys})
        index = {g: i for i, g in enumerate(groups)}
        totals = np.zeros((len(groups), len(models)))
        counts = np.zeros(len(groups))
        for key in keys:
            group = index[reference[key]['image_group']]
            counts[group] += 1
            for j, model in enumerate(models):
                assert complete[model][1][key]['image_group'] == reference[key]['image_group']
                totals[group, j] += complete[model][1][key]['correct_bit']
        points = totals.sum(0) / counts.sum()
        draws = np.empty((replicates, len(models)))
        for start in range(0, replicates, 40):
            size = min(40, replicates - start)
            weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=size)
            draws[start:start + size] = (weights @ totals) / (weights @ counts)[:, None]
        arm_points, arm_draws = {'N': points[0]}, {'N': draws[:, 0]}
        for i, arm in enumerate(('A1', 'A2', 'U')):
            columns = 1 + 3 * seed_indices + i
            arm_points[arm] = points[1 + i::3].mean()
            arm_draws[arm] = draws[np.arange(replicates)[:, None], columns].mean(1)
        for name in output['paired_contrasts']:
            left, right = name.split('-')
            output['paired_contrasts'][name][axis] = {'difference': float(arm_points[left] - arm_points[right]),
                      **intervals(arm_draws[left] - arm_draws[right], family_size=12),
                      'original_units': len(keys), 'image_clusters': len(groups)}
    folder = OUT / 'formal-report'
    folder.mkdir(exist_ok=True)
    write_json(folder / 'evaluation-results.json', output)
    lines = ['# 独立9B+122B：完整评测与配对区间', '',
             '全部15次固定1,000更新训练及16模型×16,900提示已独立验收。恢复状态已在本机核验后保存到GitHub Release；完成任务的本机优化器副本可归档腾挪。', '',
             '| 模型 | pope | grounding | consistency | mmstar |', '|---|---:|---:|---:|---:|']
    for name, scores in output['scores'].items():
        lines.append('| ' + name + ' | ' + ' | '.join(f'{scores[a]:.5f}' for a in AXES) + ' |')
    lines += ['', '| 对比 | 轴 | 差值 | 逐项95%区间 | 12项Bonferroni校正区间 |', '|---|---|---:|---|---|']
    for name, axes in output['paired_contrasts'].items():
        for axis, result in axes.items():
            low, high = result['pointwise_ci95']
            adjusted_low, adjusted_high = result['bonferroni_familywise_ci95']
            lines.append(f'| {name} | {axis} | {result["difference"]:.5f} | [{low:.5f}, {high:.5f}] | [{adjusted_low:.5f}, {adjusted_high:.5f}] |')
    lines += ['', '按训练seed和来源图像簇配对重采样；比较族为3组×4轴。名义近似95%整体区间不消除有限seed和尾部Monte Carlo不确定性。',
              '', '本实验不放行原ARP阶段C。与7B自蒸馏相比同时改变学生、教师、预处理和可用池，不将差异单独归因于容量。']
    (folder / 'evaluation-report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    print(json.dumps({'status': analyze()['status']}), flush=True)
