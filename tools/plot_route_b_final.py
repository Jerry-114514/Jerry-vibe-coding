"""Final7B figures and interval-based exploratory conclusions; no new inference."""
from datetime import datetime, timezone
import json

from experiment_common import ROOT, AXES, read_json, sha, write_json
from route_b_staged import OUT, SEEDS

FOLDER = OUT / 'completion-reports/final-figures'
SOURCE = OUT / 'completion-reports/multiplicity/multiplicity-results.json'
CONTRASTS = ('A1-A2', 'A1-U', 'A1-N')


def validated_results():
    data = read_json(SOURCE)
    expected = {'N-seed42'} | {f'{arm}-seed{seed}' for arm in ('A1', 'A2', 'U') for seed in SEEDS}
    assert data['status'] == 'all16_evaluations_complete'
    assert set(data['scores']) == expected and data['matched_seeds'] == SEEDS
    assert data['family_size'] == 12 and data['bootstrap_replicates'] == 10000
    assert set(data['paired_contrasts']) == set(CONTRASTS)
    conclusions = []
    for label in CONTRASTS:
        assert set(data['paired_contrasts'][label]) == set(AXES)
        left, right = label.split('-')
        for axis in AXES:
            result = data['paired_contrasts'][label][axis]
            delta = sum(data['scores'][f'{left}-seed{s}'][axis] -
                        data['scores'][f'{right}-seed{42 if right == "N" else s}'][axis] for s in SEEDS) / len(SEEDS)
            assert abs(delta - result['difference']) < 1e-12
            lo, hi = result['bonferroni_familywise_ci95']
            assert -1 <= lo <= hi <= 1
            direction = 'left_higher' if lo > 0 else 'right_higher' if hi < 0 else 'interval_includes_zero'
            conclusions.append({'contrast': label, 'axis': axis, 'difference': delta,
                                'adjusted_interval': [lo, hi], 'direction': direction})
    return data, conclusions


def generate():
    data, conclusions = validated_results()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    FOLDER.mkdir(parents=True, exist_ok=True)
    figure, panels = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for panel, contrast in zip(panels, CONTRASTS):
        for y, axis in enumerate(AXES):
            result = data['paired_contrasts'][contrast][axis]
            lo, hi = (100 * x for x in result['bonferroni_familywise_ci95'])
            color = '#2563eb' if lo > 0 else '#dc2626' if hi < 0 else '#666666'
            panel.plot([lo, hi], [y, y], color=color, linewidth=2)
            panel.scatter([100 * result['difference']], [y], color=color, s=34, zorder=3)
        panel.axvline(0, color='black', linestyle='--', linewidth=.8)
        panel.set(yticks=range(4), yticklabels=AXES, title=contrast, xlabel='Difference (percentage points)')
        panel.invert_yaxis()
        panel.grid(axis='x', alpha=.2)
    figure.suptitle('7B independent route B: Bonferroni intervals, 12 comparisons\n5 paired seeds + source-image bootstrap; nominal approximate family coverage 95%')
    for suffix in ('png', 'pdf'):
        figure.savefig(FOLDER / ('paired-intervals.' + suffix), dpi=180)
    plt.close(figure)
    figure, panels = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = plt.get_cmap('tab10').colors
    for panel, axis in zip(panels.flat, AXES):
        baseline = data['scores']['N-seed42'][axis] * 100
        panel.scatter([0], [baseline], color='gray', s=40)
        panel.axhline(baseline, color='gray', linewidth=.8, linestyle=':')
        for position, arm in enumerate(('A1', 'A2', 'U'), 1):
            values = [100 * data['scores'][f'{arm}-seed{s}'][axis] for s in SEEDS]
            for i, value in enumerate(values):
                panel.scatter([position + (i - 2) * .045], [value], color=colors[i], s=28,
                              label=str(SEEDS[i]) if position == 1 else None)
            panel.scatter([position], [np.mean(values)], color='black', marker='D', s=28)
        panel.set(xticks=range(4), xticklabels=['N', 'A1', 'A2', 'U'], title=axis, ylabel='Score (%)')
        panel.grid(axis='y', alpha=.2)
    panels.flat[0].legend(title='Training seed', fontsize=8)
    figure.suptitle('All 16 models: colored points = trained seeds, diamond = arm mean\nN is one fixed untrained baseline, not five training replicates')
    for suffix in ('png', 'pdf'):
        figure.savefig(FOLDER / ('four-axis-scores.' + suffix), dpi=180)
    plt.close(figure)
    record = {'status': 'complete', 'timestamp': datetime.now(timezone.utc).isoformat(),
              'source_sha256': sha(SOURCE), 'figure_source_sha256': sha(ROOT / 'tools/plot_route_b_final.py'),
              'models': data['scores'], 'matched_seeds': SEEDS, 'comparisons': conclusions,
              'interpretation': 'Exploratory interval exclusion of zero; approximate family coverage, no invented p-values; not ARP confirmation'}
    write_json(FOLDER / 'figure-data.json', record)
    lines = ['# 独立7B：最终图表与校正区间结论', '',
             '16模型全部完成，每模型16,900提示；15次训练均固定1,000更新。全部五个seed配对，12项比较按Bonferroni校正。', '',
             '教师与学生均为Qwen2.5-VL-7B，双CE序列蒸馏。A1=mmstar推理代理，A2=consistency，U=四轴等比例，N=不更新。', '',
             '![四轴与全部seed](four-axis-scores.png)', '', '![配对差值及校正区间](paired-intervals.png)', '',
             '| 模型 | pope | grounding | consistency | mmstar |', '|---|---:|---:|---:|---:|']
    for name in ['N-seed42'] + [f'{arm}-seed{s}' for s in SEEDS for arm in ('A1', 'A2', 'U')]:
        lines.append('| ' + name + ' | ' + ' | '.join(f"{data['scores'][name][a]*100:.3f}%" for a in AXES) + ' |')
    lines += ['', '| 对比 | 轴 | 差值（百分点） | 校正区间（百分点） | 区间结论 |', '|---|---|---:|---|---|']
    labels = {'left_higher': '左臂更高，区间排除0', 'right_higher': '右臂更高，区间排除0',
              'interval_includes_zero': '区间包含0，差异证据不足'}
    for row in conclusions:
        lo, hi = row['adjusted_interval']
        lines.append(f'| {row["contrast"]} | {row["axis"]} | {row["difference"]*100:.3f} | [{lo*100:.3f}, {hi*100:.3f}] | {labels[row["direction"]]} |')
    lines += ['', '按组归纳：']
    for contrast in CONTRASTS:
        separated = [row for row in conclusions if row['contrast'] == contrast and row['direction'] != 'interval_includes_zero']
        if not separated:
            lines.append(f'- {contrast}：四轴校正区间均包含0，当前实验的差异证据不足。')
        else:
            left, right = contrast.split('-')
            detail = '；'.join(f"{row['axis']}（{left}{'高于' if row['direction'] == 'left_higher' else '低于'}{right}）" for row in separated)
            lines.append(f'- {contrast}：以下校正区间排除0：{detail}；其余轴证据不足。')
    lines += ['', '校正族固定为3组对比×4轴。区间由10,000次训练seed与来源图像簇配对bootstrap生成；一致性五改写先合并原题。只有5个训练seed，尾部分位数存在Monte Carlo误差，因此整体95%覆盖是名义近似。', '',
              '包含0不证明相等；排除0只作为本独立实验的探索性差异证据，不虚构p值。多重比较校正是在部分开发结果出现后按用户要求补充，不冒充事前确认性检验。原ARP阶段C仍未放行。', '',
              '黑色菱形为五seed均值，彩色点为各次训练；N为唯一未训练基线。训练损失、显存和耗时见同Release的训练汇总。PNG供快速查看，PDF可独立导出。']
    (FOLDER / 'final-summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    print(json.dumps({'status': generate()['status']}), flush=True)
