"""Paired seed x image-cluster bootstrap for the independent completed16-model eval.

Pointwise descriptive95% intervals, not confirmatory ARP significance tests.
Identical bootstrap draws are used for all arms and contrasts within an axis.
"""
import json
import numpy as np
from experiment_common import ROOT, AXES, read_rows, read_json, units, write_json, sha
from route_b_staged import OUT, SEEDS
from verify_route_b_stage_outputs import verify


def analyze(replicates=2000):
    verify('formal')
    protocol = read_json(OUT/'protocol.json')
    manifest_path = ROOT/protocol['eval']['formal_manifest']
    manifest = read_rows(manifest_path)
    keys = [('N',42)]+[(arm,seed) for seed in SEEDS for arm in ('A1','A2','U')]
    scored = []
    hashes = {}
    for arm,seed in keys:
        folder = OUT/'formal'/f'{arm}-seed{seed}'
        marker = read_json(folder/'scores.json')
        ledger = folder/'eval.jsonl'
        assert marker['rows'] == 16900 and sha(ledger) == marker['ledger_sha256']
        rows = units(manifest, read_rows(ledger))
        scored.append({(r['axis'],r['dataset'],r['unit']):r for r in rows})
        hashes[f'{arm}-seed{seed}'] = sha(ledger)
    assert all(set(x) == set(scored[0]) for x in scored)
    rng = np.random.default_rng(2026101642)
    seed_draws = rng.integers(0,5,size=(replicates,5))
    output = {'scope':'Independent routeB exploratory development benchmark; not originalARP',
              'interval':'Pointwise95%; no claim of familywise significance',
              'bootstrap':{'replicates':replicates,'seed':2026101642,'resampling':'paired training seeds and source-image clusters within axis'},
              'ledger_sha256':hashes,'axes':{}}
    macro_draws = {name:np.zeros(replicates) for name in ('A1-N','A2-N','U-N','A1-U','A2-U')}
    macro_points = {name:0.0 for name in macro_draws}
    for axis in AXES:
        axis_keys = sorted(k for k in scored[0] if k[0] == axis)
        groups = sorted({scored[0][k]['image_group'] for k in axis_keys})
        group_index = {g:i for i,g in enumerate(groups)}
        sums = np.zeros((len(groups),len(keys)),dtype=np.float64)
        denominator = np.zeros(len(groups),dtype=np.float64)
        for k in axis_keys:
            g = group_index[scored[0][k]['image_group']]
            denominator[g] += 1
            for j,model in enumerate(scored):
                sums[g,j] += model[k]['correct_bit']
        points = sums.sum(0)/denominator.sum()
        samples = np.empty((replicates,len(keys)))
        # Bounded batches avoid a full replicates x image-cluster allocation.
        for start in range(0,replicates,40):
            size = min(40,replicates-start)
            counts = rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=size)
            samples[start:start+size] = (counts@sums)/(counts@denominator)[:,None]
        arm_samples = {}
        arm_points = {}
        for arm_index,arm in enumerate(('A1','A2','U')):
            columns = 1+3*seed_draws+arm_index
            arm_samples[arm] = samples[np.arange(replicates)[:,None],columns].mean(1)
            arm_points[arm] = points[1+arm_index::3].mean()
        arm_samples['N'] = samples[:,0]
        arm_points['N'] = points[0]
        contrasts = {}
        for name in macro_draws:
            left,right = name.split('-')
            draws = arm_samples[left]-arm_samples[right]
            point = float(arm_points[left]-arm_points[right])
            contrasts[name] = {'difference':point,'ci95':[float(x) for x in np.quantile(draws,[.025,.975])]}
            macro_draws[name] += draws/4
            macro_points[name] += point/4
        output['axes'][axis] = {'original_units':len(axis_keys),'image_clusters':len(groups),
                                'scores_seed_mean':{a:float(v) for a,v in arm_points.items()},'paired_contrasts':contrasts}
    output['macro_secondary'] = {name:{'difference':macro_points[name],
                'ci95':[float(x) for x in np.quantile(draws,[.025,.975])]} for name,draws in macro_draws.items()}
    output['macro_caveat'] = 'Within-axis cluster resampling; cross-axis shared-image covariance is not modeled. Interpret macro intervals descriptively.'
    write_json(OUT/'formal-paired-analysis.json',output)
    lines = ['# 独立7B序列蒸馏：15次训练与5090评测', '',
             '全部15次固定1,000更新训练及N+15模型的16,900提示评测已验收。以下为开发基准探索性结果，不是ARP确认性结论。', '',
             '| 轴 | 对比 | 差值 | 配对95%区间 |','|---|---|---:|---|']
    for axis,result in output['axes'].items():
        for name,c in result['paired_contrasts'].items():
            lines.append(f"| {axis} | {name} | {c['difference']:.5f} | [{c['ci95'][0]:.5f}, {c['ci95'][1]:.5f}] |")
    lines += ['', '区间由seed与来源图像簇配对重采样产生，是逐项描述性95%区间；不据此声称多重比较后的显著性。四轴间共用图像的协方差未纳入次要macro区间。', '',
              '原ARP八项阻塞与工作量估算保留在仓库route_b_execution_plan.md；本轮成功不改变C门槛状态。']
    (OUT/'formal-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete','models':len(keys),'rows_per_model':len(manifest)}),flush=True)


if __name__ == '__main__':
    analyze()
