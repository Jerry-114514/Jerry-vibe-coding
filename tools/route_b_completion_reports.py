"""Generate/upload reports on training completion and evaluation completion.

Reads real artifacts; no new training, no added seeds, no API inference.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiment_common import ROOT, AXES, read_json, read_rows, sha, write_json, units
from route_b_staged import OUT, SEEDS

JOBS = [(arm,seed) for seed in SEEDS for arm in ('A1','A2','U')]
REPORTS = OUT/'completion-reports'


def training_report():
    folder = REPORTS/'training'
    folder.mkdir(parents=True,exist_ok=True)
    results = []
    curves = {}
    for arm,seed in JOBS:
        run = OUT/'formal'/f'{arm}-seed{seed}'
        marker = read_json(run/'complete.json')
        losses = read_rows(run/'loss.jsonl')
        assert marker['actual_updates'] == len(losses) == 1000
        assert [r['step'] for r in losses] == list(range(1,1001))
        assert sha(run/'loss.jsonl') == marker['loss_sha256']
        assert sha(run/'adapter/adapter_model.safetensors') == marker['adapter_sha256']
        assert all(r['grad_norm']>0 and all(math.isfinite(r[k]) for k in ('loss','ce_gt','ce_teacher','grad_norm')) for r in losses)
        seconds = sum(r['seconds'] for r in losses)
        result = {'arm':arm,'seed':seed,'updates':1000,'training_step_seconds':seconds,
                  'seconds_per_step':seconds/1000,'reserved_peak_bytes':max(r['reserved_peak_bytes'] for r in losses),
                  'first_loss':losses[0]['loss'],'last_loss':losses[-1]['loss'],
                  'adapter_sha256':marker['adapter_sha256'],'loss_sha256':marker['loss_sha256']}
        results.append(result)
        curves[(arm,seed)] = [r['loss'] for r in losses]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes = plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for axis,arm in zip(axes,('A1','A2','U')):
        for seed in SEEDS:
            values = np.asarray(curves[(arm,seed)])
            smoothed = np.convolve(values,np.ones(20)/20,mode='valid')
            axis.plot(np.arange(20,1001),np.maximum(smoothed,1e-8),label=str(seed),linewidth=1)
        axis.set(title=arm,xlabel='Actual optimizer update',ylabel='Dual CE (20-step moving mean)',yscale='log')
        axis.grid(alpha=.2)
        axis.legend(title='Seed',fontsize=7)
    figure.savefig(folder/'loss-curves.png',dpi=180)
    plt.close(figure)
    data = {'status':'15_trainings_complete','updates':15000,'runs':results,
            'step_seconds_total':sum(r['training_step_seconds'] for r in results),
            'timestamp':datetime.now(timezone.utc).isoformat(),'evaluation_complete':False,
            'scope':'Independent routeB; noARP confirmation claim'}
    write_json(folder/'training-results.json',data)
    lines = ['# 独立7B：15次训练完成', '',
             '固定3臂×5seed×1,000次真实更新，未临时追加预算。耗时为逐步处理总和，不含装载/保存/备份/排队；账单应按实际开机时长。', '',
             '| 臂 | seed | 秒/步 | 更新耗时(分钟) | reserved GB | 末步损失 |','|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['arm']} | {r['seed']} | {r['seconds_per_step']:.3f} | {r['training_step_seconds']/60:.2f} | {r['reserved_peak_bytes']/1e9:.3f} | {r['last_loss']:.6g} |" for r in results]
    lines += ['', '![逐臂五seed损失曲线](loss-curves.png)', '', '该报告不将训练完成等同于完整评测完成；逐题评测及配对区间另行更新。']
    (folder/'training-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return folder


def read_complete_evaluations():
    manifest = read_rows(ROOT/'data/manifest.jsonl')
    complete = {}
    for arm,seed in [('N',42)]+JOBS:
        run = OUT/'formal'/f'{arm}-seed{seed}'
        if not (run/'scores.json').exists():continue
        marker = read_json(run/'scores.json')
        assert marker['rows'] == 16900 and marker['ledger_sha256'] == sha(run/'eval.jsonl')
        records = units(manifest,read_rows(run/'eval.jsonl'))
        complete[(arm,seed)] = (marker,{(r['axis'],r['dataset'],r['unit']):r for r in records})
    return complete


def paired_intervals(complete,replicates=2000):
    contrasts = {}
    for left,right in [('A1','A2'),('A1','U'),('A1','N')]:
        seeds = [s for s in SEEDS if (left,s) in complete and (right,42 if right=='N' else s) in complete]
        name = left+'-'+right
        if not seeds:
            contrasts[name] = {'status':'pending_matched_complete_evaluations','matched_seeds':[]}
            continue
        result = {'status':'available','matched_seeds':seeds,'interval':'Pointwise95%; exploratory',
                  'training_seed_uncertainty_included':len(seeds)>1,'axes':{}}
        first = complete[(left,seeds[0])][1]
        for axis in AXES:
            keys = sorted(k for k in first if k[0] == axis)
            groups = sorted({first[k]['image_group'] for k in keys})
            indexes = {g:i for i,g in enumerate(groups)}
            totals = np.zeros((len(groups),len(seeds)))
            counts = np.zeros(len(groups))
            for key in keys:
                g = indexes[first[key]['image_group']]
                counts[g] += 1
                for j,seed in enumerate(seeds):
                    totals[g,j] += complete[(left,seed)][1][key]['correct_bit']-complete[(right,42 if right=='N' else seed)][1][key]['correct_bit']
            rng = np.random.default_rng(2026101643)
            draws = np.empty(replicates)
            for start in range(0,replicates,30):
                size = min(30,replicates-start)
                weights = rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=size)
                seed_indices = rng.integers(0,len(seeds),size=(size,len(seeds)))
                means = (weights@totals)/(weights@counts)[:,None]
                draws[start:start+size] = means[np.arange(size)[:,None],seed_indices].mean(1)
            result['axes'][axis] = {'difference':float(totals.sum()/counts.sum()/len(seeds)),
                       'ci95':[float(x) for x in np.quantile(draws,[.025,.975])],
                       'image_clusters':len(groups),'original_units':len(keys)}
        contrasts[name] = result
    return contrasts


def evaluation_report():
    folder = REPORTS/'evaluation'
    folder.mkdir(parents=True,exist_ok=True)
    complete = read_complete_evaluations()
    total = 16*16900
    processed = 0
    seconds = 0
    for run in (OUT/'formal').glob('*-seed*'):
        ledger = run/'eval.jsonl'
        if not ledger.exists():continue
        for line in ledger.read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'):continue
            row = json.loads(line)
            processed += 1
            seconds += row['seconds']
    rate = seconds/processed if processed else read_json(OUT/'minimal-report.json')['measured_eval_seconds_per_prompt']
    remaining = (total-processed)*rate/3600
    result = {'status':'all_evaluations_complete' if len(complete)==16 else 'partial',
              'completed_models':len(complete),'expected_models':16,'processed_prompts':processed,
              'remaining_estimated_hours':remaining,'remaining_estimated_CNY':remaining*2.70,
              'mean_seconds_per_prompt':rate,'scores':{f'{a}-seed{s}':m[0] for (a,s),m in complete.items()},
              'paired_contrasts':paired_intervals(complete),'timestamp':datetime.now(timezone.utc).isoformat(),
              'eta_limit':'Observed mix estimate; includes neither future output-length drift nor backup time'}
    write_json(folder/'evaluation-results.json',result)
    lines = ['# 独立7B：5090评测报告', '',f"已完整评测{len(complete)}/16个模型；剩余估计{remaining:.2f}小时、{remaining*2.70:.2f}元。", '',
             '| 模型 | pope | grounding | consistency | mmstar |','|---|---:|---:|---:|---:|']
    for key,(marker,_) in complete.items():
        lines.append('| '+f'{key[0]}-seed{key[1]}'+' | '+' | '.join(f"{marker['axis_scores'][a]:.4f}" for a in AXES)+' |')
    lines += ['', '| 配对 | 轴 | 差值 | 描述性95%区间 | 匹配seed数 |','|---|---|---:|---|---:|']
    for name,c in result['paired_contrasts'].items():
        if c['status'] != 'available':
            lines.append(f'| {name} | 待配对评测完成 | — | — | 0 |');continue
        for a,r in c['axes'].items():
            lines.append(f"| {name} | {a} | {r['difference']:.5f} | [{r['ci95'][0]:.5f},{r['ci95'][1]:.5f}] | {len(c['matched_seeds'])} |")
    lines += ['', '仅使用已完整评测且同seed配对的模型；一个seed时区间只有来源图像簇不确定性，不能表示跨seed稳定性。区间未进行多重比较校正；独立实验不冒充ARP结论。']
    (folder/'evaluation-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return folder,result


def watch():
    import fcntl
    REPORTS.mkdir(parents=True,exist_ok=True)
    with (REPORTS/'observer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        trained = False
        previous = -1
        from publish_artifacts import publish
        deadline = time.monotonic()+40*3600
        while time.monotonic()<deadline:
            ready = sum((OUT/'formal'/f'{a}-seed{s}'/'complete.json').exists() for a,s in JOBS)
            if ready==15 and not trained:
                folder = training_report()
                publish(folder,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])
                write_json(REPORTS/'training-report-delivered.json',{'status':'uploaded','time':datetime.now(timezone.utc).isoformat()})
                trained = True
            count = sum((OUT/'formal'/f'{a}-seed{s}'/'scores.json').exists() for a,s in [('N',42)]+JOBS)
            if trained and count!=previous:
                folder,result = evaluation_report()
                publish(folder,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])
                previous = count
                if count==16:
                    write_json(REPORTS/'final-report-delivered.json',{'status':'uploaded','time':datetime.now(timezone.utc).isoformat()})
                    return
            if (OUT/'failure.json').exists():
                write_json(REPORTS/'observed-failure.json',read_json(OUT/'failure.json'))
                publish(REPORTS,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])
                return
            time.sleep(10)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        watch()
    elif '--snapshot' in sys.argv:
        print(evaluation_report()[1])
    else:
        token = sys.stdin.readline().strip()
        command = 'cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/route_b_completion_reports.py --worker >> runs/route-b-staged-7b-20261016/completion-reports.log 2>&1'
        subprocess.run(['screen','-dmS','route-b-reports','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Completion reports scheduled on actual15-run and16-model artifacts, no extra experiments')
