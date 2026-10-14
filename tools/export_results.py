"""E1/E2 and D4: tables, signed gain decomposition and publication figures."""
import argparse
import csv
from pathlib import Path
import numpy as np
from analyze_rulers import finite_json, ruler_vector
from experiment_common import AXES, LADDERS, ROOT, read_json, read_rows, units, write_json


def csv_file(path, rows):
    if not rows:
        return
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def net_gain(run, arm):
    manifest=read_rows(ROOT/'data/manifest.jsonl')
    base=units(manifest,[x for p in sorted((run/'ledger').glob('qwen3b-[0-9][0-9][0-9].jsonl')) for x in read_rows(p)])
    teacher=units(manifest,[x for p in sorted((run/'ledger').glob('qwen7b-[0-9][0-9][0-9].jsonl')) for x in read_rows(p)])
    baseline=units(manifest,read_rows(ROOT/'data/baselines/per_item.jsonl'))
    if arm=='N':
        post=base
    else:
        paths=sorted((run/'post'/arm).glob('qwen3b-[0-9][0-9][0-9].jsonl'))
        post=units(manifest,[x for p in paths for x in read_rows(p)])
    records=[baseline,base,teacher,post]
    scores={}
    for mode in ['wrong','exclude']:
        values=np.full((4,len(AXES)),np.nan)
        for a,axis in enumerate(AXES):
            strata=sorted({x['stratum'] for x in base if x['axis']==axis})
            components=[]
            for stratum in strata:
                indices=[i for i,x in enumerate(base) if x['axis']==axis and x['stratum']==stratum]
                if mode=='exclude':
                    indices=[i for i in indices if not any(rows[i]['parse_fail_flag'] for rows in records[1:])]
                components.append([np.mean([rows[i]['correct_bit'] for i in indices]) if indices else np.nan for rows in records])
            values[:,a]=np.mean(components,axis=0)
        scores[mode]=values
    classes=read_json(run/'analysis/qwen-classes.json')
    rows=[]
    for ruler in classes:
        value=scores[ruler['setting']['parse_fail']]
        vector,guards=ruler_vector(np.array([value[1],value[3]]),value[0],ruler['setting'],u=value[2])
        rows.append(finite_json({'arm':arm,'class_id':ruler['id'],'G_net':float(vector.sum()) if np.isfinite(vector).all() else None,
                                **{axis:vector[i] for i,axis in enumerate(AXES)},'guards':str(guards)}))
    raw=scores['wrong'][3]-scores['wrong'][1]
    return rows, {'arm':arm,'G_net_raw':float(raw.sum()),'signed_components':dict(zip(AXES,raw.tolist()))}


def export(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run)
    output=run/'paper'
    output.mkdir(parents=True,exist_ok=True)
    summary=read_json(run/'analysis/summary.json')
    t1=[]
    t2=[]
    t3=[]
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    if summary['gate'].get('synthetic_test'):
        fig.suptitle('SYNTHETIC INTEGRATION TEST — NOT EXPERIMENT RESULTS')
    for ax,(family,result) in zip(axes,summary['families'].items()):
        for row in read_rows(run/f'analysis/{family}-folding.jsonl'):
            t1.append({'family':family,**row})
        t2.append({'family':family,'classes':result['quotient_classes'],'F_scale':result['F_scale'],
                   'F_scale_CI':str(result['F_scale_CI']),'F_noise':result['F_noise'],
                   'F_noise_CI':str(result['F_noise_CI']), 'stable_axes':';'.join(result['stable_axis_set']),
                   **{f'pi_{axis}':result['pi_boot_mean'][axis] for axis in AXES}})
        for row in result['attribution']:
            t3.append({'family':family,**row})
        classes=read_json(run/f'analysis/{family}-classes.json')
        matrix=np.array([[np.nan if value is None else value for value in r['vector']] for r in classes])
        ax.imshow(matrix,aspect='auto',cmap='viridis')
        ax.set_xticks(range(4),AXES,rotation=35,ha='right')
        ax.set_ylabel('Ruler equivalence class')
        ax.set_title(family)
    fig.savefig(output/'figure1-ruler-scan.png',dpi=300)
    fig.savefig(output/'figure1-ruler-scan.svg')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for i,(family,result) in enumerate(summary['families'].items()):
        for offset,name,ci_name,color in [(-.12,'F_scale','F_scale_CI','#2563eb'),(.12,'F_noise','F_noise_CI','#d97706')]:
            y=result[name]
            lo,hi=result[ci_name]
            if y is not None and lo is not None:
                ax.errorbar(i+offset,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',color=color,
                            capsize=4,label=name if i==0 else None)
    ax.set_xticks(range(3),list(summary['families']))
    ax.set_ylabel('Argmax flip rate')
    ax.set_ylim(0,1)
    ax.legend()
    ax.set_title('Scale and noise with paired-bootstrap 95% CI')
    fig.savefig(output/'figure2-scale-vs-noise.png',dpi=300)
    fig.savefig(output/'figure2-scale-vs-noise.svg')
    plt.close(fig)
    csv_file(output/'table1-equivalence.csv',t1)
    csv_file(output/'table2-stability.csv',t2)
    csv_file(output/'table3-attribution.csv',t3)
    t4,raw=[],[]
    for arm in ['A1','A2','U','N']:
        if arm=='N' or (run/'post'/arm).exists():
            rows,components=net_gain(run,arm)
            t4.extend(rows)
            raw.append(components)
    if len(raw)==4:
        csv_file(output/'table4-arm-gains.csv',t4)
        write_json(output/'signed-gains.json',raw)
    planted=read_json(run/'analysis/planted.json')
    lower,upper=planted['flip_ratio_boundary']
    fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
    ratio=np.linspace(.1,1.3,300)
    ax.plot(ratio,ratio,label='Raw gap ratio')
    ax.plot(ratio,ratio/lower,label='Floor-corrected ratio')
    ax.axhline(1,color='gray',linestyle='--')
    ax.axvspan(lower,upper,alpha=.15,color='#d97706',label='Planted flip interval')
    ax.set_xlabel('Gap ratio')
    ax.legend()
    fig.savefig(output/'planted-boundary.png',dpi=300)
    plt.close(fig)
    write_json(output/'results.json',summary)
    import html,json
    text=html.escape(json.dumps(summary,ensure_ascii=False,indent=2))
    (output/'interactive-results.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>ARP results</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>ARP frozen-ledger results</h1><p>F_noise fixes absolute drop, no floor correction, no headroom normalization, mean OWA and parse failures scored as wrong.</p><img src="figure1-ruler-scan.png"><img src="figure2-scale-vs-noise.png"><details><summary>Full numerical results and protocol</summary><pre>'+text+'</pre></details></html>',encoding='utf-8')
    print(output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',default=str(ROOT/'runs/formal'))
    a=p.parse_args()
    export(a.run)
