"""Two-seed factorial effects; explicitly exploratory and separate from old12 tests."""
from collections import defaultdict
import math
import importlib.metadata

import numpy as np

from experiment_common import AXES,read_json,read_rows,sha,write_json
from prepare_9b_repair import OUT,OLD,OLD_EVAL,SEEDS,now
from repair_9b_codec import CONDITIONS

EFFECTS={'coordinate':[-.5,-.5,.5,.5], 'teacher':[.5,-.5,.5,-.5],
         'interaction':[-1.,1.,1.,-1.]}


def load_endpoint(condition,seed,dataset):
    if dataset=='development' and condition in ('P-D','N'):
        import run_9b_native_frontend_v41 as old
        return old.verify_evaluation('U' if condition=='P-D' else 'N',seed)
    from evaluate_9b_repair import verify_evaluation
    return verify_evaluation(condition,seed,dataset)


def report():
    folder=OUT/'report';folder.mkdir(exist_ok=True)
    rng=np.random.default_rng(2026101804);replicates=10000
    environment={n:importlib.metadata.version(n) for n in ('numpy','matplotlib')}
    assert environment==read_json(OUT/'protocol.json')['analysis_packages']
    model_keys=[(c,s) for s in SEEDS for c in CONDITIONS]
    seed_indices=rng.integers(0,len(SEEDS),size=(replicates,len(SEEDS)))
    result={'status':'fixed_factorial_endpoints_verified','timestamp_utc':now(),
        'protocol_sha256':sha(OUT/'protocol.json'),'seeds':list(SEEDS),
        'analysis_environment':environment,
        'bootstrap_replicates':replicates,'bootstrap_seed':2026101804,'family_size':24,
        'scope':'Post-result exploratory2seed follow-up; historical PD reused; heldout single-image choice subset; not ARP confirmation',
        'caveat':'Percentile bootstrap intervals are approximate; only2 training seeds cannot establish broad seed stability or equivalence',
        'datasets':{},'training':{}}
    for dataset in ('development','heldout'):
        values={key:load_endpoint(*key,dataset) for key in model_keys}
        baseline=load_endpoint('N',42,dataset)
        reference=baseline[1]
        assert all(set(v[1])==set(reference) for v in values.values())
        data={'scores':{f'{c}-seed{s}':values[(c,s)][0]['axis_scores'] for c,s in model_keys},
              'N_scores':baseline[0]['axis_scores'],'effects':{},'per_seed_effects':{},
              'errors':{f'{c}-seed{s}':{'length_capped_outputs':values[(c,s)][0]['length_capped_outputs'],
                   'ledger_sha256':values[(c,s)][0]['ledger_sha256']} for c,s in model_keys}}
        for axis in AXES:
            keys=sorted(k for k in reference if k[0]==axis)
            groups=sorted({reference[k]['image_group'] for k in keys});index={v:i for i,v in enumerate(groups)}
            totals=np.zeros((len(groups),len(model_keys)));counts=np.zeros(len(groups))
            for key in keys:
                i=index[reference[key]['image_group']];counts[i]+=1
                for j,model in enumerate(model_keys):
                    assert values[model][1][key]['image_group']==reference[key]['image_group']
                    totals[i,j]+=values[model][1][key]['correct_bit']
            point=totals.sum(0)/counts.sum();draws=np.empty((replicates,len(model_keys)))
            for start in range(0,replicates,40):
                size=min(40,replicates-start)
                weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=size)
                draws[start:start+size]=(weights@totals)/(weights@counts)[:,None]
            condition_points=np.array([point[i::4].mean() for i in range(4)])
            condition_draws=np.column_stack([draws[np.arange(replicates)[:,None],4*seed_indices+i].mean(1) for i in range(4)])
            for effect,coefficients in EFFECTS.items():
                coefficient=np.array(coefficients);distribution=condition_draws@coefficient
                alpha=.05/24
                data['effects'].setdefault(effect,{})[axis]={'difference':float(condition_points@coefficient),
                    'pointwise_ci95':[float(v) for v in np.quantile(distribution,[.025,.975])],
                    'bonferroni24_ci95':[float(v) for v in np.quantile(distribution,[alpha/2,1-alpha/2])],
                    'original_units':len(keys),'image_clusters':len(groups)}
                data['per_seed_effects'].setdefault(effect,{})[axis]={str(seed):float(point[4*i:4*i+4]@coefficient) for i,seed in enumerate(SEEDS)}
        result['datasets'][dataset]=data
    for condition,seed in model_keys:
        path=OLD/'formal'/f'U-seed{seed}' if condition=='P-D' else OUT/'training'/f'{condition}-seed{seed}'
        marker=read_json(path/'complete.json');rows=read_rows(path/'loss.jsonl')
        assert sha(path/'loss.jsonl')==marker['loss_sha256'] and [r['step'] for r in rows]==list(range(1,1001))
        assert all(math.isfinite(r['loss']) and r['grad_norm']>0 and math.isfinite(r['grad_norm']) for r in rows)
        result['training'][f'{condition}-seed{seed}']={'updates':1000,'adapter_sha256':marker['adapter_sha256'],
            'loss_sha256':marker['loss_sha256'],'last_loss':rows[-1]['loss'],
            'pure_step_seconds':sum(r['seconds'] for r in rows),
            'peak_reserved_bytes':max(r['reserved_peak_bytes'] for r in rows),
            'source':'historical fixed endpoint' if condition=='P-D' else 'new fixed endpoint'}
    from train_9b_repair import verify_training
    for c,s in model_keys:
        if c!='P-D':verify_training(c,s)
    write_json(folder/'results.json',result)
    lines=['# 9B定位失稳：坐标表示×监督2×2探索性对照','',
        '共8个固定1000步端点、2个seed，其中两个P-D为历史端点，新增6,000步。所有结果保留，不选择最优seed或检查点。',
        '旧16模型及原12项比较结果不改写；本轮24项探索性比较单独报告。只有两个seed，不把区间解释为广泛训练稳定性证明。','']
    for dataset,data in result['datasets'].items():
        lines.extend(['## '+dataset,'','| 模型 | POPE | 定位 | 一致性 | 推理轴 |','|---|---:|---:|---:|---:|'])
        lines.append('| N | '+' | '.join(f"{100*data['N_scores'][a]:.3f}%" for a in AXES)+' |')
        for model,scores in data['scores'].items():lines.append('| '+model+' | '+' | '.join(f'{100*scores[a]:.3f}%' for a in AXES)+' |')
        lines.extend(['','| 因素 | 轴 | 差值pp | 24项校正区间pp |','|---|---|---:|---|'])
        for effect,axes in data['effects'].items():
            for axis,row in axes.items():
                lo,hi=row['bonferroni24_ci95'];lines.append(f"| {effect} | {axis} | {100*row['difference']:+.3f} | [{100*lo:.3f}, {100*hi:.3f}] |")
        lines.extend(['',f'![四轴全部端点]({dataset}-scores.png)',''])
    lines.extend(['## 解释边界','',
        '坐标主效应=(N-D+N-G-P-D-P-G)/2；教师主效应=(P-D+N-D-P-G-N-G)/2；交互=(N-D-N-G)-(P-D-P-G)。',
        '表示变更包含由单位改变产生的数值文本与token长度变化，不能声称只隔离空间能力。GT-only保留完整单CE权重。',
        '复核集每轴120原题，总960提示，来自既有去重confirm角色中未见的单图子集；推理为MMMU选择题代理，不能冒称MMStar全量或完整MMMU。',
        '旧数据上的排错发生在结果观察之后；独立子集也只有两个训练seed。未有人工作为原ARP所要求的定位盲审，原ARP阶段C仍未放行。',
        '历史/新端点存在运行时序差异，坐标、教师与其他优化机制的完整因果归因仍有局限。区间跨0不证明等效。',
        '', '![24项配对区间](factorial-effects.png)', '',
        '## 训练过程与资源','',
        '不同目标表示和GT-only/双CE之间的损失量级不可直接当作能力排名；曲线用于各条件内健康检查。',
        '', '| 条件与seed | 更新 | 步骤耗时分钟 | reserved GB | 最后损失 |', '|---|---:|---:|---:|---:|'])
    for name,row in result['training'].items():
        lines.append(f"| {name} | 1000 | {row['pure_step_seconds']/60:.2f} | {row['peak_reserved_bytes']/1e9:.3f} | {row['last_loss']:.6g} |")
    lines.extend(['','步骤耗时不含加载、保存与上传；原始逐步损失、梯度、显存、耗时、恢复状态及SHA都归档Release。',
                  '', '![各条件内训练曲线](loss-curves.png)'])
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    figures(result)
    write_json(folder/'verification.json',{'status':'passed','model_endpoints':8,'new_runs':6,
        'new_updates':6000,'new_predictions':6*16900+9*960,'datasets_verified':2,'effects':24,'timestamp_utc':now()})
    return result


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'report'
    for dataset,data in result['datasets'].items():
        scores={'N':data['N_scores'],**data['scores']};names=list(scores)
        fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
        for axis,ax in zip(AXES,axes.flat):
            ax.bar(range(9),[100*scores[n][axis] for n in names],color=['#777777']+['#397bb5' if n.startswith('P-') else '#e19539' for n in names[1:]])
            ax.set_xticks(range(9),names,rotation=65,ha='right',fontsize=8);ax.set_ylim(0,100);ax.set_title(axis);ax.set_ylabel('Original-unit accuracy (%)')
        fig.suptitle('Exploratory 2-seed coordinate x supervision / '+dataset)
        for ext in ('png','pdf'):fig.savefig(folder/f'{dataset}-scores.{ext}',dpi=170)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,11),constrained_layout=True);labels=[]
    for dataset,data in result['datasets'].items():
        for effect,axes in data['effects'].items():
            for axis,row in axes.items():
                i=len(labels);labels.append(dataset+' / '+effect+' / '+axis)
                lo,hi=row['bonferroni24_ci95'];ax.plot([100*lo,100*hi],[i,i],color='#397bb5');ax.plot(100*row['difference'],i,'o',color='#333333')
    ax.axvline(0,color='#999999',linestyle='--');ax.set_yticks(range(24),labels,fontsize=8)
    ax.set_xlabel('Paired difference (percentage points)');ax.set_title('Exploratory: 2 seeds; paired seed/image bootstrap; Bonferroni family24')
    for ext in ('png','pdf'):fig.savefig(folder/f'factorial-effects.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for condition,ax in zip(CONDITIONS,axes.flat):
        for seed in SEEDS:
            path=OLD/'formal'/f'U-seed{seed}' if condition=='P-D' else OUT/'training'/f'{condition}-seed{seed}'
            values=np.array([r['loss'] for r in read_rows(path/'loss.jsonl')])
            smooth=np.convolve(values,np.ones(20)/20,mode='valid')
            ax.plot(range(20,1001),np.maximum(smooth,1e-9),label=str(seed))
        ax.set_title(condition);ax.set_yscale('log');ax.set_xlabel('Actual update');ax.set_ylabel('Within-condition loss / 20-step mean');ax.legend(title='Seed')
    fig.suptitle('Fixed1000 updates; target representations and loss definitions differ across conditions')
    for ext in ('png','pdf'):fig.savefig(folder/f'loss-curves.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':report()
