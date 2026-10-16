"""P0 exploratory reanalysis: no new inference, no retrospective training gate."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np

from analyze_rulers import ci, finite_json, grid
from experiment_common import AXES, LADDERS, ROOT, read_json, read_rows, sha, units, write_json
from rulers_v2 import paired_cluster_scores, ruler_vector, scan, signed_gain


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(finite_json(rows))


def verified_ledger(folder, key, manifest):
    paths = sorted(folder.glob(key+'-[0-9][0-9][0-9].jsonl'))
    if not paths:
        raise ValueError('No completed historical ledger for '+key)
    rows = []
    for path in paths:
        done = read_json(path.with_suffix('.complete.json'))
        meta = read_json(path.with_suffix('.meta.json'))
        if done['ledger_sha256'] != sha(path) or meta.get('smoke') or meta.get('synthetic'):
            raise ValueError('Unverified or synthetic ledger: '+str(path))
        if meta['manifest_sha256'] != sha(ROOT/'data/manifest.jsonl'):
            raise ValueError('Historical manifest mismatch')
        rows.extend(read_rows(path))
    return units(manifest, rows)


def c_reanalysis(source, output, protocol, n_boot):
    old = read_json(source/'analysis/summary.json')
    comparison, diagnostics, class_vectors = [], [], {}
    for family in LADDERS:
        cache = np.load(source/f'analysis/{family}-bootstrap.npz')
        scores = {m: cache['scores_'+m] for m in ['wrong','exclude']}
        floors = {m: cache['floors_'+m] for m in ['wrong','exclude']}
        baseline = old['families'][family]
        comparison.append({'family':family, 'analysis':'v1_published', 'F':baseline['F_scale'],
                           'F_CI_low':baseline['F_scale_CI'][0], 'F_CI_high':baseline['F_scale_CI'][1],
                           'classes':baseline['quotient_classes'], 'strict_pair_flip_rate':None,
                           'full_ties':None, 'partial_ties':None, 'invalid_classes':None,
                           'undefined_bootstraps':baseline.get('undefined_bootstraps'),
                           'axes':','.join(AXES)})
        for name, base, quotient, full_axes in [
                ('formula_only_large_legacy_quotient','large','legacy_empirical',False),
                ('v2_structural_common_axes','large','structural',True)]:
            point = scan({m:v[0] for m,v in scores.items()}, {m:v[0] for m,v in floors.items()}, protocol,
                         base, quotient, require_full_axes=full_axes)
            Fs=[]
            for b in range(1, min(n_boot+1,len(scores['wrong']))):
                value = scan({m:v[b] for m,v in scores.items()}, {m:v[b] for m,v in floors.items()}, protocol,
                             base, quotient, require_full_axes=full_axes)
                Fs.append(value['F'])
            interval=ci(Fs)
            comparison.append({'family':family,'analysis':name,'F':point['F'],
                               'F_CI_low':interval[0],'F_CI_high':interval[1],
                               'classes':len(point['classes']),'strict_pair_flip_rate':point['strict_pair_flip_rate'],
                               'full_ties':point['full_ties'],'partial_ties':point['partial_ties'],
                               'invalid_classes':point['invalid_classes'],
                               'undefined_bootstraps':sum(x is None for x in Fs),'axes':','.join(AXES)})
            class_vectors[family+'/'+name]=finite_json(point)
        for epsilon in [0,.005,.01,.02]:
            for axes in [list(range(4)),[0,2,3]]:
                result=scan({m:v[0] for m,v in scores.items()}, {m:v[0] for m,v in floors.items()},
                            {**protocol,'tie_epsilon':epsilon},axis_indices=axes)
                diagnostics.append({'family':family,'tie_epsilon':epsilon,'axes':','.join(AXES[a] for a in axes),
                                    'F':result['F'],'F_without_full_ties':result['F_without_full_ties'],
                                    'strict_pair_flip_rate':result['strict_pair_flip_rate'],
                                    'full_ties':result['full_ties'],'partial_ties':result['partial_ties'],
                                    'usable_classes':result['usable_classes'],'invalid_classes':result['invalid_classes']})
        print('C P0 complete:',family,flush=True)
    write_csv(output/'analysis_v1_vs_v2.csv',comparison)
    write_csv(output/'tie_and_guard_report.csv',diagnostics)
    write_json(output/'class-vectors.json',class_vectors)


def d_reanalysis(source, output, manifest, protocol, n_boot):
    keys=['base','teacher','A1','A2','U']
    records=[units(manifest,read_rows(ROOT/'data/baselines/per_item.jsonl')),
             verified_ledger(source/'ledger','qwen3b',manifest),
             verified_ledger(source/'ledger','qwen7b',manifest)]
    records += [verified_ledger(source/'post'/arm,'qwen3b',manifest) for arm in ['A1','A2','U']]
    identities=[(r['axis'],r['stratum'],r['unit']) for r in records[0]]
    if any([(r['axis'],r['stratum'],r['unit']) for r in rows]!=identities for rows in records):
        raise ValueError('Training arm scoring units are not paired')
    failed=np.array([[r['parse_fail_flag'] for r in rows] for rows in records],dtype=bool)
    common=~failed[1:].any(axis=0)
    coverage=[]
    scores={}
    for mode in ['wrong','exclude']:
        means=np.empty((len(records),len(AXES)))
        for a,axis in enumerate(AXES):
            strata=sorted({r['stratum'] for r in records[0] if r['axis']==axis})
            parts=[]
            for stratum in strata:
                original=[i for i,r in enumerate(records[0]) if r['axis']==axis and r['stratum']==stratum]
                indices=original if mode=='wrong' else [i for i in original if common[i]]
                parts.append([np.mean([rows[i]['correct_bit'] for i in indices]) if indices else np.nan for rows in records])
                coverage.append({'mode':mode,'axis':axis,'stratum':stratum,'original_units':len(original),
                                 'common_units':len(indices),'excluded_units':len(original)-len(indices),
                                 'common_images':len({records[0][i]['image_group'] for i in indices})})
            means[:,a]=np.mean(parts,axis=0)
        scores[mode]=means
    write_csv(output/'arm_common_support.csv',coverage)
    arm_indices={'A1':3,'A2':4,'U':5,'N':1}
    all_rows=[]
    joint_classes={}
    for nominal,setting in enumerate(grid(protocol)):
        values=scores[setting['parse_fail']]
        joint=[]
        arm_rows=[]
        for arm,index in arm_indices.items():
            vector,guards,denominator=ruler_vector(np.array([values[1],values[index]]),values[0],setting,u=values[2])
            joint.extend(None if not np.isfinite(x) else round(float(x),12) for x in vector)
            arm_rows.append({'arm':arm,'nominal_id':nominal,**setting,'joint_class_id':None,
                             'G_sensitivity':float(vector.sum()) if np.isfinite(vector).all() else None,
                             **{axis:vector[a] for a,axis in enumerate(AXES)},
                             'guards':json.dumps(guards),'denominators':json.dumps(finite_json(denominator))})
        joint_key=tuple(joint)
        if joint_key not in joint_classes:
            joint_classes[joint_key]=len(joint_classes)
        for row in arm_rows:
            row['joint_class_id']=joint_classes[joint_key]
        all_rows.extend(arm_rows)
    write_csv(output/'arm_gains_all_120.csv',all_rows)
    samples=paired_cluster_scores(records,AXES,n_boot,20261016)
    denominators=[]
    for a,axis in enumerate(AXES):
        draws=samples[1:,2,a]-samples[1:,1,a]
        interval=ci(draws)
        denominators.append({'axis':axis,'teacher_minus_pre':float(samples[0,2,a]-samples[0,1,a]),
                             'CI_low':interval[0],'CI_high':interval[1],
                             'CI_crosses_zero':interval[0]<=0<=interval[1],
                             'nonpositive_draw_fraction':float(np.mean(draws<=0)),
                             'abs_below_0_001_fraction':float(np.mean(np.abs(draws)<.001)),
                             'abs_below_0_005_fraction':float(np.mean(np.abs(draws)<.005)),
                             'abs_below_0_01_fraction':float(np.mean(np.abs(draws)<.01))})
    write_csv(output/'headroom_denominator_diagnostics.csv',denominators)
    primary={}
    distributions={}
    for arm,index in arm_indices.items():
        point=signed_gain(samples[0,1],samples[0,index],samples[0,0])
        draws=[signed_gain(sample[1],sample[index],sample[0]) for sample in samples[1:]]
        primary[arm]=finite_json({**point,'raw_CI':ci([d['G_raw'] for d in draws]),
                                 'floor_CI':ci([d['G_floor'] for d in draws]),
                                 'undefined_draws':sum(not d['defined'] for d in draws)})
        distributions[arm]=np.array([[d['G_raw'],d['G_floor']] for d in draws],dtype=float)
    comparisons=[]
    for comparator in ['A2','U','N']:
        for endpoint,column in [('G_raw',0),('G_floor',1)]:
            difference=distributions['A1'][:,column]-distributions[comparator][:,column]
            finite=difference[np.isfinite(difference)]
            interval=np.quantile(finite,[.05/(2*3),1-.05/(2*3)]).tolist() if len(finite) else [None,None]
            comparisons.append({'contrast':'A1-'+comparator,'endpoint':endpoint,
                                'point':primary['A1'][endpoint]-primary[comparator][endpoint],
                                'paired_CI_95':json.dumps(ci(difference)),
                                'bonferroni_3_CI':json.dumps(interval),
                                'training_seeds':1,'evaluation_cluster_draws':n_boot,
                                'confirmation':False})
    write_json(output/'signed-gains-v2.json',{'arms':primary,'joint_empirical_classes':len(joint_classes),
        'inference_scope':'Historical exploratory evaluation-cluster uncertainty only; one training seed; post-selection bias not removed'})
    write_csv(output/'arm_paired_contrasts.csv',comparisons)
    np.savez_compressed(output/'arm-paired-bootstrap.npz',scores=samples)
    print('D P0 complete:',len(all_rows),'arm-setting rows; common support and paired intervals',flush=True)


def main(source, output, n_boot):
    if source.resolve()==output.resolve() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError('P0 outputs must not overwrite historical run')
    output.mkdir(parents=True,exist_ok=True)
    protocol=read_json(ROOT/'configs/experiment.json')
    started=time.monotonic()
    status={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
            'source_run':str(source.relative_to(ROOT)),'n_boot':n_boot,
            'historical_exploratory_only':True,'training_allowed':False}
    write_json(output/'status.json',status)
    try:
        c_reanalysis(source,output,protocol,n_boot)
        d_reanalysis(source,output,read_rows(ROOT/'data/manifest.jsonl'),protocol,n_boot)
        status.update(status='complete',elapsed_seconds=time.monotonic()-started)
        (output/'formula_audit.md').write_text('''# P0 公式与估计量审查

相对差主定义由 small 改为 large 分母；校正相对差由 small-floor 改为 large-floor。原 v1 未覆盖。
输出分开报告：原 v1、仅更改分母的原经验商集分析、采用结构商集及四轴共同有效域的新分析。
旧 C 的抽样索引直接复用，避免把公式变化和抽样变化混为一谈；这些区间仍是旧抽样结构，非新确认分析。
新增严格唯一最大轴冲突比例、平局率、guard、去除定位轴与四档并列阈值敏感性。
D 以所有臂共同样本域扫描完整 120 设置，联合折叠训练后同值设置；主 G_floor 保留负迁移及 post-subfloor。
D 差值使用所有模型/臂共用的图像簇重采样权重，保留同图跨轴关联，附三项比较的 Bonferroni 区间。
这仍是旧数据、单训练 seed 的探索性补分析，不消除训练随机性或选择后偏差，不授予新训练 gate。
新的 C 配对噪声差值、独立数据确认、多 seed 和人工盲审仍需另行执行。
''',encoding='utf-8')
        write_json(output/'gate.json',{'decision':'indeterminate','formal':False,'training_allowed':False,
                                      'reason':'Historical exploratory P0 only; independent audit and paired noise gate not yet executed'})
    except Exception as exc:
        status.update(status='failed',error=type(exc).__name__,message=str(exc))
        raise
    finally:
        write_json(output/'status.json',status)
        print(status,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'runs/formal-20261014')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/protocol-v2/p0')
    parser.add_argument('--n-boot',type=int,default=1000)
    args=parser.parse_args()
    main(args.source,args.output,args.n_boot)
