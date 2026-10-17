"""User-approved7B remeasurement; explicit processor pixels, no retraining."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT,read_json,read_rows,write_json,write_rows,sha,units
from route_b_staged import OUT as SEVEN,SEEDS,freeze
from processor_pixel_contract_csv import dimensions,item_with_contract,score,assert_runtime_grid,cpu_7b_inputs

OUT=ROOT/'runs/7b-processor-pixel-audit-20261017'
NINE=ROOT/'runs/route-b-9b-122b-20261017/student-experiment/processor-pixel-evaluation-v3'
MODELS=[('N',42)]+[(a,s) for s in SEEDS for a in ('A1','A2','U')]
ENV=Path('/root/autodl-tmp/arp-venv/bin/python')
SOURCES=('run_7b_processor_grounding.py','processor_pixel_contract.py','processor_pixel_contract_csv.py')


def prepare():
    path=OUT/'protocol.json'
    if path.exists():
        plan=read_json(path)
        for name,digest in plan['source_sha256'].items():assert sha(ROOT/'tools'/name)==digest
        assert sha(OUT/'manifest.jsonl')==plan['manifest_sha256']
        assert sha(ROOT/'tools/route_b_staged.py')==plan['original_engine_sha256']
        return read_rows(OUT/'manifest.jsonl')
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    processor=AutoProcessor.from_pretrained(str(ROOT/'.model_cache/qwen7b'),local_files_only=True,use_fast=False,
                 min_pixels=256*28*28,max_pixels=1280*28*28)
    original=[r for r in read_rows(ROOT/'data/manifest.jsonl') if r['axis']=='grounding'];assert len(original)==2400
    sizes={};manifest=[]
    for item in original:
        if item['image_path'] not in sizes:sizes[item['image_path']]=dimensions(processor,item,'qwen7b')
        row=item_with_contract(item,sizes[item['image_path']]);batch=cpu_7b_inputs(processor,row);assert_runtime_grid(batch,row)
        row['frozen_prompt_tokens']=int(batch['input_ids'].shape[1]);assert row['frozen_prompt_tokens']<=2048
        manifest.append(row)
    OUT.mkdir(parents=True,exist_ok=True);write_rows(OUT/'manifest.jsonl',manifest);write_json(OUT/'processor-dimensions.json',sizes)
    plan={'schema':'independent.7b.processor-pixels.audit.v1','scope':'User-approved post-result measurement repair; not a preregistered ability finding',
          'timestamp':datetime.now(timezone.utc).isoformat(),'models':[f'{a}-seed{s}' for a,s in MODELS],
          'rows_per_model':2400,'total_generations':38400,'calibration_included_in38400':100,
          'coordinate_contract':'Explicit actual processor-resized-image pixels; x*original_width/processor_width,y*original_height/processor_height; original-pixel IoU>=.5',
          'same_mapping_as9b':True,'no_gt_or_arm_based_unit_selection':True,'no_training_or_teacher_calls':True,
          'N_first100_minimum_accuracy':.80,'N_first100_maximum_parse_failures':5,
          'source_sha256':{name:sha(ROOT/'tools'/name) for name in SOURCES},
          'original_engine_sha256':sha(ROOT/'tools/route_b_staged.py'),'manifest_sha256':sha(OUT/'manifest.jsonl'),
          'processor_dimensions_sha256':sha(OUT/'processor-dimensions.json'),
          'max_prompt_tokens':max(r['frozen_prompt_tokens'] for r in manifest),'max_new_tokens':128,
          'gpu_wall_hours_cap':12,'budget_never_reset':True,'family_size':4,'bootstrap_replicates':10000,'bootstrap_seed':2026101704,
          'contrasts':['A1-A2','A1-U','A1-N','U-N'],'old12_comparison_family_unchanged':True,
          'queue':'After all9B training and9B common-contract calibration, before9B16-model formal evaluation',
          'supersedes_pending_normalized1000_gpu_plan':True,'old_cpu_diagnostic_retained':True}
    assert plan['original_engine_sha256']==read_json(SEVEN/'protocol.json')['source_sha256']
    assert not torch.cuda.is_initialized();freeze(path,plan)
    return manifest


def evaluate(arm,seed,limit=None):
    manifest=prepare();target=manifest[:limit] if limit is not None else manifest
    assert limit is None or (arm=='N' and seed==42 and limit==100)
    folder=OUT/'models'/f'{arm}-seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    ledger=folder/'eval.jsonl';previous=read_rows(ledger) if ledger.exists() else []
    done={r['item_id'] for r in previous};assert len(done)==len(previous) and done<={r['item_id'] for r in target}
    end=datetime.fromisoformat(read_json(OUT/'gpu-wall-budget.json')['end_utc']).timestamp()
    from gpu_lock import gpu_lock
    import torch
    from route_b_staged import backend,inputs
    from model_backends import finite_logits_processors
    with gpu_lock():
        torch.set_num_threads(2);torch.set_grad_enabled(False)
        b=backend(None if arm=='N' else SEVEN/'formal'/f'{arm}-seed{seed}'/'adapter')
        torch.cuda.reset_peak_memory_stats()
        for item in target:
            if item['item_id'] in done:continue
            assert time.time()<end and shutil.disk_usage(OUT).free>=5*1024**3
            start=time.monotonic()
            with torch.inference_mode():
                x=inputs(b.processor,item);assert_runtime_grid(x,item)
                assert x['input_ids'].shape[1]==item['frozen_prompt_tokens']
                y=b.model.generate(**x,max_new_tokens=128,do_sample=False,logits_processor=finite_logits_processors())
                raw=b.processor.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip()
            assert torch.cuda.max_memory_reserved()<=30e9
            row={'item_id':item['item_id'],'axis':'grounding','seconds':time.monotonic()-start,**score(raw,item)}
            with ledger.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            done.add(item['item_id'])
    rows=read_rows(ledger);assert len(rows)==len(done)==len(target)
    for item,row in zip(target,rows):
        assert item['item_id']==row['item_id']
        assert all(score(row['raw_generation'],item)[k]==row[k] for k in ('correct_bit','parsed_answer','parsed_processor_pixels','parse_fail_flag','iou'))
    result={'rows':len(rows),'accuracy':sum(r['correct_bit'] for r in rows)/len(rows),
               'parse_failures':sum(r['parse_fail_flag'] for r in rows),'ledger_sha256':sha(ledger),
               'protocol_sha256':sha(OUT/'protocol.json'),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),
               'generation_seconds':sum(r['seconds'] for r in rows),'optimizer_loaded':False,'gradients_enabled':False}
    write_json(folder/('calibration.json' if limit else 'scores.json'),result)


def analyze():
    import numpy as np
    from report_route_b_multiplicity import intervals
    manifest=prepare();groups=sorted({r['image_group'] for r in manifest});positions={g:i for i,g in enumerate(groups)}
    counts=np.zeros(len(groups));sums=np.zeros((len(groups),16));scores={}
    for j,(arm,seed) in enumerate(MODELS):
        folder=OUT/'models'/f'{arm}-seed{seed}';marker=read_json(folder/'scores.json')
        assert marker['rows']==2400 and marker['ledger_sha256']==sha(folder/'eval.jsonl')
        rows=units(manifest,read_rows(folder/'eval.jsonl'))
        for row in rows:
            sums[positions[row['image_group']],j]+=row['correct_bit']
            if j==0:counts[positions[row['image_group']]]+=1
        scores[folder.name]=marker
    rng=np.random.default_rng(2026101704);seed_indices=rng.integers(0,5,size=(10000,5));samples=np.empty((10000,16))
    for start in range(0,10000,40):
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=min(40,10000-start))
        samples[start:start+len(weights)]=(weights@sums)/(weights@counts)[:,None]
    point=sums.sum(0)/counts.sum();p={'N':point[0]};draws={'N':samples[:,0]}
    for i,arm in enumerate(('A1','A2','U')):
        p[arm]=point[1+i::3].mean();draws[arm]=samples[np.arange(10000)[:,None],1+3*seed_indices+i].mean(1)
    contrasts={a+'-'+b:{'difference':float(p[a]-p[b]),**intervals(draws[a]-draws[b],family_size=4)} for a,b in (('A1','A2'),('A1','U'),('A1','N'),('U','N'))}
    result={'status':'all16_grounding_evaluations_complete','scores':scores,'paired_contrasts':contrasts,
            'U_minus_N_point':float(p['U']-p['N']),'mean_scores':{a:float(v) for a,v in p.items()},
            'family_size':4,'bootstrap_replicates':10000,'bootstrap_seed':2026101704,'coordinate_contract':read_json(OUT/'protocol.json')['coordinate_contract']}
    write_json(OUT/'results.json',result)
    lines=['# 7B统一处理器像素复测','','这是结果出现后的测量修订；不覆盖原12项检验。全部模型提示明确处理器宽高，统一换回原图计分，无按臂或GT选择单位。','','| 模型 | 定位分 | 解析失败 | reserved GB |','|---|---:|---:|---:|']
    for name,row in scores.items():lines.append(f'| {name} | {100*row["accuracy"]:.3f}% | {row["parse_failures"]} | {row["reserved_peak_bytes"]/1e9:.3f} |')
    lines+=['','| 对比 | 差值pp | 4项校正区间pp |','|---|---:|---|']
    for name,row in contrasts.items():lines.append(f'| {name} | {100*row["difference"]:.3f} | '+str([round(100*v,3) for v in row['bonferroni_familywise_ci95']])+' |')
    lines+=['',f'U−N点差为{100*result["U_minus_N_point"]:.3f}pp；与其余三组对比合并为独立4项校正族，旧12项检验保持原样。','', '即使统一提示，输出格式遵循与空间定位仍可能相互影响；不把事后修订称为原ARP确认性结论。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def publish_checked(folder,token):
    from audit_7b_grounding_contracts import publish_checked as publish
    return publish(folder,token)


def run(token):
    import fcntl
    prepare()
    with (OUT/'manager.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registration=OUT/'registration';registration.mkdir(exist_ok=True)
        for source in [OUT/'protocol.json',OUT/'manifest.jsonl',OUT/'processor-dimensions.json']+[ROOT/'tools'/n for n in SOURCES]:shutil.copyfile(source,registration/source.name)
        if not (registration/'delivery-proof.json').exists():publish_checked(registration,token)
        while not (NINE/'minimal-acceptance.json').exists():
            write_json(OUT/'queue-status.json',{'phase':'waiting_all9b_training_and9b_common_contract_calibration','gpu_used':False,'timestamp':datetime.now(timezone.utc).isoformat()})
            time.sleep(45)
        assert read_json(NINE/'minimal-acceptance.json')['status']=='passed'
        budget=OUT/'gpu-wall-budget.json'
        if not budget.exists():
            start=time.time();write_json(budget,{'start_utc':datetime.fromtimestamp(start,timezone.utc).isoformat(),
                'end_utc':datetime.fromtimestamp(start+12*3600,timezone.utc).isoformat(),'never_reset_on_restart':True})
        def child(arm,seed,limit=None):
            remaining=datetime.fromisoformat(read_json(budget)['end_utc']).timestamp()-time.time();assert remaining>0
            folder=OUT/'models'/f'{arm}-seed{seed}'
            path=folder/'eval.jsonl'
            if path.exists():
                raw=path.read_bytes()
                if raw and not raw.endswith(b'\n'):
                    boundary=raw.rfind(b'\n')+1;path.with_name('eval.incomplete-tail-'+str(time.time_ns())).write_bytes(raw[boundary:])
                    with path.open('r+b') as stream:stream.truncate(boundary)
            command=[str(ENV),'-u',str(Path(__file__)),'eval','--arm',arm,'--seed',str(seed)]
            if limit is not None:command+=['--limit',str(limit)]
            subprocess.run(command,cwd=ROOT,check=True,timeout=remaining)
        calibration=OUT/'models/N-seed42/calibration.json'
        if not calibration.exists():child('N',42,100)
        proof=read_json(calibration);assert proof['accuracy']>=.80 and proof['parse_failures']<=5,'7B fixed100-row common-contract calibration failed; never tune/retry for a score'
        for arm,seed in MODELS:
            folder=OUT/'models'/f'{arm}-seed{seed}'
            if not (folder/'scores.json').exists():child(arm,seed)
            if not (folder/'delivery-proof.json').exists():publish_checked(folder,token)
        analyze();receipt=publish_checked(OUT,token)
        complete={'status':'all16_processor_pixel_grounding_verified_and_delivered','protocol_sha256':sha(OUT/'protocol.json'),
                  'models':16,'rows_per_model':2400,'timestamp':datetime.now(timezone.utc).isoformat(),'receipt':receipt}
        proof=OUT/'terminal-proof';proof.mkdir(exist_ok=True);write_json(proof/'complete.json',complete);publish_checked(proof,token)
        write_json(OUT/'complete.json',complete)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','start','run','eval']);p.add_argument('--arm',default='N');p.add_argument('--seed',type=int,default=42);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.action=='prepare':print('Prepared CPU-only',len(prepare()))
    elif a.action=='eval':evaluate(a.arm,a.seed,a.limit)
    else:
        token=os.environ.get('GH_TOKEN') or sys.stdin.readline().strip();assert token
        if a.action=='start':
            command='cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/run_7b_processor_grounding.py run >> runs/7b-processor-pixel-audit-console.log 2>&1'
            subprocess.run(['screen','-dmS','7b-processor-pixel-audit','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
            print('7B common-contract38400-row remeasurement queued after9B calibration, before9B full evaluation')
        else:
            try:run(token)
            except Exception as error:write_json(OUT/'stopped.json',{'type':type(error).__name__,'message':str(error),'timestamp':datetime.now(timezone.utc).isoformat()});raise
