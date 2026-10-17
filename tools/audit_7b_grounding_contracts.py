"""Separate post-result coordinate audit; never overwrite the frozen7B trial."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT,read_json,read_rows,write_json,write_rows,sha,parse_answer,score_generation,units
from route_b_staged import OUT as SEVEN,SEEDS,freeze
from evaluate_9b_native import native_item,score_native
from scoring import box_iou

OUT=ROOT/'runs/7b-grounding-contract-audit-20261017'
MODELS=[('N',42)]+[(a,s) for s in SEEDS for a in ('A1','A2','U')]
TAG='arp-protocol-v2-20261016'
ENV=Path('/root/autodl-tmp/arp-venv/bin/python')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert sha(ROOT/'tools/route_b_staged.py')==read_json(SEVEN/'protocol.json')['source_sha256']
    original=[r for r in read_rows(ROOT/'data/manifest.jsonl') if r['axis']=='grounding']
    assert len(original)==len({r['item_id'] for r in original})==2400
    native=[native_item(r) for r in original]
    manifest=OUT/'normalized1000-manifest.jsonl'
    if not manifest.exists():write_rows(manifest,native)
    assert read_rows(manifest)==native
    plan={'scope':'User-required post-result measurement audit; not preregistered ability evidence',
          'models':[f'{a}-seed{s}' for a,s in MODELS],'rows_per_model':2400,'new_generations':38400,
          'cpu_contracts':['original_image_pixels','normalized1000','processor_resized_pixels'],
          'no_per_item_or_per_arm_best_contract_selection':True,'no_training_or_teacher_calls':True,
          'native_generations':'Explicit0..1000 prompt for every model, then original pixel IoU>=.5',
          'interpretation':'Qwen2.5-VL defaults to absolute pixels; normalized prompt is instruction/coordinate sensitivity, not its native contract',
          'official_source':'https://qwenlm.github.io/blog/qwen2.5-vl/',
          'original_engine_sha256':sha(ROOT/'tools/route_b_staged.py'),
          'source_sha256':sha(Path(__file__)),'normalized_manifest_sha256':sha(manifest),
          'original_manifest_sha256':sha(ROOT/'data/manifest.jsonl'),
          'gpu_queue':'After9B full native evaluation and final delivery, using global GPU lock; CPU audit starts now',
          'gpu_wall_hours_cap':12,'max_new_tokens':128,'max_visual_tokens':1280,'max_total_tokens':2048,
          'statistical_family':'3 grounding-only paired contrasts A1-A2,A1-U,A1-N,10000 paired seed/image bootstrap,Bonferroni3',
          'bootstrap_seed':2026101704,'old12_comparison_family_unchanged':True,
          'stops':['nonfinite generation logits','OOM','reserved>30GB','input truncation','SHA mismatch','free disk<5GiB','new audit wall budget exhausted']}
    freeze(OUT/'protocol.json',plan)
    return original,native


def resized_score(raw,item,width,height):
    parsed=parse_answer(raw,{'answer_type':'bbox','width':width,'height':height})
    pixel=[v*(item['width']/width if i%2==0 else item['height']/height) for i,v in enumerate(parsed)] if parsed is not None else None
    overlap=box_iou(pixel,item['answer_key']) if pixel is not None else None
    return {'parsed_answer':pixel,'parse_fail_flag':int(parsed is None),'correct_bit':int(overlap>=.5) if overlap is not None else 0,'iou':overlap}


def preserve_tail(path):
    assert path.resolve().is_relative_to(OUT.resolve())
    if not path.exists():return
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        boundary=raw.rfind(b'\n')+1
        path.with_name(path.name+'.incomplete-tail-'+str(time.time_ns())).write_bytes(raw[boundary:])
        with path.open('r+b') as stream:stream.truncate(boundary)


def cpu_audit():
    original,_=prepare()
    import torch
    torch.set_num_threads(2)
    from transformers import AutoProcessor
    from PIL import Image
    processor=AutoProcessor.from_pretrained(str(ROOT/'.model_cache/qwen7b'),local_files_only=True,use_fast=False,
                 min_pixels=256*28*28,max_pixels=1280*28*28)
    dims_path=OUT/'processor-dimensions.json'
    dims=read_json(dims_path) if dims_path.exists() else {}
    for item in original:
        key=item['image_path']
        if key in dims:continue
        with Image.open(ROOT/key) as image:
            assert image.size==(item['width'],item['height'])
            x=processor.image_processor(images=[image.convert('RGB')],return_tensors='pt',
                 size={'shortest_edge':256*28*28,'longest_edge':1280*28*28},
                 min_pixels=256*28*28,max_pixels=1280*28*28)
        grid=x['image_grid_thw'][0].tolist();patch=processor.image_processor.patch_size
        dims[key]={'width':grid[2]*patch,'height':grid[1]*patch,'grid_thw':grid,'patch_size':patch}
        if len(dims)%100==0:write_json(dims_path,dims)
    write_json(dims_path,dims)
    summary={};latency=[]
    for arm,seed in MODELS:
        source=SEVEN/'formal'/f'{arm}-seed{seed}'
        assert read_json(source/'scores.json')['ledger_sha256']==sha(source/'eval.jsonl')
        rows={r['item_id']:r for r in read_rows(source/'eval.jsonl') if r['axis']=='grounding'}
        assert set(rows)=={r['item_id'] for r in original}
        evidence=[]
        for item in original:
            old=rows[item['item_id']];raw=old['raw_generation'];d=dims[item['image_path']]
            pixel=score_generation(raw,item)
            assert all(old[k]==pixel[k] for k in ('correct_bit','parsed_answer','parse_fail_flag','iou'))
            contracts={'original_image_pixels':pixel,'normalized1000':score_native(raw,item),
                       'processor_resized_pixels':resized_score(raw,item,d['width'],d['height'])}
            evidence.append({'item_id':item['item_id'],'raw_generation':raw,'original_size':[item['width'],item['height']],
                             'processed_size':[d['width'],d['height']],'contracts':contracts})
            latency.append(old['seconds'])
        folder=OUT/'cpu'/source.name;folder.mkdir(parents=True,exist_ok=True);write_rows(folder/'scores.jsonl',evidence)
        summary[source.name]={c:{'accuracy':sum(r['contracts'][c]['correct_bit'] for r in evidence)/2400,
                               'parse_failures':sum(r['contracts'][c]['parse_fail_flag'] for r in evidence)}
                              for c in ('original_image_pixels','normalized1000','processor_resized_pixels')}
        write_json(folder/'provenance.json',{'old_raw_sha256':sha(source/'eval.jsonl'),'rows':2400,
                   'posthoc_diagnostic_only':True,'no_best_contract_selection':True})
    result={'status':'passed','models':16,'rows_per_model':2400,'contracts':summary,
            'processed_size_differs':sum((dims[r['image_path']]['width'],dims[r['image_path']]['height'])!=(r['width'],r['height']) for r in original),
            'mean_old_grounding_seconds':sum(latency)/len(latency),'estimated38400_pure_seconds':sum(latency),
            'interpretation':'Alternative rescoring is sensitivity diagnosis, not corrected primary scores; new explicit-contract generations pending'}
    write_json(OUT/'cpu-audit.json',result)
    lines=['# 7B定位坐标审计：CPU阶段','','同一批旧输出按三种预先列明的契约分别诊断，禁止挑选较高分。此表不替换原报告，也不把统一千分比当作Qwen2.5-VL原生坐标。',
           '', '| 模型 | 原图像素 | 0–1000诊断 | 处理器像素诊断 |','|---|---:|---:|---:|']
    for model,contracts in summary.items():lines.append('| '+model+' | '+' | '.join(f'{100*contracts[c]["accuracy"]:.3f}%' for c in ('original_image_pixels','normalized1000','processor_resized_pixels'))+' |')
    lines+=['',f'2400题中{result["processed_size_differs"]}题的处理器网格尺寸不同于原图；这本身不能证明模型遵循哪一个坐标提示。',
            f'旧定位累计生成{sum(latency)/3600:.2f}小时；新明确0–1000提示可能改变回答长度，不能保证相同时延。',
            '', 'Qwen官方说明Qwen2.5-VL使用绝对坐标：https://qwenlm.github.io/blog/qwen2.5-vl/ 。必须区分原图像素、处理器像素和归一化坐标。格式适配是否解释U增益，仍须结合新固定提示复测；不能按旧输出中最高诊断分下结论。']
    (OUT/'cpu-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('CPU coordinate audit complete; normalized prompt GPU remeasurement remains pending',flush=True)


def evaluate(arm,seed):
    _,manifest=prepare()
    folder=OUT/'normalized'/f'{arm}-seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    ledger=folder/'eval.jsonl';previous=read_rows(ledger) if ledger.exists() else []
    done={r['item_id'] for r in previous};assert len(done)==len(previous) and done<={r['item_id'] for r in manifest}
    budget=read_json(OUT/'gpu-wall-budget.json');end=datetime.fromisoformat(budget['end_utc']).timestamp()
    from gpu_lock import gpu_lock
    import torch
    from route_b_staged import backend,inputs
    from model_backends import finite_logits_processors
    with gpu_lock():
        torch.set_num_threads(2);torch.set_grad_enabled(False)
        b=backend(None if arm=='N' else SEVEN/'formal'/f'{arm}-seed{seed}'/'adapter')
        torch.cuda.reset_peak_memory_stats()
        for item in manifest:
            if item['item_id'] in done:continue
            assert time.time()<end and shutil.disk_usage(OUT).free>=5*1024**3
            start=time.monotonic()
            with torch.inference_mode():
                x=inputs(b.processor,item)
                y=b.model.generate(**x,max_new_tokens=128,do_sample=False,logits_processor=finite_logits_processors())
                raw=b.processor.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip()
            assert torch.cuda.max_memory_reserved()<=30e9
            row={'item_id':item['item_id'],'axis':'grounding','seconds':time.monotonic()-start,**score_native(raw,item)}
            with ledger.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            done.add(item['item_id'])
    rows=read_rows(ledger);assert len(rows)==len(done)==2400
    for item,row in zip(manifest,rows):
        assert item['item_id']==row['item_id']
        assert all(score_native(row['raw_generation'],item)[k]==row[k] for k in ('correct_bit','parsed_answer','parse_fail_flag','iou'))
    write_json(folder/'scores.json',{'rows':2400,'accuracy':sum(r['correct_bit'] for r in rows)/2400,
               'parse_failures':sum(r['parse_fail_flag'] for r in rows),'ledger_sha256':sha(ledger),
               'protocol_sha256':sha(OUT/'protocol.json'),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),
               'generation_seconds':sum(r['seconds'] for r in rows),'optimizer_loaded':False,'gradients_enabled':False})


def analyze():
    import numpy as np
    from report_route_b_multiplicity import intervals
    original,_=prepare()
    groups=sorted({r['image_group'] for r in units(original,[{'item_id':r['item_id'],'axis':'grounding','correct_bit':0,'parse_fail_flag':0} for r in original])})
    pos={g:i for i,g in enumerate(groups)};counts=np.zeros(len(groups));sums=np.zeros((len(groups),16))
    scores={}
    for j,(arm,seed) in enumerate(MODELS):
        folder=OUT/'normalized'/f'{arm}-seed{seed}';marker=read_json(folder/'scores.json')
        assert marker['ledger_sha256']==sha(folder/'eval.jsonl')
        ledger=read_rows(folder/'eval.jsonl');rows=units(original,ledger)
        for row in rows:
            sums[pos[row['image_group']],j]+=row['correct_bit']
            if j==0:counts[pos[row['image_group']]]+=1
        scores[folder.name]=marker
    rng=np.random.default_rng(2026101704);seeds=rng.integers(0,5,size=(10000,5));samples=np.empty((10000,16))
    for start in range(0,10000,40):
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=min(40,10000-start))
        samples[start:start+len(weights)]=(weights@sums)/(weights@counts)[:,None]
    points=sums.sum(0)/counts.sum();p={'N':points[0]};draws={'N':samples[:,0]}
    for i,arm in enumerate(('A1','A2','U')):
        p[arm]=points[1+i::3].mean();draws[arm]=samples[np.arange(10000)[:,None],1+3*seeds+i].mean(1)
    contrasts={a+'-'+b:{'difference':float(p[a]-p[b]),**intervals(draws[a]-draws[b],family_size=3)} for a,b in (('A1','A2'),('A1','U'),('A1','N'))}
    write_json(OUT/'normalized-results.json',{'status':'all16_grounding_evaluations_complete','scores':scores,'paired_contrasts':contrasts,
               'family_size':3,'bootstrap_replicates':10000,'bootstrap_seed':2026101704,'post_result_sensitivity_not_original12_test_family':True})


def publish_checked(folder,token):
    from publish_artifacts import publish
    from resilient_9b_delivery import network_retry
    from finish_7b_safely import verify_figure_receipt
    receipt=network_retry(lambda:publish(folder,TAG,token=token),'7b-ground-upload:'+folder.name)
    proof=network_retry(lambda:verify_figure_receipt(receipt,token),'7b-ground-server-sha:'+folder.name)
    write_json(folder/'delivery-proof.json',{'status':'passed','receipt':receipt,'fresh_server_check':proof})
    return receipt


def run(token):
    import fcntl
    prepare()
    with (OUT/'manager.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registration=OUT/'registration';registration.mkdir(exist_ok=True)
        for source in (OUT/'protocol.json',OUT/'normalized1000-manifest.jsonl',Path(__file__)):shutil.copyfile(source,registration/source.name)
        if not (registration/'delivery-proof.json').exists():publish_checked(registration,token)
        if not (OUT/'cpu-audit.json').exists():cpu_audit()
        publish_checked(OUT,token)
        gate=ROOT/'runs/route-b-9b-122b-20261017/student-experiment/native-evaluation-v2/finalization/complete.json'
        while not gate.exists():
            write_json(OUT/'queue-status.json',{'phase':'cpu_audit_complete_waiting9b_verified_delivery','no_gpu_used':True,'timestamp':datetime.now(timezone.utc).isoformat()})
            time.sleep(45)
        assert read_json(gate)['status']=='full_native9b_content_and_delivery_verified'
        budget=OUT/'gpu-wall-budget.json'
        if not budget.exists():
            start=time.time();write_json(budget,{'start_utc':datetime.fromtimestamp(start,timezone.utc).isoformat(),
                'end_utc':datetime.fromtimestamp(start+12*3600,timezone.utc).isoformat(),'never_reset_on_restart':True})
        for arm,seed in MODELS:
            folder=OUT/'normalized'/f'{arm}-seed{seed}'
            if not (folder/'scores.json').exists():
                remaining=datetime.fromisoformat(read_json(budget)['end_utc']).timestamp()-time.time();assert remaining>0
                preserve_tail(folder/'eval.jsonl')
                subprocess.run([str(ENV),'-u',str(Path(__file__)),'eval','--arm',arm,'--seed',str(seed)],cwd=ROOT,check=True,timeout=remaining)
            if not (folder/'delivery-proof.json').exists():publish_checked(folder,token)
        analyze();receipt=publish_checked(OUT,token)
        write_json(OUT/'complete.json',{'status':'all16_grounding_audit_and_release_complete','timestamp':datetime.now(timezone.utc).isoformat()})
        terminal=OUT/'terminal-proof';terminal.mkdir(exist_ok=True)
        shutil.copyfile(OUT/'complete.json',terminal/'complete.json')
        write_json(terminal/'final-receipt.json',receipt);publish_checked(terminal,token)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','run','cpu','eval']);parser.add_argument('--arm',default='N');parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.action=='cpu':cpu_audit()
    elif args.action=='eval':evaluate(args.arm,args.seed)
    else:
        token=os.environ.get('GH_TOKEN') or sys.stdin.readline().strip();assert token
        if args.action=='start':
            command='cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/audit_7b_grounding_contracts.py run >> runs/7b-grounding-contract-audit-console.log 2>&1'
            subprocess.run(['screen','-dmS','7b-grounding-audit','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
            print('Mandatory7B coordinate audit launched on CPU;38400 normalized generations queued after9B verified delivery')
        else:
            try:run(token)
            except Exception as error:
                write_json(OUT/'stopped.json',{'type':type(error).__name__,'message':str(error),'timestamp':datetime.now(timezone.utc).isoformat()});raise
