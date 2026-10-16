"""Frozen two-checkpoint Qwen audit, gated on completed input calibration.

This is not the requested three-checkpoint expansion and never authorizes D.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

from experiment_common import ROOT,AXES,read_json,read_rows,write_json,write_rows,sha,units
from scoring_v2 import score_generation_v2
from coordinate_contracts import score_processed_grounding
from axis_aggregation_v2 import summarize_units

SOURCE=ROOT/'runs/protocol-v2-data/clean-role-manifests-v4'
NATIVE=ROOT/'runs/protocol-v2/calibration/processed-pixel-grounding-v1'
OUT=ROOT/'runs/protocol-v2/audit/qwen-clean-v1'


def prepare():
    if read_json(SOURCE/'verification.json')['status']!='passed': raise ValueError('Clean input verification missing')
    manifest=read_rows(SOURCE/'audit.jsonl')
    done=read_json(SOURCE/'complete.json')
    if sha(SOURCE/'audit.jsonl')!=done['manifests']['audit']['sha256']: raise ValueError('Audit manifest changed')
    design={'schema':'arp.qwen-clean-two-point-audit.v1','role':'audit_only',
            'models':['qwen3b','qwen7b'],'three_checkpoint_expansion_complete':False,
            'manifest_sha256':sha(SOURCE/'audit.jsonl'),'prompt_rows':len(manifest),
            'axis_unit_counts':{a:len({r['bootstrap_unit'] for r in manifest if r['axis']==a}) for a in AXES},
            'input_protocol':'Same labelled BF16 multi-image input and aggregate visual cap as verified calibration',
            'grounding_contract':'Explicit processed-image pixels; fixed mapping using actual patch grid',
            'aggregation':'Equal datasets, equal subsets within each dataset, then four axes kept separate',
            'native_calibration_minimum_parse_rate':.95,'calibration_control_replay':'Exact raw-output match on first grounding and largest multi-image calibration item',
            'baseline_seed':2026101609,'baseline_policy':{'pope':'always no','consistency':'always A',
                'grounding':'Original train-only per-dataset median bbox area and image-centred box in original pixels',
                'mmmu_choice':'Per-item keyed uniform choice over all actual options','mmmu_open':'Abstain, counted incorrect'},
            'baseline_statistics_sha256':sha(ROOT/'data/baseline_train_statistics.json'),
            'source_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [
                Path(__file__),ROOT/'tools/audit_input_backend.py',ROOT/'tools/protocol_v2_backend.py',
                ROOT/'tools/run_processed_grounding.py',ROOT/'tools/coordinate_contracts.py',
                ROOT/'tools/scoring_v2.py',ROOT/'tools/axis_aggregation_v2.py']},
            'confirmation_inference':False,'formal_training_allowed':False,
            'formal_gate_pending':['New audit and paired C inference','Grounding blind human review','Required larger-model access']}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'design.json').exists() and read_json(OUT/'design.json')!=design:
        raise ValueError('Frozen audit design differs; use a new version')
    write_json(OUT/'design.json',design)
    if not (OUT/'baseline.jsonl').exists():
        from scoring import center_box
        training_stats=read_json(ROOT/'data/baseline_train_statistics.json')
        rows=[]
        for item in manifest:
            if item['axis']=='pope': raw='no'
            elif item['axis']=='consistency': raw='A'
            elif item['axis']=='grounding':
                raw=json.dumps(center_box(item['width'],item['height'],training_stats[item['dataset']]['median_bbox_area_px2']))
            elif item['options']:
                seed=hashlib.sha256(('2026101609:'+item['bootstrap_unit']).encode()).hexdigest()
                raw=random.Random(seed).choice(list(item['options']))
            else: raw=''
            rows.append({'checkpoint':'degenerate-v2','item_id':item['item_id'],'axis':item['axis'],**score_generation_v2(raw,item)})
        write_rows(OUT/'baseline.jsonl',rows)
        write_json(OUT/'baseline-summary.json',{'axis_scores':summarize_units(units(manifest,rows)),
                   'ledger_sha256':sha(OUT/'baseline.jsonl'),'training_allowed':False})
    return manifest


def wait_calibration(state):
    path=NATIVE/'status.json'
    while True:
        source=read_json(path)
        if source['status']=='complete': break
        if source['status']=='failed': raise RuntimeError('Native coordinate calibration failed')
        pid=source.get('pid')
        if not pid or not Path('/proc/'+str(pid)).exists(): raise RuntimeError('Native calibration process missing')
        state.update(status='waiting',waiting_for_pid=pid,updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(OUT/'status.json',state);time.sleep(20)
    for model in ('qwen3b','qwen7b'):
        done=read_json(NATIVE/model/'complete.json')
        if done['rows']!=120 or done['ledger_sha256']!=sha(NATIVE/model/'ledger.jsonl'):
            raise ValueError('Native calibration completion is not verified')
        if read_json(NATIVE/model/'verification.json')['status']!='passed': raise ValueError('Native score verification failed')
        if 1-done['parse_failures']/120 < .95:
            write_json(OUT/'measurement-gate.json',{'decision':'stop','model':model,'reason':'Native grounding parse rate below frozen .95 minimum','formal_training_allowed':False})
            return False
    write_json(OUT/'measurement-gate.json',{'decision':'allow_audit_only','native_parse_rate_threshold':.95,
               'human_grounding_review_complete':False,'formal_training_allowed':False})
    return True


def score(raw,item,usage):
    if item['axis']=='grounding': return score_processed_grounding(raw,item,usage['processed_width'],usage['processed_height'])
    return score_generation_v2(raw,item)


def infer(model):
    import torch
    from gpu_lock import gpu_lock
    from precision_bridge import memory_worker
    from audit_input_backend import AuditBackend
    manifest=prepare();folder=OUT/model;folder.mkdir(parents=True,exist_ok=True)
    metadata={'model':model,'design_sha256':sha(OUT/'design.json'),'manifest_sha256':sha(SOURCE/'audit.jsonl'),
              'model_revision':read_json(ROOT/'configs/models.lock.json')['models'][model]['revision']}
    if (folder/'meta.json').exists() and read_json(folder/'meta.json')!=metadata: raise ValueError('Audit metadata changed')
    write_json(folder/'meta.json',metadata)
    if (folder/'complete.json').exists():
        if read_json(folder/'complete.json')['ledger_sha256']!=sha(folder/'ledger.jsonl'): raise ValueError('Complete audit changed')
        return
    previous=read_rows(folder/'ledger.jsonl') if (folder/'ledger.jsonl').exists() else []
    done={r['item_id'] for r in previous}
    if len(done)!=len(previous) or not done<={r['item_id'] for r in manifest}: raise ValueError('Audit resume coverage invalid')
    state={'status':'running','worker_script':Path(__file__).name,'pid':os.getpid(),'model':model,
           'expected_prompts':len(manifest),'completed_prompts':len(done),'formal_training_allowed':False}
    write_json(folder/'status.json',state)
    with gpu_lock():
        deadline=time.monotonic()+45
        while torch.cuda.mem_get_info()[0]<21*1024**3 and time.monotonic()<deadline: time.sleep(1)
        if torch.cuda.mem_get_info()[0]<21*1024**3: raise RuntimeError('GPU not otherwise idle')
        torch.cuda.set_per_process_memory_fraction(min(1.,23*1024**3/torch.cuda.mem_get_info()[1]));torch.cuda.reset_peak_memory_stats()
        stop=threading.Event();monitor=threading.Thread(target=memory_worker,args=(stop,folder/'nvidia-smi.jsonl',model),daemon=True);monitor.start()
        try:
            backend=AuditBackend(model)
            cal=read_rows(SOURCE/'calibration.jsonl')
            grounding=next(r for r in cal if r['axis']=='grounding')
            multi=max(cal,key=lambda r:len(r['image_paths']))
            expected_native={r['item_id']:r for r in read_rows(NATIVE/model/'ledger.jsonl')}
            expected_general={r['item_id']:r for r in read_rows(ROOT/'runs/protocol-v2/calibration/clean-inputs-v5'/model/'ledger.jsonl')}
            controls=[]
            for item,expected in [(grounding,expected_native[grounding['item_id']]),(multi,expected_general[multi['item_id']])]:
                raw=backend.generate(item)
                if raw!=expected['raw_generation']: raise ValueError('Audit backend failed exact calibration raw-output replay')
                controls.append({'item_id':item['item_id'],'exact_raw_match':True})
            write_json(folder/'calibration-replay.json',{'status':'passed','controls':controls})
            with (folder/'ledger.jsonl').open('a',encoding='utf-8') as stream:
                for item in manifest:
                    if item['item_id'] in done: continue
                    tick=time.monotonic();raw=backend.generate(item)
                    row={'checkpoint':model,'item_id':item['item_id'],'axis':item['axis'],
                         **score(raw,item,backend.last_usage),'usage':backend.last_usage,'elapsed_seconds':time.monotonic()-tick}
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();done.add(item['item_id'])
                    state.update(completed_prompts=len(done),updated_utc=datetime.now(timezone.utc).isoformat());write_json(folder/'status.json',state)
                    if len(done)%100==0: print(model,len(done),'/',len(manifest),flush=True)
            rows=read_rows(folder/'ledger.jsonl');scored_units=units(manifest,rows);lookup={r['item_id']:r for r in manifest}
            for row in rows:
                actual=score(row['raw_generation'],lookup[row['item_id']],row['usage'])
                if any(actual[k]!=row[k] for k in ('correct_bit','parse_fail_flag','parsed_answer','iou')): raise ValueError('Independent audit score replay differs')
            marker={'status':'complete','rows':len(rows),'original_units':len(scored_units),
                    'axis_scores':summarize_units(scored_units),'ledger_sha256':sha(folder/'ledger.jsonl'),
                    'manifest_sha256':metadata['manifest_sha256'],'formal_training_allowed':False,
                    'parse_failures_by_axis':{a:sum(r['parse_fail_flag'] for r in scored_units if r['axis']==a) for a in AXES},
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'completed_utc':datetime.now(timezone.utc).isoformat()}
            write_json(folder/'verification.json',{'status':'passed','checks':['exact_unique_coverage','independent_rescore','fixed_coordinate_contract','pre-audit_calibration_replay'],'training_allowed':False})
            write_json(folder/'complete.json',marker);state.update(marker)
        except Exception as error: state.update(status='failed',error_type=type(error).__name__);raise
        finally: stop.set();monitor.join(timeout=6);write_json(folder/'status.json',state)


def worker():
    prepare();state={'status':'waiting','worker_script':Path(__file__).name,'pid':os.getpid(),'completed_models':[],'formal_training_allowed':False}
    try:
        if not wait_calibration(state):
            state.update(status='stopped_at_measurement_gate');return
        state['status']='running'
        for model in ('qwen3b','qwen7b'):
            state.update(current_model=model,updated_utc=datetime.now(timezone.utc).isoformat());write_json(OUT/'status.json',state)
            subprocess.run([sys.executable,'-u',str(Path(__file__)),'--model',model],cwd=ROOT,check=True)
            state['completed_models'].append(model)
            from publish_artifacts import publish
            publish(OUT/model,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])
        state.update(status='complete',current_model=None,completed_utc=datetime.now(timezone.utc).isoformat());write_json(OUT/'complete.json',state)
    except Exception as error: state.update(status='failed',error_type=type(error).__name__);raise
    finally: write_json(OUT/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--model',choices=['qwen3b','qwen7b']);args=parser.parse_args()
    if args.model: infer(args.model)
    elif args.worker: worker()
    else:
        token=sys.stdin.readline().strip()
        if not token: raise SystemExit('GitHub backup credential required')
        if '.arp-qwen-clean-audit' in subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout: raise SystemExit('Audit worker already running')
        prepare()
        command=('source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/run_clean_audit.py --worker >> '+str(OUT/'console.log')+' 2>&1')
        subprocess.run(['screen','-dmS','arp-qwen-clean-audit','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Started frozen Qwen audit queue, admitted only after native calibration; no confirmation inference or training',flush=True)
