"""Explicit native Qwen2.5 processed-pixel grounding calibration only."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from experiment_common import ROOT,read_json,read_rows,sha,write_json
from coordinate_contracts import score_processed_grounding

SOURCE=ROOT/'runs/protocol-v2-data/clean-role-manifests-v4'
OUT=ROOT/'runs/protocol-v2/calibration/processed-pixel-grounding-v1'
PROMPT=('Locate the object referred to by: {expression}. Return only one JSON array '
        '[x_min, y_min, x_max, y_max] in absolute pixel coordinates on the processed image. '
        'The processed image shown to you is {width} pixels wide and {height} pixels high. '
        'Use these processed-image pixel coordinates, not coordinates on another resolution.')


def infer(model):
    import torch
    from gpu_lock import gpu_lock
    from precision_bridge import memory_worker
    from protocol_v2_backend import QwenV2Backend,processor_inputs
    from model_backends import finite_logits_processors
    manifest=[r for r in read_rows(SOURCE/'calibration.jsonl') if r['axis']=='grounding']
    if len(manifest)!=120: raise ValueError('Native coordinate calibration must retain all 120 frozen units')
    folder=OUT/model;folder.mkdir(parents=True,exist_ok=True)
    meta={'model':model,'coordinate_contract':'processed_pixels','prompt_template':PROMPT,
          'source_manifest_sha256':sha(SOURCE/'calibration.jsonl'),'rows':120,'formal_training_allowed':False,
          'contract_source':'https://github.com/QwenLM/Qwen3-VL/issues/866',
          'implementation_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
              (Path(__file__),ROOT/'tools/protocol_v2_backend.py',ROOT/'tools/coordinate_contracts.py')}}
    if (folder/'meta.json').exists() and read_json(folder/'meta.json')!=meta:
        raise ValueError('Native coordinate protocol changed; use a new version')
    write_json(folder/'meta.json',meta)
    if (folder/'complete.json').exists():
        if read_json(folder/'complete.json')['ledger_sha256']!=sha(folder/'ledger.jsonl'): raise ValueError('Ledger changed')
        return
    previous=read_rows(folder/'ledger.jsonl') if (folder/'ledger.jsonl').exists() else []
    done={r['item_id'] for r in previous}
    if len(done)!=len(previous) or not done<={r['item_id'] for r in manifest}: raise ValueError('Invalid partial ledger')
    state={'status':'running','worker_script':Path(__file__).name,'pid':os.getpid(),'model':model,
           'expected_prompts':120,'completed_prompts':len(done),'formal_training_allowed':False}
    write_json(folder/'status.json',state)
    with gpu_lock():
        deadline=time.monotonic()+45
        while torch.cuda.mem_get_info()[0]<21*1024**3 and time.monotonic()<deadline: time.sleep(1)
        if torch.cuda.mem_get_info()[0]<21*1024**3: raise RuntimeError('GPU not idle')
        torch.cuda.set_per_process_memory_fraction(min(1.,23*1024**3/torch.cuda.mem_get_info()[1]))
        torch.cuda.reset_peak_memory_stats()
        stop=threading.Event();monitor=threading.Thread(target=memory_worker,args=(stop,folder/'nvidia-smi.jsonl',model),daemon=True);monitor.start()
        try:
            backend=QwenV2Backend(model)
            with (folder/'ledger.jsonl').open('a',encoding='utf-8') as stream:
                for item in manifest:
                    if item['item_id'] in done: continue
                    _,initial_usage=processor_inputs(backend.processor,item)
                    grid=initial_usage['image_grid_thw']
                    if len(grid)!=1 or grid[0][0]!=1: raise ValueError('Grounding expects one static image')
                    patch=backend.processor.image_processor.patch_size
                    height,width=int(grid[0][1]*patch),int(grid[0][2]*patch)
                    prompt=PROMPT.format(expression=item['referring_expression'],width=width,height=height)
                    inputs,usage=processor_inputs(backend.processor,{**item,'question':prompt})
                    if usage['image_grid_thw']!=grid: raise ValueError('Prompt rendering changed image geometry')
                    inputs={k:v.to('cuda',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k,v in inputs.items()}
                    tick=time.monotonic()
                    with torch.inference_mode():
                        output=backend.model.generate(**inputs,max_new_tokens=128,do_sample=False,num_beams=1,
                                                      logits_processor=finite_logits_processors())
                    raw=backend.processor.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                    usage.update(processed_width=width,processed_height=height,
                                 output_tokens=int(output.shape[1]-inputs['input_ids'].shape[1]))
                    row={'checkpoint':model,'item_id':item['item_id'],'axis':'grounding',
                         **score_processed_grounding(raw,item,width,height),'usage':usage,
                         'actual_prompt':prompt,'elapsed_seconds':time.monotonic()-tick}
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    done.add(item['item_id']);state.update(completed_prompts=len(done),updated_utc=datetime.now(timezone.utc).isoformat());write_json(folder/'status.json',state)
                    if len(done)%20==0: print(model,len(done),'/120',flush=True)
            rows=read_rows(folder/'ledger.jsonl');lookup={r['item_id']:r for r in manifest}
            if len(rows)!=120 or {r['item_id'] for r in rows}!=set(lookup): raise ValueError('Incomplete original-unit coverage')
            for row in rows:
                scored=score_processed_grounding(row['raw_generation'],lookup[row['item_id']],row['usage']['processed_width'],row['usage']['processed_height'])
                if any(scored[k]!=row[k] for k in ('correct_bit','parse_fail_flag','parsed_answer','iou')): raise ValueError('Score replay differs')
            strata=sorted({(r['dataset'],r.get('subset','')) for r in manifest})
            score=sum(sum(r['correct_bit'] for r in rows if (lookup[r['item_id']]['dataset'],lookup[r['item_id']].get('subset',''))==s)/sum((r['dataset'],r.get('subset',''))==s for r in manifest) for s in strata)/len(strata)
            marker={'status':'complete','rows':120,'axis_scores':{'grounding':score},
                    'parse_failures':sum(r['parse_fail_flag'] for r in rows),'ledger_sha256':sha(folder/'ledger.jsonl'),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'formal_training_allowed':False,'completed_utc':datetime.now(timezone.utc).isoformat()}
            write_json(folder/'verification.json',{'status':'passed','checks':['exact_120_unit_coverage','fixed_geometry_mapping','independent_score_replay'],'formal_training_allowed':False})
            write_json(folder/'complete.json',marker);state.update(marker)
        except Exception as error:
            state.update(status='failed',error_type=type(error).__name__);raise
        finally:
            stop.set();monitor.join(timeout=6);write_json(folder/'status.json',state)


def worker():
    OUT.mkdir(parents=True,exist_ok=True)
    state={'status':'running','worker_script':Path(__file__).name,'pid':os.getpid(),'completed_models':[],'formal_training_allowed':False}
    try:
        for model in ('qwen3b','qwen7b'):
            state.update(current_model=model,updated_utc=datetime.now(timezone.utc).isoformat());write_json(OUT/'status.json',state)
            subprocess.run([sys.executable,'-u',str(Path(__file__)),'--model',model],cwd=ROOT,check=True)
            state['completed_models'].append(model)
            from publish_artifacts import publish
            publish(OUT/model,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])
        state.update(status='complete',current_model=None,completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(OUT/'complete.json',state)
    except Exception as error: state.update(status='failed',error_type=type(error).__name__);raise
    finally: write_json(OUT/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--model',choices=['qwen3b','qwen7b']);args=parser.parse_args()
    if args.model: infer(args.model)
    elif args.worker: worker()
    else:
        token=sys.stdin.readline().strip()
        if not token: raise SystemExit('GitHub backup credential required')
        if '.arp-native-processed-grounding' in subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout: raise SystemExit('Native grounding already running')
        OUT.mkdir(parents=True,exist_ok=True)
        command=('source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/run_processed_grounding.py --worker >> '+str(OUT/'console.log')+' 2>&1')
        subprocess.run(['screen','-dmS','arp-native-processed-grounding','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Started explicit processed-pixel coordinate calibration, all 120 grounding units for both Qwen models',flush=True)
