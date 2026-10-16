"""GPU-bounded calibration on frozen clean roles; never reads confirmation answers."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from experiment_common import ROOT, AXES, read_json, read_rows, sha, units, write_json
from scoring_v2 import score_generation_v2

SOURCE = ROOT/'runs/protocol-v2-data/clean-role-manifests-v4'
OUT = ROOT/'runs/protocol-v2/calibration/clean-inputs-v5'


def verify(folder):
    manifest = read_rows(SOURCE/'calibration.jsonl')
    rows = read_rows(folder/'ledger.jsonl')
    scored = units(manifest,rows)
    by_id = {row['item_id']:row for row in manifest}
    for row in rows:
        actual = score_generation_v2(row['raw_generation'],by_id[row['item_id']])
        for key in ('parsed_answer','correct_bit','parse_fail_flag','iou'):
            if actual[key] != row[key]: raise ValueError('Independent score replay differs')
        if row['usage']['image_count'] != len(by_id[row['item_id']]['image_paths']):
            raise ValueError('An ordered source image was dropped')
        if row['usage']['visual_tokens'] > 1280: raise ValueError('Visual token cap exceeded')
        if row['usage']['output_tokens'] > 128: raise ValueError('Output cap exceeded')
    result = {'status':'passed','rows':len(rows),'original_units':len(scored),
              'ledger_sha256':sha(folder/'ledger.jsonl'),'formal_training_allowed':False,
              'checks':['exact_unique_coverage','independent_rescore','all_ordered_images_present','visual_and_output_caps']}
    write_json(folder/'verification.json',result)
    return scored


def infer(model):
    import torch
    from gpu_lock import gpu_lock
    from precision_bridge import memory_worker
    from protocol_v2_backend import QwenV2Backend, BACKEND_VERSION
    folder = OUT/model
    folder.mkdir(parents=True,exist_ok=True)
    manifest = read_rows(SOURCE/'calibration.jsonl')
    source = read_json(SOURCE/'complete.json')
    if sha(SOURCE/'calibration.jsonl') != source['manifests']['calibration']['sha256']:
        raise ValueError('Calibration manifest digest differs')
    metadata = {'model':model,'backend':BACKEND_VERSION,'manifest_sha256':sha(SOURCE/'calibration.jsonl'),
                'role':'calibration_only','formal_training_allowed':False,
                'implementation_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
                    (ROOT/'tools/protocol_v2_backend.py',ROOT/'tools/scoring_v2.py',ROOT/'tools/vendor/mmmu_eval_utils.py')},
                'max_visual_tokens':1280,'max_output_tokens':128,'dtype':'bfloat16','coordinate_contract':'pixels'}
    if (folder/'meta.json').exists() and read_json(folder/'meta.json') != metadata:
        raise ValueError('Calibration implementation changed; use a new version')
    write_json(folder/'meta.json',metadata)
    if (folder/'complete.json').exists():
        if read_json(folder/'complete.json')['ledger_sha256'] != sha(folder/'ledger.jsonl'):
            raise ValueError('Completed ledger changed')
        verify(folder)
        return
    previous = read_rows(folder/'ledger.jsonl') if (folder/'ledger.jsonl').exists() else []
    done = {row['item_id'] for row in previous}
    if len(done) != len(previous) or not done <= {row['item_id'] for row in manifest}:
        raise ValueError('Invalid partial coverage')
    state = {'status':'running','worker_script':Path(__file__).name,'pid':os.getpid(),'model':model,
             'expected_prompts':len(manifest),'completed_prompts':len(done),'formal_training_allowed':False}
    write_json(folder/'status.json',state)
    with gpu_lock():
        deadline = time.monotonic()+45
        while torch.cuda.mem_get_info()[0] < 21*1024**3 and time.monotonic() < deadline: time.sleep(1)
        free,total = torch.cuda.mem_get_info()
        if free < 21*1024**3: raise RuntimeError('Calibration requires an otherwise idle GPU')
        torch.cuda.set_per_process_memory_fraction(min(1.0,23*1024**3/total))
        torch.cuda.reset_peak_memory_stats()
        stop = threading.Event()
        monitor = threading.Thread(target=memory_worker,args=(stop,folder/'nvidia-smi.jsonl',model),daemon=True)
        monitor.start()
        try:
            backend = QwenV2Backend(model)
            with (folder/'ledger.jsonl').open('a',encoding='utf-8') as stream:
                for item in manifest:
                    if item['item_id'] in done: continue
                    tick = time.monotonic()
                    raw = backend.generate(item)
                    row = {'checkpoint':model,'item_id':item['item_id'],'axis':item['axis'],
                           **score_generation_v2(raw,item),'usage':backend.last_usage,
                           'elapsed_seconds':time.monotonic()-tick}
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    done.add(item['item_id'])
                    state.update(completed_prompts=len(done),updated_utc=datetime.now(timezone.utc).isoformat())
                    write_json(folder/'status.json',state)
                    if len(done)%25==0: print(model,len(done),'/',len(manifest),flush=True)
            scored = verify(folder)
            scores = {}
            for axis in AXES:
                strata = sorted({row['stratum'] for row in scored if row['axis']==axis})
                scores[axis] = sum(sum(row['correct_bit'] for row in scored if row['axis']==axis and row['stratum']==s)/sum(row['axis']==axis and row['stratum']==s for row in scored) for s in strata)/len(strata)
            done_marker = {'status':'complete','rows':len(done),'original_units':len(scored),'axis_scores':scores,
                           'parse_failures_by_axis':{a:sum(row['parse_fail_flag'] for row in scored if row['axis']==a) for a in AXES},
                           'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                           'ledger_sha256':sha(folder/'ledger.jsonl'),'manifest_sha256':metadata['manifest_sha256'],
                           'formal_training_allowed':False,'completed_utc':datetime.now(timezone.utc).isoformat()}
            write_json(folder/'complete.json',done_marker)
            state.update(done_marker)
        except Exception as error:
            state.update(status='failed',error_type=type(error).__name__)
            raise
        finally:
            stop.set();monitor.join(timeout=6)
            write_json(folder/'status.json',state)


def worker():
    OUT.mkdir(parents=True,exist_ok=True)
    state={'status':'running','worker_script':Path(__file__).name,'pid':os.getpid(),'completed_models':[],
           'formal_training_allowed':False,'scope':'Clean-input development calibration; no audit/confirmation inference'}
    try:
        for model in ('qwen3b','qwen7b'):
            state.update(current_model=model,updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(OUT/'status.json',state)
            subprocess.run([sys.executable,'-u',str(ROOT/'tools/run_clean_calibration.py'),'--model',model],cwd=ROOT,check=True)
            state['completed_models'].append(model)
            if os.environ.get('GH_TOKEN'):
                from publish_artifacts import publish
                publish(OUT/model,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])
        state.update(status='complete',current_model=None,completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(OUT/'complete.json',state)
    except Exception as error:
        state.update(status='failed',error_type=type(error).__name__)
        raise
    finally: write_json(OUT/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--model',choices=['qwen3b','qwen7b']);args=parser.parse_args()
    if args.model: infer(args.model)
    elif args.worker: worker()
    else:
        token=sys.stdin.readline().strip()
        if not token: raise SystemExit('Completion-backup credential required')
        if '.arp-v5-clean-calibration' in subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout:
            raise SystemExit('Calibration worker already running')
        OUT.mkdir(parents=True,exist_ok=True)
        command=('source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; '
                 'exec /root/autodl-tmp/arp-venv/bin/python -u tools/run_clean_calibration.py --worker >> '+str(OUT/'console.log')+' 2>&1')
        subprocess.run(['screen','-dmS','arp-v5-clean-calibration','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Started serial Qwen 3B/7B clean calibration, 960 prompts each; completion verification and GitHub backup enabled',flush=True)
