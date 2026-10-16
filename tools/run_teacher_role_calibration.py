"""Bounded, frozen calibration for the assigned API teacher/control and audit access.

Uses public calibration images only. Never reads confirmation or training data.
An ambiguous paid request stops resume; no automatic request retries.
"""
import argparse
import base64
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests
from PIL import Image
from experiment_common import ROOT, AXES, read_json, read_rows, write_json, write_rows, sha, units
from coordinate_contracts import score_normalized_grounding
from scoring_v2 import score_generation_v2
from axis_aggregation_v2 import summarize_units

OUT = ROOT/'runs/protocol-v2/calibration/teacher-roles-v3'
SOURCE = ROOT/'runs/protocol-v2-data/clean-role-manifests-v4/calibration.jsonl'


def rank(value):
    return hashlib.sha256(('20261016:teacher-roles-v1:'+value).encode()).hexdigest()


def freeze():
    roles = read_json(ROOT/'configs/api_experiment_roles.json')
    teacher = read_json(ROOT/'configs/api_teacher.json')
    if teacher['model'] != roles['teacher'] or teacher['calibration_control_model'] != roles['teacher_control']:
        raise ValueError('Teacher configuration and roles disagree')
    grouped = defaultdict(list)
    for row in read_rows(SOURCE):
        grouped[(row['axis'],row['bootstrap_unit'])].append(row)
    selected = []
    for axis in AXES:
        strata = defaultdict(list)
        for (a,unit), rows in grouped.items():
            if a == axis:
                strata[(rows[0]['dataset'],rows[0].get('subset',''))].append((unit,rows))
        pools = [deque(sorted(strata[s],key=lambda p:rank(p[0]))) for s in sorted(strata)]
        count = 0
        while count < 6:
            for pool in pools:
                if pool and count < 6:
                    _, rows = pool.popleft()
                    selected.extend(sorted(rows,key=lambda r:r['item_id']))
                    count += 1
            if not any(pools) and count < 6:
                raise ValueError('Insufficient calibration units')
    if len(selected) != 48:
        raise ValueError('Expected 24 original units, 48 prompts')
    manifest = []
    for row in selected:
        if row['axis'] == 'grounding':
            expression = row['question'].split('Locate the object referred to by: ',1)[1].split('. Return only',1)[0]
            row = {**row,'question':'Locate the object referred to by: '+expression+
                   '. Return only a JSON array containing exactly one object with keys bbox_2d '
                   '([x_min, y_min, x_max, y_max]) and label. Coordinates must be normalized to '
                   '[0, 1000] relative to the full image. Do not use pixel coordinates.'}
        manifest.append(row)
    OUT.mkdir(parents=True,exist_ok=True)
    design = {'schema':'arp.teacher-role-calibration.v3','role':'engineering_calibration_only',
              'models':[roles['teacher'],roles['teacher_control']], 'audit_access_models':roles['audit_api_nodes'],
              'source_manifest_sha256':sha(SOURCE),'original_units_per_axis':6,'prompts_per_model':48,
              'selection':'Outcome-independent hash ranking, round-robin sorted strata',
              'max_requests':98,'max_image_pixels':4000000,'max_encoded_image_bytes':8000000,'image_policy':'Preserve original PNG unless over budget; deterministic aspect-preserving resize; record transmitted hashes and sizes','max_tokens':128,'temperature':0,'enable_thinking':False,
              'grounding_contract':'normalized_1000; explicit singleton bbox_2d wrapper; no unit inference',
              'formal_training_allowed':False,'confirmation':False,'training_cache':False,
              'note':'Small engineering subset; historical MMBench consistency calibration; not a capability ranking',
              'source_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tools/coordinate_contracts.py',ROOT/'tools/scoring_v2.py',ROOT/'tools/axis_aggregation_v2.py']}}
    if (OUT/'design.json').exists():
        if read_json(OUT/'design.json') != design or read_rows(OUT/'manifest.jsonl') != manifest:
            raise ValueError('Frozen calibration changed; use a new run version')
    else:
        write_json(OUT/'design.json',design)
        write_rows(OUT/'manifest.jsonl',manifest)
    return manifest,design,teacher


def content(item):
    images = item.get('image_paths') or [item['image_path']]
    if not 1 <= len(images) <= 7:
        raise ValueError('Unexpected image count')
    blocks, hashes = [], []
    for index,rel in enumerate(images,1):
        path = (ROOT/rel).resolve()
        if not path.is_relative_to((ROOT/'data').resolve()):
            raise ValueError('Image outside public project data')
        with Image.open(path) as im:
            if im.width*im.height > 16000000:
                raise ValueError('Source decode pixel budget exceeded')
            image = im.convert('RGB')
        source_size = list(image.size)
        if image.width*image.height > 4000000:
            scale = (4000000/(image.width*image.height))**.5
            image = image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))),Image.Resampling.LANCZOS)
        for _ in range(8):
            buffer = io.BytesIO()
            image.save(buffer,format='PNG')
            encoded = buffer.getvalue()
            if len(encoded) <= 8000000:
                break
            image = image.resize((max(1,int(image.width*.8)),max(1,int(image.height*.8))),Image.Resampling.LANCZOS)
        else:
            raise ValueError('Encoded image byte budget exceeded')
        blocks.extend([{'type':'text','text':'Image '+str(index)+':'},
                       {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(encoded).decode()}}])
        hashes.append({'path':rel,'sha256':sha(path),'source_size':source_size,'transmitted_size':list(image.size),'transmitted_png_sha256':hashlib.sha256(encoded).hexdigest()})
    blocks.append({'type':'text','text':item['question']})
    return blocks,hashes


def score(raw,item):
    return score_normalized_grounding(raw,item) if item['axis']=='grounding' else score_generation_v2(raw,item)


def worker():
    import fcntl
    manifest,design,config = freeze()
    with (OUT/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for item in manifest:
            content(item)
        write_json(OUT/'input-preflight.json',{'status':'passed','prompt_rows':len(manifest),'max_image_pixels':4000000,'max_encoded_image_bytes':8000000,'image_policy':'Preserve original PNG unless over budget; deterministic aspect-preserving resize; record transmitted hashes and sizes'})
        state = {'status':'running','worker_script':Path(__file__).name,'pid':os.getpid(),
                 'completed_models':[],'formal_training_allowed':False}
        session = requests.Session();session.trust_env=False
        session.headers['Authorization'] = 'Bearer '+os.environ[config['api_key_env']]
        base = os.environ.get('base_url') or config['base_url']
        if base.rstrip('/') != config['base_url'].rstrip('/'):
            raise ValueError('Base URL differs from frozen configuration')
        endpoint = base.rstrip('/')+'/chat/completions'
        write_json(OUT/'public-provenance.json',{'all_inputs_from_public_benchmarks':True,
                   'private_user_files_included':False,'answer_keys_sent':False,'role':'calibration_only'})
        try:
            access = OUT/'audit-access.json'
            if not access.exists():
                # Record the initial stage before paid requests so an interruption is visible.
                marker = OUT/'audit-access-inflight.json'
                if marker.exists(): raise RuntimeError('Ambiguous access probe; reconcile before resume')
                write_json(marker,{'models':design['audit_access_models'],'automatic_retry':False})
                buffer=io.BytesIO();Image.new('RGB',(128,128),'red').save(buffer,format='PNG')
                synthetic=[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()}},
                           {'type':'text','text':'What is the dominant color? Answer with one English word.'}]
                rows=[]
                for model in design['audit_access_models']:
                    response=session.post(endpoint,json={'model':model,'max_tokens':8,'temperature':0,
                                          'messages':[{'role':'user','content':synthetic}]},timeout=120)
                    body=response.json();row={'requested_model':model,'http_status':response.status_code}
                    if response.status_code==200:
                        row.update(returned_model=body.get('model'),raw_generation=body['choices'][0]['message'].get('content'),usage=body.get('usage'))
                    else: row['error_code']=body.get('error',{}).get('code')
                    rows.append(row)
                    write_json(OUT/'audit-access-partial.json',{'models':rows})
                write_json(access,{'checked_utc':datetime.now(timezone.utc).isoformat(),'models':rows,
                                  'commercial_parameter_order_verified':False,'formal_training_allowed':False})
                marker.unlink()
            for model in design['models']:
                folder=OUT/model;folder.mkdir(exist_ok=True)
                ledger=folder/'ledger.jsonl';inflight=folder/'inflight.json'
                rows=read_rows(ledger) if ledger.exists() else []
                ids={r['item_id'] for r in rows}
                if len(ids)!=len(rows) or not ids<={r['item_id'] for r in manifest}:
                    raise ValueError('Invalid resumed coverage')
                if inflight.exists():
                    if read_json(inflight)['item_id'] not in ids: raise RuntimeError('Unresolved paid request; no retry')
                    inflight.unlink()
                state.update(current_model=model,completed_prompts=len(rows),expected_prompts=48)
                for item in manifest:
                    if item['item_id'] in ids: continue
                    state['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(OUT/'status.json',state)
                    blocks,hashes=content(item)
                    write_json(inflight,{'item_id':item['item_id'],'model':model,'started_utc':state['updated_utc']})
                    tick=time.monotonic()
                    response=session.post(endpoint,json={'model':model,'max_tokens':128,'temperature':0,
                        'enable_thinking':False,'messages':[{'role':'user','content':blocks}]},timeout=120)
                    if response.status_code!=200:
                        write_json(folder/'http_failure.json',{'http_status':response.status_code,'item_id':item['item_id']})
                        raise RuntimeError('API failed; no paid retry')
                    body=response.json();choice=body['choices'][0];raw=choice['message'].get('content') or ''
                    row={'checkpoint':model,'returned_model':body.get('model'),'item_id':item['item_id'],
                         'axis':item['axis'],**score(raw,item),'usage':body.get('usage'),'request_id':body.get('id'),
                         'finish_reason':choice.get('finish_reason'),'image_hashes':hashes,
                         'actual_prompt':item['question'],'started_utc':state['updated_utc'],'elapsed_seconds':time.monotonic()-tick}
                    with ledger.open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                    rows.append(row);ids.add(item['item_id']);inflight.unlink()
                    state['completed_prompts']=len(rows)
                    write_json(OUT/'status.json',state)
                    print(model,len(rows),'/48',flush=True)
                for row,item in zip(rows,manifest):
                    if row['item_id']!=item['item_id'] or row['returned_model']!=model:
                        raise ValueError('Coverage or returned model differs')
                    rescored=score(row['raw_generation'],item)
                    if any(row[k]!=rescored[k] for k in ('correct_bit','parsed_answer','parse_fail_flag','iou')):
                        raise ValueError('Independent replay differs')
                summary={'status':'complete','model':model,'rows':len(rows),'original_units':24,
                         'axis_scores':summarize_units(units(manifest,rows)),
                         'parse_failures_by_axis':{a:sum(r['parse_fail_flag'] for r in rows if r['axis']==a) for a in AXES},
                         'truncated_responses':sum(r['finish_reason']=='length' for r in rows),
                         'reported_tokens':sum(r['usage']['total_tokens'] for r in rows),
                         'ledger_sha256':sha(ledger),'manifest_sha256':sha(OUT/'manifest.jsonl'),
                         'formal_training_allowed':False,'confirmation':False}
                write_json(folder/'verification.json',{'status':'passed','checks':['coverage','returned_model','independent_score_replay']})
                write_json(folder/'complete.json',summary)
                state['completed_models'].append(model)
            state.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
            write_json(OUT/'complete.json',state)
            if os.environ.get('GH_TOKEN'):
                from publish_artifacts import publish
                publish(OUT,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])
        except Exception as error:
            state.update(status='failed',error_type=type(error).__name__)
            raise
        finally:
            write_json(OUT/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker: worker()
    else:
        token=sys.stdin.readline().strip()
        if not token or not os.environ.get('DASHSCOPE_API_KEY'):
            raise SystemExit('API and GitHub credentials required')
        if '.arp-teacher-roles-v3' in subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout:
            raise SystemExit('Role calibration already running')
        freeze()
        command='cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/run_teacher_role_calibration.py --worker >> '+str(OUT/'console.log')+' 2>&1'
        subprocess.run(['screen','-dmS','arp-teacher-roles-v3','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Started bounded 122B teacher / 397B control calibration; audit access probes only; training gated')
