"""Run the calibrated commercial audit node on the untouched frozen audit split."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests
from experiment_common import ROOT, AXES, read_json, read_rows, write_json, write_rows, sha, units
from axis_aggregation_v2 import summarize_units
from run_teacher_role_calibration import content, score

OUT=ROOT/'runs/protocol-v2/audit/api-qwen3-vl-plus-v1'
CAL=ROOT/'runs/protocol-v2/calibration/api-audit-qwen3-vl-plus-v1'
SOURCE=ROOT/'runs/protocol-v2-data/clean-role-manifests-v4/audit.jsonl'


def freeze():
    config=read_json(ROOT/'configs/api_audit.json')
    if config['model'] not in read_json(ROOT/'configs/api_experiment_roles.json')['audit_api_nodes']:
        raise ValueError('Audit configuration differs from assigned roles')
    acceptance=read_json(CAL/'complete.json')
    if acceptance['measurement_gate']!='allow_audit_only' or acceptance['ledger_sha256']!=sha(CAL/'ledger.jsonl') or read_json(CAL/'verification.json')['status']!='passed':
        raise ValueError('Completed calibration has not admitted this audit')
    if acceptance['model']!=config['model']: raise ValueError('Calibrated model differs')
    manifest=[]
    for item in read_rows(SOURCE):
        if item['axis']=='grounding':
            expression=item['question'].split('Locate the object referred to by: ',1)[1].split('. Return only',1)[0]
            item={**item,'question':'Locate the object referred to by: '+expression+
                  '. Return only a JSON array containing exactly one object with keys bbox_2d '
                  '([x_min, y_min, x_max, y_max]) and label. Coordinates must be normalized to '
                  '[0, 1000] relative to the full image. Do not use pixel coordinates.'}
        manifest.append(item)
    if len(manifest)!=9600: raise ValueError('Expected 9600 audit prompts')
    OUT.mkdir(parents=True,exist_ok=True)
    design={'schema':'arp.commercial-audit.v1','config':config,'prompts':9600,'calibration_ledger_sha256':acceptance['ledger_sha256'],'max_cumulative_tokens':12000000,
            'source_manifest_sha256':sha(SOURCE),'formal_training_allowed':False,
            'interpretation':'Cross-generation commercial node, not a known-size Qwen2.5 checkpoint',
            'measurement_gate':'Calibration admitted audit only; report all audit failures and truncations without prompt changes',
            'source_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tools/run_teacher_role_calibration.py',ROOT/'tools/coordinate_contracts.py',ROOT/'tools/scoring_v2.py',ROOT/'tools/axis_aggregation_v2.py']}}
    if (OUT/'design.json').exists():
        if read_json(OUT/'design.json')!=design or read_rows(OUT/'manifest.jsonl')!=manifest:
            raise ValueError('Frozen audit calibration changed')
    else:
        write_json(OUT/'design.json',design);write_rows(OUT/'manifest.jsonl',manifest)
    return config,manifest


def worker():
    import fcntl
    config,manifest=freeze()
    with (OUT/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'running','worker_script':Path(__file__).name,'pid':os.getpid(),
               'model':config['model'],'expected_prompts':9600,'formal_training_allowed':False}
        try:
            state.update(status='preflight',completed_prompts=0,expected_prompts=9600)
            write_json(OUT/'status.json',state)
            seen=set()
            for item in manifest:
                for rel in item.get('image_paths') or [item['image_path']]:
                    if rel not in seen:
                        content({'image_paths':[rel],'question':''});seen.add(rel)
            state['status']='running'
            write_json(OUT/'input-preflight.json',{'status':'passed','prompts':9600,'unique_image_paths':len(seen)})
            base=os.environ.get('base_url') or config['base_url']
            if base.rstrip('/')!=config['base_url'].rstrip('/'): raise ValueError('Base URL changed')
            session=requests.Session();session.trust_env=False
            session.headers['Authorization']='Bearer '+os.environ[config['api_key_env']]
            ledger=OUT/'ledger.jsonl';inflight=OUT/'inflight.json'
            rows=read_rows(ledger) if ledger.exists() else []
            ids={r['item_id'] for r in rows}
            if len(ids)!=len(rows) or not ids<={r['item_id'] for r in manifest}: raise ValueError('Invalid resumed coverage')
            if inflight.exists():
                if read_json(inflight)['item_id'] not in ids: raise RuntimeError('Unresolved paid request; do not retry')
                inflight.unlink()
            used=sum(r['usage']['total_tokens'] for r in rows)
            for item in manifest:
                if item['item_id'] in ids: continue
                if used+65536>12000000: raise RuntimeError('Frozen cumulative token budget reached')
                blocks,hashes=content(item)
                state.update(completed_prompts=len(rows),updated_utc=datetime.now(timezone.utc).isoformat())
                write_json(OUT/'status.json',state);write_json(inflight,{'item_id':item['item_id'],'started_utc':state['updated_utc']})
                tick=time.monotonic()
                response=session.post(base.rstrip('/')+'/chat/completions',json={
                    'model':config['model'],'max_tokens':config['max_tokens'],'temperature':config['temperature'],
                    'enable_thinking':config['enable_thinking'],'messages':[{'role':'user','content':blocks}]},timeout=120)
                if response.status_code!=200:
                    write_json(OUT/'http_failure.json',{'http_status':response.status_code,'item_id':item['item_id']})
                    raise RuntimeError('API failed; no paid retry')
                body=response.json();choice=body['choices'][0];raw=choice['message'].get('content') or ''
                row={'checkpoint':config['model'],'returned_model':body.get('model'),'axis':item['axis'],'item_id':item['item_id'],
                     **score(raw,item),'usage':body['usage'],'finish_reason':choice.get('finish_reason'),
                     'request_id':body.get('id'),'started_utc':state['updated_utc'],'image_hashes':hashes,
                     'actual_prompt':item['question'],'elapsed_seconds':time.monotonic()-tick}
                with ledger.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                rows.append(row);ids.add(item['item_id']);inflight.unlink()
                used+=row['usage']['total_tokens']
                state['completed_prompts']=len(rows);write_json(OUT/'status.json',state)
                if len(rows)%100==0: print(config['model'],len(rows),'/9600',flush=True)
            scored=units(manifest,rows)
            for row in rows:
                item=next(r for r in manifest if r['item_id']==row['item_id'])
                rescored=score(row['raw_generation'],item)
                if row['returned_model']!=config['model'] or any(row[k]!=rescored[k] for k in ('correct_bit','parsed_answer','parse_fail_flag','iou')):
                    raise ValueError('Model identity or independent score replay differs')
            failures={a:sum(r['parse_fail_flag'] for r in rows if r['axis']==a) for a in AXES}
            counts={a:sum(r['axis']==a for r in rows) for a in AXES}
            truncated=sum(r['finish_reason']=='length' for r in rows)
            admitted=not truncated and all(1-failures[a]/counts[a]>=.95 for a in AXES)
            marker={'status':'complete','rows':9600,'model':config['model'],'original_units':4800,
                    'axis_scores':summarize_units(scored),'parse_failures_by_axis':failures,'truncated_responses':truncated,
                    'reported_tokens':sum(r['usage']['total_tokens'] for r in rows),'ledger_sha256':sha(ledger),
                    'measurement_gate':'audit_measurement_acceptable' if admitted else 'audit_measurement_failed','formal_C_computed':False,
                    'formal_training_allowed':False,'confirmation':False}
            write_json(OUT/'verification.json',{'status':'passed','checks':['exact_coverage','model_identity','independent_score_replay']})
            write_json(OUT/'complete.json',marker);state.update(marker)
            if os.environ.get('GH_TOKEN'):
                from publish_artifacts import publish
                publish(OUT,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])
        except Exception as error:
            state.update(status='failed',error_type=type(error).__name__);raise
        finally: write_json(OUT/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker: worker()
    else:
        token=sys.stdin.readline().strip()
        if not token or not os.environ.get('DASHSCOPE_API_KEY'): raise SystemExit('API and GitHub credentials required')
        if '.arp-api-audit-full' in subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout: raise SystemExit('Calibration already running')
        freeze()
        command='cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/run_api_audit.py --worker >> '+str(OUT/'console.log')+' 2>&1'
        subprocess.run(['screen','-dmS','arp-api-audit-full','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Started frozen 9600-prompt qwen3-vl-plus audit; calibrated, 12M token cap, no training')
