"""Probe exact 3B/7B API access before any purported numerical bridge."""
import base64
from datetime import datetime, timezone
import io
import json
import os

import requests
from PIL import Image
from experiment_common import ROOT, read_json, write_json, sha


def main():
    design=read_json(ROOT/'configs/precision_bridge.json')
    endpoint=read_json(ROOT/'configs/api_audit.json')
    base=os.environ.get('base_url') or endpoint['base_url']
    if base.rstrip('/')!=endpoint['base_url'].rstrip('/'): raise ValueError('Base URL differs')
    locked=read_json(ROOT/'configs/models.lock.json')['models']
    for key,model in design['models'].items():
        if locked[key]['revision']!=model['local_revision'] or locked[key]['repo_id']!=model['repo_id']:
            raise ValueError('Local checkpoint changed')
    session=requests.Session();session.trust_env=False
    session.headers['Authorization']='Bearer '+os.environ[endpoint['api_key_env']]
    buffer=io.BytesIO();Image.new('RGB',(128,128),'red').save(buffer,format='PNG')
    url='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
    results=[]
    for key,model in design['models'].items():
        row={'local_model':key,'requested_model':model['api_model'],'local_revision':model['local_revision'],
             'synthetic_image_only':True,'automatic_retry':False,'exact_model_access_verified':False}
        try:
            response=session.post(base.rstrip('/')+'/chat/completions',json={
                'model':model['api_model'],'temperature':0,'max_tokens':8,'messages':[{'role':'user','content':[
                    {'type':'image_url','image_url':{'url':url}},
                    {'type':'text','text':'What is the dominant color? Answer with one English word.'}]}]},timeout=120)
            row['http_status']=response.status_code
            body=response.json()
            if response.status_code==200:
                raw=body['choices'][0]['message'].get('content') or ''
                row.update(returned_model=body.get('model'),raw_generation=raw,usage=body.get('usage'),
                           exact_model_access_verified=body.get('model')==model['api_model'] and raw.strip().lower().rstrip('.')=='red')
            else: row['error_code']=body.get('error',{}).get('code')
        except Exception as error: row['error_type']=type(error).__name__
        results.append(row)
    ready=all(row['exact_model_access_verified'] for row in results)
    report={'checked_utc':datetime.now(timezone.utc).isoformat(),'purpose':'same_checkpoint_precision_bridge_access',
            'design_sha256':sha(ROOT/'configs/precision_bridge.json'),'models':results,
            'status':'ready_for_same_item_inference' if ready else 'blocked_exact_model_api_access',
            'four_axis_comparison_completed':False,'bridge_passed':False,
            'merge_local_and_api_into_ladder_allowed':False,'formal_training_allowed':False,
            'qwen3_vl_plus_is_not_a_3b_or_7b_precision_bridge':True}
    path=ROOT/'runs/protocol-v2/calibration/same-checkpoint-bridge/access-check.json'
    write_json(path,report)
    print('ARP_BRIDGE='+json.dumps(report,ensure_ascii=False))


if __name__=='__main__': main()
