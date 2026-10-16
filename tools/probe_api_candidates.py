"""Bounded synthetic-image access checks for named large Qwen API teachers."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path

import requests
from PIL import Image, ImageDraw


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/api_teacher.json').read_text(encoding='utf-8'))
    base = os.environ.get('base_url') or config['base_url']
    if base.rstrip('/') != config['base_url'].rstrip('/'):
        raise ValueError('Base URL differs from configuration')
    key = os.environ[config['api_key_env']]
    image = Image.new('RGB', (256, 256), 'white')
    ImageDraw.Draw(image).rectangle((64,64,191,191),fill='red')
    buffer = io.BytesIO()
    image.save(buffer,format='PNG')
    image_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    models = ['qwen3.5-397b-a17b','qwen3.5-122b-a10b','qwen3.5-35b-a3b','qwen3.6-35b-a3b']

    def probe(model):
        result = {'requested_model':model,'automatic_retry':False,'coordinate_contract':'normalized_1000'}
        session = requests.Session()
        session.trust_env = False
        session.headers['Authorization'] = 'Bearer '+key
        payload = {'model':model,'temperature':0,'max_tokens':128,'enable_thinking':False,
                   'response_format':{'type':'json_object'},'messages':[{'role':'user','content':[
                       {'type':'image_url','image_url':{'url':image_url}},
                       {'type':'text','text':'Locate the red rectangle. Return a JSON object with keys color '
                        '(one English word) and bbox ([x_min, y_min, x_max, y_max]). Normalize all coordinates '
                        'to [0, 1000] relative to the full image. Output JSON only.'}]}]}
        try:
            response = session.post(base.rstrip('/')+'/chat/completions',json=payload,timeout=120)
            result['http_status'] = response.status_code
            body = response.json()
            if response.status_code != 200:
                result['error_code'] = body.get('error',{}).get('code')
                return result
            choice = body['choices'][0]
            result.update(returned_model=body.get('model'),usage=body.get('usage'),
                          raw_generation=choice['message'].get('content'),finish_reason=choice.get('finish_reason'))
            result['structured_grounding_passed'] = False
            try:
                answer = json.loads(result['raw_generation'])
                box = answer['bbox']
                valid = (len(box)==4 and all(type(x) in (int,float) and 0<=x<=1000 for x in box)
                         and box[0]<box[2] and box[1]<box[3])
                if valid:
                    intersection = max(0,min(box[2],750)-max(box[0],250))*max(0,min(box[3],750)-max(box[1],250))
                    result['iou'] = intersection/((box[2]-box[0])*(box[3]-box[1])+250000-intersection)
                    result['structured_grounding_passed'] = answer['color'].lower()=='red' and result['iou']>=.5
            except (ValueError,TypeError,KeyError):
                result['parse_failed'] = True
        except Exception as error:
            result['error_type'] = type(error).__name__
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(probe,models))
    report = {'checked_utc':datetime.now(timezone.utc).isoformat(),'purpose':'synthetic_large_teacher_access',
              'max_requests':4,'max_output_tokens_per_request':128,'formal_training_allowed':False,
              'top_logprobs_capability':'not_tested','models':results}
    path = root/'runs/protocol-v2/api-validation'/('large-teachers-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ARP_API_CHECK='+json.dumps({'report_path':str(path),**report},ensure_ascii=False))


if __name__ == '__main__':
    main()
