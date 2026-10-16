"""Small remote API capability probe. Never persist credentials or request headers."""
import base64
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/api_teacher.json').read_text(encoding='utf-8'))
    key = os.environ.get(config['api_key_env'])
    if not key:
        raise SystemExit('API key missing from process environment')
    base = os.environ.get('base_url') or os.environ.get('DASHSCOPE_BASE_URL') or config['base_url']
    if base.rstrip('/') != config['base_url'].rstrip('/'):
        raise SystemExit('Environment Base URL differs from approved configuration')
    session = requests.Session()
    session.trust_env = False
    session.headers['Authorization'] = 'Bearer ' + key
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'base_url': base,
              'purpose': 'synthetic_visual_capability_probe', 'formal_training_allowed': False}
    response = session.get(base.rstrip('/') + '/models', timeout=45)
    result['models_http_status'] = response.status_code
    if response.status_code != 200:
        raise SystemExit('Model-list authentication failed: HTTP ' + str(response.status_code))
    models = [row['id'] for row in response.json().get('data', [])]
    result['listed_model_count'] = len(models)
    result['requested_checkpoint_models_listed'] = {
        name: name in models for name in ('qwen2.5-vl-32b-instruct', 'qwen2.5-vl-72b-instruct')}
    model = 'qwen3-vl-plus'
    if model not in models:
        raise SystemExit('Visual probe model not listed')
    result['requested_model'] = model
    result['immutable_model_version_confirmed'] = False
    buffer = io.BytesIO()
    Image.new('RGB', (128, 128), (255, 0, 0)).save(buffer, format='PNG')
    image_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    payload = {'model': model, 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': image_url}},
        {'type': 'text', 'text': 'What is the dominant color in this image? Answer with one English word.'}]}],
        'temperature': 0, 'max_tokens': 32, 'enable_thinking': False,
        'logprobs': True, 'top_logprobs': 5}
    result['attempts'] = []
    for use_logprobs in (True, False):
        if not use_logprobs:
            payload.pop('logprobs', None)
            payload.pop('top_logprobs', None)
        response = session.post(base.rstrip('/') + '/chat/completions', json=payload, timeout=120)
        body = response.json()
        attempt = {'requested_logprobs': use_logprobs, 'http_status': response.status_code}
        if response.status_code != 200:
            error = body.get('error', {})
            attempt['error_code'] = error.get('code')
            result['attempts'].append(attempt)
            if response.status_code == 400 and use_logprobs:
                continue
            break
        result['attempts'].append(attempt)
        choice = body['choices'][0]
        result.update({'returned_model': body.get('model'), 'request_id': body.get('id'),
                       'answer': choice['message'].get('content'), 'usage': body.get('usage'),
                       'finish_reason': choice.get('finish_reason')})
        content = (choice.get('logprobs') or {}).get('content') or []
        result['logprobs_returned'] = bool(content)
        result['max_returned_top_logprobs'] = max((len(t.get('top_logprobs', [])) for t in content), default=0)
        result['visual_probe_passed'] = (result['answer'] or '').strip().lower().rstrip('.') == 'red'
        result['original_top50_kd_compatible'] = result['max_returned_top_logprobs'] >= 50
        break
    directory = root / 'runs/protocol-v2/api-validation'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('probe-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ARP_API_CHECK=' + json.dumps({'report_path': str(path), **result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
