"""Small remote API capability probe. Never persist credentials or request headers."""
import base64
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageDraw


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
    model = config['model']
    result['configured_model_listed'] = model in models
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
    if result.get('visual_probe_passed'):
        image = Image.new('RGB', (256, 256), 'white')
        ImageDraw.Draw(image).rectangle((64, 64, 191, 191), fill='red')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
        structured = {'model': model, 'temperature': 0, 'max_tokens': 128, 'enable_thinking': False,
                      'response_format': {'type': 'json_object'},
                      'messages': [{'role': 'user', 'content': [
                          {'type': 'image_url', 'image_url': {'url': image_url}},
                          {'type': 'text', 'text': 'Locate the red rectangle. Return only a JSON object '
                           'with keys color (one English word) and bbox ([x_min, y_min, x_max, y_max]). '
                           'Coordinates must be normalized to [0, 1000] relative to the full image.'}]}]}
        native_box = config.get('grounding_output_schema') == 'bbox_2d_singleton'
        if native_box:
            structured.pop('response_format')
            structured['messages'][0]['content'][1]['text'] = (
                'Locate the red rectangle. Return only a JSON array containing exactly one object '
                'with keys bbox_2d ([x_min, y_min, x_max, y_max]) and label. '
                'Normalize all coordinates to [0, 1000] relative to the full image.')
        response = session.post(base.rstrip('/') + '/chat/completions', json=structured, timeout=120)
        check = {'http_status': response.status_code, 'coordinate_contract': 'normalized_1000',
                 'passed': False, 'automatic_retry': False}
        if response.status_code == 200:
            body = response.json()
            choice = body['choices'][0]
            raw = choice['message'].get('content')
            check.update(raw_generation=raw, returned_model=body.get('model'), usage=body.get('usage'),
                         request_id=body.get('id'), finish_reason=choice.get('finish_reason'))
            try:
                if native_box:
                    from experiment_common import parse_answer
                    box = parse_answer(raw, {'answer_type': 'bbox', 'width': 1000, 'height': 1000})
                    if box is None:
                        raise ValueError('Invalid native box schema')
                else:
                    answer = json.loads(raw)
                    box = answer['bbox']
                valid = (len(box) == 4 and all(type(x) in (int, float) and 0 <= x <= 1000 for x in box)
                         and box[0] < box[2] and box[1] < box[3])
                if valid:
                    intersection = max(0, min(box[2], 750) - max(box[0], 250)) * max(0, min(box[3], 750) - max(box[1], 250))
                    union = (box[2]-box[0])*(box[3]-box[1]) + 500*500 - intersection
                    check['iou'] = intersection / union
                    check['passed'] = check['iou'] >= .5 and (native_box or answer['color'].lower() == 'red')
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                check['parse_failed'] = True
        else:
            check['error_code'] = response.json().get('error', {}).get('code')
        result['structured_grounding_probe'] = check
    directory = root / 'runs/protocol-v2/api-validation'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('probe-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ARP_API_CHECK=' + json.dumps({'report_path': str(path), **result}, ensure_ascii=False))
    if not result.get('visual_probe_passed') or not result.get('structured_grounding_probe', {}).get('passed'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
