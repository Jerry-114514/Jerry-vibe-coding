"""Post-hoc CPU-only coordinate diagnostic; primary pixel scores stay unchanged."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import sys

from experiment_common import ROOT, read_json, read_rows, write_json, write_rows, sha, parse_answer
from scoring import box_iou
from route_b_9b_loop import OUT, MINIMAL_MANIFEST, freeze

FOLDER = OUT / 'coordinate-diagnostic'
SOURCE = 'https://swift.readthedocs.io/en/v4.0/BestPractices/Qwen3_5-Best-Practice.html'


def register():
    path = FOLDER / 'diagnostic-plan.json'
    if path.exists():
        plan = read_json(path)
        assert plan['source_sha256'] == sha(Path(__file__))
        return plan
    plan = {'timestamp': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
            'scope': 'Post-hoc measurement diagnostic after observing N/A1 minimum pixel scores and output units',
            'primary_protocol_sha256': sha(OUT / 'protocol.json'), 'phase': 'minimal',
            'models': ['N-seed42', 'A1-seed42', 'A2-seed42', 'U-seed42'],
            'conversion': 'For every model/output, parse strictly in0..1000, then x*original_width/1000 and y*original_height/1000; never choose units from GT/IoU or per-item score',
            'iou_threshold': .5, 'extra_inference_or_training': False,
            'primary_scores_changed': False, 'documentation': SOURCE,
            'interpretation': 'Pixel primary includes instruction/coordinate adherence. Native-unit sensitivity is not a substitute primary score; U also trains pixel-target adherence.'}
    freeze(path, plan)
    return plan


def generate():
    plan = register()
    assert plan['primary_protocol_sha256'] == sha(OUT / 'protocol.json')
    from verify_9b_student_outputs import verify_evaluation
    manifest = read_rows(MINIMAL_MANIFEST)
    lookup = {r['item_id']: r for r in manifest}
    output, models, original_hashes = [], {}, {}
    for arm in ('N', 'A1', 'A2', 'U'):
        marker, _ = verify_evaluation(arm, 42, phase='minimal')
        folder = OUT / 'minimal' / f'{arm}-seed42'
        original_hashes[arm] = {name: sha(folder / name) for name in ('scores.json', 'eval.jsonl')}
        rows = [r for r in read_rows(folder / 'eval.jsonl') if r['axis'] == 'grounding']
        assert len(rows) == 100 and len({r['item_id'] for r in rows}) == 100
        correct, failures = 0, 0
        for row in rows:
            item = lookup[row['item_id']]
            box = parse_answer(row['raw_generation'], {'answer_type': 'bbox', 'width': 1000, 'height': 1000})
            pixel = [v * (item['width'] if i % 2 == 0 else item['height']) / 1000 for i, v in enumerate(box)] if box is not None else None
            iou = box_iou(pixel, item['answer_key']) if pixel is not None else None
            bit = int(iou >= .5) if iou is not None else 0
            correct += bit
            failures += int(pixel is None)
            output.append({'model': f'{arm}-seed42', 'item_id': row['item_id'],
                           'raw_generation': row['raw_generation'], 'width': item['width'], 'height': item['height'],
                           'ground_truth_pixels': item['answer_key'], 'primary_correct': row['correct_bit'],
                           'primary_parse_fail': row['parse_fail_flag'], 'fixed1000_pixels': pixel,
                           'fixed1000_iou': iou, 'fixed1000_correct': bit})
        models[arm] = {'grounding_rows': len(rows), 'primary_pixel_score': marker['axis_scores']['grounding'],
                       'primary_pixel_parse_failures': sum(r['parse_fail_flag'] for r in rows),
                       'fixed1000_diagnostic_score': correct / len(rows), 'fixed1000_parse_failures': failures,
                       'primary_ledger_sha256': marker['ledger_sha256']}
        assert models[arm]['primary_pixel_score'] == sum(r['correct_bit'] for r in rows) / 100
    write_rows(FOLDER / 'diagnostic-ledger.jsonl', output)
    result = {'status': 'all_four_minimum_models_diagnosed', 'plan_sha256': sha(FOLDER / 'diagnostic-plan.json'),
              'manifest_sha256': sha(MINIMAL_MANIFEST), 'models': models,
              'diagnostic_ledger_sha256': sha(FOLDER / 'diagnostic-ledger.jsonl'),
              'primary_scores_and_ledgers_sha256': original_hashes, 'primary_files_unchanged': True,
              'post_hoc_not_confirmatory': True, 'extra_inference_or_training': False}
    for arm, files in original_hashes.items():
        assert all(sha(OUT / 'minimal' / f'{arm}-seed42' / name) == digest for name, digest in files.items())
    write_json(FOLDER / 'diagnostic-results.json', result)
    lines = ['# 9B最小循环：定位坐标口径诊断', '',
             '已有N/A1像素定位低分触发的事后CPU诊断；没有追加推理或训练，原始像素评分/输出未更改。对所有四模型、全部100定位题一律使用同一个0–1000换算，不按答案选择坐标系。', '',
             '| 模型 | 冻结像素主分 | 像素解析失败/100 | 固定0–1000诊断分 | 诊断解析失败/100 |', '|---|---:|---:|---:|---:|']
    for arm, model in models.items():
        lines.append(f'| {arm} | {100*model["primary_pixel_score"]:.1f}% | {model["primary_pixel_parse_failures"]} | {100*model["fixed1000_diagnostic_score"]:.1f}% | {model["fixed1000_parse_failures"]} |')
    lines += ['', f'[ms-swift技术文档]({SOURCE})说明Qwen3.5原生bbox使用0–1000坐标。当前冻结提示明确要求原图像素；模型给出原生坐标会被严格像素解析器拒绝或得到错误IoU。因此像素低分同时反映坐标指令遵循，不能直接推为定位能力低。', '',
              '冻结训练目标把教师框转换为原图像素，U含定位训练，而A1/A2不含；后续像素分数变化可能包含坐标格式适应。诊断转换也可能错误解释已经遵守像素指令的回答，所以不能从两口径按题/模型取较高分，也不能用诊断替换主终点。', '',
              '四轴原结果、冻结协议、训练预算和正式入场规则保持原样。本诊断与逐题两口径记录单独归档；它不是事前确认性分析，不放行原ARP阶段C。']
    (FOLDER / 'diagnostic-report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'generate', 'publish'])
    args = parser.parse_args()
    if args.action == 'register':
        print(json.dumps(register()), flush=True)
    else:
        generate()
        if args.action == 'publish':
            from publish_artifacts import publish
            token = os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
            assert token, 'Credential only through encrypted stdin/inherited environment'
            publish(FOLDER, 'arp-9b-122b-20261017', token=token)
