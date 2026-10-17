"""CPU-only final verification/figures after the frozen 7B audit completes."""
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT, read_json, read_rows, sha, write_json

OUT = ROOT / 'runs/7b-processor-pixel-audit-20261017'
DEST = OUT / 'final-report'


def overlap(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/area if area > 0 else 0.


def verify():
    completion = read_json(OUT / 'complete.json')
    assert completion['status'] == 'all16_processor_pixel_grounding_verified_and_delivered'
    plan = read_json(OUT / 'protocol.json')
    assert completion['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert plan['family_size'] == 4 and plan['total_generations'] == 38400
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / 'tools' / name) == digest
    manifest = read_rows(OUT / 'manifest.jsonl')
    assert len(manifest) == 2400 and sha(OUT / 'manifest.jsonl') == plan['manifest_sha256']
    result = read_json(OUT / 'results.json')
    proofs = []
    for name in plan['models']:
        folder = OUT / 'models' / name
        marker = read_json(folder / 'scores.json')
        assert (folder / 'delivery-proof.json').exists()
        ledger = read_rows(folder / 'eval.jsonl')
        assert len(ledger) == len({row['item_id'] for row in ledger}) == 2400
        assert sha(folder / 'eval.jsonl') == marker['ledger_sha256']
        assert marker['protocol_sha256'] == sha(OUT / 'protocol.json')
        assert marker['optimizer_loaded'] is False and marker['gradients_enabled'] is False
        assert result['scores'][name] == marker
        for item, row in zip(manifest, ledger):
            assert item['item_id'] == row['item_id'] and row['axis'] == 'grounding'
            box = row['parsed_processor_pixels']
            if box is None:
                assert row['parse_fail_flag'] == 1 and row['correct_bit'] == 0
                assert row['parsed_answer'] is None and row['iou'] is None
                continue
            assert len(box) == 4 and all(math.isfinite(x) for x in box)
            assert 0 <= box[0] < box[2] <= item['processor_width']
            assert 0 <= box[1] < box[3] <= item['processor_height']
            mapped = [v * item['width' if i % 2 == 0 else 'height'] /
                      item['processor_width' if i % 2 == 0 else 'processor_height'] for i, v in enumerate(box)]
            assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(mapped, row['parsed_answer']))
            iou = overlap(mapped, item['answer_key'])
            assert math.isclose(iou, row['iou'], abs_tol=1e-9)
            assert row['parse_fail_flag'] == 0 and row['correct_bit'] == int(iou >= .5)
        assert marker['accuracy'] == sum(row['correct_bit'] for row in ledger)/2400
        assert marker['parse_failures'] == sum(row['parse_fail_flag'] for row in ledger)
        proofs.append({'model': name, 'rows': 2400, 'raw_sha256': marker['ledger_sha256']})
    assert result['family_size'] == 4 and result['bootstrap_replicates'] == 10000
    assert set(result['paired_contrasts']) == {'A1-A2', 'A1-U', 'A1-N', 'U-N'}
    return result, {'status': 'passed', 'rows_verified': 38400, 'models': proofs,
                    'independent_mapping_and_iou': True, 'optimizer_loaded': False,
                    'new_predictions': 0, 'source_sha256': sha(Path(__file__)),
                    'protocol_sha256': sha(OUT / 'protocol.json'),
                    'result_sha256': sha(OUT / 'results.json'),
                    'timestamp_utc': datetime.now(timezone.utc).isoformat()}


def create_figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(result['scores'])
    palette = {'N': '#666666', 'A1': '#2878b5', 'A2': '#df8f44', 'U': '#35a07b'}
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].bar(range(16), [100*result['scores'][n]['accuracy'] for n in names],
                color=[palette[n.split('-')[0]] for n in names])
    axes[0].set_xticks(range(16), names, rotation=65, ha='right')
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Grounding accuracy (%)')
    axes[0].set_title('7B: explicit processor-pixel contract, 2,400 items per model')
    comparisons = list(result['paired_contrasts'])
    for i, name in enumerate(comparisons):
        row = result['paired_contrasts'][name]
        low, high = [100*v for v in row['bonferroni_familywise_ci95']]
        point = 100*row['difference']
        axes[1].plot([low, high], [i, i], color='#2878b5', linewidth=3)
        axes[1].plot(point, i, 'o', color='#222222')
    axes[1].axvline(0, color='#777777', linestyle='--')
    axes[1].set_yticks(range(4), comparisons)
    axes[1].set_xlabel('Paired difference (percentage points)')
    axes[1].set_title('Five seeds; 10,000 paired seed/image-cluster bootstrap draws; Bonferroni family=4')
    fig.savefig(DEST / 'grounding-final.png', dpi=170)
    fig.savefig(DEST / 'grounding-final.pdf')
    plt.close(fig)


def finish(token):
    import fcntl
    OUT.mkdir(exist_ok=True)
    with (OUT / 'figure-finalizer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (DEST / 'complete.json').exists():
            return
        result, proof = verify()
        DEST.mkdir(exist_ok=True)
        create_figures(result)
        for filename in ('protocol.json', 'results.json', 'report.md', 'schema-amendment.json'):
            shutil.copyfile(OUT / filename, DEST / filename)
        write_json(DEST / 'verification.json', proof)
        with (DEST / 'report.md').open('a', encoding='utf-8') as stream:
            stream.write('\n![16模型定位分数与四项校正区间](grounding-final.png)\n\n'
                         '独立CPU复核全部38,400行的身份、固定坐标映射、IoU、汇总及归档标记；不新增推理或训练。\n')
        from audit_7b_grounding_contracts import publish_checked
        receipt = publish_checked(DEST, token)
        write_json(DEST / 'complete.json', {'status': 'verified_figures_and_report_delivered',
                   'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'receipt': receipt})
        print('7B final grounding figures independently verified and delivered', flush=True)


if __name__ == '__main__':
    token = os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
    assert token
    if '--start' in sys.argv:
        command = ('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u '
                   'tools/finish_7b_processor_report.py >> runs/7b-processor-figure-finalizer.log 2>&1')
        subprocess.run(['screen', '-dmS', '7b-processor-figures', 'bash', '-c', command],
                       env={**os.environ, 'GH_TOKEN': token}, check=True)
        print('CPU-only final figure task queued after all16 delivered models')
    else:
        try:
            while not (OUT / 'complete.json').exists():
                if (OUT / 'stopped.json').exists():
                    raise RuntimeError('Numerical audit stopped; final report cannot claim completion')
                time.sleep(45)
            finish(token)
        except Exception as error:
            write_json(OUT / 'figure-finalizer-stopped.json', {'type': type(error).__name__,
                       'message': str(error), 'timestamp_utc': datetime.now(timezone.utc).isoformat()})
            raise
