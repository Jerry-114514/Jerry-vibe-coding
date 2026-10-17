"""Observe completed9B runs, audit content and fresh Release coverage; no GPU."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT, AXES, read_json, sha, write_json
from route_b_9b_loop import OUT, SEEDS

FINAL = OUT / 'finalization'
CONTRASTS = ('A1-A2', 'A1-U', 'A1-N')


def script_live(name, action=None):
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if argv and b'python' in Path(os.fsdecode(argv[0])).name.encode():
            arguments = [os.fsdecode(a) for a in argv[1:] if a]
            if any(Path(a).name == name for a in arguments) and (action is None or action in arguments):
                return True
    return False


def readiness():
    models = [('N', 42)] + [(a, s) for s in SEEDS for a in ('A1', 'A2', 'U')]
    required = [OUT / 'status.json', OUT / 'minimal-acceptance.json', OUT / 'formal-plan.json',
                OUT / 'formal-independent-verification.json', OUT / 'formal-report/evaluation-results.json']
    required += [OUT / 'formal' / f'{a}-seed{s}' / 'scores.json' for a, s in models]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    phase = read_json(OUT / 'status.json').get('phase') if (OUT / 'status.json').exists() else None
    live = script_live('run_9b_formal_when_ready.py')
    return {'ready': not missing and phase == 'all9b_formal_stages_complete' and not live,
            'missing': missing, 'formal_controller_live': live, 'phase': phase}


def validate_statistics(data, plan, evaluations):
    expected = {'N-seed42'} | {f'{a}-seed{s}' for s in SEEDS for a in ('A1', 'A2', 'U')}
    assert data['status'] == 'all16_verified' and set(data['scores']) == expected
    assert data['matched_seeds'] == SEEDS
    assert data['family_size'] == plan['family_size'] == 12
    assert data['bootstrap_replicates'] == plan['bootstrap_replicates'] == 10000
    assert data['bootstrap_seed'] == plan['bootstrap_seed'] == 2026101703
    assert set(data['paired_contrasts']) == set(CONTRASTS)
    for (arm, seed), (marker, _) in evaluations.items():
        key = f'{arm}-seed{seed}'
        assert data['scores'][key] == marker['axis_scores']
        assert data['ledger_sha256'][key] == marker['ledger_sha256']
    conclusions = []
    for label in CONTRASTS:
        left, right = label.split('-')
        assert set(data['paired_contrasts'][label]) == set(AXES)
        for axis, row in data['paired_contrasts'][label].items():
            delta = sum(data['scores'][f'{left}-seed{s}'][axis] -
                        data['scores'][f'{right}-seed{42 if right == "N" else s}'][axis] for s in SEEDS) / 5
            assert abs(delta - row['difference']) < 1e-12
            lo, hi = row['pointwise_ci95']
            low, high = row['bonferroni_familywise_ci95']
            assert all(math.isfinite(v) for v in (lo, hi, low, high, delta))
            assert -1 <= low <= lo <= hi <= high <= 1
            conclusions.append({'contrast': label, 'axis': axis, 'difference': delta,
                                'adjusted_interval': [low, high],
                                'direction': 'left_higher' if low > 0 else 'right_higher' if high < 0 else 'includes_zero'})
    return conclusions


def summary(data, conclusions):
    lines = ['# 独立9B+122B：最终分数与交付证据', '',
             '15次固定1000更新训练、16模型各16900提示均经独立核验。以下为全部五seed，未按分数选择检查点或追加预算。', '',
             '| 模型 | pope | grounding | consistency | mmstar |', '|---|---:|---:|---:|---:|']
    names = ['N-seed42'] + [f'{a}-seed{s}' for s in SEEDS for a in ('A1', 'A2', 'U')]
    for name in names:
        lines.append('| ' + name + ' | ' + ' | '.join(f'{data["scores"][name][a]*100:.3f}%' for a in AXES) + ' |')
    lines += ['', '| 对比 | 轴 | 差值（百分点） | 校正区间（百分点） | 区间结论 |', '|---|---|---:|---|---|']
    labels = {'left_higher': '左臂更高', 'right_higher': '右臂更高', 'includes_zero': '包含0，证据不足'}
    for row in conclusions:
        low, high = row['adjusted_interval']
        lines.append(f'| {row["contrast"]} | {row["axis"]} | {100*row["difference"]:.3f} | [{100*low:.3f}, {100*high:.3f}] | {labels[row["direction"]]} |')
    lines += ['', '区间使用已登记的10000次五seed/来源图像簇配对bootstrap及12项Bonferroni校正，族覆盖为名义近似；有限seed与尾部分位数存在不确定性。N为唯一固定未训练基线，不是五次独立训练。', '',
              '仅为开发基准上的独立9B+122B扩展，原ARP阶段C保持未放行。与7B实验同时改变学生、教师、预处理和可用池，不能单独归因为模型容量。包含0不证明相等，不虚构p值。', '',
              '训练曲线、逐步损失/梯度/显存/耗时、断点恢复证据、原始评测与统计均按Release清单归档；已经腾挪的完成任务恢复状态须按恢复指针从已核验资产取回。']
    return '\n'.join(lines) + '\n'


def finalize(token):
    assert readiness()['ready'], 'All formal work and final metadata delivery must finish first'
    from verify_9b_student_outputs import verify_formal
    from verify_9b_teacher_delivery import verify as verify_teacher
    from audit_route_b_completion import existing_code_unchanged
    from run_9b_formal_when_ready import metadata_backup, verify_receipt
    from audit_9b_student_delivery import audit
    from publish_artifacts import publish
    _, evaluations = verify_formal(return_evaluations=True)
    teacher = verify_teacher(token)
    plan = read_json(OUT / 'formal-plan.json')
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / 'tools' / name) == digest
    data = read_json(OUT / 'formal-report/evaluation-results.json')
    conclusions = validate_statistics(data, plan, evaluations)
    minimal = read_json(OUT / 'minimal-acceptance.json')
    assert minimal['status'] == 'passed' and len(minimal['five_checks']) == 5 and all(minimal['five_checks'].values())
    assert read_json(ROOT / 'runs/protocol-v2/paired-gate-development/gate.json')['training_allowed'] is False
    proof_folder = FINAL / 'verified-evidence'
    proof_folder.mkdir(parents=True, exist_ok=True)
    content = {'status': 'passed', 'training_runs': 15, 'updates_per_run': 1000,
               'evaluation_models': 16, 'prompts_per_model': 16900, 'matched_seeds': SEEDS,
               'family_size': 12, 'teacher_eligible_rows': teacher['eligible_rows'],
               'original_arp_gate_unpassed': True, 'original_python_sources_unchanged': existing_code_unchanged(),
               'source_sha256': sha(Path(__file__)), 'statistics_sha256': sha(OUT / 'formal-report/evaluation-results.json'),
               'formal_plan_sha256': sha(OUT / 'formal-plan.json'), 'conclusions': conclusions}
    write_json(proof_folder / 'content-verification.json', content)
    (proof_folder / 'final-summary.md').write_text(summary(data, conclusions), encoding='utf-8')
    metadata_backup(OUT, token)
    delivery = audit(token, strict=True)
    assert delivery['status'] == 'passed_delivery_coverage'
    shutil.copyfile(OUT / 'final-delivery-verification.json', proof_folder / 'delivery-verification.json')
    receipt = publish(proof_folder, 'arp-9b-122b-20261017', token=token)
    server = verify_receipt(receipt, token)
    result = {'status': 'full_9b_content_and_delivery_verified', 'timestamp': datetime.now(timezone.utc).isoformat(),
              'verified_files': delivery['verified_files'], 'required_files': delivery['required_files'],
              'evidence_delivery': receipt, 'evidence_server_verification': server,
              'not_original_arp': True, 'no_training_or_inference_started': True}
    write_json(FINAL / 'complete.json', result)
    print(json.dumps(result), flush=True)


def watch(token):
    import fcntl
    FINAL.mkdir(parents=True, exist_ok=True)
    with (FINAL / 'observer.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (FINAL / 'complete.json').exists():
            state = readiness()
            write_json(FINAL / 'queue-status.json', state)
            if state['ready']:
                finalize(token)
                return
            if (OUT / 'formal-manager-stopped.json').exists() or (OUT / 'minimal-manager-stopped.json').exists():
                raise RuntimeError('Student controller stopped; preserve logs and inspect before restart')
            time.sleep(45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'once', 'watch', 'start'])
    args = parser.parse_args()
    if args.action == 'status':
        print(json.dumps(readiness()), flush=True)
    else:
        token = os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
        assert token, 'Credential required only through encrypted stdin or inherited environment'
        if args.action == 'start':
            assert not script_live('finalize_9b_delivery.py', 'watch'), 'Observer already running'
            command = ('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u '
                       'tools/finalize_9b_delivery.py watch >> '
                       'runs/route-b-9b-122b-20261017/student-experiment/finalization-console.log 2>&1')
            subprocess.run(['screen', '-dmS', 'route-b-9b-finalization', 'bash', '-c', command],
                           env={**os.environ, 'GH_TOKEN': token}, check=True)
            print('9B final content and fresh server delivery audit queued; no GPU or budget changes')
        else:
            try:
                watch(token) if args.action == 'watch' else finalize(token)
            except Exception as error:
                write_json(FINAL / 'stopped.json', {'type': type(error).__name__, 'message': str(error),
                           'timestamp': datetime.now(timezone.utc).isoformat(), 'completion_unproven': True})
                raise
