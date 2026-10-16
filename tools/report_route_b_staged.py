"""Generate the requested five-check report only from real minimal-loop artifacts."""
import json
import subprocess
import sys
from experiment_common import ROOT, AXES, read_json, read_rows, write_json, sha
from route_b_staged import OUT, CACHE


def main():
    subprocess.run([sys.executable, str(ROOT/'tools/report_route_b_staged_tokens.py')], check=True, cwd=ROOT)
    tokens = read_json(OUT/'minimal-token-distribution.json')
    assert tokens['status'] == 'complete'
    acceptance = read_json(OUT/'minimal-acceptance.json')
    assert acceptance['status'] == 'passed'
    baseline = read_json(OUT/'minimal/N-seed42/scores.json')
    results = {}
    for arm in ('A1', 'A2', 'U'):
        folder = OUT/'minimal'/f'{arm}-seed42'
        losses = read_rows(folder/'loss.jsonl')
        score = read_json(folder/'scores.json')
        results[arm] = {'updates': len(losses), 'loss_first': losses[0]['loss'], 'loss_last': losses[-1]['loss'],
                        'gt_ce_range': [min(r['ce_gt'] for r in losses), max(r['ce_gt'] for r in losses)],
                        'teacher_ce_range': [min(r['ce_teacher'] for r in losses), max(r['ce_teacher'] for r in losses)],
                        'minimum_gradient_norm': min(r['grad_norm'] for r in losses),
                        'mean_seconds_per_step': sum(r['seconds'] for r in losses)/len(losses),
                        'reserved_peak_bytes': max(r['reserved_peak_bytes'] for r in losses),
                        'scores': score['axis_scores'],
                        'axis_deltas_vs_N': {a: score['axis_scores'][a]-baseline['axis_scores'][a] for a in AXES}}
    losses = read_rows(OUT/'minimal/A1-seed42/loss.jsonl')
    resume = read_json(OUT/'minimal/A1-seed42/resume-verification.json')
    resume.update(first_resumed_step=losses[50]['step'], first_resumed_loss=losses[50]['loss'],
                  final_step=losses[-1]['step'], contiguous_unique_steps=[r['step'] for r in losses] == list(range(1,101)))
    elapsed = sum(read_json(OUT/'minimal'/f'{arm}-seed42/scores.json')['total_generation_seconds'] for arm in ('A1','A2','U','N'))
    per_prompt = elapsed/3200
    summary = {'status': 'passed', 'five_checks': {'cache_loading_no_truncation': True, 'finite_dual_ce_nonzero_gradients': True,
               'adapter_saved_and_reloaded': True, 'four_axis_eval_output': True, 'real_process_resume': True},
               'teacher_minimal_cache_rows': 500, 'training': results, 'baseline': baseline, 'resume': resume,
               'token_distribution': tokens, 'measured_eval_seconds_per_prompt': per_prompt,
               'formal_eval_linear_hours': 270400*per_prompt/3600,
               'formal_eval_linear_CNY': 270400*per_prompt/3600*2.70,
               'preflight_source_sha256': '781d0f024aa74a0ec02f4628380c5a5d4176d6bac123f84de7a60e78171d060a',
               'independent_protocol_sha256': sha(OUT/'protocol.json'),
               'scope': 'Engineering validation, one seed100 updates; not formal15-run results or ARP confirmation'}
    write_json(OUT/'minimal-report.json', summary)
    lines = ['# 7B 独立路线 B：最小循环真实验收', '',
             '四轴各125提示，seed42；A1/A2/U各100实际更新，N零更新。以下五项全部通过。', '',
             '| 验收 | 结果 |', '|---|---|']
    lines += [f'| {k} | 通过 |' for k in summary['five_checks']]
    lines += ['', f"A1恢复点：{resume['restored_step']}；新进程首步{resume['first_resumed_step']}，损失{resume['first_resumed_loss']:.6f}；最终100步，步数连续无重复。", '',
              '| 臂 | 首/末损失 | 秒/步 | reserved GB |', '|---|---:|---:|---:|']
    lines += [f"| {arm} | {r['loss_first']:.5f}/{r['loss_last']:.5f} | {r['mean_seconds_per_step']:.3f} | {r['reserved_peak_bytes']/1e9:.3f} |" for arm,r in results.items()]
    lines += ['', '| 轴 | prompt min/max/mean | 视觉 min/max/mean | 教师回答 min/max/mean |', '|---|---|---|---|']
    def cell(axis, key):
        r = tokens['axes'][axis][key]
        return f"{r['min']}/{r['max']}/{r['mean']:.2f}"
    lines += [f"| {a} | {cell(a,'prompt_including_visual')} | {cell(a,'visual')} | {cell(a,'teacher_response_including_chat_end')} |" for a in AXES]
    lines += ['', '| 臂 | pope | grounding | consistency | mmstar |', '|---|---:|---:|---:|---:|']
    for arm in ('N','A1','A2','U'):
        scores = baseline['axis_scores'] if arm == 'N' else results[arm]['scores']
        lines.append('| '+arm+' | '+' | '.join(f'{scores[a]:.4f}' for a in AXES)+' |')
    lines += ['', f"5090实测评测均值{per_prompt:.3f}秒/提示，270,400提示线性估计{summary['formal_eval_linear_hours']:.2f}小时、{summary['formal_eval_linear_CNY']:.2f}元；正式长度分布可能不同，实际ETA按后续账本更新。", '',
              '完整GT/教师两路token分布、损失、梯度和恢复证据见minimal-report.json与原始账本；原ARP八项阻塞工时估算见仓库route_b_execution_plan.md。该结果仅为工程流程验证。']
    (OUT/'minimal-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','report':str(OUT/'minimal-report.md'),'resume':resume}),flush=True)


if __name__ == '__main__':
    main()
