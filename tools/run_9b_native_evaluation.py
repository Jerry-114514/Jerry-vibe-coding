"""Night-mode: wait fixed training, native minimum calibration,16 queued evals."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from experiment_common import ROOT, AXES, read_json, read_rows, write_json, sha
from route_b_9b_loop import OUT, SEEDS
from evaluate_9b_native import NATIVE, prepare, verify_evaluation
from preflight_9b_full import ENV
from run_9b_formal_when_ready import remaining_budget, metadata_backup as original_metadata_backup, preserve_incomplete_journal_tail
from resilient_9b_delivery import network_retry


def metadata_backup(folder,token):
    return network_retry(lambda:original_metadata_backup(folder,token),"native-metadata:"+str(folder.relative_to(ROOT)))


def report_tick(token,force=False):
    try:
        from report_9b_native_progress import maybe_publish
        maybe_publish(token,force=force)
    except Exception as error:
        write_json(NATIVE/'progress-report-error.json',{'type':type(error).__name__,'message':str(error),
                   'timestamp':datetime.now(timezone.utc).isoformat(),'experiment_continues':True})


def child(command,token,timeout=None):
    with subprocess.Popen(command,cwd=ROOT) as process:
        expired=threading.Event()
        def expire():
            expired.set()
            if process.poll() is None:process.terminate()
        timer=threading.Timer(timeout,expire) if timeout is not None else None
        if timer is not None:timer.daemon=True;timer.start()
        try:
            while True:
                try:
                    code=process.wait(timeout=45)
                    assert not expired.is_set(), 'Registered evaluation wall budget exhausted'
                    assert code==0, ('Child failed',code,command)
                    return
                except subprocess.TimeoutExpired:
                    report_tick(token)
        finally:
            if timer is not None:timer.cancel()


def training_health():
    from verify_9b_student_outputs import verify_training
    results=[]
    for seed in SEEDS:
        for arm in ('A1','A2','U'):
            proof=verify_training(arm,seed,allow_offloaded=True)
            rows=read_rows(OUT/'formal'/f'{arm}-seed{seed}'/'loss.jsonl')
            gradients=sorted(r['grad_norm'] for r in rows)
            assert all(math.isfinite(g) and g>0 for g in gradients)
            results.append({'arm':arm,'seed':seed,'updates':len(rows),'max_grad_norm_before_clip':max(gradients),
                            'median_grad_norm_before_clip':gradients[len(gradients)//2],
                            'gradient_clipping_threshold':1.0,'finite_dual_ce_and_gradients':True,
                            'reserved_peak_bytes':max(r['reserved_peak_bytes'] for r in rows),
                            'recovery_sha256':proof['resume_sha256'],
                            'recovery_local':(OUT/'formal'/f'{arm}-seed{seed}'/'resume.pt').exists()})
    result={'status':'passed','runs':results,'updates':15000,'no_unhandled_training_failure':True,
            'scope':'All continuous1000-update logs, actual optimizer/recovery tensors or prior verified archive chain; preclip gradient magnitudes reported, not silently called zero-spike'}
    write_json(NATIVE/'training-health.json',result)
    return result


def minimum_report():
    legacy=read_json(OUT/'minimal-acceptance.json')
    assert legacy['status']=='passed'
    scores={};timings={}
    for arm in ('N','A1','A2','U'):
        marker,_=verify_evaluation(arm,42,phase='minimal')
        scores[arm]=marker['axis_scores'];timings[arm]=marker['new_generation_seconds']
    # Engineering sanity threshold fixed in source/registration before native
    # predictions: the user's expected ~80% should not remain at the1% floor.
    nrows=read_rows(NATIVE/'minimal/N-seed42/eval.jsonl')
    ground=[r for r in nrows if r['axis']=='grounding']
    assert len(ground)==100 and sum(r['parse_fail_flag'] for r in ground)<=5
    assert scores['N']['grounding']>=.80, 'N native-coordinate sanity score below80%; inspect, never tune prompts/retry budget to reach a score'
    proof={'status':'passed','five_checks':legacy['five_checks'],
           'old_engineering_acceptance_sha256':sha(OUT/'minimal-acceptance.json'),
           'measurement_protocol_sha256':sha(NATIVE/'protocol.json'),'scores':scores,
           'resume':legacy['resume'],'first_resumed_loss':legacy['first_resumed_loss'],
           'new_grounding_generations':400,'reused_unchanged_other_axis_generations':2800,
           'calibration_new_generation_seconds':timings,
           'N_grounding_sanity_threshold':.80,'original_pixel_outputs_retained':True,
           'scope':'Original five engineering checks plus user-amended native-coordinate measurement calibration'}
    write_json(NATIVE/'minimal-acceptance.json',proof)
    lines=['# 9B最小循环五项验收与坐标校准','','旧工程循环的五项检查已通过；本报告补充用户要求的显式0–1000测量校准。三臂各100步训练不重复，N不训练。每模型仅新生成100条定位题，其余700条未变提示的原始输出按摘要复用。','','| 验收 | 结果 |','|---|---|']
    labels=['教师缓存/学生输入无截断','双CE有限、梯度非零','适配器保存并在新进程加载','四轴输出及原始输出重评分一致','A1新进程从50步恢复到100步']
    for label in labels:lines.append(f'| {label} | 通过；旧工程证据与本次校准分别保留 |')
    lines += ['',f'恢复后首损失：`{legacy["first_resumed_loss"]:.12g}`。N定位校准为{100*scores["N"]["grounding"]:.1f}%，解析失败{sum(r["parse_fail_flag"] for r in ground)}/100。','','| 模型 | pope | grounding | consistency | mmstar |','|---|---:|---:|---:|---:|']
    for arm,s in scores.items():lines.append('| '+arm+' | '+' | '.join(f'{100*s[a]:.2f}%' for a in AXES)+' |')
    lines += ['','## 测量校准','','所有模型的定位问题明确要求输出0–1000，x乘原宽/1000、y乘原高/1000后与原像素标注计算IoU≥0.5。没有按答案猜坐标系，没有对不同训练臂采用不同解析器。原始文本与换算框分别保存。旧U曾学会像素输出，因此不能对旧U输出直接统一缩放后冒充正确评测。','','本次400条新增定位推理由用户要求的校准单独登记，最多15分钟GPU活动时间；不是追加训练预算，也不冒充旧两小时工程预检的一部分。80%为新预测前固定的N工程合理性检查，低于阈值即停，不为达到分数调提示或重试。原ARP门槛仍未放行。']
    (NATIVE/'minimal-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return proof


def live_training_controller():
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:argv=p.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if argv and b'python' in argv[0] and any(Path(a.decode()).name=='continue_9b_training_only.py' for a in argv[1:] if a):return True
    return False


def run(token):
    import fcntl
    NATIVE.mkdir(parents=True,exist_ok=True)
    with (NATIVE/'evaluation-manager.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run([str(ENV),'-u',str(ROOT/'tools/evaluate_9b_native.py'),'prepare'],cwd=ROOT,check=True,timeout=1800)
        from publish_artifacts import publish
        registration=NATIVE/'registration-user80pct-cloud-recovery';registration.mkdir(exist_ok=True)
        for name in ('protocol.json','minimal-manifest.jsonl','formal-manifest.jsonl'):
            shutil.copyfile(NATIVE/name,registration/name)
        for name in ('evaluate_9b_native.py','run_9b_native_evaluation.py','analyze_9b_native.py','resilient_9b_delivery.py','verify_9b_recovery_availability.py'):
            shutil.copyfile(ROOT/'tools'/name,registration/name)
        shutil.copyfile(NATIVE/'night-mode-amendment.json',registration/'night-mode-amendment.json')
        if not (registration/'github-storage.json').exists():network_retry(lambda:publish(registration,'arp-9b-122b-20261017',token=token),'register-user80pct')
        while not (OUT/'formal-training-complete.json').exists() or live_training_controller():
            assert not (OUT/'training-only-stopped.json').exists(),'Training failed; no calibration/evaluation launch'
            write_json(NATIVE/'queue-status.json',{'phase':'waiting_all15_trainings_and_recovery_checks','pid':os.getpid(),'gpu_used':False})
            report_tick(token)
            time.sleep(45)
        from verify_9b_recovery_availability import verify as verify_recovery_availability
        verify_recovery_availability(token)
        training_health()
        if not (NATIVE/'minimal-acceptance.json').exists():
            # GPU lock waiting time precedes the fixed15min active budget.
            child([str(ENV),'-u',str(ROOT/'tools/evaluate_9b_native.py'),'calibrate'],token)
            minimum_report()
            metadata_backup(NATIVE,token)
        order=read_json(OUT/'formal-plan.json')['run_order']
        models=[('N',42)]+[(j['arm'],j['seed']) for j in order]
        for arm,seed in models:
            folder=NATIVE/'formal'/f'{arm}-seed{seed}'
            if not (folder/'scores.json').exists():
                preserve_incomplete_journal_tail(folder/'eval.jsonl')
                child([str(ENV),'-u',str(ROOT/'tools/evaluate_9b_native.py'),'eval','--phase','formal',
                       '--arm',arm,'--seed',str(seed)],token,timeout=remaining_budget('formal-evaluation',40))
            marker,_=verify_evaluation(arm,seed)
            receipt=metadata_backup(folder,token)
            write_json(folder/'eval-delivery.json',{'ledger_sha256':marker['ledger_sha256'],'receipt':receipt,
                       'exact_row_count':16900,'original_units':13700,'rescore_verified':True})
            report_tick(token)
        from analyze_9b_native import analyze
        analyze()
        write_json(NATIVE/'status.json',{'phase':'all_native_evaluations_complete','models':16,'not_original_arp':True})
        metadata_backup(OUT,token)
        report_tick(token,force=True)
        print('All16 native-coordinate evaluations, fixed paired analysis and process data delivered',flush=True)


if __name__=='__main__':
    token=os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
    assert token,'Credential only through encrypted stdin/inherited environment'
    if '--start' in sys.argv:
        command=('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/run_9b_native_evaluation.py >> '
                 'runs/route-b-9b-122b-20261017/student-experiment/native-evaluation-console.log 2>&1')
        subprocess.run(['screen','-dmS','route-b-9b-native-eval','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Native-coordinate evaluation queued after all15 fixed trainings;400 calibration prompts and16 full models registered')
    else:
        try:run(token)
        except Exception as error:
            write_json(NATIVE/'stopped.json',{'type':type(error).__name__,'message':str(error),
                       'timestamp':datetime.now(timezone.utc).isoformat(),'inspect_actual_process_before_restart':True})
            raise
