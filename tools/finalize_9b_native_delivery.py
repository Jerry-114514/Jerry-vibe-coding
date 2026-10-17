"""Independent terminal content/delivery audit for user-amended native9B eval."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT, AXES, read_json, write_json, sha
from route_b_9b_loop import OUT, SEEDS
from evaluate_9b_native import NATIVE, verify_formal
from finalize_9b_delivery import script_live, validate_statistics, summary

FINAL=NATIVE/'finalization'


def ready():
    missing=[f'{a}-seed{s}' for a,s in [('N',42)]+[(a,s) for s in SEEDS for a in ('A1','A2','U')]
             if not (NATIVE/'formal'/f'{a}-seed{s}'/'scores.json').exists()]
    phase=read_json(NATIVE/'status.json').get('phase') if (NATIVE/'status.json').exists() else None
    live=script_live('run_9b_native_evaluation.py') or script_live('continue_9b_training_only.py')
    return {'ready':not missing and phase=='all_native_evaluations_complete' and not live,
            'missing_models':missing,'phase':phase,'actual_controllers_live':live}


def finalize(token):
    assert ready()['ready']
    _,evaluations=verify_formal(return_evaluations=True)
    from verify_9b_teacher_delivery import verify as teacher_audit
    teacher=teacher_audit(token)
    plan=read_json(OUT/'formal-plan.json')
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/'tools'/name)==digest
    native=read_json(NATIVE/'protocol.json')
    for name,digest in native['source_sha256'].items():assert sha(ROOT/'tools'/name)==digest
    data=read_json(NATIVE/'formal-report/evaluation-results.json')
    conclusions=validate_statistics(data,plan,evaluations)
    minimum=read_json(NATIVE/'minimal-acceptance.json')
    assert minimum['status']=='passed' and all(minimum['five_checks'].values())
    assert minimum['scores']['N']['grounding']>=native['N_grounding_minimum_accuracy']
    assert read_json(NATIVE/'training-health.json')['status']=='passed'
    assert all((OUT/'formal'/f'{a}-seed{s}'/'resume.pt').exists() for s in SEEDS for a in ('A1','A2','U')),'Keep all9B recovery states per night-mode instruction'
    from audit_route_b_completion import existing_code_unchanged
    assert read_json(ROOT/'runs/protocol-v2/paired-gate-development/gate.json')['training_allowed'] is False
    proof=FINAL/'verified-evidence';proof.mkdir(parents=True,exist_ok=True)
    content={'status':'passed','training_runs':15,'updates_per_run':1000,'evaluation_models':16,
             'prompts_per_model':16900,'matched_seeds':SEEDS,'family_size':12,'teacher_eligible_rows':teacher['eligible_rows'],
             'all15_recovery_states_retained_locally':True,'original_arp_gate_unpassed':True,
             'existing_python_sources_unchanged':existing_code_unchanged(),'conclusions':conclusions,
             'native_protocol_sha256':sha(NATIVE/'protocol.json'),
             'finalizer_source_sha256':sha(Path(__file__)),'validator_source_sha256':sha(ROOT/'tools/finalize_9b_delivery.py'),
             'statistics_sha256':sha(NATIVE/'formal-report/evaluation-results.json')}
    write_json(proof/'content-verification.json',content)
    text=summary(data,conclusions)+'\n## 测量校准\n\n本版本定位提示对所有模型明确指定0–1000，按原图宽高统一换算为像素，再计算IoU。N小样本校准与旧工程五项验收分别保存；旧U已学会像素输出，故不能对旧输出盲目统一缩放。旧分数和本版本分数不混合。修订由用户在旧最小循环后要求，不冒充训练前登记。\n'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,panels=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for panel,label in zip(panels,('A1-A2','A1-U','A1-N')):
        for y,axis in enumerate(AXES):
            row=data['paired_contrasts'][label][axis];low,high=row['bonferroni_familywise_ci95']
            color='#dc2626' if high<0 else '#2563eb' if low>0 else '#666666'
            panel.plot([100*low,100*high],[y,y],color=color,linewidth=2)
            panel.scatter([100*row['difference']],[y],color=color,s=30)
        panel.axvline(0,color='black',linestyle='--',linewidth=.8)
        panel.set(yticks=range(4),yticklabels=AXES,title=label,xlabel='Difference (percentage points)')
        panel.invert_yaxis();panel.grid(axis='x',alpha=.2)
    figure.suptitle('9B +122B independent extension: native0-1000 evaluation\n5 paired seeds,12 Bonferroni intervals; nominal approximate95% family coverage')
    for suffix in ('png','pdf'):figure.savefig(proof/('paired-intervals.'+suffix),dpi=180)
    plt.close(figure)
    (proof/'final-summary.md').write_text(text+'\n![12项校正区间](paired-intervals.png)\n',encoding='utf-8')
    for source in (NATIVE/'minimal-report.md',NATIVE/'training-health.json',NATIVE/'recovery-restoration.json'):
        shutil.copyfile(source,proof/source.name)
    from run_9b_formal_when_ready import metadata_backup,verify_receipt
    from audit_9b_student_delivery import audit
    from publish_artifacts import publish
    metadata_backup(OUT,token)
    delivery=audit(token,strict=True)
    shutil.copyfile(OUT/'final-delivery-verification.json',proof/'delivery-verification.json')
    receipt=publish(proof,'arp-9b-122b-20261017',token=token)
    server=verify_receipt(receipt,token)
    write_json(FINAL/'complete.json',{'status':'full_native9b_content_and_delivery_verified',
               'timestamp':datetime.now(timezone.utc).isoformat(),'required_files':delivery['required_files'],
               'verified_files':delivery['verified_files'],'evidence_delivery':receipt,'fresh_server_verification':server,
               'not_original_arp':True,'no_extra_training_or_inference':True})
    print('Native9B full content and fresh Release delivery verified; all15 recovery states retained',flush=True)


def watch(token):
    import fcntl
    FINAL.mkdir(parents=True,exist_ok=True)
    with (FINAL/'observer.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (FINAL/'complete.json').exists():
            state=ready();write_json(FINAL/'queue-status.json',state)
            if state['ready']:finalize(token);return
            if (NATIVE/'stopped.json').exists() or (OUT/'training-only-stopped.json').exists():
                raise RuntimeError('Inspect stopped experiment; no premature completion claim')
            time.sleep(45)


if __name__=='__main__':
    token=os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
    assert token,'Credential only through encrypted stdin/inherited environment'
    if '--start' in sys.argv:
        command=('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/finalize_9b_native_delivery.py '
                 '>> runs/route-b-9b-122b-20261017/student-experiment/native-finalization-console.log 2>&1')
        subprocess.run(['screen','-dmS','route-b-9b-native-finalization','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Native9B final content, measurement, retained recovery and fresh server delivery audit queued')
    else:
        try:watch(token)
        except Exception as error:
            write_json(FINAL/'stopped.json',{'type':type(error).__name__,'message':str(error),
                       'timestamp':datetime.now(timezone.utc).isoformat(),'completion_unproven':True})
            raise
