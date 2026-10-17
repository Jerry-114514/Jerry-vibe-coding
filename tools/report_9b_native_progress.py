"""Night-mode progress and08:00 Beijing brief, from actual remote artifacts."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import AXES, read_json, write_json
from route_b_9b_loop import OUT, SEEDS
from evaluate_9b_native import NATIVE

MORNING = datetime(2026,10,18,0,0,tzinfo=timezone.utc)


def rows(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n') and line.strip()]


def collect():
    training=[];seconds=0;updates=0
    for seed in SEEDS:
        for arm in ('A1','A2','U'):
            folder=OUT/'formal'/f'{arm}-seed{seed}'
            ledger=rows(folder/'loss.jsonl')
            seconds+=sum(r['seconds'] for r in ledger);updates+=len(ledger)
            training.append({'model':folder.name,'updates':len(ledger),'complete':(folder/'complete.json').exists(),
                             'last_loss':ledger[-1]['loss'] if ledger else None,
                             'reserved_peak_bytes':max((r['reserved_peak_bytes'] for r in ledger),default=0),
                             'recovery_local':(folder/'resume.pt').exists(),
                             'recovery_archived':(folder/'training-delivery.json').exists()})
    models=[('N',42)]+[(a,s) for s in SEEDS for a in ('A1','A2','U')]
    scores={};processed={a:0 for a in AXES};timing={a:[] for a in AXES};partial=[]
    for arm,seed in models:
        folder=NATIVE/'formal'/f'{arm}-seed{seed}'
        ledger=rows(folder/'eval.jsonl')
        for row in ledger:processed[row['axis']]+=1;timing[row['axis']].append(row['seconds'])
        if (folder/'scores.json').exists():scores[folder.name]=read_json(folder/'scores.json')['axis_scores']
        elif ledger:partial.append({'model':folder.name,'rows':len(ledger)})
    fallback={a:[] for a in AXES}
    for arm in ('N','A1','A2','U'):
        for row in rows(NATIVE/'minimal'/f'{arm}-seed42'/'eval.jsonl'):fallback[row['axis']].append(row['seconds'])
    counts={'pope':9000,'grounding':2400,'consistency':4000,'mmstar':1500}
    eta=0.;available=True
    for axis in AXES:
        sample=timing[axis] or fallback[axis]
        if not sample:available=False;break
        eta+=(16*counts[axis]-processed[axis])*sum(sample)/len(sample)
    calibration=read_json(NATIVE/'minimal-acceptance.json') if (NATIVE/'minimal-acceptance.json').exists() else None
    failure_path=NATIVE/'failed-calibration-report/diagnostic.json'
    failure=read_json(failure_path) if failure_path.exists() else None
    stops=[str(p.relative_to(OUT)) for p in OUT.glob('*stopped.json')]
    if (NATIVE/'stopped.json').exists():stops.append('processor-pixel-evaluation-v3/stopped.json')
    seven=OUT.parents[1]/'7b-processor-pixel-audit-20261017'
    seven_scores={};seven_rows=0;seven_seconds=0
    for folder in (seven/'models').glob('*'):
        ledger=rows(folder/'eval.jsonl');seven_rows+=len(ledger);seven_seconds+=sum(r['seconds'] for r in ledger)
        if (folder/'scores.json').exists():seven_scores[folder.name]=read_json(folder/'scores.json')['accuracy']
    return {'timestamp_utc':datetime.now(timezone.utc).isoformat(),'training':training,'training_complete':sum(r['complete'] for r in training),
            'seven_processor_audit':{'complete_models':len(seven_scores),'processed_rows':seven_rows,'scores':seven_scores,
                 'remaining_pure_hours':(38400-seven_rows)*seven_seconds/seven_rows/3600 if seven_rows else None,
                 'stopped':read_json(seven/'stopped.json') if (seven/'stopped.json').exists() else None},
            'updates':updates,'remaining_pure_training_hours':(15000-updates)*seconds/updates/3600 if updates else None,
            'native_calibration_status':calibration['status'] if calibration else failure['status'] if failure else 'pending',
            'native_calibration_failure':failure,
            'native_calibration_scores':calibration['scores'] if calibration else None,
            'formal_evaluations_complete':len(scores),'scores':scores,'partial_models':partial,
            'processed_prompts':sum(processed.values()),'remaining_generation_hours':eta/3600 if available else None,
            'eta_caveat':'Axis-weighted observed latency; excludes future output-length drift, loading, archive verification and upload; unavailable until each axis has samples',
            'free_GiB':shutil.disk_usage(OUT).free/1024**3,'stopped_markers':stops,'not_original_arp':True}


def render(data):
    text=['# 9B夜间实验简报','',f'采集UTC时间：{data["timestamp_utc"]}（北京时间加8小时）。', '',
          f'固定训练完成{data["training_complete"]}/15，真实更新{data["updates"]}/15,000；处理器像素坐标最小循环状态：{data["native_calibration_status"]}。正式评测完成{data["formal_evaluations_complete"]}/16，已生成{data["processed_prompts"]}/270,400提示。磁盘空闲{data["free_GiB"]:.2f}GiB。', '',
          '| 训练 | 已更新 | 末步双CE | reserved GB | 恢复状态 |','|---|---:|---:|---:|---|']
    for row in data['training']:
        loss=f'{row["last_loss"]:.6g}' if row['last_loss'] is not None else '待运行'
        state='本地保留' if row['recovery_local'] else '旧归档恢复链' if row['recovery_archived'] else '待保存'
        text.append(f'| {row["model"]} | {row["updates"]} | {loss} | {row["reserved_peak_bytes"]/1e9:.3f} | {state} |')
    text += ['','| 已完成正式评测模型 | pope | grounding | consistency | mmstar |','|---|---:|---:|---:|---:|']
    for model,scores in data['scores'].items():text.append('| '+model+' | '+' | '.join(f'{100*scores[a]:.3f}%' for a in AXES)+' |')
    if not data['scores']:text += ['','尚无完整正式评测，不能用最小循环分数代替正式结果。']
    seven=data['seven_processor_audit']
    text += ['',f'7B统一处理器像素复测：{seven["complete_models"]}/16模型、{seven["processed_rows"]}/38,400条；独立继续，不因9B校准失败而等待。',
             '7B已完成模型定位分：'+json.dumps(seven['scores'],ensure_ascii=False)]
    if data.get('native_calibration_failure'):
        failure=data['native_calibration_failure']
        registered=failure['scores']['registered_processor_pixels']
        text += ['',f'9B共同坐标校准未通过：N准确率{registered["accuracy"]*100:.1f}%，解析失败{registered["parse_failures"]}/100；正式评测保持停止。',
                 '同输出0–1000解释的86%仅作事后单位诊断，不是通过。原校准预算不重置，不为达到门槛反复调提示。']
    if seven['remaining_pure_hours'] is not None:text += [f'7B剩余纯生成约{seven["remaining_pure_hours"]:.2f}小时，另加装载、校验及上传。']
    if seven['stopped']:text += ['7B审计停止记录：'+json.dumps(seven['stopped'],ensure_ascii=False)]
    eta=data['remaining_generation_hours']
    text += ['',f'剩余纯生成预计{eta:.2f}小时；不含装载/上传/校验，未来回答长度变化会改变估计。' if eta is not None else '四轴时延样本尚不足，正式评测剩余时间暂不可靠，不给固定完成时刻。', '',
             '## 五项验收与测量校准','','正式训练的旧工程五项验收与新处理器像素坐标校准分开保存。坐标校准后才开始本版本正式评测；100条N定位题先检查合理性，再补充三个适配器的定位测量。五项最终报告见同包minimal-report.md（存在时）。原ARP阶段C保持未放行。']
    if data['stopped_markers']:text += ['', '需要检查的停止记录：'+', '.join(data['stopped_markers'])]
    return '\n'.join(text)+'\n'


def maybe_publish(token,force=False):
    import fcntl
    NATIVE.mkdir(parents=True,exist_ok=True)
    with (NATIVE/'progress-write.lock').open('w') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        now=datetime.now(timezone.utc)
        destination=OUT/'night-reports'/('final-native-v2' if force else 'morning-20261018')
        due=force or now>=MORNING and not (destination/'published.json').exists()
        progress=NATIVE/'progress.json'
        if not due and progress.exists() and time.time()-progress.stat().st_mtime<600:return
        data=collect();write_json(progress,data)
        (NATIVE/'progress.md').write_text(render(data),encoding='utf-8')
        if not due:return
        destination.mkdir(parents=True,exist_ok=True)
        write_json(destination/'brief.json',data)
        (destination/'brief.md').write_text(render(data),encoding='utf-8')
        for source in [NATIVE/'minimal-report.md',NATIVE/'minimal-acceptance.json',NATIVE/'training-health.json',
                       OUT/'formal-report/training-results.json',OUT/'formal-report/training-report.md',OUT/'formal-report/loss-curves.png']:
            if source.exists():shutil.copyfile(source,destination/source.name)
        # Carry failure and syntax-only measurement repairs as explicit evidence,
        # rather than silently substituting them for successful calibration.
        for label,source in [
            ('9b-failed-calibration.json',NATIVE/'failed-calibration-report/diagnostic.json'),
            ('9b-failed-calibration.md',NATIVE/'failed-calibration-report/report.md'),
            ('7b-csv-schema-amendment.json',OUT.parents[1]/'7b-processor-pixel-audit-20261017/schema-amendment.json'),
            ('7b-calibration.json',OUT.parents[1]/'7b-processor-pixel-audit-20261017/calibration-proof/calibration.json')]:
            if source.exists():shutil.copyfile(source,destination/label)
        from publish_artifacts import publish
        receipt=publish(destination,'arp-9b-122b-20261017',token=token)
        write_json(destination/'published.json',{'status':'uploaded_and_server_digest_checked','timestamp':now.isoformat(),'receipt':receipt})


if __name__=='__main__':
    token=os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
    assert token,'Credential only through encrypted stdin/inherited environment'
    if '--start' in sys.argv:
        command=('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/report_9b_native_progress.py '
                 '>> runs/route-b-9b-122b-20261017/student-experiment/morning-report-console.log 2>&1')
        subprocess.run(['screen','-dmS','route-b-9b-morning-report','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('08:00 BeijingOct18 brief queued from actual artifacts; quiet while unchanged')
    else:
        while True:
            try:
                maybe_publish(token)
                if datetime.now(timezone.utc)>=MORNING and (OUT/'night-reports/morning-20261018/published.json').exists():break
            except Exception as error:
                write_json(NATIVE/'morning-report-error.json',{'type':type(error).__name__,'message':str(error),'timestamp':datetime.now(timezone.utc).isoformat()})
            time.sleep(45)
