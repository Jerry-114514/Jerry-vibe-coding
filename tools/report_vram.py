"""Measured GPU memory evidence; never impute missing historical peaks."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
from experiment_common import ROOT, MODELS, read_json, sha, write_json

RUN=ROOT/'runs/formal-20261014'
SAMPLES=ROOT/'runs/monitor/vram-samples.jsonl'

def sample():
    status=read_json(RUN/'status.json')
    rows=list(csv.reader(subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).splitlines()))
    processes=list(csv.reader(subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()))
    return {'utc':datetime.now(timezone.utc).isoformat(),'stage':status.get('stage'),
        'stage_status':status.get('status'),
        'gpus':[{'index':int(r[0]),'name':r[1].strip(),'total_mib':int(r[2]),
                 'used_mib':int(r[3]),'free_mib':int(r[4]),'utilization_percent':int(r[5])} for r in rows],
        'compute_processes':[{'pid':int(r[0]),'used_mib':int(r[1])} for r in processes if len(r)==2 and r[1].strip().isdigit()]}

def monitor():
    import fcntl
    SAMPLES.parent.mkdir(parents=True,exist_ok=True)
    with (SAMPLES.parent/'vram-monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (SAMPLES.parent/'vram-stop').exists():
            try:
                value=sample()
                with SAMPLES.open('a',encoding='utf-8') as out:
                    out.write(json.dumps(value,ensure_ascii=False)+'\n')
                write_json(SAMPLES.parent/'vram-latest.json',value)
                if value['stage']=='E' and value['stage_status']=='complete':
                    break
            except Exception as error:
                print(datetime.now(timezone.utc).isoformat(),repr(error),flush=True)
            time.sleep(15)

def report():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=ROOT/'runs/vram-reports'/stamp
    destination.mkdir(parents=True,exist_ok=False)
    rows=[]
    def record(stage,workload,path=None,note=''):
        value=read_json(path) if path and path.exists() else {}
        peak=value.get('peak_vram_bytes')
        if path and path.exists():
            target=destination/'evidence'/path.relative_to(RUN)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
        rows.append({'stage':stage,'workload':workload,'peak_allocated_bytes':peak,
            'peak_allocated_gib':round(peak/1024**3,4) if peak is not None else None,
            'measurement':'torch.cuda.max_memory_allocated' if peak is not None else 'not historically measured',
            'source':str(path.relative_to(ROOT)) if path and path.exists() else None,
            'source_sha256':sha(path) if path and path.exists() else None,'note':note})
    for key in MODELS:
        record('B',key,RUN/'ledger'/f'{key}-000.complete.json','Completed full 16,900-item inference')
    record('C','CPU analysis',note='NumPy CPU analysis does not invoke CUDA. Historical board memory was not sampled; missing is not zero.')
    record('D1','Training-set construction',note='CPU-only preparation; no historical board sample.')
    record('D2','Qwen7B teacher cache',RUN/'teacher-cache/complete.json',
           'Completed 3,545 teacher entries; original completion file did not record a GPU peak.')
    for arm in ['A1','A2','U','N']:
        record('D3',arm,RUN/'arms'/arm/'complete.json',
               'No-op control; no model training' if arm=='N' else '1,000 actual optimizer updates when complete; pending runs have no peak yet')
        if arm!='N':
            record('D4',arm,RUN/'post'/arm/'qwen3b-000.complete.json',
                   'Full post-training evaluation peak becomes available only after completion')
    record('D5','Homogeneous-floor control',note='CPU analysis; no historical board sample.')
    current=sample()
    finished=current['stage']=='E' and current['stage_status']=='complete'
    samples=[]
    if SAMPLES.exists():
        for line in SAMPLES.read_text(encoding='utf-8').splitlines(keepends=True):
            if line.endswith('\n'):
                samples.append(json.loads(line))
    if not samples or samples[-1]!=current:
        samples.append(current)
    observed={}
    for value in samples:
        key=value['stage'] or 'unknown'
        observed[key]=max(observed.get(key,0),max(g['used_mib'] for g in value['gpus']))
    summary={'created_utc':current['utc'],'experiment_run':str(RUN.relative_to(ROOT)),
        'rows':rows,'current_nvidia_smi':current,'sampled_board_peaks_mib':observed,
        'sample_interval_seconds':15,'sample_count':len(samples),'experiment_complete':finished,
        'definitions':{'allocated_peak':'PyTorch process peak allocated tensor memory; excludes CUDA context and allocator reserved cache.',
                       'board_used':'nvidia-smi board total used memory including reserved memory and all processes.',
                       'sampled_peak':'Maximum observed since monitoring began; not a retrospective or guaranteed continuous peak.',
                       'units':'GiB=1024^3 bytes; MiB=1024^2 bytes; nvidia-smi uses MiB.'},
        'limitations':['C and earlier D2 have no historical nvidia-smi trace.']+
                      ([] if finished else ['D is ongoing; pending runs have no final peak yet.'])}
    write_json(destination/'vram-summary.json',summary)
    with (destination/'peak-allocated.csv').open('w',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (destination/'nvidia-smi-samples.jsonl').open('w',encoding='utf-8') as out:
        for value in samples:
            out.write(json.dumps(value,ensure_ascii=False)+'\n')
    lines=['# B / C / D 显存占用报告','',f'采集时间（UTC）：{current["utc"]}',
        '','历史峰值来自完成文件中的 `torch.cuda.max_memory_allocated()`；GiB = 1024³ 字节。',
        '该数值不包含 CUDA 上下文和 PyTorch 缓存池保留显存，不能直接等同于 nvidia-smi 总占用。',
        '','| 阶段 | 任务 | PyTorch 峰值（GiB） |','|---|---|---:|']
    for row in rows:
        lines.append(f'| {row["stage"]} | {row["workload"]} | {row["peak_allocated_gib"] if row["peak_allocated_gib"] is not None else "未采集 / 尚未完成（详见 JSON）"} |')
    gpu=current['gpus'][0]
    lines += ['',f'当前 {current["stage"]}：nvidia-smi 总占用 {gpu["used_mib"]} MiB（{gpu["used_mib"]/1024:.2f} GiB），总容量 {gpu["total_mib"]} MiB。',
        '','C 为 CPU 分析，不调用 CUDA；没有历史板卡采样，不能据此填写“总显存占用为 0”。',
        'D2 已完成，但当时未记录峰值，因此不会用后来采样冒充历史峰值。',
        '后续 D 阶段每 15 秒采样一次；采样最大值只是监测开始后的观测值。',
        'D 已全部完成；报告包含 A1、A2、U 的训练与完整评测峰值。' if finished else 'D 尚未全部完成，本报告会在后续阶段补充。',
        '','文件：`vram-summary.json`、`peak-allocated.csv`、`nvidia-smi-samples.jsonl`、`evidence/` 原始完成记录。']
    (destination/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(destination),flush=True)
    print(json.dumps({'B_and_D_completed_peaks':[{k:r[k] for k in ['stage','workload','peak_allocated_gib']} for r in rows if r['peak_allocated_bytes'] is not None],
                      'current_stage':current['stage'],'board_used_mib':gpu['used_mib'],'sample_count':len(samples)},ensure_ascii=False),flush=True)
    return destination

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['monitor','report'])
    args=parser.parse_args()
    monitor() if args.action=='monitor' else report()
