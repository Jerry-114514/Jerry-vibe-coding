"""Compact authoritative process, step, delivery and memory snapshot."""
from datetime import datetime, timezone
import json
import os
import subprocess
from experiment_common import ROOT, read_json, read_rows
from route_b_staged import OUT, SEEDS


def alive(pid):
    try:
        os.kill(int(pid),0)
        return True
    except (ValueError,TypeError,ProcessLookupError,PermissionError):
        return False


def snapshot():
    state = read_json(OUT/'status.json') if (OUT/'status.json').exists() else {}
    rows = []
    for seed in SEEDS:
        for arm in ('A1','A2','U'):
            folder = OUT/'formal'/f'{arm}-seed{seed}'
            if not folder.exists():continue
            ledger = folder/'loss.jsonl'
            # A liveappend may be between write and flush; ignore only an
            # incomplete observation tail, never alter the training ledger.
            loss = [json.loads(line) for line in ledger.read_bytes().splitlines(keepends=True)
                    if line.endswith(b'\n')] if ledger.exists() else []
            done = (folder/'complete.json').exists()
            rows.append({'arm':arm,'seed':seed,'steps':len(loss),'complete':done,
                         'backup_verified':(folder/'github-storage.json').exists(),
                         'mean_step_seconds':sum(r['seconds'] for r in loss)/len(loss) if loss else None,
                         'peak_reserved_GB':max((r['reserved_peak_bytes'] for r in loss),default=0)/1e9,
                         'eval_complete':(folder/'scores.json').exists()})
    processes = subprocess.run(['ps','-eo','pid,ppid,etime,args'],capture_output=True,text=True,check=True).stdout
    current = [line.strip() for line in processes.splitlines() if any(
        x in line for x in ('tools/run_route_b_storage_safe.py --worker','tools/route_b_staged.py train',
                           'tools/route_b_staged.py eval','tools/route_b_completion_reports.py --worker',
                           'tools/prepare_qwen35_9b_after_7b.py --worker')) and 'SCREEN ' not in line]
    result = {'utc':datetime.now(timezone.utc).isoformat(),'state':state,'state_worker_alive':alive(state.get('pid')),
              'trained_runs':sum(r['complete'] for r in rows),'expected_runs':15,
              'verified_training_backups':sum(r['backup_verified'] for r in rows),'runs':rows,
              'live_processes':current,'nvidia_smi':subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                  '--format=csv,noheader'],capture_output=True,text=True).stdout.strip()}
    if (OUT/'failure.json').exists():result['failure']=read_json(OUT/'failure.json')
    for label,path in [('training_report',OUT/'completion-reports/training-report-delivered.json'),
                       ('final_report',OUT/'completion-reports/final-report-delivered.json'),
                       ('9b_priority',ROOT/'runs/qwen35-9b-infrastructure-20261016/priority-preparation-status.json')]:
        if path.exists():result[label]=read_json(path)
    return result


if __name__=='__main__':
    print(json.dumps(snapshot(),ensure_ascii=False),flush=True)
