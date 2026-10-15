"""Read-only process/output heartbeat; no automatic scientific-error bypass."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from experiment_common import ROOT, write_json

destination=ROOT/'runs/monitor'
destination.mkdir(parents=True,exist_ok=True)
stop=destination/'stop'
previous=None
while not stop.exists():
    processes=subprocess.run(['ps','-eo','pid,etimes,args'],capture_output=True,text=True,check=True).stdout.splitlines()
    live=[]
    for line in processes:
        parts=line.strip().split(None,2)
        if len(parts)!=3 or not parts[0].isdigit():
            continue
        pid,elapsed,command=parts
        if int(pid)==os.getpid():
            continue
        if '/arp-venv/bin/python' in command and any('tools/'+name+'.py' in command for name in
                ['preflight','pipeline','run_inference','train_distill','analyze_rulers','prepare_training']):
            live.append({'pid':int(pid),'elapsed_seconds':int(elapsed),'command':command})
    run=ROOT/'runs/formal-20261014'
    try:
        status=json.loads((run/'status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        status={}
    outputs={str(path.relative_to(ROOT)):{'bytes':path.stat().st_size,'mtime':path.stat().st_mtime}
             for pattern in ['ledger/*.jsonl','arms/*/loss.jsonl','teacher-cache/complete.json','analysis/gate.json']
             for path in run.glob(pattern)}
    state='running' if live else 'stopped'
    if status.get('status')=='failed':
        state='failure'
    elif not live and status.get('status') in ['complete','stopped-at-gate']:
        state=status['status']
    heartbeat={'utc':datetime.now(timezone.utc).isoformat(),'state':state,'processes':live,'stage':status,'outputs':outputs}
    write_json(destination/'heartbeat.json',heartbeat)
    transition=(state,status.get('stage'),status.get('status'))
    if transition!=previous:
        with (destination/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(heartbeat,ensure_ascii=False)+'\n')
        print(heartbeat['utc'],transition,flush=True)
        previous=transition
    time.sleep(60)
print('User requested monitoring stop',flush=True)
