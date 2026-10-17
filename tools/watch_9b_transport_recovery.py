"""Resume failed CPU orchestration only for transient transport failures."""
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from experiment_common import ROOT,read_json,write_json
from route_b_9b_loop import OUT
from finalize_9b_delivery import script_live

HERE=OUT/'transport-watchdog'
CONTROL=Path('/root/autodl-tmp/arp-venv/bin/python')


def transient(record):
    kind=record.get('type');message=record.get('message','')
    if kind=='HTTPError':
        match=re.search(r'\b([45]\d\d)\b',message)
        return bool(match and (int(match[1]) in (408,429) or int(match[1])>=500))
    return kind in ('ConnectionError','ReadTimeout','ConnectTimeout','Timeout','ChunkedEncodingError')


def restart(name,marker,token):
    assert not script_live(name)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    history=HERE/'history';history.mkdir(parents=True,exist_ok=True)
    marker.replace(history/(stamp+'-'+name+'.json'))
    with (HERE/(name+'.log')).open('a') as log:
        subprocess.Popen([str(CONTROL),'-u',str(ROOT/'tools'/name)],cwd=ROOT,
                         env={**os.environ,'GH_TOKEN':token},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(HERE/'latest-restart.json',{'timestamp':stamp,'script':name,'reason':'Recognized transient transport only',
               'training_budget_reset':False,'sampling_or_updates_changed':False})


def run(token):
    import fcntl
    HERE.mkdir(exist_ok=True)
    with (HERE/'watchdog.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (OUT/'processor-pixel-evaluation-v3/finalization/complete.json').exists():
            marker=OUT/'training-only-stopped.json'
            if marker.exists() and not script_live('continue_9b_training_only.py') and not script_live('route_b_9b_loop.py'):
                record=read_json(marker)
                budget=read_json(OUT/'formal-training-wall-budget.json')
                if transient(record) and datetime.now(timezone.utc)<datetime.fromisoformat(budget['end_utc']):
                    restart('continue_9b_training_only.py',marker,token)
                else:write_json(HERE/'attention.json',{'unhandled_training_stop':record,'never_retries_validation_or_oom':True})
            native=OUT/'processor-pixel-evaluation-v3'
            marker=native/'stopped.json'
            if marker.exists() and not script_live('run_9b_native_evaluation.py') and not (OUT/'training-only-stopped.json').exists():
                record=read_json(marker)
                recovered_training=(HERE/'latest-restart.json').exists()
                dependency_stop=record.get('type')=='AssertionError' and record.get('message')=='Training failed; no calibration/evaluation launch'
                if transient(record) or (recovered_training and dependency_stop):restart('run_9b_native_evaluation.py',marker,token)
            marker=native/'finalization/stopped.json'
            if marker.exists() and not script_live('finalize_9b_native_delivery.py') and not (native/'stopped.json').exists() and not (OUT/'training-only-stopped.json').exists():
                record=read_json(marker)
                dependency_stop=record.get('message')=='Inspect stopped experiment; no premature completion claim'
                if transient(record) or ((HERE/'latest-restart.json').exists() and dependency_stop):restart('finalize_9b_native_delivery.py',marker,token)
            write_json(HERE/'heartbeat.json',{'timestamp':datetime.now(timezone.utc).isoformat(),
                       'training_controller_live':script_live('continue_9b_training_only.py'),
                       'evaluation_controller_live':script_live('run_9b_native_evaluation.py')})
            time.sleep(30)


if __name__=='__main__':
    token=os.environ.get('GH_TOKEN') or sys.stdin.readline().strip();assert token
    if '--start' in sys.argv:
        command='cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/watch_9b_transport_recovery.py >> runs/route-b-9b-122b-20261017/student-experiment/transport-watchdog.log 2>&1'
        subprocess.run(['screen','-dmS','9b-transport-watchdog','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Transient transport watchdog queued; never retries OOM, nonfinite loss or failed measurement checks')
    else:run(token)
