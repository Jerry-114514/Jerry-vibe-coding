"""Archive superseded evidence before rerunning the complete precision-v2 preflight."""
from datetime import datetime, timezone
from experiment_common import ROOT
from pathlib import Path
import subprocess

sessions=subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout
if '.arp-setup' in sessions or '.arp-formal' in sessions:
    raise RuntimeError('Stop: an existing pipeline session is still present')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive=ROOT/'runs'/'superseded-precision-v1'/stamp
archive.mkdir(parents=True,exist_ok=False)
paths=['runs/formal-20261014','runs/preflight/teacher-cache',
       'runs/preflight/lora-smoke-8','runs/preflight/adapted-ledger',
       'runs/preflight/ledger','runs/preflight/ready.json']
for relative in paths:
    source=(ROOT/relative).resolve()
    if not source.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Archive source outside project runs')
    if source.exists():
        destination=archive/Path(relative).relative_to('runs')
        destination.parent.mkdir(parents=True,exist_ok=True)
        source.rename(destination)
print('Preserved all superseded outputs at',archive,flush=True)
subprocess.run(['/root/autodl-tmp/arp-venv/bin/python','tools/start_recovery.py'],cwd=ROOT,check=True)
