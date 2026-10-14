"""Start formal work under screen; token remains in process memory, never files."""
import os
import subprocess
import sys
from experiment_common import ROOT, read_json

if __name__=='__main__':
    if read_json(ROOT/'runs/preflight/ready.json')['status']!='ready':
        raise RuntimeError('Preflight is not ready')
    existing=subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout
    if '.arp-formal' in existing:
        raise RuntimeError('A formal screen session already exists')
    token=sys.stdin.readline().strip()
    if not token:
        raise RuntimeError('GitHub credential is required for automatic artifact backup')
    environment={**os.environ,'GH_TOKEN':token}
    subprocess.run(['screen','-DmS','arp-formal','bash',str(ROOT/'scripts/formal.sh')],cwd=ROOT,env=environment,check=True)
    print('Started screen session arp-formal',flush=True)
