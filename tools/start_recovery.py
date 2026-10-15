"""Resume setup under screen and start the formal pipeline only after preflight."""
import os
import subprocess
import sys
from experiment_common import ROOT

if __name__=='__main__':
    sessions=subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout
    if '.arp-setup' in sessions or '.arp-formal' in sessions:
        raise RuntimeError('An ARP setup/formal screen session already exists')
    subprocess.run(['screen','-DmS','arp-setup','bash',str(ROOT/'scripts/recover_setup.sh')],
                   cwd=ROOT,check=True)
    print('Started arp-setup: resume -> full preflight -> formal pipeline; outputs stay on AutoDL')
