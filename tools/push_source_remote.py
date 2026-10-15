"""Push a local code bundle through AutoDL's reliable GitHub network route."""
import base64
import os
import subprocess
import sys
from experiment_common import ROOT

if __name__=='__main__':
    token=sys.stdin.readline().strip()
    if not token:
        raise RuntimeError('GitHub credential required')
    subprocess.run(['git','fetch',str(ROOT/'tmp/autodl/source.bundle'),
                    'codex/arp-autodl:refs/heads/codex/arp-autodl'],cwd=ROOT,check=True)
    auth=base64.b64encode(('x-access-token:'+token).encode()).decode()
    environment={**os.environ,'GIT_TERMINAL_PROMPT':'0','GIT_CONFIG_COUNT':'1',
                 'GIT_CONFIG_KEY_0':'http.https://github.com/.extraheader',
                 'GIT_CONFIG_VALUE_0':'Authorization: Basic '+auth}
    subprocess.run(['git','-c','http.version=HTTP/1.1','push','origin','refs/heads/codex/arp-autodl'],
                   cwd=ROOT,env=environment,check=True)
    print('Pushed codex/arp-autodl from AutoDL',flush=True)
