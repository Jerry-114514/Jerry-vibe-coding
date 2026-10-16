"""Repair unavailable configuredpip mirror using officialPyPI, without touching7B."""
import os
import subprocess
import sys
import time
from experiment_common import ROOT, read_json, write_json
import prepare_qwen35_9b as setup

original_run = subprocess.run


def official_pypi_run(args, **kwargs):
    if isinstance(args,list) and 'pip' in args and 'install' in args:
        args = [*args,'--index-url','https://pypi.org/simple']
    return original_run(args,**kwargs)


def worker():
    setup.subprocess.run = official_pypi_run
    setup.environment()
    write_json(setup.OUT/'pip-mirror-repair.json',{'status':'passed','previous_index':'ConfiguredAliyun mirror returned no distributions',
               'new_index':'https://pypi.org/simple','environment':str(setup.ENV),'old7b_environment_unchanged':True})
    deadline = time.monotonic()+8*3600
    while not (setup.OUT/'weights-ready.json').exists():
        if time.monotonic()>deadline:
            raise RuntimeError('9B download deadline reached; keep resumable partials')
        time.sleep(10)
    minimum = ROOT/'runs/route-b-staged-7b-20261016/minimal-acceptance.json'
    while not minimum.exists():
        if (minimum.parent/'failure.json').exists() or time.monotonic()>deadline:
            raise RuntimeError('Keep9B GPU smoke pending until7B accepts')
        time.sleep(10)
    original_run([str(setup.ENV/'bin/python'),'-u',str(ROOT/'tools/smoke_qwen35_9b.py')],check=True,cwd=ROOT)
    write_json(setup.OUT/'infrastructure-complete.json',{'status':'passed','weights':read_json(setup.OUT/'weights-ready.json'),
               'environment':read_json(setup.OUT/'environment-ready.json'),'gpu_smoke':read_json(setup.OUT/'gpu-smoke.json'),
               'note':'Initialmirrorfailure repaired via separate job; originalfailure log retained'})
    if os.environ.get('GH_TOKEN'):
        from publish_artifacts import publish
        publish(setup.OUT,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        token = sys.stdin.readline().strip()
        command = 'source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/prepare_qwen35_9b_official_index.py --worker >> runs/qwen35-9b-infrastructure-20261016/pip-repair.log 2>&1'
        original_run(['screen','-dmS','qwen35-env-repair','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('StartedofficialPyPI isolated environment installation; weight download continues')
