"""Honor7B formal priority:9B GPU smoke waits until all15 training runs finish."""
import os
import subprocess
import sys
import time
import importlib.metadata
from experiment_common import ROOT, read_json, write_json
from prepare_qwen35_9b import OUT, ENV


def worker():
    deadline = time.monotonic()+12*3600
    formal = ROOT/'runs/route-b-staged-7b-20261016/formal'
    seeds = [42,137,256,512,1024]
    while time.monotonic()<deadline:
        weights = (OUT/'weights-ready.json').exists()
        environment = (OUT/'environment-ready.json').exists()
        training = sum((formal/f'{arm}-seed{seed}'/'complete.json').exists() for seed in seeds for arm in ('A1','A2','U'))
        write_json(OUT/'priority-preparation-status.json',{'weights_verified':weights,'environment_verified':environment,
                   'completed7b_trainings':training,'expected7b_trainings':15,'9b_gpu_smoke':'deferred_until7b15_trainings_complete'})
        if weights and environment:
            write_json(OUT/'cpu-infrastructure-complete.json',{'status':'passed_weights_and_environment',
                       'model':read_json(OUT/'weights-ready.json'),'environment':read_json(OUT/'environment-ready.json'),
                       'gpu_smoke_deferred_for7b_priority':True})
            if training==15:break
        time.sleep(10)
    else:
        raise RuntimeError('Keep9B GPU smoke pending;7B priority deadline reached')
    subprocess.run([str(ENV/'bin/python'),'-u',str(ROOT/'tools/smoke_qwen35_9b.py')],check=True,cwd=ROOT)
    original = read_json(OUT/'environment.lock.json')['shared_readonly_runtime']
    assert all(importlib.metadata.version(name)==version for name,version in original.items())
    result = {'status':'passed','weights':read_json(OUT/'weights-ready.json'),
              'environment':read_json(OUT/'environment-ready.json'),'gpu_smoke':read_json(OUT/'gpu-smoke.json'),
              'old7b_environment_unchanged':True,'training_memory_preflight_complete':False,
              'historical_mirror_install_failure':'Repaired via officialPyPI in isolated environment; logs retained'}
    write_json(OUT/'infrastructure-complete.json',result)
    write_json(OUT/'status.json',{'phase':'infrastructure_ready','training_memory_preflight_complete':False})
    if os.environ.get('GH_TOKEN'):
        from publish_artifacts import publish
        publish(OUT,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        token = sys.stdin.readline().strip()
        command = 'cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/prepare_qwen35_9b_after_7b.py --worker >> runs/qwen35-9b-infrastructure-20261016/priority-preparation.log 2>&1'
        subprocess.run(['screen','-dmS','qwen35-priority','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('9B weights/environment continue;GPU smoke deferred until15 formal7B trainings complete')
