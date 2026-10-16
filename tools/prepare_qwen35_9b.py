"""Prepare pinned official9B weights and a separate Transformers environment.

No changes to the live7B virtual environment. CPU/network work only; GPU smoke
waits for the7B minimal-loop acceptance marker and takes the shared GPU lock.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import requests

from experiment_common import ROOT, read_json, read_rows, sha, write_json

OUT = ROOT/'runs/qwen35-9b-infrastructure-20261016'
MODEL = ROOT/'.model_cache/qwen35_9b'
ENV = Path('/root/autodl-tmp/arp-qwen35-venv')
REPO = 'Qwen/Qwen3.5-9B'


def metadata():
    from huggingface_hub import HfApi
    OUT.mkdir(parents=True,exist_ok=True)
    lock = OUT/'model.lock.json'
    if lock.exists():
        return read_json(lock)
    info = HfApi(endpoint='https://hf-mirror.com').model_info(REPO,files_metadata=True)
    files = []
    for f in info.siblings:
        if not f.rfilename.endswith(('.safetensors','.json','.jinja','.txt','.model')):
            continue
        lfs = f.lfs
        digest = lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
        files.append({'file':f.rfilename,'bytes':f.size,'sha256':digest,'git_blob_sha1':getattr(f,'blob_id',None)})
    result = {'repo_id':REPO,'revision':info.sha,'files':files,
              'weights_bytes':sum(f['bytes'] for f in files if f['file'].endswith('.safetensors')),
              'source':'Official Qwen repository; mirror orModelScope accepted only with pinned SHA256 match',
              'model_card':'https://huggingface.co/Qwen/Qwen3.5-9B'}
    write_json(lock,result)
    return result


def download(spec):
    import hashlib
    from download_ranges import download_ranges
    MODEL.mkdir(parents=True,exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    for entry in spec['files']:
        name = entry['file']
        if name.endswith('.safetensors'):
            continue
        target = MODEL/name
        if not target.exists():
            response = session.get(f'https://hf-mirror.com/{REPO}/resolve/{spec["revision"]}/{name}',timeout=(20,120))
            response.raise_for_status()
            content = response.content
            if entry['sha256']:
                assert hashlib.sha256(content).hexdigest() == entry['sha256']
            elif entry['git_blob_sha1']:
                assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() == entry['git_blob_sha1']
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
    weights = [e for e in spec['files'] if e['file'].endswith('.safetensors')]
    for i,entry in enumerate(weights):
        if shutil.disk_usage(MODEL).free < sum(e['bytes'] for e in weights[i:] if not (MODEL/e['file']).exists())+12*1024**3:
            raise RuntimeError('Need12GiB reserve for ongoing7B checkpoints; do not evict live experiment data')
        alternate = f'https://modelscope.cn/api/v1/models/{REPO}/repo?Revision=master&FilePath={entry["file"]}'
        try:
            download_ranges(alternate,MODEL/entry['file'],entry['bytes'],entry['sha256'],workers=4,direct=True)
        except RuntimeError:
            url = f'https://hf-mirror.com/{REPO}/resolve/{spec["revision"]}/{entry["file"]}'
            download_ranges(url,MODEL/entry['file'],entry['bytes'],entry['sha256'],workers=4,direct=True)
    write_json(MODEL/'weights.verified.json',{'revision':spec['revision'],'files':weights})
    write_json(OUT/'weights-ready.json',{'status':'passed','bytes':spec['weights_bytes'],'revision':spec['revision']})


def environment():
    versions = {name:importlib.metadata.version(name) for name in ('torch','torchvision','transformers','peft')}
    if not (ENV/'bin/python').exists():
        subprocess.run([sys.executable,'-m','venv',str(ENV)],check=True)
    site = next((ENV/'lib').glob('python*/site-packages'))
    original_site = str(Path(importlib.metadata.distribution('torch').locate_file('')).resolve())
    # The shared path is read-only to this installer; all upgrades go toENV first.
    (site/'shared-existing-runtime.pth').write_text(original_site+'\n')
    session = requests.Session()
    session.trust_env = False
    lock = OUT/'environment.lock.json'
    if lock.exists():
        pinned = read_json(lock)
    else:
        pinned = {'shared_readonly_runtime':versions,'packages':{}}
        for name in ('transformers','peft'):
            r = session.get(f'https://pypi.org/pypi/{name}/json',timeout=30)
            r.raise_for_status()
            pinned['packages'][name] = r.json()['info']['version']
        write_json(lock,pinned)
    constraint = OUT/'shared-runtime-constraints.txt'
    constraint.write_text('torch=='+versions['torch']+'\ntorchvision=='+versions['torchvision']+'\n')
    subprocess.run([str(ENV/'bin/python'),'-m','pip','install','--no-cache-dir','--upgrade',
                    '--constraint',str(constraint),*[f'{n}=={v}' for n,v in pinned['packages'].items()]],check=True)
    probe = 'import json,torch,transformers,peft;from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor; print(json.dumps({"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__,"qwen35_class_import":True}))'
    result = subprocess.run([str(ENV/'bin/python'),'-c',probe],capture_output=True,text=True)
    (OUT/'environment-probe.log').write_text(result.stdout+'\n'+result.stderr)
    result.check_returncode()
    assert {n:importlib.metadata.version(n) for n in versions} == versions
    freeze = subprocess.run([str(ENV/'bin/python'),'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (OUT/'requirements.frozen.txt').write_text(freeze.stdout)
    write_json(OUT/'environment-ready.json',{'status':'passed','path':str(ENV),
               'probe':json.loads(result.stdout.strip().splitlines()[-1]),'old_environment_unchanged':True})


def worker():
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        spec = metadata()
        if shutil.disk_usage(ROOT).free < spec['weights_bytes']+12*1024**3:
            raise RuntimeError('Insufficient disk for9B plus12GiB7B experiment reserve')
        write_json(OUT/'status.json',{'phase':'preparing_weights_and_environment','pid':os.getpid(),'weights_bytes':spec['weights_bytes']})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(download,spec),pool.submit(environment)]
            for future in futures:
                future.result()
        write_json(OUT/'status.json',{'phase':'weights_and_environment_ready_gpu_smoke_pending','revision':spec['revision']})
        deadline = time.monotonic()+8*3600
        minimum = ROOT/'runs/route-b-staged-7b-20261016/minimal-acceptance.json'
        failure = ROOT/'runs/route-b-staged-7b-20261016/failure.json'
        while not minimum.exists():
            if failure.exists() or time.monotonic()>deadline:
                raise RuntimeError('7B minimal loop not accepted; keep9B GPU smoke pending')
            time.sleep(10)
        subprocess.run([str(ENV/'bin/python'),'-u',str(ROOT/'tools/smoke_qwen35_9b.py')],check=True,cwd=ROOT)
        write_json(OUT/'status.json',{'phase':'infrastructure_ready','training_preflight_complete':False})
    except Exception as error:
        write_json(OUT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        if os.environ.get('GH_TOKEN'):
            from publish_artifacts import publish
            publish(OUT,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        token = sys.stdin.readline().strip()
        if not token:
            raise RuntimeError('Backup credential required')
        OUT.mkdir(parents=True,exist_ok=True)
        if 'qwen35-infra' in subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout:
            raise RuntimeError('Infrastructure job already exists')
        command = 'source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/prepare_qwen35_9b.py --worker >> runs/qwen35-9b-infrastructure-20261016/console.log 2>&1'
        subprocess.run(['screen','-dmS','qwen35-infra','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Started9B CPU/network infrastructure preparation;7B GPU work has priority')
