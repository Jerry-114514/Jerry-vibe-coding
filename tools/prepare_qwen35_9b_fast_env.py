"""Fetch the large publicwheel with verified ranges, then resume isolated install."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import subprocess
import sys
import time
import requests
from experiment_common import ROOT, read_json, write_json
import prepare_qwen35_9b_official_index as repair


def wheel():
    folder = repair.setup.OUT/'wheels'
    folder.mkdir(exist_ok=True)
    version = read_json(repair.setup.OUT/'environment.lock.json')['packages']['transformers']
    session = requests.Session();session.trust_env = False
    result = session.get(f'https://pypi.org/pypi/transformers/{version}/json',timeout=30)
    result.raise_for_status()
    spec = next(r for r in result.json()['urls'] if r['packagetype'] == 'bdist_wheel')
    destination = folder/spec['filename']
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == spec['digests']['sha256']:
        return destination
    partial = destination.with_suffix('.partial')
    with partial.open('wb') as stream:
        stream.truncate(spec['size'])
    chunk = 256*1024
    def part(index):
        lo = index*chunk;hi = min(spec['size']-1,lo+chunk-1)
        for attempt in range(3):
            try:
                s = requests.Session()
                with s.get(spec['url'],headers={'Range':f'bytes={lo}-{hi}'},timeout=(15,45)) as response:
                    assert response.status_code == 206 and response.headers['Content-Range'] == f'bytes {lo}-{hi}/{spec["size"]}'
                    content = response.content
                    assert len(content) == hi-lo+1
                with partial.open('r+b') as stream:
                    stream.seek(lo);stream.write(content)
                return
            except (requests.RequestException,AssertionError):
                if attempt == 2:raise
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(part,range((spec['size']+chunk-1)//chunk)))
    assert hashlib.sha256(partial.read_bytes()).hexdigest() == spec['digests']['sha256']
    partial.replace(destination)
    write_json(repair.setup.OUT/'wheel-download.json',{'file':spec['filename'],'sha256':spec['digests']['sha256'],
               'bytes':spec['size'],'source':'OfficialPyPI metadata and CDN; exact ranged byte coverage'})
    return destination


def worker():
    destination = wheel()
    original_run = repair.original_run
    def localwheel_run(args,**kwargs):
        if isinstance(args,list) and 'pip' in args and 'install' in args:
            args = [str(destination) if isinstance(x,str) and x.startswith('transformers==') else x for x in args]
        return original_run(args,**kwargs)
    repair.original_run = localwheel_run
    repair.worker()


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        token = sys.stdin.readline().strip()
        command = 'source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/prepare_qwen35_9b_fast_env.py --worker >> runs/qwen35-9b-infrastructure-20261016/fast-env.log 2>&1'
        subprocess.run(['screen','-dmS','qwen35-fast-env','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('StartedSHA-verified ranged wheel fetch and isolated environment installation')
