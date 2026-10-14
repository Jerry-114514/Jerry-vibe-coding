"""Remote resumable range downloader with explicit length and SHA-256 checks."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import requests
from experiment_common import sha, write_json


def _download_ranges(url, destination, size, expected_sha=None, workers=8, direct=False):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != size or (expected_sha and sha(destination)!=expected_sha):
            raise ValueError('Existing download digest mismatch: '+destination.name)
        return destination
    partial=destination.with_suffix(destination.suffix+'.range-partial')
    journal=destination.with_suffix(destination.suffix+'.ranges.json')
    chunk=16*1024*1024
    spec={'size':size,'sha256':expected_sha,'chunk':chunk}
    state=json.loads(journal.read_text()) if journal.exists() else {**spec,'done':[]}
    if state.get('size') is None and not state.get('done') and (not partial.exists() or partial.stat().st_size==0):
        state={**spec,'done':[]}
    if any(state[k]!=v for k,v in spec.items()):
        raise ValueError('Range resume metadata mismatch')
    if not partial.exists() or partial.stat().st_size==0:
        with partial.open('wb') as f:
            f.truncate(size)
        state['done']=[]
    write_json(journal,state)
    lock=threading.Lock()
    def get(index):
        if index in state['done']:
            return
        lo=index*chunk
        hi=min(size-1,lo+chunk-1)
        last=None
        for attempt in range(5):
            try:
                session=requests.Session()
                session.trust_env=not direct
                start=time.monotonic()
                with session.get(url,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(30,120)) as response:
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {lo}-{hi}/{size}':
                        raise RuntimeError('Server did not honor byte range: '+str(response.status_code))
                    count=0
                    with partial.open('r+b',buffering=0) as f:
                        f.seek(lo)
                        for data in response.iter_content(1024*1024):
                            if count+len(data)>hi-lo+1:
                                raise RuntimeError('Range exceeded expected length')
                            f.write(data)
                            count+=len(data)
                        if count != hi-lo+1:
                            raise RuntimeError('Incomplete range')
                with lock:
                    state['done'].append(index)
                    write_json(journal,state)
                    if len(state['done'])%8==0 or len(state['done'])==((size+chunk-1)//chunk):
                        print(f'{destination.name}: {min(len(state["done"])*chunk,size)/1024**3:.2f}/{size/1024**3:.2f} GiB',flush=True)
                return
            except (requests.RequestException, RuntimeError) as error:
                last=type(error).__name__
                time.sleep(min(2**attempt,10))
        raise RuntimeError('Range download failed after retries: '+destination.name+' '+str(index)+' '+last)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(get,range((size+chunk-1)//chunk)))
    if partial.stat().st_size != size or (expected_sha and sha(partial)!=expected_sha):
        raise ValueError('Downloaded file SHA256 mismatch: '+destination.name)
    partial.replace(destination)
    return destination


def download_ranges(url, destination, size, expected_sha=None, workers=8, direct=False):
    import fcntl
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.with_suffix(destination.suffix+'.download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            return _download_ranges(url,destination,size,expected_sha,workers,direct)
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)
