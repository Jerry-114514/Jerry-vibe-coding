"""Store necessary artifacts directly from AutoDL in GitHub Release assets."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import tarfile
from urllib.parse import quote
import requests
from experiment_common import ROOT, sha, write_json

REPO='Jerry-114514/Jerry-vibe-coding'


def publish(run, tag, token=None, include_training=False, include_dataset=False, include_teacher=False):
    token=token or os.environ.get('GH_TOKEN')
    if not token:
        token=sys.stdin.readline().strip()
    if not token:
        raise RuntimeError('GitHub credential required')
    session=requests.Session()
    session.trust_env=False
    session.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                            'X-GitHub-Api-Version':'2022-11-28'})
    base=f'https://api.github.com/repos/{REPO}'
    found=session.get(base+'/releases/tags/'+quote(tag,safe=''),timeout=30)
    if found.status_code==404:
        response=session.post(base+'/releases',json={'tag_name':tag,'target_commitish':'codex/arp-autodl',
                              'name':'ARP '+tag,'body':'Remote AutoDL experiment artifacts. Smoke results are validation only, not formal experimental results.',
                              'draft':False,'prerelease':True},timeout=30)
        response.raise_for_status()
        release=response.json()
    else:
        found.raise_for_status()
        release=found.json()
    run=Path(run).resolve()
    if not run.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Only experiment run folders may be exported')
    paths=[run,ROOT/'configs']
    if include_training:
        paths.append(ROOT/'data/training')
    if include_dataset:
        paths += [ROOT/'data/images',ROOT/'data/raw']
    export=ROOT/'.exports'
    export.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=export/f'{tag}-{stamp}.tar.gz'
    allowed=[]
    for directory in paths:
        for path in sorted(directory.rglob('*')):
            blocked=['.cache','__pycache__']
            if not include_teacher:
                blocked.append('teacher-cache')
            if directory==ROOT/'data/training':
                blocked.append('raw')
            if path.is_file() and not path.is_symlink() and not any(x in path.parts for x in blocked):
                allowed.append(path)
    inventory={'repository':REPO,'tag':tag,'created_utc':stamp,
               'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in allowed]}
    inventory_path=export/f'{tag}-{stamp}-inventory.json'
    write_json(inventory_path,inventory)
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for path in allowed:
            tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    # Binary assets are split well below GitHub's per-asset limit; never Git blobs.
    uploads=[inventory_path]
    if archive.stat().st_size<=1024**3:
        uploads.append(archive)
    else:
        with archive.open('rb') as source:
            for index in range((archive.stat().st_size+1024**3-1)//1024**3):
                part=archive.with_name(archive.name+f'.part{index:03d}')
                with part.open('wb') as out:
                    remaining=1024**3
                    while remaining:
                        block=source.read(min(8*1024**2,remaining))
                        if not block:
                            break
                        out.write(block)
                        remaining-=len(block)
                uploads.append(part)
    uploaded=[]
    upload_url=release['upload_url'].split('{')[0]
    for path in uploads:
        with path.open('rb') as payload:
            response=session.post(upload_url,params={'name':path.name},data=payload,
                                  headers={'Content-Type':'application/octet-stream','Content-Length':str(path.stat().st_size)},timeout=(30,600))
        response.raise_for_status()
        result=response.json()
        uploaded.append({'name':path.name,'bytes':path.stat().st_size,'sha256':sha(path),
                         'url':result['browser_download_url']})
        print('Uploaded:',path.name,path.stat().st_size,flush=True)
    report={'release_url':release['html_url'],'assets':uploaded,'storage':'AutoDL -> GitHub; no local download'}
    write_json(run/'github-storage.json',report)
    print(report['release_url'],flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--include-training',action='store_true')
    p.add_argument('--include-dataset',action='store_true')
    p.add_argument('--include-teacher-cache',action='store_true')
    a=p.parse_args()
    publish(a.run,a.tag,include_training=a.include_training,include_dataset=a.include_dataset,include_teacher=a.include_teacher_cache)
