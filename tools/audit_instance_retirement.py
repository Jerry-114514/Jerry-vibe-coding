"""Independent, non-destructive final backup and restore audit for AutoDL retirement."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/arp')
OUT=ROOT/'runs/instance-retirement-20261020'
REPO='Jerry-114514/Jerry-vibe-coding'
TAG='arp-instance-retirement-20261020'
TAGS=['arp-formal-20261014','arp-protocol-v2-20261016','arp-9b-122b-20261017',TAG]
SECRET=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:ws-)?[A-Za-z0-9_.-]{25,}|hf_[A-Za-z0-9]{30,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)')
SKIP_PARTS={'.git','__pycache__','.pytest_cache','node_modules','.cache'}

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_name(p.name+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            block=f.read(8*1024**2)
            if not block:break
            h.update(block)
    return h.hexdigest()
def log(message):print(now(),message,flush=True)
def selected():
    for directory in ['runs','data','configs','tools','scripts','tests','docs']:
        for parent,dirs,files in os.walk(ROOT/directory,followlinks=False):
            dirs[:]=[d for d in dirs if d not in SKIP_PARTS and not (Path(parent)/d).is_symlink() and Path(parent)/d!=OUT]
            for name in sorted(files):
                path=Path(parent)/name
                if path.is_symlink() or name in ['.DS_Store','.gpu.lock'] or name.endswith(('.pyc','.lock')):continue
                if name.startswith('.env') or name in ['credentials','id_ed25519','known_hosts']:continue
                yield path
    for name in ['README.md','.gitattributes','.gitignore','requirements.txt','requirements-experiment.txt']:
        if (ROOT/name).is_file():yield ROOT/name

def prepare():
    OUT.mkdir(exist_ok=True)
    environments={}
    for name in ['arp-venv','arp-qwen35-venv']:
        py=ROOT.parent/name/'bin/python'
        result=subprocess.run([str(py),'-m','pip','freeze'],capture_output=True,text=True)
        # No credential-bearing URL is persisted in environment exports.
        frozen=re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@',r'\1<redacted>@',result.stdout)
        assert not SECRET.search(frozen.encode()),'Environment export contains a secret-like token'
        (OUT/f'{name}-requirements.txt').write_text(frozen)
        environments[name]={'python':subprocess.run([str(py),'--version'],capture_output=True,text=True).stdout.strip(),'freeze_exit_code':result.returncode}
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    extra={}
    for folder in [Path('/root'),Path('/root/autodl-fs')]:
        if folder.exists():extra[str(folder)]=[{'name':p.name,'directory':p.is_dir(),'symlink':p.is_symlink()} for p in folder.iterdir() if not p.name.startswith('.')]
    write(OUT/'environment.json',{'utc':now(),'gpu':gpu,'environments':environments,'os_release':Path('/etc/os-release').read_text(),'outside_project_listing':extra,'excluded':['credentials, SSH keys, .bashrc and environment-variable values','reinstallable virtual-environment binaries','public pretrained base-weight binaries; pinned metadata retained','reproducible package/download caches and .exports duplicates']})
    entries=[];links=[];inode_cache={};suspicious=[]
    for index,path in enumerate(selected()):
        stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in inode_cache:inode_cache[key]=sha(path)
        after=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            stat=after;inode_cache[key]=sha(path)
        relative=path.relative_to(ROOT).as_posix()
        if stat.st_size<32*1024**2 and path.suffix.lower() in ['.json','.jsonl','.txt','.md','.py','.sh','.log','.yaml','.yml','.toml']:
            if SECRET.search(path.read_bytes()):suspicious.append(relative)
        entries.append({'path':relative,'bytes':stat.st_size,'sha256':inode_cache[key],'mtime_ns':stat.st_mtime_ns})
        if index%5000==0:log(f'hashed {index} selected files')
    for directory in ['data','runs']:
        for parent,dirs,files in os.walk(ROOT/directory,followlinks=False):
            dirs[:]=[d for d in dirs if d not in SKIP_PARTS and Path(parent)/d!=OUT]
            for name in list(dirs)+files:
                p=Path(parent)/name
                if p.is_symlink():links.append({'path':p.relative_to(ROOT).as_posix(),'target':str(p.resolve()),'exists':p.exists()})
    model=[]
    for cache in ['.model_cache','.hf_cache']:
        base=ROOT/cache
        if not base.exists():continue
        for parent,dirs,files in os.walk(base,followlinks=False):
            dirs[:]=[d for d in dirs if d not in SKIP_PARTS]
            for name in files:
                path=Path(parent)/name
                if not path.exists():continue
                size=path.stat().st_size
                row={'path':path.relative_to(ROOT).as_posix(),'bytes':size,'symlink':path.is_symlink()}
                if size<4*1024**2 and path.suffix in ['.json','.txt','.md','.model']:
                    content=path.read_bytes()
                    if SECRET.search(content):continue
                    dest=OUT/'base-model-metadata'/cache.lstrip('.')/path.relative_to(base)
                    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content);row['sha256']=sha(path)
                model.append(row)
    write(OUT/'base-model-inventory.json',model)
    write(OUT/'selected-files.json',{'utc':now(),'files':entries,'symlinks':links,'suspicious_text_files':suspicious,'unique_inodes':len(inode_cache)})
    totals=Counter()
    for row in entries:totals[row['path'].split('/')[0]]+=row['bytes']
    write(OUT/'prepare-summary.json',{'utc':now(),'files':len(entries),'logical_bytes':sum(x['bytes'] for x in entries),'directory_bytes':dict(totals),'suspicious_text_files':suspicious,'symlinks':len(links),'free_gib':shutil.disk_usage(ROOT).free/2**30})
    log(f'prepared {len(entries)} files; suspicious paths: {suspicious}')

def github(token):
    import requests
    session=requests.Session();session.trust_env=False
    session.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    return session
def get(session,url,**kwargs):
    for attempt in range(3):
        try:
            response=session.get(url,timeout=(15,40),**kwargs);response.raise_for_status();return response
        except Exception:
            if attempt==2:raise
            time.sleep(min(30,2**attempt))
def all_assets(session):
    assets={};release_urls={}
    for tag in TAGS:
        url=f'https://api.github.com/repos/{REPO}/releases/tags/{tag}'
        response=session.get(url,timeout=45)
        if response.status_code==404:continue
        response.raise_for_status();release=response.json();release_urls[tag]=release['html_url']
        for page in range(1,100):
            rows=get(session,f'https://api.github.com/repos/{REPO}/releases/{release["id"]}/assets',params={'per_page':100,'page':page}).json()
            for row in rows:assets[(tag,row['name'])]=row
            if len(rows)<100:break
        else:raise RuntimeError('Asset pagination did not terminate')
    return assets,release_urls

def coverage(token):
    snapshot=read(OUT/'selected-files.json');session=github(token);assets,urls=all_assets(session)
    wanted={(x['sha256'],x['bytes']) for x in snapshot['files']};locations={};invalid=[]
    endpoint_hashes={}
    for row in snapshot['files']:
        if not row['path'].startswith('runs/'):continue
        basename=Path(row['path']).name
        if basename not in ['complete.json','scores.json','resume-offloaded.json'] and 'restore-pointer' not in row['path']:continue
        try:marker=read(ROOT/row['path'])
        except (OSError,ValueError):continue
        if not isinstance(marker,dict):continue
        for key in ['adapter_sha256','loss_sha256','resume_sha256','recovery_sha256','ledger_sha256']:
            value=marker.get(key)
            if isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value):endpoint_hashes.setdefault(value,[]).append({'marker':row['path'],'key':key})
        def pointer_hashes(value):
            if isinstance(value,dict):
                digest=value.get('sha256')
                if isinstance(digest,str) and value.get('original_path') and re.fullmatch('[0-9a-f]{64}',digest):endpoint_hashes.setdefault(digest,[]).append({'marker':row['path'],'key':'restore-pointer'})
                for child in value.values():pointer_hashes(child)
            elif isinstance(value,list):
                for child in value:pointer_hashes(child)
        pointer_hashes(marker)
    endpoint_found={}
    local={}
    for root in [ROOT/'runs',ROOT/'.exports']:
        for p in root.rglob('*inventory*.json'):
            if not p.is_file() or OUT in p.parents:continue
            try:local.setdefault(sha(p),p)
            except OSError:pass
    manifests=[(key,asset) for key,asset in assets.items() if asset['name'].endswith('-inventory.json')]
    manifests.sort(key=lambda x:x[1]['created_at'],reverse=True)
    used=[]
    for index,((tag,name),asset) in enumerate(manifests):
        digest=asset.get('digest','')
        if not digest.startswith('sha256:'):invalid.append({'asset':name,'reason':'server digest absent'});continue
        expected=digest.split(':',1)[1];cache=OUT/'server-inventories'/name
        if expected in local:content=local[expected].read_bytes()
        elif cache.exists() and sha(cache)==expected:content=cache.read_bytes()
        else:
            log('fetch inventory '+name)
            try:content=get(session,asset['url'],headers={'Accept':'application/octet-stream'}).content
            except Exception as error:
                invalid.append({'asset':name,'reason':'inventory temporarily unavailable','error_type':type(error).__name__})
                log('inventory unavailable; do not credit coverage: '+name)
                continue
            assert hashlib.sha256(content).hexdigest()==expected,('Inventory server SHA mismatch',name)
            cache.parent.mkdir(exist_ok=True);cache.write_bytes(content)
        assert len(content)==asset['size'] and hashlib.sha256(content).hexdigest()==expected
        data=json.loads(content)
        binary=data.get('binary_assets')
        if not binary:invalid.append({'asset':name,'reason':'no archive-part mapping'});continue
        good=True
        for part in binary:
            actual=assets.get((tag,part['name']))
            if not actual or actual['size']!=part['bytes'] or actual.get('digest')!='sha256:'+part['sha256']:good=False;break
        if not good:invalid.append({'asset':name,'reason':'archive asset missing or server SHA mismatch'});continue
        archive={'release_url':urls[tag],'inventory_asset':name,'inventory_sha256':expected,'archive':data.get('archive'),'binary_assets':binary}
        used.append(archive)
        for row in data.get('files',[]):
            key=(row.get('sha256'),row.get('bytes'))
            if key in wanted and key not in locations:
                locations[key]={**archive,'archive_member':row['path']}
            if row.get('sha256') in endpoint_hashes and row['sha256'] not in endpoint_found:
                endpoint_found[row['sha256']]={**archive,'archive_member':row['path'],'bytes':row['bytes']}
        if index%25==0:log(f'checked {index}/{len(manifests)} inventories; {len(locations)}/{len(wanted)} unique contents located')
    covered=[];missing=[]
    for row in snapshot['files']:
        key=(row['sha256'],row['bytes'])
        if key in locations:covered.append({**row,'restore_location':locations[key]})
        else:missing.append(row)
    write(OUT/'coverage.json',{'utc':now(),'status':'covered' if not missing else 'missing','selected_files':len(snapshot['files']),'covered_files':len(covered),'missing_files':len(missing),'missing_bytes':sum(x['bytes'] for x in missing),'server_assets':len(assets),'inventories_checked':len(manifests),'unusable_inventories':invalid,'release_urls':urls})
    write(OUT/'restore-map.json',{'files':covered})
    write(OUT/'missing-files.json',{'files':missing})
    write(OUT/'endpoint-digest-verification.json',{'utc':now(),'required_unique_digests':len(endpoint_hashes),'verified_unique_digests':len(endpoint_found),'missing':{h:refs for h,refs in endpoint_hashes.items() if h not in endpoint_found},'restore_locations':endpoint_found,'includes_offloaded_states':True})
    write(OUT/'server-asset-snapshot.json',{'utc':now(),'assets':[{k:v for k,v in a.items() if k in ['name','size','digest','browser_download_url','created_at','id']} for a in assets.values()]})
    log(f'coverage {len(covered)}/{len(snapshot["files"])}; missing {len(missing)} files, {sum(x["bytes"] for x in missing)/2**30:.3f} GiB')

def backup(token):
    from publish_artifacts import publish
    from collections import defaultdict
    missing=read(OUT/'missing-files.json')['files'];snapshot=read(OUT/'selected-files.json')
    assert not snapshot['suspicious_text_files'],'Resolve secret-like text before publishing selected artifacts'
    unique={}
    for row in missing:unique.setdefault((row['sha256'],row['bytes']),row)
    batches=[];batch=[];size=0
    for row in sorted(unique.values(),key=lambda x:(-x['bytes'],x['path'])):
        if batch and size+row['bytes']>512*1024**2:batches.append(batch);batch=[];size=0
        batch.append(row);size+=row['bytes']
    if batch:batches.append(batch)
    for index,rows in enumerate(batches):
        folder=OUT/'gap-batches'/f'batch-{index:03d}'
        if (folder/'delivery-proof.json').exists():continue
        folder.mkdir(parents=True,exist_ok=True)
        for row in rows:
            source=ROOT/row['path'];target=folder/'files'/row['path']
            assert source.resolve().is_relative_to(ROOT.resolve()) and source.stat().st_size==row['bytes']
            assert sha(source)==row['sha256'],('Source changed since snapshot',row['path'])
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                try:os.link(source,target)
                except OSError:shutil.copy2(source,target)
        write(folder/'restore-index.json',{'files':[{'archive_relative':'files/'+x['path'],'restore_to':x['path'],'bytes':x['bytes'],'sha256':x['sha256']} for x in rows]})
        receipt=publish(folder,TAG,token=token,include_teacher=True)
        session=github(token);assets,_=all_assets(session)
        for a in receipt['assets']:
            actual=assets[(TAG,a['name'])]
            assert actual['size']==a['bytes'] and actual.get('digest')=='sha256:'+a['sha256']
        write(folder/'delivery-proof.json',{'status':'passed','receipt':receipt,'fresh_server_size_and_sha256_verified':True,'utc':now()})
        log(f'gap batch {index+1}/{len(batches)} delivered')
    write(OUT/'gap-backup-complete.json',{'utc':now(),'batches':len(batches),'unique_files':len(unique),'unique_bytes':sum(x['bytes'] for x in unique.values()),'no_original_files_deleted':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','coverage','backup']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:
        token=sys.stdin.readline().strip();assert token,'Token must be supplied through encrypted stdin'
        if args.action=='coverage':coverage(token)
        else:backup(token)
