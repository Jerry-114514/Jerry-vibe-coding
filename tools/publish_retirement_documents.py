"""Publish human-readable final backup entry points without modifying research files."""
from pathlib import Path
import gzip
import hashlib
import json
import sys
import time

ROOT=Path('/root/autodl-tmp/arp')
sys.path.insert(0,str(ROOT/'tools'))
from audit_instance_retirement import OUT,TAG,REPO,read,write,github,now,sha

token=sys.stdin.readline().strip();assert token
ready=read(OUT/'release-ready.json');assert ready['status']=='passed' and ready['missing_files']==0
folder=OUT/'release-documents';folder.mkdir(exist_ok=True)
docs=ROOT/'tmp/autodl/retirement-documents'
mapping=OUT/'release-audit/restore-map.compact.json';compressed=folder/'INSTANCE_RESTORE_MAP.compact.json.gz'
compressed.write_bytes(gzip.compress(mapping.read_bytes(),compresslevel=6,mtime=0))
files={'INSTANCE_RELEASE_READY.json':OUT/'release-ready.json','INSTANCE_RESTORE_MAP.compact.json.gz':compressed,'INSTANCE_RETIREMENT_REPORT.md':docs/'README.md','RESTORE.md':docs/'RESTORE.md','RESTORE_TOOL.py':ROOT/'tools/restore_instance_retirement.py','RESTORE_TOOL_TESTS.json':docs/'restore-tool-tests.json'}
session=github(token);base=f'https://api.github.com/repos/{REPO}'
response=session.get(base+'/releases/tags/'+TAG,timeout=45);response.raise_for_status();release=response.json()
existing={}
for page in range(1,100):
    response=session.get(base+f'/releases/{release["id"]}/assets',params={'page':page,'per_page':100},timeout=45);response.raise_for_status();rows=response.json();existing.update({r['name']:r for r in rows})
    if len(rows)<100:break
assets=[]
for label,path in files.items():
    digest=sha(path);name=label
    if name in existing:
        if existing[name]['digest']=='sha256:'+digest and existing[name]['size']==path.stat().st_size:
            assets.append({'name':name,'bytes':path.stat().st_size,'sha256':digest,'url':existing[name]['browser_download_url']});continue
        name=Path(label).stem+'-'+str(time.time_ns())+Path(label).suffix
    with path.open('rb') as stream:
        response=session.post(release['upload_url'].split('{')[0],params={'name':name},data=stream,headers={'Content-Type':'application/octet-stream','Content-Length':str(path.stat().st_size)},timeout=(30,180));response.raise_for_status();actual=response.json()
    fresh=session.get(actual['url'],timeout=45);fresh.raise_for_status();record=fresh.json()
    assert record['size']==path.stat().st_size and record.get('digest')=='sha256:'+digest
    assets.append({'name':name,'bytes':path.stat().st_size,'sha256':digest,'url':record['browser_download_url']});print('Verified final document:',name,flush=True)
body=f'''Final release-before-retirement backup. Status: PASSED.

- {ready['selected_files']:,} existing research files covered; zero missing.
- {ready['previously_offloaded_files_added']} previously offloaded files added to the restore map.
- {ready['verified_endpoint_digests']}/{ready['required_endpoint_digests']} endpoint/ledger/recovery digests verified against current GitHub assets.
- Actual GitHub archive roundtrip: {ready['restore_roundtrip_members']} members verified, three adapter families readable and finite.
- Raw results, teacher caches, adapters, recovery states, failure records and candidate data are retained across this and the three historical releases.

Direct entry points: INSTANCE_RELEASE_READY.json, INSTANCE_RESTORE_MAP.compact.json.gz, INSTANCE_RETIREMENT_REPORT.md and RESTORE.md.

Public pretrained weight binaries and reinstallable environment binaries are not mirrored. Exact metadata and dependency pins are archived. Credentials are excluded. This is not a full virtual-machine image. The original ARP scientific gate remains unchanged.
'''
response=session.patch(base+f'/releases/{release["id"]}',json={'name':'AutoDL instance retirement — verified research backup (2026-10-20)','body':body},timeout=45);response.raise_for_status()
write(folder/'documents-delivery-proof.json',{'status':'passed','utc':now(),'release_url':release['html_url'],'assets':assets,'fresh_server_size_and_sha256_verified':True})
print('Release entry points and description updated',flush=True)
