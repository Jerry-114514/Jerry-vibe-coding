"""Publish a compact reproducibility capsule and test an actual GitHub restore."""
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path('/root/autodl-tmp/arp')
sys.path.insert(0,str(ROOT/'tools'))
from audit_instance_retirement import OUT,TAG,SECRET,sha,write,read,github,all_assets,now

def verified_publish(folder,token):
    from publish_artifacts import publish
    receipt=publish(folder,TAG,token=token,include_teacher=True)
    session=github(token);assets,_=all_assets(session)
    for record in receipt['assets']:
        actual=assets[(TAG,record['name'])]
        assert actual['size']==record['bytes'] and actual.get('digest')=='sha256:'+record['sha256']
    write(folder/'delivery-proof.json',{'status':'passed','utc':now(),'receipt':receipt,'fresh_server_size_and_sha256_verified':True})
    return receipt

def capsule(token):
    folder=OUT/'reproduction-capsule';folder.mkdir(exist_ok=True)
    originals={}
    for base in [OUT/'reproducibility',ROOT/'configs',ROOT/'tools',ROOT/'scripts',ROOT/'tests']:
        for path in base.rglob('*'):
            if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
            relative=path.relative_to(ROOT);dest=folder/'files'/relative
            assert path.stat().st_size<40*1024**2
            data=path.read_bytes();assert not SECRET.search(data),('Secret-like capsule source',relative.as_posix())
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
            originals[(Path('files')/relative).as_posix()]={'restore_to':relative.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}
    for name in ['environment.json','arp-venv-requirements.txt','arp-qwen35-venv-requirements.txt','prepare-summary.json','quiesced-monitors.json']:
        path=OUT/name;dest=folder/'audit-metadata'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
        originals[dest.relative_to(folder).as_posix()]={'restore_to':'retirement-metadata/'+name,'bytes':path.stat().st_size,'sha256':sha(path)}
    paths=[ROOT/'runs/route-b-staged-7b-20261016/formal/A1-seed42/adapter/adapter_model.safetensors',ROOT/'runs/route-b-9b-122b-20261017/student-experiment/formal/A1-seed42/adapter/adapter_model.safetensors',OUT.parent/'9b-u137-factorial-20261018/training/N-G-seed42/adapter/adapter_model.safetensors']
    for path in paths:
        assert path.is_file(),str(path)
        for source in [path,path.parent/'adapter_config.json']:
            relative=source.relative_to(ROOT);dest=folder/'files'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():os.link(source,dest)
            originals[dest.relative_to(folder).as_posix()]={'restore_to':relative.as_posix(),'bytes':source.stat().st_size,'sha256':sha(source)}
    write(folder/'restore-index.json',{'utc':now(),'files':originals,'scope':'Exact running source, model metadata/tokenizers, environment pins and representative7B/9B/factorial adapters. Full checkpoint coverage is independently audited.'})
    receipt=verified_publish(folder,token)
    binary=[a for a in receipt['assets'] if not a['name'].endswith('-inventory.json')]
    assert len(binary)==1 and binary[0]['name'].endswith('.tar.gz')
    session=github(token);assets,_=all_assets(session);asset=assets[(TAG,binary[0]['name'])]
    download=OUT/'restore-test'/asset['name'];download.parent.mkdir(exist_ok=True)
    digest=hashlib.sha256()
    response=session.get(asset['browser_download_url'],params={'restore_check':str(time.time_ns())},timeout=(30,90),stream=True);response.raise_for_status()
    with download.open('wb') as f:
        for chunk in response.iter_content(1024**2):
            if chunk:f.write(chunk);digest.update(chunk)
    assert download.stat().st_size==binary[0]['bytes'] and digest.hexdigest()==binary[0]['sha256']
    prefix=folder.relative_to(ROOT).as_posix()+'/'
    checked=[];adapters=[]
    import torch
    from safetensors.torch import load as load_safetensors
    with tarfile.open(download,'r:gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(prefix):continue
            relative=member.name[len(prefix):]
            if relative not in originals:continue
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            data=archive.extractfile(member).read();expected=originals[relative]
            assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256']
            checked.append(relative)
            if relative.endswith('adapter_model.safetensors'):
                tensors=load_safetensors(data)
                assert tensors and all(torch.isfinite(v).all().item() for v in tensors.values())
                adapters.append({'archive_member':member.name,'tensors':len(tensors),'all_finite':True})
    assert set(checked)==set(originals)
    assert len(adapters)==3
    result={'status':'passed','utc':now(),'downloaded_from_github':True,'archive_bytes':download.stat().st_size,'archive_sha256':digest.hexdigest(),'verified_members':len(checked),'adapters':adapters,'cuda_initialized':torch.cuda.is_initialized(),'no_extraction_over_original_files':True,'receipt':receipt}
    write(OUT/'restore-test/result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='receipt'}),flush=True)

def finalize(token):
    coverage=read(OUT/'coverage.json');endpoints=read(OUT/'endpoint-digest-verification.json');restored=read(OUT/'restore-test/result.json')
    assert coverage['missing_files']==0 and not endpoints['missing'] and restored['status']=='passed'
    snapshot=read(OUT/'selected-files.json');changed=[]
    for row in snapshot['files']:
        p=ROOT/row['path']
        if not p.is_file() or p.stat().st_size!=row['bytes'] or p.stat().st_mtime_ns!=row['mtime_ns']:changed.append(row['path'])
    assert not changed,('Files changed after final snapshot',changed)
    active=[]
    names={b'train_9b_repair.py',b'evaluate_9b_repair.py',b'run_9b_repair.py',b'run_9b_native_frontend_v41.py'}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=path.read_bytes().split(bytes([0]))
            if any(arg.rsplit(b'/',1)[-1] in names for arg in args):active.append(path.parent.name)
        except OSError:pass
    assert not active,('Experiment processes still active',active)
    report=OUT/'release-audit';report.mkdir(exist_ok=True)
    for name in ['coverage.json','coverage-before-backup.json','prepare-summary.json','selected-files.json','gap-backup-complete.json','endpoint-digest-verification.json','environment.json','arp-venv-requirements.txt','arp-qwen35-venv-requirements.txt','base-model-inventory.json','quiesced-monitors.json']:
        if (OUT/name).exists():shutil.copy2(OUT/name,report/name)
    shutil.copy2(OUT/'restore-test/result.json',report/'restore-roundtrip.json')
    shutil.copy2(OUT/'reproduction-capsule/delivery-proof.json',report/'reproduction-capsule-delivery-proof.json')
    mapping=read(OUT/'restore-map.json');archives={};ids={};files=[]
    present={row['path'] for row in mapping['files']};offloaded={}
    def add_offloaded(path,digest,size):
        if path in present:return
        relative=Path(path)
        assert not relative.is_absolute() and '..' not in relative.parts and path.startswith('runs/')
        location=endpoints['restore_locations'][digest]
        assert location['bytes']==size
        offloaded[path]={'path':path,'bytes':size,'sha256':digest,'restore_location':location,'previously_offloaded':True}
    def walk_pointers(value):
        if isinstance(value,dict):
            if value.get('original_path') and value.get('sha256') and isinstance(value.get('bytes'),int):add_offloaded(value['original_path'],value['sha256'],value['bytes'])
            for child in value.values():walk_pointers(child)
        elif isinstance(value,list):
            for child in value:walk_pointers(child)
    for row in snapshot['files']:
        basename=Path(row['path']).name
        if basename=='resume-offloaded.json':
            marker=read(ROOT/row['path']);add_offloaded((Path(row['path']).parent/'resume.pt').as_posix(),marker['resume_sha256'],marker['bytes'])
        elif 'restore-pointer' in row['path']:walk_pointers(read(ROOT/row['path']))
    mapping['files'].extend(offloaded.values())
    for row in mapping['files']:
        location=row['restore_location'];key=(location['release_url'],location['archive']['name'])
        if key not in ids:
            ids[key]=str(len(ids));archives[ids[key]]={k:v for k,v in location.items() if k!='archive_member'}
        files.append({'path':row['path'],'bytes':row['bytes'],'sha256':row['sha256'],'archive_id':ids[key],'archive_member':location['archive_member']})
    write(report/'restore-map.compact.json',{'schema':'instance-retirement-content-map-v1','utc':now(),'archives':archives,'files':files})
    shutil.copy2(ROOT/'tools/restore_instance_retirement.py',report/'restore_instance_retirement.py')
    checks={'status':'passed','utc':now(),'selected_files':len(snapshot['files']),'restore_map_files':len(files),'previously_offloaded_files_added':len(offloaded),'selected_logical_bytes':sum(r['bytes'] for r in files),'all_selected_contents_fresh_server_verified':True,'missing_files':0,'required_endpoint_digests':endpoints['required_unique_digests'],'verified_endpoint_digests':endpoints['verified_unique_digests'],'includes_offloaded_recovery_states':True,'restore_roundtrip_members':restored['verified_members'],'restored_adapter_families':len(restored['adapters']),'no_active_experiment_processes':True,'unchanged_since_final_hash_snapshot':True,'no_original_data_deleted':True,'user_releases_instance':True,'whole_machine_image':False,'public_base_weights_mirrored':False,'credentials_archived':False,'free_gib':shutil.disk_usage(ROOT).free/2**30,'restore_map_sha256':sha(report/'restore-map.compact.json')}
    write(report/'release-ready.json',checks)
    report_receipt=verified_publish(report,token)
    final={**checks,'report_receipt':report_receipt,'capsule_receipt':read(OUT/'reproduction-capsule/delivery-proof.json')['receipt']}
    terminal=OUT/'terminal-proof';write(terminal/'release-ready.json',final)
    verified_publish(terminal,token)
    write(OUT/'release-ready.json',final)
    print(json.dumps(checks),flush=True)

if __name__=='__main__':
    token=sys.stdin.readline().strip();assert token
    if len(sys.argv)>1 and sys.argv[1]=='finalize':finalize(token)
    else:capsule(token)
