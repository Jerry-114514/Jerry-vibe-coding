"""Capture reproducibility metadata and installed Python-source modifications only."""
import base64
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path('/root/autodl-tmp/arp')
OUT=ROOT/'runs/instance-retirement-20261020'
sys.path.insert(0,str(ROOT/'tools'))
from audit_instance_retirement import SECRET,sha,write,now

def models():
    records=[]
    for name,base in [('project-model-cache',ROOT/'.model_cache'),('historical-model-cache',Path('/root/arp-model-cache'))]:
        if not base.exists():continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path.stat().st_size>32*1024**2:continue
            if path.suffix not in ['.json','.txt','.md','.py','.model','.jinja','.jinja2','.tiktoken','.spm','.bpe','.vocab','.yaml','.yml'] or path.name.endswith('.ranges.json'):continue
            data=path.read_bytes()
            assert not SECRET.search(data),('Do not archive secret-like model metadata',str(path))
            target=OUT/'reproducibility'/name/path.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            records.append({'original_path':str(path),'stored_relative':target.relative_to(OUT).as_posix(),'bytes':len(data),'sha256':sha(path)})
    test=Path('/root/test_api.py')
    if test.exists():
        raw=test.read_bytes();redacted=SECRET.sub(b'REDACTED_USE_ENVIRONMENT_VARIABLE',raw)
        target=OUT/'reproducibility/external-scripts/test_api.py';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(redacted)
        records.append({'original_path':str(test),'stored_relative':target.relative_to(OUT).as_posix(),'bytes':len(redacted),'sha256':sha(target),'credential_patterns_redacted':raw!=redacted})
    write(OUT/'reproducibility/model-and-external-source-index.json',{'utc':now(),'files':records,'excluded_external_data':'/root/data contains only a public MNIST example archive; tf-logs is empty','base_weights':'Public pretrained safetensors excluded; retain pinned identities, metadata and existing verification hashes'})

def runtime(label):
    result=[]
    for name in ['torch','transformers','peft','safetensors','qwen-vl-utils','accelerate']:
        try:dist=metadata.distribution(name)
        except metadata.PackageNotFoundError:continue
        row={'name':name,'version':dist.version,'checked_source_files':0,'modified_source_files':[]}
        recorded=set()
        for item in dist.files or []:
            path=Path(dist.locate_file(item));recorded.add(path)
            if path.suffix not in ['.py','.json','.txt'] or not item.hash or item.hash.mode!='sha256' or not path.is_file():continue
            row['checked_source_files']+=1
            data=path.read_bytes();actual=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
            if actual==item.hash.value:continue
            assert not SECRET.search(data),('Modified runtime file contains secret-like text',name,str(item))
            relative=Path(str(item));assert '..' not in relative.parts and not relative.is_absolute()
            dest=OUT/'reproducibility/runtime-patches'/label/name/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            row['modified_source_files'].append({'installed_relative':str(item),'stored_relative':dest.relative_to(OUT).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'wheel_record_sha256_urlsafe':item.hash.value})
        result.append(row)
    write(OUT/f'reproducibility/{label}-runtime-integrity.json',{'utc':now(),'packages':result,'scope':'Python/JSON/text files with wheel RECORD SHA256 in six critical packages; unchanged package binaries are reinstallable, not archived'})

if __name__=='__main__':
    if sys.argv[1]=='models':models()
    else:runtime(sys.argv[1])
