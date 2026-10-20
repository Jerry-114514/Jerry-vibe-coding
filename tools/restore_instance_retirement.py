"""Offline, hash-checked restoration from the final per-file archive map."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from urllib.parse import quote

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def safe_relative(value):
    p=PurePosixPath(value)
    if p.is_absolute() or not p.parts or '..' in p.parts or '\\' in value:raise ValueError('Unsafe restore path')
    return p

def groups(mapping,prefixes):
    result={}
    for row in mapping['files']:
        safe_relative(row['path'])
        if prefixes and not any(row['path'].startswith(p) for p in prefixes):continue
        if 'restore_location' in row:location=row['restore_location']
        else:location={**mapping['archives'][str(row['archive_id'])],'archive_member':row['archive_member']}
        archive=location['archive'];key=(location['release_url'],archive['name'])
        group=result.setdefault(key,{'location':location,'files':[]});group['files'].append({**row,'restore_location':location})
    return result

def plan(mapping,prefixes):
    assets={};count=0
    for group in groups(mapping,prefixes).values():
        location=group['location'];count+=len(group['files'])
        base=location['release_url'].replace('/releases/tag/','/releases/download/')
        for item in location['binary_assets']:
            safe_relative(item['name'])
            assets[item['name']]={**item,'url':base+'/'+quote(item['name'],safe='')}
    return {'files_to_restore':count,'download_bytes':sum(x['bytes'] for x in assets.values()),'assets':list(assets.values()),'download_separately':'This tool never requests or stores API credentials.'}

class JoinedParts(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.current=None
    def readable(self):return True
    def read(self,n=-1):
        if n<0:raise ValueError('Unbounded archive reads are disabled')
        out=bytearray()
        while len(out)<n:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:break
            data=self.current.read(n-len(out))
            if data:out.extend(data)
            else:self.current.close();self.current=None
        return bytes(out)
    def close(self):
        if self.current:self.current.close()
        super().close()

def restore(mapping,archive_dir,destination,prefixes=()):
    archive_dir=archive_dir.resolve();destination=destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    restored=skipped=0
    for group in groups(mapping,prefixes).values():
        location=group['location'];parts=sorted(location['binary_assets'],key=lambda x:x['name'])
        combined=hashlib.sha256();size=0;paths=[]
        for part in parts:
            safe_relative(part['name']);path=archive_dir/part['name']
            if not path.resolve().is_relative_to(archive_dir) or path.is_symlink():raise ValueError('Unsafe archive part')
            if path.stat().st_size!=part['bytes'] or file_hash(path)!=part['sha256']:raise ValueError('Archive part hash/size mismatch: '+part['name'])
            paths.append(path)
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024**2),b''):combined.update(chunk);size+=len(chunk)
        if size!=location['archive']['bytes'] or combined.hexdigest()!=location['archive']['sha256']:raise ValueError('Combined archive hash/size mismatch')
        needed={}
        for row in group['files']:
            safe_relative(row['restore_location']['archive_member'])
            target=destination.joinpath(*safe_relative(row['path']).parts)
            if not target.resolve().is_relative_to(destination) or target.is_symlink():raise ValueError('Restore destination escapes root')
            if target.exists():
                if target.is_file() and target.stat().st_size==row['bytes'] and file_hash(target)==row['sha256']:skipped+=1;continue
                raise FileExistsError('Refusing to overwrite a different existing file: '+row['path'])
            needed.setdefault(row['restore_location']['archive_member'],[]).append(row)
        with JoinedParts(paths) as joined,tarfile.open(fileobj=joined,mode='r|gz') as archive:
            for member in archive:
                if member.name not in needed:continue
                if not member.isfile():raise ValueError('Expected regular archive member')
                rows=needed.pop(member.name)
                if any(row['bytes']!=member.size for row in rows):raise ValueError('Member size mismatch')
                first=destination.joinpath(*safe_relative(rows[0]['path']).parts);first.parent.mkdir(parents=True,exist_ok=True)
                temporary=first.with_name(first.name+'.retirement-restore-tmp')
                digest=hashlib.sha256()
                try:
                    with temporary.open('xb') as output,archive.extractfile(member) as source:
                        for chunk in iter(lambda:source.read(1024**2),b''):output.write(chunk);digest.update(chunk)
                    if digest.hexdigest()!=rows[0]['sha256']:raise ValueError('Member hash mismatch')
                    try:os.link(temporary,first)
                    except OSError:
                        # Some Windows or mounted filesystems disallow hard links.
                        # Exclusive creation still refuses a concurrent overwrite.
                        with temporary.open('rb') as source,first.open('xb') as output:
                            for chunk in iter(lambda:source.read(1024**2),b''):output.write(chunk)
                    temporary.unlink();restored+=1
                except Exception:
                    if temporary.is_file() and temporary.resolve().is_relative_to(destination):temporary.unlink()
                    raise
                for row in rows[1:]:
                    if row['sha256']!=rows[0]['sha256']:raise ValueError('Alias content mismatch')
                    target=destination.joinpath(*safe_relative(row['path']).parts);target.parent.mkdir(parents=True,exist_ok=True)
                    with first.open('rb') as source,target.open('xb') as output:
                        for chunk in iter(lambda:source.read(1024**2),b''):output.write(chunk)
                    restored+=1
        if needed:raise ValueError('Required members missing from verified archive')
    return {'status':'passed','restored_files':restored,'already_matching_files':skipped,'destination':str(destination),'existing_different_files_overwritten':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','restore']);p.add_argument('--map',required=True);p.add_argument('--prefix',action='append',default=[]);p.add_argument('--archives');p.add_argument('--destination');a=p.parse_args()
    mapping=json.loads(Path(a.map).read_text())
    if a.action=='plan':result=plan(mapping,a.prefix)
    else:
        if not a.archives or not a.destination:p.error('restore requires --archives and --destination')
        result=restore(mapping,Path(a.archives),Path(a.destination),a.prefix)
    print(json.dumps(result,indent=2))
