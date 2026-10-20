import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from restore_instance_retirement import restore,plan

def fixture(tmp_path):
    content=b'{"updates":1000}\n'
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        member=tarfile.TarInfo('backup/endpoint.json');member.size=len(content);archive.addfile(member,io.BytesIO(content))
    data=stream.getvalue();parts=[];folder=tmp_path/'archives';folder.mkdir()
    for i,chunk in enumerate([data[:len(data)//2],data[len(data)//2:]]):
        name=f'test.tar.gz.part{i:03d}';(folder/name).write_bytes(chunk);parts.append({'name':name,'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
    location={'release_url':'https://github.com/test/project/releases/tag/test','archive':{'name':'test.tar.gz','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()},'binary_assets':parts,'archive_member':'backup/endpoint.json'}
    row={'path':'runs/run/complete.json','bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'restore_location':location}
    return folder,{'files':[row]},content

def test_multipart_roundtrip_and_idempotence(tmp_path):
    folder,mapping,content=fixture(tmp_path);dest=tmp_path/'restored'
    assert plan(mapping,[])['files_to_restore']==1
    assert restore(mapping,folder,dest)['restored_files']==1
    assert (dest/'runs/run/complete.json').read_bytes()==content
    assert restore(mapping,folder,dest)['already_matching_files']==1

def test_corrupt_part_rejected_before_restore(tmp_path):
    folder,mapping,_=fixture(tmp_path);(folder/'test.tar.gz.part000').write_bytes(b'bad')
    with pytest.raises(ValueError,match='part hash'):restore(mapping,folder,tmp_path/'restored')
    assert not (tmp_path/'restored/runs/run/complete.json').exists()

def test_compact_map_and_prefix_selection(tmp_path):
    folder,mapping,content=fixture(tmp_path);row=mapping['files'][0];location=row.pop('restore_location');member=location.pop('archive_member')
    row.update({'archive_id':0,'archive_member':member});mapping['archives']={'0':location}
    assert plan(mapping,['unselected/'])['files_to_restore']==0
    assert restore(mapping,folder,tmp_path/'restored',['runs/'])['restored_files']==1
    assert (tmp_path/'restored/runs/run/complete.json').read_bytes()==content

def test_path_traversal_and_overwrite_refused(tmp_path):
    folder,mapping,_=fixture(tmp_path);mapping['files'][0]['path']='../escape'
    with pytest.raises(ValueError,match='Unsafe restore path'):restore(mapping,folder,tmp_path/'restored')
    mapping['files'][0]['path']='runs/run/complete.json';target=tmp_path/'restored/runs/run/complete.json';target.parent.mkdir(parents=True);target.write_bytes(b'existing work')
    with pytest.raises(FileExistsError):restore(mapping,folder,tmp_path/'restored')
    assert target.read_bytes()==b'existing work'
