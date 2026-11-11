"""Read only text-label columns of the original ScienceQA Parquet via HTTP ranges."""
import argparse
import gzip
import io
import json
from pathlib import Path
import urllib.request

import pyarrow.parquet as pq

from audit_archived_image_fingerprints_and_7b_cache import from_archive
from experiment_common import read_json, write_json


def audit(cache, source):
    original = next(x for x in json.loads(gzip.decompress((cache/'map.json.gz').read_bytes()))['files']
                    if x['path'].startswith('data/training/raw/scienceqa/') and x['path'].endswith('.parquet'))
    assert original['sha256']==source['sha256'] and original['bytes']==source['bytes']
    assert Path(original['path']).name==Path(source['path']).name
    assert len(source['revision'])==40 and all(ch in '0123456789abcdef' for ch in source['revision'])
    url = 'https://huggingface.co/datasets/derek-thomas/ScienceQA/resolve/'+source['revision']+'/'+source['path']
    items = from_archive(cache,'094432','/clean-role-manifests-v4/train.jsonl')
    wanted = {int(r['item_id'].rsplit(':',1)[-1]):r for r in items if r['axis']=='mmstar'}

    class RangeFile(io.RawIOBase):
        def __init__(self):
            self.pos = self.n = self.calls = 0

        def readable(self):
            return True

        def seekable(self):
            return True

        def tell(self):
            return self.pos

        def seek(self, offset, whence=0):
            self.pos = offset+(0 if whence==0 else self.pos if whence==1 else source['bytes'])
            return self.pos

        def readinto(self, b):
            start = self.pos
            end = min(start+len(b),source['bytes'])-1
            if end<start:
                return 0
            assert end-start+1<=16*1024**2
            req = urllib.request.Request(url+'?audit_range='+str(start),headers={
                'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'research-annotation-audit'})
            with urllib.request.urlopen(req,timeout=25) as r:
                assert r.status==206
                assert r.headers.get('Content-Range')==f"bytes {start}-{end}/{source['bytes']}"
                raw = r.read(end-start+2)
            assert len(raw)==end-start+1
            b[:len(raw)] = raw
            self.pos += len(raw)
            self.n += len(raw)
            self.calls += 1
            return len(raw)

    raw = RangeFile()
    parquet = pq.ParquetFile(raw)
    cols = ['question','choices','answer','hint']
    source_rows = {}
    offset = 0
    for i in range(parquet.num_row_groups):
        n = parquet.metadata.row_group(i).num_rows
        ordinals = [k for k in wanted if offset<=k<offset+n]
        if ordinals:
            table = parquet.read_row_group(i,columns=cols,use_threads=False).to_pylist()
            for k in ordinals:
                source_rows[k] = table[k-offset]
        offset += n
    assert len(source_rows)==len(wanted)==600
    for key,r in wanted.items():
        row = source_rows[key]
        opts = list('ABCD'[:len(row['choices'])])
        prompt = (('Context: '+row['hint']+'\n') if row.get('hint') else '')+row['question']+'\n'
        prompt += '\n'.join(f'{k}. {v}' for k,v in zip(opts,row['choices']))+'\nReturn only the option letter.'
        assert (prompt,opts[row['answer']],opts)==(r['question'],r['answer_key'],r['options']),key
    return {'source':source,'range_bytes_read':raw.n,'http_range_requests':raw.calls,
            'exact_parquet_row_matches':len(source_rows),'columns':cols,
            'scope':'Pinned file identity via LFS SHA/size metadata and verified byte ranges; no image column or full binary downloaded.'}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('source_identity',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    out = audit(a.cache,read_json(a.source_identity))
    write_json(a.output,out)
    print(out)
