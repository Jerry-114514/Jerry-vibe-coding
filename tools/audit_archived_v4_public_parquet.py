"""Audit selected AI2D/MMMU labels at exact pinned source rows, without image columns."""
import argparse
import ast
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import io
import json
from pathlib import Path
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

import pyarrow.parquet as pq

from audit_archived_image_fingerprints_and_7b_cache import from_archive
from audit_archived_training_chain import table
from audit_v41_archived_dataflow import digest
from experiment_common import ROOT, read_json, write_json


class RangeFile(io.RawIOBase):
    def __init__(self,url,size):
        self.url,self.size = url,size
        self.pos = self.n = self.calls = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self,offset,whence=0):
        self.pos = offset+(0 if whence==0 else self.pos if whence==1 else self.size)
        return self.pos

    def readinto(self,b):
        start,end = self.pos,min(self.pos+len(b),self.size)-1
        if end<start:
            return 0
        assert end-start+1<=16*1024**2
        for attempt in range(3):
            req = urllib.request.Request(self.url+f'?audit_range={start}-{end}',headers={
                'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'arp-v4-annotation-audit'})
            try:
                with urllib.request.urlopen(req,timeout=25) as r:
                    assert r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}'
                    raw = r.read(end-start+2)
                if len(raw)==end-start+1:
                    break
                raise OSError('Truncated HTTP range')
            except (urllib.error.URLError,OSError):
                if attempt==2:
                    raise
                time.sleep(attempt+1)
        b[:len(raw)] = raw
        self.pos += len(raw)
        self.n += len(raw)
        self.calls += 1
        return len(raw)


def get_file(dataset,entry,requested):
    ai2d = dataset['repo'].endswith('/ai2d')
    url = 'https://huggingface.co/datasets/'+dataset['repo']+'/resolve/'+dataset['revision']+'/'+urllib.parse.quote(entry['path'])
    with RangeFile(url,entry['bytes']) as source:
        parquet = pq.ParquetFile(source)
        cols = ['question','options','answer'] if ai2d else ['id','question','options','answer','question_type']
        raw = parquet.read(columns=cols,use_threads=False).to_pylist() if requested else []
        return parquet.metadata.num_rows,raw,source.n,source.calls


def audit(cache,output,workers):
    proof = read_json(ROOT/'docs/results/protocol-v2-20261016/data-preparation/candidate-index/complete.json')
    with tarfile.open(next(cache.glob('*085925Z.tar.gz'))) as archive:
        raw = table(archive,'configs/independent_data_sources.json')
        assert digest(raw)==proof['sources_config_sha256']
    datasets = json.loads(raw)['datasets']
    index_raw = gzip.decompress((cache/'map.json.gz').read_bytes())
    assert digest(index_raw)==read_json(ROOT/'docs/instance-retirement-20261020/release-ready.json')['restore_map_sha256']
    mapped = {x['path']:x for x in json.loads(index_raw)['files']}
    keys = {}
    role_counts = Counter()
    originals = {r['item_id']:r for r in from_archive(cache,'085925','/candidates.jsonl')}
    for role in ('calibration','audit','confirm'):
        selected = from_archive(cache,'094432','/'+role+'.jsonl')
        seen = set()
        for prompt in selected:
            if 'source_repo' not in prompt or prompt['base_item_id'] in seen:
                continue
            seen.add(prompt['base_item_id'])
            r = originals[prompt['base_item_id']]
            assert r['item_id'] not in keys
            keys[r['item_id']] = role,r
            role_counts[role] += 1
    requests = defaultdict(list)
    for role,r in keys.values():
        requests[r['source_repo'],r['source_file']].append((role,r))
    jobs = []
    for d in datasets:
        for entry in d['files']:
            p = 'data/protocol-v2/raw/'+d['repo'].replace('/','--')+'/'+entry['path']
            expected = mapped[p]
            assert expected['bytes']==entry['bytes'] and expected['sha256']==entry['sha256']
            selected = requests[d['repo'],entry['path']]
            assert all(r['source_revision']==d['revision'] for _,r in selected)
            # Need row counts even for unselected shards to convert the global source ordinal.
            jobs.append((d,entry,selected))
    fetched = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(get_file,d,e,bool(q)):(d,e) for d,e,q in jobs}
        for future in as_completed(futures):
            d,e = futures[future]
            key = d['repo'],e['path']
            try:
                fetched[key] = future.result()
            except Exception as err:
                errors['/'.join(key)] = type(err).__name__
            print(len(fetched)+len(errors),'/',len(jobs),key, 'failed' if '/'.join(key) in errors else 'read',flush=True)
    success, partial = Counter(),Counter()
    mismatches = []
    for d in datasets:
        base = 0
        for e in d['files']:
            key = d['repo'],e['path']
            if key not in fetched:
                base = None
                continue
            count,raw,_,_ = fetched[key]
            by_id = {str(r.get('id')):r for r in raw} if not d['repo'].endswith('/ai2d') else {}
            for role,r in requests[key]:
                i = r['source_row']-base if base is not None else None
                if i is not None:
                    assert 0<=i<count
                    row = raw[i]
                elif by_id:
                    row = by_id[r['item_id'].removeprefix('mmmu:')]
                else:
                    continue
                ai2d = not by_id
                original_id = 'ai2d:test:'+str(r['source_row']) if ai2d else 'mmmu:'+str(row['id'])
                answer = chr(65+int(row['answer'])) if ai2d else row['answer']
                options = row.get('options')
                options = ast.literal_eval(options) if isinstance(options,str) else options
                opts = {chr(65+j):v for j,v in enumerate(options or [])}
                if (original_id,row['question'],answer,opts)!=(
                        r['item_id'],r['question'],r['answer_key'],r['options']):
                    mismatches.append(r['item_id'])
                elif i is None:
                    partial[role]+=1
                else:
                    success[role]+=1
            if base is not None:
                base += count
    result = {'expected_by_role':dict(role_counts),'exact_source_row_matches':dict(success),
            'source_id_only_matches_after_missing_shard':dict(partial),'mismatch_ids':mismatches,
            'shard_errors':errors,'shards_read':len(fetched),'shards_expected':len(jobs),
            'range_bytes':sum(v[2] for v in fetched.values()),'http_ranges':sum(v[3] for v in fetched.values()),
            'scope':'Pinned source text, options and labels; image columns not fetched. Shard identities tied to frozen lock and retirement map.'}
    write_json(output,result)
    print(json.dumps(result,indent=2))
    assert not mismatches
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    audit(a.cache,a.output,a.workers)
