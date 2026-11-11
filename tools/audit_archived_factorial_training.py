"""Verify any restored 9B factorial training ledger against frozen input and receipts."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

from experiment_common import ROOT, read_json, read_rows, sha, write_json


def verify(cache):
    run = cache/'selective/arp-9b-122b-20261017/runs/9b-u137-factorial-20261018'
    mapping = {r['path']:r for r in json.loads(gzip.decompress((cache/'map.json.gz').read_bytes()))['files']}
    official = read_json(ROOT/'docs/results/protocol-v2-20261016/9b-repair-final/report/results.json')
    protocols = sha(run/'protocol.json')
    assert protocols==official['protocol_sha256']
    results = {}
    for folder in sorted((run/'training').glob('*')):
        name = folder.name
        condition,seed = name.split('-seed')
        if not all((folder/file).is_file() for file in ('loss.jsonl','train.meta.json','complete.json','verification.json')):
            continue
        base = f'runs/9b-u137-factorial-20261018/training/{name}/'
        for file in ('loss.jsonl','train.meta.json','complete.json','verification.json'):
            assert sha(folder/file)==mapping[base+file]['sha256']
        prior = read_json(folder/'verification.json')
        complete = read_json(folder/'complete.json')
        meta = read_json(folder/'train.meta.json')
        assert meta['protocol_sha256']==protocols
        assert meta['condition']==condition and meta['seed']==int(seed)
        assert complete['actual_updates']==prior['actual_updates']==1000
        frozen_path = run/f'registration/training-inputs/{name}.jsonl'
        assert sha(frozen_path)==mapping[f'runs/9b-u137-factorial-20261018/training-inputs/{name}.jsonl']['sha256']
        frozen = read_rows(frozen_path)
        ledger = read_rows(folder/'loss.jsonl')
        assert len(frozen)==len({x['item_id'] for x in frozen})==600
        assert [r['item_id'] for r in frozen]==meta['sampling_ids']
        assert [r['step'] for r in ledger]==list(range(1,1001))
        axis_counts = Counter()
        for i,row in enumerate(ledger):
            ref = frozen[i%len(frozen)]
            assert row['item_id']==ref['item_id'] and row['axis']==ref['axis']
            assert row['actual_optimizer_update'] and row['grad_norm']>0
            assert all(math.isfinite(row[k]) for k in ('grad_norm','loss','ce_gt'))
            dual = condition.endswith('-D')
            assert row['teacher_forward_computed']==dual
            if dual:
                assert math.isfinite(row['ce_teacher'])
                target = .5*(row['ce_teacher']+row['ce_gt'])
            else:
                assert row['ce_teacher'] is None
                target = row['ce_gt']
            assert abs(row['loss']-target)<1e-5
            axis_counts[row['axis']]+=1
        assert dict(axis_counts)==prior['axes'] and set(axis_counts.values())=={250}
        assert complete['loss_sha256']==official['training'][name]['loss_sha256']==sha(folder/'loss.jsonl')
        for field,file in [('adapter_sha256','adapter/adapter_model.safetensors'),('resume_sha256','resume.pt')]:
            assert complete[field]==mapping[base+file]['sha256']
            if (folder/file).is_file():
                assert sha(folder/file)==complete[field]
        evaluation_links = {}
        for split in ('development','heldout'):
            eval_meta = run/'evaluations'/split/name/'eval.meta.json'
            proof = read_json(eval_meta)
            assert proof['adapter_sha256']==complete['adapter_sha256']
            assert proof['protocol_sha256']==protocols
            evaluation_links[split] = sha(eval_meta)
        results[name] = {'actual_updates':1000,'axes':dict(axis_counts),'dual':condition.endswith('-D'),
                         'loss_sha256':complete['loss_sha256'],'adapter_sha256':complete['adapter_sha256'],
                         'resume_sha256':complete['resume_sha256'],
                         'evaluation_metadata_sha256':evaluation_links,
                         'historical_state_verification':prior,
                         'fresh_scope':'per-step ledger and artifact digests; no fresh tensor reload'}
    return {'verified_runs':len(results),'expected_new_runs':6,'runs':results}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    evidence=verify(a.cache)
    assert evidence['verified_runs']>0
    write_json(a.output,evidence)
    print('Verified',evidence['verified_runs'],'factorial training ledgers.')
