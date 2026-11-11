"""Verify original v1 top-50 cache and the actual D-phase sample schedule."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import random

from audit_archived_image_fingerprints_and_7b_cache import from_archive
from audit_v41_archived_dataflow import digest
from experiment_common import AXES, ROOT, read_json, read_rows, sha, write_json


def jsonl_sha(rows):
    return digest(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n'
                          for x in rows).encode('utf-8'))


def audit(cache):
    import torch
    torch.set_num_threads(1)
    raw = gzip.decompress((cache/'map.json.gz').read_bytes())
    assert digest(raw)==read_json(ROOT/'docs/instance-retirement-20261020/release-ready.json')['restore_map_sha256']
    mapping = {r['path']:r for r in json.loads(raw)['files']}
    selected = cache/'selective/arp-formal-20261014'
    (run,) = (selected/'runs/github-snapshots').glob('*003911Z')
    inv = read_json(next(cache.glob('*004018Z-inventory.json')))
    archived = {x['path']:x for x in inv['files']}

    def confirmed(path):
        assert sha(path)==archived[path.relative_to(selected).as_posix()]['sha256']
        return path

    root = run/'teacher-cache'
    meta = read_json(confirmed(root/'cache.meta.json'))
    done = read_json(confirmed(root/'complete.json'))
    vocab = set(read_json(confirmed(root/'tokenizer.json'))['vocab'].values())
    frozen_pool = from_archive(cache,'094432','/clean-role-manifests-v4/train.jsonl')
    pool = [{k:v for k,v in r.items() if k not in ('role','protocol_v2_role')} for r in frozen_pool]
    assert jsonl_sha(pool)==done['manifest_sha256']==meta['manifest_sha256']==mapping['data/training/pool.jsonl']['sha256']
    assert len(pool)==done['items']==len(done['files'])==3545
    assert meta['top_k']==50 and meta['teacher']=='qwen7b'
    axis_counts, endings = Counter(),Counter()
    raw_id_fields = 0
    for row in pool:
        name = digest(row['item_id'].encode())+'.pt'
        path = confirmed(root/name)
        assert sha(path)==done['files'][name]
        x = torch.load(path,map_location='cpu',weights_only=True)
        assert x['item_id']==row['item_id'] and x['response'].strip()
        assert x['image_sha256']==mapping[row['image_path']]['sha256']
        t = x['targets'].reshape(-1)
        values, ids, z = x['top_logits'],x['top_indices'],x['log_normalizer']
        assert len(t)>0 and values.shape==ids.shape==(len(t),50) and z.shape==(len(t),)
        assert torch.equal(t,x['input_ids'][:,x['prompt_length']:].reshape(-1))
        assert torch.isfinite(values).all() and torch.isfinite(z).all()
        assert set(ids.flatten().tolist()) <= vocab and set(t.tolist())<=vocab
        assert ((values-z[:,None]).exp().sum(-1)<=1.0001).all()
        raw_id_fields += any(k in x for k in ('generation_ids','generated_token_ids','finish_reason'))
        axis_counts[row['axis']] += 1
        endings[int(t[-1])] += 1

    summary = read_json(confirmed(run/'analysis/summary.json'))
    rank = sorted(AXES,key=lambda x:(-summary['families']['qwen']['pi_boot_mean'][x],AXES.index(x)))
    size = min(axis_counts.values())
    chosen = {'A1':[r for r in pool if r['axis']==rank[0]],
              'A2':[r for r in pool if r['axis']==rank[1]],
              'U':[r for a in AXES for r in [x for x in pool if x['axis']==a][:size]],'N':[]}
    plan = read_json(ROOT/'configs/experiment.json')['training']
    train = {}
    for arm,seq in chosen.items():
        folder = run/'arms'/arm
        config = read_json(confirmed(folder/'train.meta.json'))
        marker = read_json(confirmed(folder/'complete.json'))
        assert config['protocol']==plan
        assert config['cache_meta_sha256']==sha(root/'cache.meta.json')
        assert jsonl_sha(seq)==config['manifest_sha256']==archived[(run/f'training/{arm}.jsonl').relative_to(selected).as_posix()]['sha256']
        if arm=='N':
            assert marker['effective_updates']==0 and marker['no_op'] is True
            train[arm]={'updates':0,'no_op':True}
            continue
        ledger = read_rows(confirmed(folder/'loss.jsonl'))
        order = list(range(len(seq)))
        random.Random(plan['seed']).shuffle(order)
        assert marker['effective_updates']==len(ledger)==plan['steps']==1000
        actual_axes = Counter()
        retries = 0
        for i,r in enumerate(ledger):
            row = seq[order[i%len(order)]]
            assert r['step']==i+1 and r['item_id']==row['item_id']
            assert all(math.isfinite(r[k]) for k in ('loss','ce','kd','grad_norm'))
            target = (1-plan['kd_weight'])*r['ce']+plan['kd_weight']*r['kd']
            assert abs(r['loss']-target)<1e-5
            assert r['grad_norm']>0 and r['amp_retries']>=0
            actual_axes[row['axis']] += 1
            retries += r['amp_retries']
        train[arm]={'updates':len(ledger),'axis_counts':dict(actual_axes),'amp_retries':retries,
                    'frozen_samples':len(seq),'loss_sha256':sha(folder/'loss.jsonl')}
    return {'teacher_cache_rows':len(pool),'axis_counts':dict(axis_counts),'retokenized_target_last_ids':dict(endings),
            'raw_generation_or_finish_metadata_count':raw_id_fields,'legacy_nontruncation_certified':False,
            'teacher_cache_protocol':meta,'training':train,'ranked_axes':rank,
            'scope':'All v1 top-50 tensors and per-step D logs loaded from verified archive; optimizer tensors were not reloaded.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    r=audit(a.cache)
    write_json(a.output,r)
    print('Verified',r['teacher_cache_rows'],'old cache tensors and original D ledgers.')
