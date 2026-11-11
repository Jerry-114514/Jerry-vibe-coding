"""Check archived image fingerprints and replay all BF16 7B EOS cache tensors."""
import argparse
from collections import Counter, defaultdict
import gzip
import io
import json
from pathlib import Path
import tarfile

from audit_archived_training_chain import image_paths, rows, table
from audit_v41_archived_dataflow import digest
from experiment_common import ROOT, read_json, write_json


def from_archive(cache, stamp, suffix):
    with tarfile.open(next(cache.glob('*'+stamp+'Z.tar.gz'))) as tar:
        return rows(table(tar,suffix))


def fingerprints(cache, mapping):
    train = from_archive(cache,'094432','/clean-role-manifests-v4/train.jsonl')
    histories = from_archive(cache,'085925','/historical-image-fingerprints.jsonl')
    selected = from_archive(cache,'094432','/selected-image-assets.jsonl')
    facts = {}
    for f in histories+selected:
        if f['path'] in facts:
            assert facts[f['path']] == f
        facts[f['path']] = f
    roles = {k:from_archive(cache,'094432','/clean-role-manifests-v4/'+k+'.jsonl')
             for k in ('calibration','audit','confirm')}
    roles['historical_v41'] = from_archive(cache,'070813','/formal-manifest.jsonl')
    factorial = cache/'selective/arp-9b-122b-20261017/runs/9b-u137-factorial-20261018/heldout-manifest.jsonl'
    roles['factorial_heldout'] = rows(factorial.read_bytes())

    def collect(items):
        paths = image_paths(items)
        representative, axes = {}, defaultdict(set)
        for r in items:
            for p in image_paths([r]):
                f = facts[p]
                assert f['source_sha256'] == mapping[p]['sha256'] and f['bytes'] == mapping[p]['bytes']
                pixel = f['pixel_sha256']
                if pixel in representative:
                    assert representative[pixel]['dhash64']==f['dhash64']
                else:
                    representative[pixel] = f
                axes[pixel].add(r['axis'])
        return paths,representative,axes

    paths,train_facts,train_axes = collect(train)
    output,pairs = {},{}
    for role,items in roles.items():
        other,eval_facts,eval_axes = collect(items)
        counts,axis_counts = Counter(),Counter()
        found = []
        right = [(pix,f,int(f['dhash64'],16)) for pix,f in eval_facts.items()]
        for lp,lf in train_facts.items():
            left_hash = int(lf['dhash64'],16)
            for rp,rf,right_hash in right:
                distance = (left_hash^right_hash).bit_count()
                if distance>4:
                    continue
                counts[distance] += 1
                common = sorted(train_axes[lp]&eval_axes[rp])
                for a in common:
                    axis_counts[a] += 1
                found.append({'train_path':lf['path'],'eval_path':rf['path'],'distance':distance,
                              'train_axes':sorted(train_axes[lp]),'eval_axes':sorted(eval_axes[rp]),
                              'shared_axes':common,'same_rgb':lp==rp})
        output[role] = {'eval_paths':len(other),'eval_distinct_rgb':len(eval_facts),
                        'train_eval_rgb_intersection':len(train_facts.keys() & eval_facts.keys()),
                        'dhash_distance_le4_candidate_pairs':len(found),
                        'distance_counts':dict(counts),'within_same_axis_candidates':dict(axis_counts)}
        pairs[role] = sorted(found,key=lambda p:(p['distance'],p['train_path'],p['eval_path']))
    return {'train_paths':len(paths),'train_distinct_rgb':len(train_facts),'comparisons':output,
            'pairs':pairs,'scope':'Historical decoded RGB and dHash verified against archive source SHA256; not freshly decoded. dHash hits are review candidates, not proven duplicate images.'}


def audit_cache(cache,mapping):
    import torch
    torch.set_num_threads(1)
    path = next(cache.glob('*140316Z.tar.gz'))
    inv = read_json(next(cache.glob('*140316Z-inventory.json')))
    assert digest(path.read_bytes())==inv['archive']['sha256']
    ref = read_json(ROOT/'docs/results/protocol-v2-20261016/route-b-full-eos-cache-independent-verification.json')
    config_raw = (cache/'qwen7b-pinned-generation-config.json').read_bytes()
    eos = set(json.loads(config_raw)['eos_token_id'])
    train = from_archive(cache,'094432','/clean-role-manifests-v4/train.jsonl')
    root = 'runs/route-b-7b-20261016/teacher-cache-bf16-eos/'
    targets = {root+digest(r['item_id'].encode())+'.pt':r for r in train}
    assert len(targets)==len(train)==3545
    stats = defaultdict(Counter)
    with tarfile.open(path) as tar:
        marker = table(tar,'/teacher-cache-bf16-eos/complete.json')
        assert digest(marker)==ref['cache_inventory_sha256']
        manifest = json.loads(marker)
        assert manifest['items']==len(targets)
        count = 0
        for member in tar:
            if member.name not in targets:
                continue
            row = targets.pop(member.name)
            raw = tar.extractfile(member).read()
            assert digest(raw)==manifest['files'][Path(member.name).name]==mapping[member.name]['sha256']
            entry = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            assert entry['item_id']==row['item_id'] and entry['response'].strip()
            assert entry['image_sha256']==mapping[row['image_path']]['sha256']
            generated = entry['generation_ids'].reshape(-1)
            assert 0 < generated.numel() <= 128 and int(generated[-1]) in eos
            assert entry['finish_reason']=='eos'
            target = entry['targets'].reshape(-1)
            logits,indices,normalizer = entry['top_logits'],entry['top_indices'],entry['log_normalizer']
            assert logits.shape==indices.shape==(target.numel(),50) and normalizer.shape==(target.numel(),)
            assert target.numel()>0
            assert torch.equal(entry['input_ids'][:,entry['prompt_length']:].reshape(-1),target)
            assert torch.isfinite(logits).all() and torch.isfinite(normalizer).all()
            assert (indices>=0).all()
            assert ((logits-normalizer[:,None]).exp().sum(-1)<=1.0001).all()
            stats[row['axis']]['rows'] += 1
            stats[row['axis']]['target_tokens'] += int(target.numel())
            stats[row['axis']]['max_raw_generated_tokens'] = max(stats[row['axis']]['max_raw_generated_tokens'],generated.numel())
            stats[row['axis']]['raw_final_token_'+str(int(generated[-1]))] += 1
            count += 1
        assert not targets and count==3545
    return {'rows':count,'by_axis':dict(stats),'official_pinned_eos':sorted(eos),
            'generation_config_sha256':digest(config_raw),'archive_sha256':inv['archive']['sha256'],
            'scope':'All BF16 EOS teacher tensors loaded on CPU; older v1 cache has different provenance.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    raw = gzip.decompress((a.cache/'map.json.gz').read_bytes())
    assert digest(raw)==read_json(ROOT/'docs/instance-retirement-20261020/release-ready.json')['restore_map_sha256']
    mapping = {r['path']:r for r in json.loads(raw)['files']}
    result = {'images':fingerprints(a.cache,mapping),'7b_bf16_eos_cache':audit_cache(a.cache,mapping)}
    write_json(a.output,result)
    print(json.dumps({'image_comparisons':result['images']['comparisons'],'7b_cache':result['7b_bf16_eos_cache']},indent=2))
