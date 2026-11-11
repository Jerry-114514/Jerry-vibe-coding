"""Audit training ledgers and cross-role image content by archived byte hashes."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re
from statistics import median
import tarfile

from audit_v41_archived_dataflow import digest
from experiment_common import parse_answer
from scoring import box_iou

AXES = ('pope','grounding','consistency','mmstar')


def table(tar, suffix):
    matches = [m for m in tar.getmembers() if m.name.endswith(suffix) and m.isfile()]
    assert matches, suffix
    versions = [tar.extractfile(m).read() for m in matches]
    assert len({digest(r) for r in versions}) == 1, suffix
    return versions[0]


def rows(raw):
    return [json.loads(x) for x in raw.splitlines() if x.strip()]


def image_paths(items):
    return {p for r in items for p in (r.get('image_paths') or [r['image_path']])}


def image_overlap(mapping, train, eval_roles):
    result = {}
    train_paths = image_paths(train)
    assert train_paths <= mapping.keys()
    train_hashes = {mapping[p]['sha256'] for p in train_paths}
    def coco_ids(paths):
        return {int(m.group(1)) for p in paths
                if (m := re.search(r'(?i)/COCO_(?:train|val)2014_(\d+)\.(?:jpg|jpeg|png)$',p))}
    train_coco = coco_ids(train_paths)
    for role, items in eval_roles.items():
        paths = image_paths(items)
        assert paths <= mapping.keys(), (role,len(paths-mapping.keys()))
        hashes = {mapping[p]['sha256'] for p in paths}
        result[role] = {'eval_paths':len(paths),'eval_unique_byte_hashes':len(hashes),
                        'shared_paths':len(paths & train_paths),
                        'shared_byte_hashes':len(hashes & train_hashes),
                        'coco_ids_from_filenames':len(coco_ids(paths)),
                        'shared_coco_ids_from_filenames':len(coco_ids(paths) & train_coco)}
    return {'training_paths':len(train_paths),'training_unique_byte_hashes':len(train_hashes),
            'training_coco_ids_from_filenames':len(train_coco),
            'comparisons':result,'basis':'Frozen manifest paths joined to verified restore-map file SHA256. No image decoding.'}


def verify_runs(tar, prefix, mapping, pool):
    result = {}
    lookup = {x['item_id']:x for x in pool}
    assert len(lookup) == len(pool)
    protocol_raw = table(tar, prefix+'/protocol.json')
    for seed in (42,137,256,512,1024):
        for arm in ('A1','A2','U'):
            name = f'{arm}-seed{seed}'
            root = prefix+'/formal/'+name
            meta = json.loads(table(tar,root+'/train.meta.json'))
            marker = json.loads(table(tar,root+'/complete.json'))
            loss = table(tar,root+'/loss.jsonl')
            assert meta['protocol_sha256'] == digest(protocol_raw)
            assert (meta['seed'],meta['arm'],meta['phase'],meta['target_steps']) == (seed,arm,'formal',1000)
            assert digest(loss) == marker['loss_sha256']
            assert digest(loss) == mapping[root+'/loss.jsonl']['sha256']
            raw = rows(loss)
            order = meta['sampling_ids']
            assert len(order) == len(set(order)) == 600 and set(order) <= lookup.keys()
            assert marker['actual_updates'] == len(raw) == 1000
            for i,r in enumerate(raw):
                item_id = order[i%len(order)]
                assert (r['step'],r['item_id'],r['axis']) == (i+1,item_id,lookup[item_id]['axis'])
                assert r['actual_optimizer_update'] and r['grad_norm'] > 0
                assert all(math.isfinite(r[k]) for k in ('loss','grad_norm','ce_gt','ce_teacher','seconds'))
                assert abs(r['loss']-.5*r['ce_gt']-.5*r['ce_teacher']) < 1e-5
            counts = dict(Counter(r['axis'] for r in raw))
            assert counts == ({a:250 for a in AXES} if arm=='U' else {'mmstar' if arm=='A1' else 'consistency':1000})
            weights = mapping[root+'/adapter/adapter_model.safetensors']
            assert weights['sha256'] == marker['adapter_sha256']
            resume = mapping[root+'/resume.pt']
            if 'resume_sha256' in marker:
                assert marker['resume_sha256'] == resume['sha256']
            result[name] = {'steps':len(raw),'axes':counts,'unique_sample_ids':len(order),
                            'unique_image_paths':len(image_paths([lookup[i] for i in order])),
                            'loss_sha256':digest(loss),'adapter_sha256':weights['sha256'],
                            'recovery_sha256':resume['sha256'],
                            'recovery_link':'same digest in endpoint and restore map' if 'resume_sha256' in marker
                            else 'restore map only; endpoint marker has no recovery digest'}
    return result


def verify_teacher(teacher_root, mapping, eligible, pool_raw):
    marker = json.loads(next(teacher_root.rglob('teacher-eligible-complete.json')).read_text(encoding='utf-8'))
    raw = next(teacher_root.rglob('teacher.jsonl')).read_bytes()
    assert digest(raw) == marker['teacher_ledger_sha256'] == mapping[
        'runs/route-b-9b-122b-20261017/teacher.jsonl']['sha256']
    assert digest(pool_raw) == marker['eligible_pool_sha256']
    items = {r['item_id']:r for r in eligible}
    records = rows(raw)
    assert len(records) == len(items) == len({r['item_id'] for r in records}) == marker['cache_rows_retained']
    assert {r['item_id'] for r in records} == items.keys()
    assert len({r['request_id'] for r in records}) == len(records)
    counts = defaultdict(Counter)
    teacher_iou = {}
    for r in records:
        item = items[r['item_id']]
        axis = item['axis']
        assert r['axis'] == axis and r['finish_reason']=='stop' and r['returned_model']=='qwen3.5-122b-a10b'
        assert r['raw_generation'].strip()
        paths = item.get('image_paths') or [item['image_path']]
        assert [{k:x[k] for k in ('path','sha256')} for x in r['image_hashes']] == [
            {'path':p,'sha256':mapping[p]['sha256']} for p in paths]
        for image in r['image_hashes']:
            assert len(image['transmitted_png_sha256']) == 64
            assert all(v>0 for v in image['source_size']+image['transmitted_size'])
        counts[axis]['records'] += 1
        counts[axis]['actual_prompt_differs_from_pool'] += r['actual_prompt'] != item['question']
        if axis=='grounding':
            assert r['image_hashes'][0]['source_size'] == [item['width'],item['height']]
            box = parse_answer(r['raw_generation'],{'answer_type':'bbox','width':1000,'height':1000})
            assert box is not None
            pixels = [round(v*item['width' if i%2==0 else 'height']/1000,3) for i,v in enumerate(box)]
            iou = box_iou(pixels,item['answer_key'])
            teacher_iou[r['item_id']] = iou
            counts[axis]['agrees_with_gt_at_iou50'] += iou>=.5
        else:
            answer = parse_answer(r['raw_generation'],item)
            counts[axis]['strict_unparseable'] += answer is None
            counts[axis]['strict_agrees_with_gt'] += answer is not None and answer==item['answer_key']
    return {'ledger_sha256':digest(raw),'by_axis':dict(counts),'grounding_median_iou':median(teacher_iou.values()),
            'grounding_targets_below_iou50':sum(v<.5 for v in teacher_iou.values()),
            'scope':'API stop metadata, image SHA and text targets. The provider token IDs and an internal teacher EOS are not in this cache.'}


def audit(folder, evidence_root):
    mapping_raw = gzip.decompress((folder/'map.json.gz').read_bytes())
    release = json.loads((evidence_root/'release-ready.json').read_text(encoding='utf-8'))
    assert digest(mapping_raw) == release['restore_map_sha256']
    index = json.loads(mapping_raw)
    mapping = {r['path']:r for r in index['files']}
    role_prefix = '/clean-role-manifests-v4/'
    with tarfile.open(next(folder.glob('*094432Z.tar.gz'))) as tar:
        pool_raw = table(tar,role_prefix+'train.jsonl')
        assert digest(pool_raw) == mapping['runs/protocol-v2-data/clean-role-manifests-v4/train.jsonl']['sha256']
        original_pool = rows(pool_raw)
        roles = {role:rows(table(tar,role_prefix+role+'.jsonl')) for role in ('calibration','audit','confirm')}
    with tarfile.open(next(folder.glob('*070813Z.tar.gz'))) as tar:
        raw = table(tar,'/formal-manifest.jsonl')
        historical_manifest = rows(raw)
        roles['historical_v41'] = historical_manifest
    teacher_root = folder/'selective'/'arp-9b-122b-20261017'
    pool_raw = next(teacher_root.rglob('eligible-pool.jsonl')).read_bytes()
    assert digest(pool_raw) == mapping['runs/route-b-9b-122b-20261017/eligible-pool.jsonl']['sha256']
    eligible = rows(pool_raw)
    assert len(original_pool) == 3545 and len(eligible) == 3540
    old = {r['item_id']:r for r in original_pool}
    assert len(old) == len(original_pool)
    key_matches = set(old) & {r['item_id'] for r in eligible}
    assert len(key_matches) == len(eligible)
    for row in eligible:
        assert all(row[k] == old[row['item_id']][k] for k in ('axis','image_path','question','answer_key'))
    with tarfile.open(next(folder.glob('*091121Z.tar.gz'))) as tar:
        run7 = verify_runs(tar,'runs/route-b-staged-7b-20261016',mapping,original_pool)
    with tarfile.open(next(folder.glob('*140724Z.tar.gz'))) as tar:
        run9 = verify_runs(tar,'runs/route-b-9b-122b-20261017/student-experiment',mapping,eligible)
    teacher = verify_teacher(teacher_root,mapping,eligible,pool_raw)
    return {'map_sha256':digest(mapping_raw),'images_v4_pool':image_overlap(mapping,original_pool,roles),
            'images_9b_eligible':image_overlap(mapping,eligible,roles),
            'eligible_matches_v4_pool':len(key_matches),
            'excluded_item_axes':dict(Counter(r['axis'] for k,r in old.items() if k not in key_matches)),
            '7b':run7,'9b':run9,'teacher_122b':teacher,
            'scope':'Metadata and complete loss ledgers replayed; adapter and resume bytes not downloaded or tensor-tested.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('retirement_evidence',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    result = audit(a.cache,a.retirement_evidence)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('7b','9b')},indent=2))
    print('Replayed',len(result['7b'])+len(result['9b']),'training runs.')
