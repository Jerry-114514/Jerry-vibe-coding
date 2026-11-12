"""Replay v4 graph/roles and check COCO labels directly against the frozen sources."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import tarfile

from audit_archived_image_fingerprints_and_7b_cache import from_archive
from audit_archived_training_chain import rows, table
from audit_v41_archived_dataflow import digest
from experiment_common import AXES, ROOT, read_json, sha, write_json
from freeze_clean_candidates_v3 import conservative_components, legacy_consistency_calibration, repair_role_quotas
from independent_manifest_v2 import expand_item, select_axis
from prepare_coco_candidates_v2 import rank
from refer_api import load_original


def verify_coco(candidates, mapping):
    counts = Counter()
    for dataset, version in [('refcoco','unc'),('refcoco+','unc'),('refcocog','umd')]:
        rel = f'data/raw/refer/{dataset}.zip'
        assert sha(ROOT/rel)==mapping[rel]['sha256']
        refs,instances = load_original(ROOT/rel,dataset,version)
        ref_by_id = {r['ref_id']:r for r in refs}
        anns = {r['id']:r for r in instances['annotations']}
        imgs = {r['id']:r for r in instances['images']}
        for row in candidates:
            if row['axis']!='grounding' or row['dataset']!=dataset:
                continue
            ref = ref_by_id[row['source_ref_id']]
            img = imgs[ref['image_id']]
            sent = min(ref['sentences'],key=lambda s:rank(str(s['sent_id'])))
            x,y,w,h = anns[ref['ann_id']]['bbox']
            assert ref['split'] in ('test','testA','testB') and row['source_split']==ref['split']
            assert row['source_sentence_id']==sent['sent_id'] and row['referring_expression']==sent['sent']
            assert row['answer_key']==[x,y,x+w,y+h] and 0<=x<x+w<=img['width'] and 0<=y<y+h<=img['height']
            assert row['source_image_id']==img['id'] and row['image_paths']==['data/protocol-v2/coco-images/'+img['file_name']]
            assert [row['width'],row['height']]==[img['width'],img['height']]
            counts['grounding/'+dataset]+=1
        if dataset=='refcoco':
            present = defaultdict(set)
            for annotation in instances['annotations']:
                present[annotation['image_id']].add(annotation['category_id'])
            categories = {r['id']:r['name'] for r in instances['categories']}
            for row in candidates:
                if row['axis']!='pope':
                    continue
                iid,cid = row['source_image_id'],row['source_category_id']
                assert row['answer_key']==('yes' if cid in present[iid] else 'no')
                assert row['question']==f'Is there a {categories[cid]} in the image? Answer with yes or no only.'
                assert row['image_paths']==['data/protocol-v2/coco-images/'+imgs[iid]['file_name']]
                counts['pope/'+row['subset']]+=1
    assert sum(counts.values())==len(candidates)
    return dict(counts)


def audit(cache):
    evidence = ROOT/'docs/results/protocol-v2-20261016/data-preparation/clean-role-manifests-v4'
    done = read_json(evidence/'complete.json')
    design = read_json(evidence/'design.json')
    raw = gzip.decompress((cache/'map.json.gz').read_bytes())
    assert digest(raw)==read_json(ROOT/'docs/instance-retirement-20261020/release-ready.json')['restore_map_sha256']
    mapping = {r['path']:r for r in json.loads(raw)['files']}
    with tarfile.open(next(cache.glob('*085925Z.tar.gz'))) as archive:
        hf = table(archive,'/candidates.jsonl')
        historical = table(archive,'/historical-image-fingerprints.jsonl')
        assert digest(hf)==design['candidate_inputs']['hf']
        assert digest(historical)==design['historical_inventory_sha256']
    with tarfile.open(next(cache.glob('*091457Z.tar.gz'))) as archive:
        coco = table(archive,'/coco-candidates.jsonl')
        assert digest(coco)==design['candidate_inputs']['coco']
    assets = (from_archive(cache,'085925','/image-assets.jsonl')+
              from_archive(cache,'091457','/coco-image-assets.jsonl'))
    candidates = rows(hf)+rows(coco)
    assigned,excluded,edges,graph = conservative_components(candidates,assets,rows(historical),design['dhash_radius'])
    assigned,moves = repair_role_quotas(assigned,design['targets'])
    hashes = {}
    exposure = {}
    proof = {}
    historical_keys = {r['pixel_sha256'] for r in rows(historical)}
    blocked = {graph.find(p) for p in historical_keys}
    with tarfile.open(next(cache.glob('*094432Z.tar.gz'))) as archive:
        for name,computed in [('near-component-edges',edges),('exclusions',excluded)]:
            assert rows(table(archive,'/'+name+'.jsonl'))==computed
            proof[name]=len(computed)
        assert json.loads(table(archive,'/component-role-moves.json'))==moves
        for role,count in design['targets'].items():
            originals = [r for axis in AXES for r in (
                legacy_consistency_calibration(rows(historical)) if role=='calibration' and axis=='consistency'
                else select_axis([r for r in assigned if r['role']==role and r['axis']==axis],count))]
            prompts = [x for r in originals for x in expand_item(r)]
            frozen = table(archive,'/clean-role-manifests-v4/'+role+'.jsonl')
            assert prompts==rows(frozen) and digest(frozen)==done['manifests'][role]['sha256']
            hashes[role]=digest(frozen)
            exposure[role] = dict(Counter(r['dataset'] for r in originals))
            assert all(graph.find(r['image_pixel_sha256'][0]) not in blocked for r in originals
                       if r['dataset']!='mmbench-legacy-calibration')
    return {'candidate_originals':len(candidates),'assigned_originals':len(assigned),
            'edge_and_exclusion_counts':proof,'quota_repairs':len(moves),
            'reproduced_role_hashes':hashes,'selected_originals_by_role_dataset':exposure,
            'raw_coco_candidate_labels':verify_coco(rows(coco),mapping),
            'scope':'Replay of frozen candidates, dHash graph, role repair, sampling and prompt expansion. No new human review or model outcomes.'}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    report=audit(a.cache)
    write_json(a.output,report)
    print(json.dumps(report,indent=2))
