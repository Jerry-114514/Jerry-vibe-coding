"""Read-only round trip from archived prompts/labels to their raw source annotations."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from audit_archived_image_fingerprints_and_7b_cache import from_archive
from experiment_common import ROOT, read_json, read_rows, sha, write_json
from prepare_phase_a import TEMPLATES, image_bytes
from refer_api import load_original


def audit(cache):
    restore = gzip.decompress((cache/'map.json.gz').read_bytes())
    assert hashlib.sha256(restore).hexdigest() == read_json(
        ROOT/'docs/instance-retirement-20261020/release-ready.json')['restore_map_sha256']
    by_path = {r['path']:r for r in json.loads(restore)['files']}
    checked = {}

    def raw(rel):
        path = ROOT/rel
        assert sha(path) == by_path[rel]['sha256'], rel
        checked[rel] = by_path[rel]['sha256']
        return path

    (run,) = (cache/'selective/arp-formal-20261014/runs/github-snapshots').glob('*003911Z')
    manifest_path = run/'frozen-inputs/data/manifest.jsonl'
    assert sha(manifest_path) == by_path['data/manifest.jsonl']['sha256']
    evaluation = read_rows(manifest_path)
    train = from_archive(cache,'094432','/clean-role-manifests-v4/train.jsonl')
    eval_by_id = {r['item_id']:r for r in evaluation}
    assert len(eval_by_id)==len(evaluation)==16900
    assert len({r['item_id'] for r in train})==len(train)==3545
    counts = Counter()
    popex = {}
    for subset in ('random','popular','adversarial'):
        for row in read_rows(raw(f'data/raw/pope/coco_pope_{subset}.json')):
            item = eval_by_id[f"pope:{subset}:{row['question_id']}"]
            assert item['axis']=='pope' and item['subset']==subset
            assert item['question']==row['text']+' Answer with yes or no only.'
            assert item['answer_key']==row['label'].lower()
            assert Path(item['image_path']).name==row['image']
            counts['eval_pope'] += 1
            popex[item['item_id']] = True
    assert len(popex)==9000
    refer_details = {}
    for dataset, split_by in [('refcoco','unc'),('refcoco+','unc'),('refcocog','umd')]:
        refs, instances = load_original(raw(f'data/raw/refer/{dataset}.zip'),dataset,split_by)
        ref = {r['ref_id']:r for r in refs}
        anns = {a['id']:a for a in instances['annotations']}
        imgs = {im['id']:im for im in instances['images']}
        f = {}
        for scope, source, split in [('eval',evaluation,'val'),('train',train,'train')]:
            chosen = [r for r in source if r['axis']=='grounding' and r['dataset']==dataset]
            for row in chosen:
                _,rid,sid = row['item_id'].rsplit(':',2)
                annotation = ref[int(rid)]
                sentence = min(annotation['sentences'],key=lambda x:x['sent_id'])
                assert sentence['sent_id']==int(sid) and annotation['split']==split
                img = imgs[annotation['image_id']]
                x,y,w,h = anns[annotation['ann_id']]['bbox']
                assert row['answer_key']==[x,y,x+w,y+h]
                assert (row['width'],row['height'])==(img['width'],img['height'])
                assert Path(row['image_path']).name==img['file_name']
                assert 0<=x<x+w<=img['width'] and 0<=y<y+h<=img['height']
                expected = (f"Locate the object referred to by: {sentence['sent']}. Return only its bounding box as "
                            f"[x_min, y_min, x_max, y_max] in original-image pixel coordinates. The image is "
                            f"{img['width']} pixels wide and {img['height']} pixels high.")
                assert row['question']==expected
                if scope=='eval':
                    assert (row['ref_id'],row['ann_id'],row['image_id'],row['sent_id'])==(
                        int(rid),annotation['ann_id'],annotation['image_id'],int(sid))
                counts[scope+'_grounding'] += 1
            f[scope] = len(chosen)
        assert f=={'eval':800,'train':200}
        refer_details[dataset] = f
        if dataset=='refcoco':
            present = defaultdict(set)
            category = {r['id']:r['name'] for r in instances['categories']}
            for ann in instances['annotations']:
                present[ann['image_id']].add(ann['category_id'])
            for item in (r for r in train if r['axis']=='pope'):
                iid,cid = (int(x) for x in item['item_id'].split(':')[-2:])
                assert item['answer_key']==('yes' if cid in present[iid] else 'no')
                assert item['question']==f'Is there a {category[cid]} in the image? Answer with yes or no only.'
                assert Path(item['image_path']).name==imgs[iid]['file_name']
                counts['train_pope']+=1
    star = pq.read_table(raw('data/raw/mmstar/mmstar.parquet')).to_pylist()
    for s in star:
        item = eval_by_id[f"mmstar:{s['index']}"]
        assert item['question']==s['question']+'\nReturn only the option letter.'
        assert item['answer_key']==s['answer'] and item['category']==s['category'] and item['l2_category']==s['l2_category']
        assert hashlib.sha256(image_bytes(s)).hexdigest()==by_path[item['image_path']]['sha256']
        counts['eval_mmstar']+=1
    bench = {str(r['index']):r for r in pq.read_table(raw('data/raw/mmbench/dev-00000-of-00001.parquet')).to_pylist()}
    for scope,source in [('eval',evaluation),('train',train)]:
        for item in (r for r in source if r['axis']=='consistency'):
            key,n = re.search(r':(\d+):p([1-5])$',item['item_id']).groups()
            r = bench[key]
            opts = {k:r[k] for k in 'ABCD' if r.get(k) and r[k].strip().lower() not in ('nan','none','null')}
            hint = (r.get('hint') or '').strip()
            prefix = 'Context: '+hint+'\n' if hint and hint.lower() not in ('nan','none','null') else ''
            target = '\n'.join(f'{k}. {v}' for k,v in opts.items())
            assert item['question']==prefix+TEMPLATES[int(n)-1].format(question=r['question'])+'\n'+target+'\nReturn only the option letter.'
            assert item['answer_key']==r['answer'] and item['options']==list(opts)
            assert hashlib.sha256(image_bytes(r)).hexdigest()==by_path[item['image_path']]['sha256']
            counts[scope+'_consistency'] += 1
    science = None
    f = cache/'scienceqa-official-problems.json'
    s = cache/'scienceqa-official-pid_splits.json'
    if f.is_file() and s.is_file():
        raw_problems = read_json(f)
        split = read_json(s)
        indexed = defaultdict(set)
        for pid in split['train']:
            r = raw_problems[str(pid)]
            options = list('ABCD'[:len(r['choices'])])
            prompt = (('Context: '+r['hint']+'\n') if r.get('hint') else '') + r['question'] + '\n'
            prompt += '\n'.join(f'{k}. {v}' for k,v in zip(options,r['choices']))+'\nReturn only the option letter.'
            if len(options)==len(r['choices']) and r.get('image'):
                indexed[prompt].add((options[r['answer']],str(pid)))
        checked_items = [r for r in train if r['axis']=='mmstar']
        no_match, multiple, disagreement = [], [], []
        for r in checked_items:
            options = indexed[r['question']]
            labels = {label for label,_ in options}
            if not options:
                no_match.append(r['item_id'])
            elif r['answer_key'] not in labels:
                disagreement.append(r['item_id'])
            elif len(labels)>1 or len(options)>1:
                multiple.append(r['item_id'])
            else:
                counts['train_scienceqa_unique_prompt_answer'] += 1
        science = {'official_problems_sha256':sha(f),'official_split_sha256':sha(s),'pool_rows':len(checked_items),
                   'unmatched':no_match,'ambiguous':multiple,'disagreement':disagreement,
                   'scope':'Match exact prompt/choices and label to official train problem JSON. Does not verify HF parquet ordinal or image SHA.'}
    assert {k:counts[k] for k in ('eval_pope','eval_grounding','eval_mmstar','eval_consistency')} == {
        'eval_pope':9000,'eval_grounding':2400,'eval_mmstar':1500,'eval_consistency':4000}
    assert counts['train_pope']==600 and counts['train_grounding']==600 and counts['train_consistency']==1745
    return {'source_bytes_sha256':checked,'matched':dict(counts),'refer_splits':refer_details,
            'scienceqa':science,'scope':'Historical evaluation and v4 training sources. POPE training no means absent from source annotations, not verified physical absence.'}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    result=audit(a.cache)
    write_json(a.output,result)
    print(result['matched'])
    if result['scienceqa']:
        print({k:len(v) for k,v in result['scienceqa'].items() if isinstance(v,list)})
