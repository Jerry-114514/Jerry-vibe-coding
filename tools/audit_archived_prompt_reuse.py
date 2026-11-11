"""Identify cross-image MMBench prompt reuse and quantify its influence."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import unicodedata

from audit_archived_image_fingerprints_and_7b_cache import from_archive
from audit_archived_paired_intervals import aliases, load, paired
from experiment_common import write_json


def index(rows):
    result = defaultdict(list)
    for row in rows:
        if row['axis']=='consistency':
            key = unicodedata.normalize('NFKC',row['question'])
            result[' '.join(key.casefold().split())].append(row)
    return result


def audit(cache):
    mapping = {r['path']:r for r in json.loads(gzip.decompress((cache/'map.json.gz').read_bytes()))['files']}
    train = from_archive(cache,'094432','/clean-role-manifests-v4/train.jsonl')
    manifest,scoring = load(next(cache.glob('*070813Z.tar.gz')),'/native-frontend-evaluation-v41',
                             '/formal-manifest.jsonl','/formal/')
    a,b = index(train),index(manifest)
    matches = []
    for key in sorted(a.keys()&b.keys()):
        for x in a[key]:
            for y in b[key]:
                same = mapping[x['image_path']]['sha256']==mapping[y['image_path']]['sha256']
                matches.append({'train_id':x['item_id'],'eval_id':y['item_id'],
                    'train_unit':x['bootstrap_unit'],'eval_unit':y['bootstrap_unit'],
                    'same_label':x['answer_key']==y['answer_key'],'same_image_bytes':same})
    flagged = {r['eval_unit'] for r in matches}
    # Retain original RNG consumption across axes so interval draws stay comparable.
    axes = ('pope','grounding','consistency','mmstar')
    comps = ('A1-A2','A1-U','A1-N')
    cut = [[r for r in s if not (r['axis']=='consistency' and r['unit'] in flagged)] for s in scoring]
    clusters = aliases(manifest,mapping)
    results = {}
    for scope,group in [('full',scoring),('exclude_prompt_reuse',cut)]:
        x = paired(group,axes,2026101703,12,comps,clusters)
        results[scope] = {k:v['consistency'] for k,v in x.items()}
    return {'normalized_distinct_prompt_keys':len(a.keys()&b.keys()),
            'train_distinct_units':len({r['train_unit'] for r in matches}),
            'eval_distinct_units':len(flagged),'cross_image_prompt_pairs':len(matches),
            'pairs_same_label':sum(r['same_label'] for r in matches),
            'pairs_different_label':sum(not r['same_label'] for r in matches),
            'pairs_same_image_bytes':sum(r['same_image_bytes'] for r in matches),
            'comparisons':results,'match_index':matches,
            'label_prior':{'scienceqa_train':dict(Counter(r['answer_key'] for r in train if r['axis']=='mmstar')),
                           'mmstar_eval':dict(Counter(r['answer_key'] for r in manifest if r['axis']=='mmstar'))},
            'scope':'Common templates can be semantically appropriate on different images. This is prompt reuse, not proof of sample leakage; exclusion is post-hoc.'}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    out=audit(a.cache)
    write_json(a.output,out)
    print({k:v for k,v in out.items() if k!='match_index'})
