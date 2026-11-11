"""Reproduce registered 7B/9B intervals and stress-test exact-image aliases."""
import argparse
import gzip
import json
from pathlib import Path
import tarfile

import numpy as np

from audit_archived_training_chain import image_paths, rows, table
from experiment_common import AXES, ROOT, read_json, units, write_json

MODELS = ['N-seed42'] + [f'{a}-seed{s}' for s in (42,137,256,512,1024) for a in ('A1','A2','U')]


def load(path, prefix, manifest_suffix, ledger_stem):
    with tarfile.open(path) as tar:
        manifest = rows(table(tar,prefix+manifest_suffix))
        ledgers = [rows(table(tar,prefix+ledger_stem+m+'/eval.jsonl')) for m in MODELS]
    return manifest,[units(manifest,ledger) for ledger in ledgers]


def aliases(manifest, mapping):
    """Join image groups if any image SHA matches, transitively within axis."""
    parent = {}
    seen = {}
    def find(key):
        parent.setdefault(key,key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for item in manifest:
        axis, group = item['axis'],item['image_group']
        key = (axis,group)
        find(key)
        for path in image_paths([item]):
            image = (axis,mapping[path]['sha256'])
            if image in seen:
                p,q = sorted((find(key),find(seen[image])))
                parent[q] = p
            else:
                seen[image] = key
    return {k:find(k)[1] for k in parent}


def paired(units_by_model, axes, seed, family, names, remap=None):
    rng = np.random.default_rng(seed)
    seed_indices = rng.integers(0,5,size=(10000,5))
    out = {n:{} for n in names}
    base = units_by_model[0]
    for axis in axes:
        positions = [i for i,r in enumerate(base) if r['axis']==axis]
        records = [base[i] for i in positions]
        clusters = [r['image_group'] for r in records]
        if remap:
            clusters = [remap[axis,g] for g in clusters]
        groups = sorted(set(clusters))
        index = {g:i for i,g in enumerate(groups)}
        totals = np.zeros((len(groups),16))
        counts = np.zeros(len(groups))
        for i,g in zip(positions,clusters):
            row = base[i]
            counts[index[g]] += 1
            for j,model in enumerate(units_by_model):
                other = model[i]
                assert (row['axis'],row['dataset'],row['unit'],row['image_group']) == (
                    other['axis'],other['dataset'],other['unit'],other['image_group'])
                totals[index[g],j] += other['correct_bit']
        points = totals.sum(0)/counts.sum()
        draws = np.empty((10000,16))
        for start in range(0,10000,40):
            w = rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=40)
            draws[start:start+40] = (w@totals)/(w@counts)[:,None]
        point,sample = {'N':points[0]},{'N':draws[:,0]}
        for i,arm in enumerate(('A1','A2','U')):
            point[arm] = points[1+i::3].mean()
            sample[arm] = draws[np.arange(10000)[:,None],1+3*seed_indices+i].mean(1)
        for n in names:
            a,b = n.split('-')
            values = sample[a]-sample[b]
            alpha = .05/family
            out[n][axis] = {'difference':float(point[a]-point[b]),
                'pointwise_ci95':np.quantile(values,[.025,.975]).tolist(),
                'bonferroni_familywise_ci95':np.quantile(values,[alpha/2,1-alpha/2]).tolist(),
                'image_clusters':len(groups),'original_units':len(records)}
    return out


def assert_matches(out,saved,axis):
    for k,x in out.items():
        for a,r in x.items():
            expected = saved[k][a] if axis is None else saved[k]
            for f in ('difference','pointwise_ci95','bonferroni_familywise_ci95'):
                assert np.allclose(r[f],expected[f],atol=1e-12,rtol=0),(k,a,f,r[f],expected[f])


def audit(cache):
    evidence = ROOT/'docs/results/protocol-v2-20261016'
    proof = read_json(evidence/'9b-v41-final/formal-report/evaluation-results.json')
    proof7 = read_json(evidence/'7b-processor-final/results.json')
    mapping = {r['path']:r for r in json.loads(gzip.decompress((cache/'map.json.gz').read_bytes()))['files']}
    manifest,m9 = load(next(cache.glob('*070813Z.tar.gz')),'/native-frontend-evaluation-v41',
                       '/formal-manifest.jsonl','/formal/')
    m7_manifest,m7 = load(next(cache.glob('*213408Z.tar.gz')),'/7b-processor-pixel-audit-20261017',
                       '/manifest.jsonl','/models/')
    p9 = paired(m9,AXES,proof['bootstrap_seed'],12,tuple(proof['paired_contrasts']))
    p7 = paired(m7,['grounding'],proof7['bootstrap_seed'],4,tuple(proof7['paired_contrasts']))
    assert_matches(p9,proof['paired_contrasts'],None)
    assert_matches(p7,proof7['paired_contrasts'],'grounding')
    merged = paired(m9,AXES,proof['bootstrap_seed'],12,tuple(proof['paired_contrasts']),aliases(manifest,mapping))
    return {'9b_registered':p9,'7b_grounding_registered':p7,'9b_byte_alias_cluster_sensitivity':merged,
            'scope':'A post-hoc sensitivity; exact archived file SHA only, not RGB or semantic near-duplicates.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    r = audit(a.cache)
    write_json(a.output,r)
    print('Registered intervals reproduced: 9B 12, 7B grounding 4.')
    for key in r['9b_registered']:
        print(key,'MMStar registered / byte-merged',r['9b_registered'][key]['mmstar'],
              r['9b_byte_alias_cluster_sensitivity'][key]['mmstar'])
