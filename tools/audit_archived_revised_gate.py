"""Replay all six revised C analyses against archived historical inputs."""
import argparse
import math
from pathlib import Path

from experiment_common import LADDERS, ROOT, read_json, read_rows, sha, units, write_json
from paired_gate_v2 import analyze_family


def equal(a,b):
    if isinstance(a,dict):
        return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float):
        return math.isclose(a,b,abs_tol=1e-12,rel_tol=0)
    return a==b


def replay(cache,output):
    evidence = ROOT/'docs/results/protocol-v2-20261016/paired-gate-development'
    expected = read_json(evidence/'summary.json')
    historical = read_json(ROOT/'docs/results/final-20261016/analysis/gate.json')
    (run,) = (cache/'selective/arp-formal-20261014/runs/github-snapshots').glob('*003911Z')
    p = run/'frozen-inputs/data/manifest.jsonl'
    assert sha(p)==historical['manifest_sha256']
    manifest = read_rows(p)
    protocol = ROOT/'configs/experiment.json'
    assert sha(protocol)==historical['protocol_sha256']
    baseline = units(manifest,read_rows(run/'frozen-inputs/data/baselines/per_item.jsonl'))
    result = {}
    for family,keys in LADDERS.items():
        records = [baseline]
        for key in keys:
            rel = f'ledger/{key}-000.jsonl'
            path = run/rel
            assert sha(path)==historical['ledger_sha256'][rel]
            records.append(units(manifest,read_rows(path)))
        for suffix,axes in [('all_axes',[0,1,2,3]),('without_grounding',[0,2,3])]:
            name = family+'/'+suffix
            ref = expected[name]
            got,_ = analyze_family(records,read_json(protocol),ref['n_outer'],ref['n_inner'],
                                   seed=ref['seed'],axes=axes,
                                   progress=lambda i:print(name,i,flush=True))
            differences = [k for k,v in got.items() if not equal(v,ref[k])]
            assert not differences,(name,differences)
            result[name]=got
            write_json(output,{'results':result,'matched':len(result),'total':len(expected),
                               'formal_training_allowed':False})
    assert set(result)==set(expected)
    print('All six revised C analyses reproduced.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    replay(a.cache,a.output)
