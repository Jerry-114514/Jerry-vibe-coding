"""Replay the original 7B protocol and separate prompt from scoring changes."""
import argparse
import json
from pathlib import Path
from statistics import mean
import tarfile

from audit_archived_paired_intervals import MODELS, assert_matches, paired
from audit_archived_training_chain import rows, table
from audit_v41_archived_dataflow import digest
from experiment_common import AXES, ROOT, read_json, read_rows, score_generation, sha, units, write_json
from processor_pixel_contract_csv import score as score_processor


def replay(cache):
    old_path = next(cache.glob('*091121Z.tar.gz'))
    new_path = next(cache.glob('*213408Z.tar.gz'))
    (run,) = (cache/'selective/arp-formal-20261014/runs/github-snapshots').glob('*003911Z')
    manifest_path = run/'frozen-inputs/data/manifest.jsonl'
    manifest = read_rows(manifest_path)
    lookup = {r['item_id']:r for r in manifest}
    evidence = ROOT/'docs/results/protocol-v2-20261016'
    saved = read_json(evidence/'route-b-final/multiplicity-results.json')
    plan = read_json(evidence/'route-b-final/reporting-amendment.json')
    scoring,models = [],{}
    with tarfile.open(old_path) as old, tarfile.open(new_path) as new:
        pm = rows(table(new,'/7b-processor-pixel-audit-20261017/manifest.jsonl'))
        proc = {r['item_id']:r for r in pm}
        assert proc.keys()=={r['item_id'] for r in manifest if r['axis']=='grounding'}
        for key,item in proc.items():
            assert all(item[k]==lookup[key][k] for k in ('width','height','answer_key','image_path'))
        for name in MODELS:
            suffix = '/route-b-staged-7b-20261016/formal/'+name
            raw = table(old,suffix+'/eval.jsonl')
            assert digest(raw)==saved['ledger_sha256'][name]
            assert sha(manifest_path)==json.loads(table(old,suffix+'/eval.meta.json'))['manifest_sha256']
            ledger = rows(raw)
            assert len(ledger)==len(lookup)==16900
            for r in ledger:
                expected = score_generation(r['raw_generation'],lookup[r['item_id']])
                assert all(r[k]==v for k,v in expected.items())
            collapsed = units(manifest,ledger)
            scoring.append(collapsed)
            by_axis = {a:mean(r['correct_bit'] for r in collapsed if r['axis']==a) for a in AXES}
            assert by_axis==saved['scores'][name]
            old_ground = {r['item_id']:r for r in ledger if r['axis']=='grounding'}
            new_ground = {r['item_id']:r for r in rows(table(new,'/models/'+name+'/eval.jsonl'))}
            assert old_ground.keys()==new_ground.keys()==proc.keys()
            models[name] = {'historical_scores':by_axis,
                'historical_generations_historical_pixel_score':by_axis['grounding'],
                'historical_generations_processor_pixel_score':mean(score_processor(r['raw_generation'],proc[k])['correct_bit'] for k,r in old_ground.items()),
                'processor_generations_historical_pixel_score':mean(score_generation(r['raw_generation'],lookup[k])['correct_bit'] for k,r in new_ground.items()),
                'processor_generations_processor_pixel_score':mean(score_processor(r['raw_generation'],proc[k])['correct_bit'] for k,r in new_ground.items())}
    contrasts = paired(scoring,AXES,plan['random_seed'],12,tuple(saved['paired_contrasts']))
    assert_matches(contrasts,saved['paired_contrasts'],None)
    keys = [k for k in models['N-seed42'] if k!='historical_scores']
    matrix = {k:mean(models[m][k] for m in MODELS if m.startswith('A1-'))-
                mean(models[m][k] for m in MODELS if m.startswith('U-')) for k in keys}
    return {'historical_rows_replayed':len(MODELS)*len(manifest),'registered_intervals_replayed':12,
            'endpoints':models,'A1_minus_U_cross_scoring_matrix':matrix,
            'scope':'Counterfactual reinterpretation of fixed raw texts is diagnostic only. Archived generators used different prompts; no per-item unit choice.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    data = replay(a.cache)
    write_json(a.output,data)
    print(data['A1_minus_U_cross_scoring_matrix'])
