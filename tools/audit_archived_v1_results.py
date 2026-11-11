"""Re-score v1 B/D ledgers and reproduce the saved C inputs and ruler classes."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

import analyze_rulers as rulers
import export_results
from experiment_common import AXES, LADDERS, MODELS, ROOT, read_json, read_rows, score_generation, sha, units, write_json


def audit(cache):
    evidence = ROOT/'docs/results/final-20261016'
    inventories = list(cache.glob('arp-formal-*-20261016T004018Z-inventory.json'))
    assert len(inventories) == 1
    inventory = {r['path']:r for r in read_json(inventories[0])['files']}
    (run,) = (cache/'selective/arp-formal-20261014/runs/github-snapshots').glob('*003911Z')
    analysis = run/'analysis'
    root = run/'frozen-inputs'
    manifest_path = root/'data/manifest.jsonl'
    baseline = root/'data/baselines/per_item.jsonl'
    baseline.parent.mkdir(exist_ok=True)
    baseline.write_bytes((ROOT/'data/baselines/per_item.jsonl').read_bytes())
    for path in (manifest_path,baseline):
        assert sha(path) == inventory[path.relative_to(cache/'selective/arp-formal-20261014').as_posix()]['sha256']
    gate = read_json(analysis/'gate.json')
    assert sha(manifest_path) == gate['manifest_sha256']
    assert sha(ROOT/'configs/experiment.json') == gate['protocol_sha256']
    protocol = read_json(ROOT/'configs/experiment.json')
    manifest = read_rows(manifest_path)
    lookup = {x['item_id']:x for x in manifest}
    assert len(manifest)==len(lookup)==16900

    def replay(path):
        assert sha(path) == inventory[path.relative_to(cache/'selective/arp-formal-20261014').as_posix()]['sha256']
        rows = read_rows(path)
        assert len(rows)==len({r['item_id'] for r in rows})==len(lookup)
        for row in rows:
            item = lookup[row['item_id']]
            assert row['axis']==item['axis']
            expected = score_generation(row['raw_generation'],item)
            assert all(row[k]==v for k,v in expected.items()), row['item_id']
        return rows

    ledgers = {key:replay(run/f'ledger/{key}-000.jsonl') for key in MODELS}
    base_scores = {}
    with (evidence/'base-model-scores.csv').open(encoding='utf-8',newline='') as f:
        for expected in csv.DictReader(f):
            model = expected['model']
            unit = units(manifest,ledgers[model])
            recomputed = {}
            for axis in AXES:
                strata = sorted({x['stratum'] for x in unit if x['axis']==axis})
                recomputed[axis] = float(np.mean([
                    np.mean([x['correct_bit'] for x in unit if x['axis']==axis and x['stratum']==s])
                    for s in strata]))
            assert len(unit)==13700 and all(math.isclose(recomputed[a],float(expected[a]),abs_tol=1e-12) for a in AXES)
            base_scores[model] = {'rows':len(ledgers[model]),'scores':recomputed}
    rulers.ROOT = export_results.ROOT = root
    families = {}
    for family in LADDERS:
        saved = np.load(analysis/f'{family}-bootstrap.npz')
        scores,floors = rulers.bootstrap_scores(manifest,ledgers,family,gate['n_boot'],protocol['seed'])
        for mode in ('wrong','exclude'):
            assert np.allclose(scores[mode],saved['scores_'+mode],atol=1e-12,rtol=0,equal_nan=True)
            assert np.allclose(floors[mode],saved['floors_'+mode],atol=1e-12,rtol=0,equal_nan=True)
        point = rulers.scan({m:scores[m][0] for m in scores},{m:floors[m][0] for m in floors},protocol)
        assert rulers.finite_json(point['classes']) == read_json(analysis/f'{family}-classes.json')
        assert point['mapping'] == read_rows(analysis/f'{family}-folding.jsonl')
        assert len(point['mapping'])==120
        summary = read_json(analysis/'summary.json')['families'][family]
        assert rulers.ci(saved['F']) == summary['F_scale_CI']
        assert rulers.ci(saved['noise_outer_rates']) == summary['F_noise_CI']
        families[family] = {'classes':len(point['classes']),'settings':120,'paired_score_draws':gate['n_boot'],
                            'F_scale_CI_from_saved_F':summary['F_scale_CI'],'F_noise_CI_from_saved_draws':summary['F_noise_CI']}
        saved.close()
    report = {r['arm']:r for r in read_json(evidence/'signed-gains.json')}
    with (evidence/'table4-arm-gains.csv').open(encoding='utf-8',newline='') as f:
        saved_rows = {(r['arm'],int(r['class_id'])):r for r in csv.DictReader(f)}
    post = {}
    for arm in ('A1','A2','U','N'):
        post_path = run/f'post/{arm}/qwen3b-000.jsonl'
        raw = replay(post_path) if arm!='N' else ledgers['qwen3b']
        if arm=='N' and post_path.exists():
            assert replay(post_path) == raw
        recalculated,point = export_results.net_gain(run,arm)
        assert point==report[arm]
        for row in recalculated:
            value = saved_rows[arm,row['class_id']]['G_net']
            assert (value=='' and row['G_net'] is None) or (
                row['G_net'] is not None and math.isclose(float(value),row['G_net'],abs_tol=1e-12))
        post[arm] = {'rows':len(raw),'gain':point,'ruler_classes':len(recalculated),
                     'reuse_base':arm=='N'}
    return {'B':base_scores,'C':families,'D':post,'historical_gate':gate['decision'],
            'scope':'v1 paired score/floor draws and 120 class maps replayed. F/noise intervals checked from saved draws; nested draws not regenerated. D tensor/optimizer state not loaded.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    r = audit(a.cache)
    write_json(a.output,r)
    print('Replayed',sum(x['rows'] for x in r['B'].values()),'B rows,',
          sum(x['rows'] for x in r['D'].values() if not x['reuse_base']),'D rows and',len(r['C']),'C families.')
