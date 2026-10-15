"""Read-only B/C replay audit, including every saved bootstrap draw."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import re
import numpy as np
from experiment_common import ROOT, MODELS, LADDERS, AXES, read_rows, read_json, sha, score_generation, units, write_json
from analyze_rulers import bootstrap_scores, scan, finite_json, ci, planted, verify_go
from model_backends import BACKEND_VERSION

def verify(run):
    run=ROOT/run
    protocol=read_json(ROOT/'configs/experiment.json')
    manifest=read_rows(ROOT/'data/manifest.jsonl')
    items={row['item_id']:row for row in manifest}
    frozen=read_json(ROOT/'data/experiment.lock.json')
    for relative,digest in frozen['files'].items():
        assert sha(ROOT/relative)==digest, 'Frozen input mismatch: '+relative
    summary=read_json(run/'analysis/summary.json')
    assert set(summary['families'])==set(LADDERS)
    gate=verify_go(run)
    assert gate==summary['gate'] and gate['n_boot']==1000
    ledgers={}
    report={'verified_utc':datetime.now(timezone.utc).isoformat(),'B':{},'C':{},'status':'checking'}
    for key in MODELS:
        path=run/'ledger'/f'{key}-000.jsonl'
        rows=read_rows(path)
        meta=read_json(path.with_suffix('.meta.json'))
        marker=read_json(path.with_suffix('.complete.json'))
        assert len(rows)==len(items)==16900
        assert {row['item_id'] for row in rows}==set(items)
        assert marker['rows']==len(rows) and marker['ledger_sha256']==sha(path)
        assert meta['selected_rows']==16900 and not meta['smoke'] and not meta.get('synthetic')
        assert meta['inference_backend_version']==BACKEND_VERSION
        assert meta['model']==read_json(ROOT/'configs/models.lock.json')['models'][key]
        for row in rows:
            item=items[row['item_id']]
            assert row['axis']==item['axis'] and row['numerics_checked']
            rescored=score_generation(row['raw_generation'],item)
            assert all(row[field]==value for field,value in rescored.items()), 'Rescoring mismatch: '+row['item_id']
        collapsed=units(manifest,rows)
        report['B'][key]={'rows':len(rows),'scoring_units':len(collapsed),
            'axis_rows':dict(Counter(row['axis'] for row in rows)),
            'parse_failures':sum(row['parse_fail_flag'] for row in rows),
            'sha256':sha(path),'all_rows_rescored':True}
        ledgers[key]=rows
        print('Verified B:',key,len(rows),'rows',flush=True)
    for family in LADDERS:
        saved=np.load(run/f'analysis/{family}-bootstrap.npz')
        # Regenerate all paired draws from the frozen manifest and recorded seed.
        scores,floors=bootstrap_scores(manifest,ledgers,family,1000,protocol['seed'])
        for mode in ['wrong','exclude']:
            assert np.allclose(scores[mode],saved['scores_'+mode],atol=1e-12,rtol=0,equal_nan=True)
            assert np.allclose(floors[mode],saved['floors_'+mode],atol=1e-12,rtol=0,equal_nan=True)
        point=scan({m:scores[m][0] for m in scores},{m:floors[m][0] for m in floors},protocol)
        result=summary['families'][family]
        assert finite_json(point['classes'])==read_json(run/f'analysis/{family}-classes.json')
        assert point['mapping']==read_rows(run/f'analysis/{family}-folding.jsonl')
        assert len(point['mapping'])==120
        assert len(result['attribution'])==5
        assert saved['pi'].shape==(1000,4) and saved['F'].shape==(1000,)
        assert saved['homogeneous_pi'].shape==(1000,4)
        assert saved['noise_outer_rates'].shape==(1000,)
        assert np.allclose(saved['pi'].mean(axis=0),list(result['pi_boot_mean'].values()),atol=1e-12)
        assert ci(saved['F'])==result['F_scale_CI']
        assert ci(saved['noise_outer_rates'])==result['F_noise_CI']
        assert result['c_a']==dict(zip(AXES,floors['wrong'][0].tolist()))
        assert result['u_a']==dict(zip(AXES,scores['wrong'][0,-1].tolist()))
        report['C'][family]={'paired_draws_reproduced':1000,'ruler_settings':120,
            'saved_bootstrap_rows':1000,'classes_reproduced':len(point['classes']),
            'F_scale':result['F_scale'],'F_scale_CI':result['F_scale_CI'],
            'F_noise':result['F_noise'],'F_noise_CI':result['F_noise_CI']}
        print('Verified C:',family,'1000 paired draws, classes and confidence intervals',flush=True)
        saved.close()
    assert planted()==read_json(run/'analysis/planted.json')
    log=(run/'console.log').read_text(encoding='utf-8')
    completions={key:bool(re.search(re.escape(key)+r' 16900/16900\b',log)) for key in MODELS}
    assert all(completions.values()), 'Missing final inference log record'
    errors=[line for line in log.splitlines() if re.match(r'^(Traceback \(most recent call last\):|(?:RuntimeError|FloatingPointError|CalledProcessError|torch\.OutOfMemoryError):)',line)]
    assert not errors, 'Exception in current formal run log'
    report.update(status='passed',gate_decision=gate['decision'],total_rows=sum(len(x) for x in ledgers.values()),
                  formal_log_completion_records=completions,current_formal_log_exception_count=len(errors),
                  scope='B and C complete; this report does not claim D completion')
    write_json(run/'bc-verification.json',report)
    print('B/C verification PASSED:',report['total_rows'],'rows; formal gate:',gate['decision'],flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='runs/formal-20261014')
    verify(p.parse_args().run)
