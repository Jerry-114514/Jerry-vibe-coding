"""Independently audit full B/C/D evidence; partial runs remain incomplete."""
import argparse
import math
from pathlib import Path
from experiment_common import ROOT, MODELS, AXES, read_json, read_rows, sha, write_json
from model_backends import BACKEND_VERSION


def audit(run):
    run=Path(run)
    manifest=read_rows(ROOT/'data/manifest.jsonl')
    expected={row['item_id'] for row in manifest}
    checks=[]
    def check(name, operation):
        try:
            detail=operation()
            checks.append({'requirement':name,'passed':True,'evidence':detail})
            return True
        except Exception as error:
            checks.append({'requirement':name,'passed':False,'reason':str(error)[:400]})
            return False
    def require(condition,message):
        if not condition:
            raise ValueError(message)
    def ledger(directory,key):
        paths=sorted(directory.glob(f'{key}-[0-9][0-9][0-9].jsonl'))
        require(bool(paths),'Ledger missing')
        rows=[]
        for path in paths:
            meta=read_json(path.with_suffix('.meta.json'))
            complete=read_json(path.with_suffix('.complete.json'))
            require(not meta['smoke'] and not meta.get('synthetic'), 'Nonformal ledger')
            require(meta['manifest_sha256']==sha(ROOT/'data/manifest.jsonl'),'Manifest changed')
            require(meta.get('inference_backend_version')==BACKEND_VERSION,'Numerical backend mismatch')
            require(complete['ledger_sha256']==sha(path),'Completion checksum mismatch')
            part=read_rows(path)
            require(complete['rows']==len(part),'Completion row count mismatch')
            rows.extend(part)
        require(len(rows)==len(expected) and {row['item_id'] for row in rows}==expected,'Incomplete or duplicate item coverage')
        require(all(row.get('numerics_checked') and row['correct_bit'] in [0,1] and row['parse_fail_flag'] in [0,1] for row in rows),'Numerical/scoring evidence missing')
        return {'rows':len(rows),'files':{str(path.relative_to(run)):sha(path) for path in paths}}
    for key in MODELS:
        check('B3-'+key,lambda key=key:ledger(run/'ledger',key))
    def analysis():
        import numpy as np
        summary=read_json(run/'analysis/summary.json')
        gate=read_json(run/'analysis/gate.json')
        require(gate==summary['gate'],'Summary/gate mismatch')
        require(gate['formal'] and not gate.get('synthetic_test') and gate['n_boot']==1000,'Nonformal bootstrap')
        require(gate['protocol_sha256']==sha(ROOT/'configs/experiment.json'),'Protocol mismatch')
        require(gate['manifest_sha256']==sha(ROOT/'data/manifest.jsonl'),'Manifest mismatch')
        for relative,digest in gate['ledger_sha256'].items():
            require(sha(run/relative)==digest,'Ledger changed after analysis')
        for family,result in summary['families'].items():
            require(len(read_rows(run/f'analysis/{family}-folding.jsonl'))==120,'Incomplete ruler mapping')
            require(len(result['attribution'])==5,'Missing coordinate attribution')
            with np.load(run/f'analysis/{family}-bootstrap.npz') as values:
                require(values['pi'].shape==(1000,len(AXES)),'Bootstrap pi coverage mismatch')
                require(values['homogeneous_pi'].shape==(1000,len(AXES)),'D5 bootstrap coverage mismatch')
            require('arm_selection_defined' in result['homogeneous_floor_control'],'D5 selection evidence missing')
        planted=read_json(run/'analysis/planted.json')
        require(planted['homogeneous_recovered'] and planted['heterogeneous_flip_recovered'],'Planted recovery failed')
        require(abs(planted['flip_ratio_boundary'][0]-(1-.5)/(1-.02))<1e-12,'Planted boundary mismatch')
        return {'decision':gate['decision'],'n_boot':1000,'families':list(summary['families'])}
    c=check('C1-C5-and-D5',analysis)
    gate=read_json(run/'analysis/gate.json') if c else {}
    if gate.get('decision')=='go':
        def teacher():
            complete=read_json(run/'teacher-cache/complete.json')
            pool=ROOT/'data/training/pool.jsonl'
            require(complete['manifest_sha256']==sha(pool),'Teacher manifest mismatch')
            require(complete['items']==len(read_rows(pool)),'Teacher coverage mismatch')
            meta=read_json(run/'teacher-cache/cache.meta.json')
            require(meta.get('inference_backend_version')==BACKEND_VERSION and meta['top_k']==50,'Teacher protocol mismatch')
            for name,digest in complete['files'].items():
                require(sha(run/'teacher-cache'/name)==digest,'Teacher checksum mismatch')
            return {'items':complete['items'],'top_k':50}
        check('D2-teacher',teacher)
        def training(arm):
            folder=run/'arms'/arm
            complete=read_json(folder/'complete.json')
            meta=read_json(folder/'train.meta.json')
            require(not complete['smoke'] and meta['steps']==1000,'Nonformal training budget')
            require(meta['manifest_sha256']==sha(run/'training'/f'{arm}.jsonl'),'Training manifest changed')
            if arm=='N':
                require(complete['effective_updates']==0 and complete['no_op'],'Invalid no-op arm')
            else:
                require(complete['effective_updates']==1000,'Optimizer update budget incomplete')
                losses=read_rows(folder/'loss.jsonl')
                require([row['step'] for row in losses]==list(range(1,1001)),'Loss history incomplete')
                require(all(math.isfinite(row[field]) for row in losses for field in ['loss','ce','kd','grad_norm']),'Nonfinite training statistics')
                require((folder/'adapter/adapter_model.safetensors').is_file(),'Adapter missing')
                require((folder/'resume.pt').is_file(),'Resume state missing')
            return complete
        for arm in ['A1','A2','U','N']:
            check('D3-'+arm,lambda arm=arm:training(arm))
            if arm!='N':
                check('D4-'+arm,lambda arm=arm:ledger(run/'post'/arm,'qwen3b'))
        def gains():
            signed=read_json(run/'paper/signed-gains.json')
            require({row['arm'] for row in signed}=={'A1','A2','U','N'},'Signed gains incomplete')
            for row in signed:
                require(set(row['signed_components'])==set(AXES),'Per-axis components missing')
                require(abs(row['G_net_raw']-sum(row['signed_components'].values()))<1e-12,'Signed gain sum mismatch')
            require((run/'paper/table4-arm-gains.csv').is_file(),'Ruler-class gains missing')
            require((run/'training/homogeneous_floor_arm_selection.json').is_file(),'D5 arm selection comparison missing')
            return {'arms':4,'signed_axes':len(AXES)}
        check('D4-gains-and-D5-selection',gains)
    else:
        checks.append({'requirement':'D-gate','passed':c and gate.get('decision')=='no-go',
                       'evidence':gate.get('decision','C incomplete'),
                       'reason':'D requires a formal go; indeterminate remains unresolved'})
    result={'status':'complete' if all(row['passed'] for row in checks) else 'incomplete',
            'gate_decision':gate.get('decision'),'checks':checks}
    write_json(run/'audit.json',result)
    print(result,flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default=str(ROOT/'runs/formal-20261014'))
    audit(parser.parse_args().run)
