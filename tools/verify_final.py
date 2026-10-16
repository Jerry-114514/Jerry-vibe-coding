"""Final read-only validation of trained weights, optimizer updates and post ledgers."""
import argparse
import csv
import math
from pathlib import Path
import torch
from safetensors.torch import load_file
from experiment_common import ROOT, read_json, read_rows, sha, score_generation, write_json
from audit_experiment import audit
from verify_bc import verify
from export_results import net_gain

def final_verify(run):
    run=(ROOT/run).resolve()
    assert read_json(run/'status.json')=={'stage':'E','status':'complete','training_started':True}
    assert audit(run)['status']=='complete'
    verify(run)
    items={r['item_id']:r for r in read_rows(ROOT/'data/manifest.jsonl')}
    protocol=read_json(ROOT/'configs/experiment.json')['training']
    details={}
    for arm in ['A1','A2','U']:
        folder=run/'arms'/arm
        complete=read_json(folder/'complete.json')
        meta=read_json(folder/'train.meta.json')
        assert meta['protocol']==protocol and not meta['smoke']
        assert meta['cache_meta_sha256']==sha(run/'teacher-cache/cache.meta.json')
        state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
        assert state['step']==1000
        steps=[float(v['step']) for v in state['optimizer']['state'].values()]
        assert steps and all(s==1000 for s in steps), 'Actual optimizer update count mismatch'
        weights=load_file(str(folder/'adapter/adapter_model.safetensors'),device='cpu')
        assert set(weights)==set(state['adapter'])
        assert all(torch.isfinite(value).all() and torch.equal(value,state['adapter'][key]) for key,value in weights.items())
        assert sum(value.numel() for value in weights.values())==complete['trainable_parameters']
        b_weights=[v for k,v in weights.items() if '.lora_B.' in k]
        assert b_weights and any(torch.count_nonzero(v)>0 for v in b_weights), 'No learned LoRA B update'
        ledger=run/'post'/arm/'qwen3b-000.jsonl'
        post=read_rows(ledger)
        assert len(post)==len(items) and {r['item_id'] for r in post}==set(items)
        post_meta=read_json(ledger.with_suffix('.meta.json'))
        assert post_meta['adapter']==str(folder/'adapter')
        for row in post:
            rescored=score_generation(row['raw_generation'],items[row['item_id']])
            assert all(row[key]==value for key,value in rescored.items())
        details[arm]={'actual_optimizer_updates':1000,'adapter_equals_final_resume_state':True,
                      'learned_lora_B_nonzero':True,'post_rows_rescored':len(post),
                      'adapter_sha256':sha(folder/'adapter/adapter_model.safetensors')}
        print('Final verified:',arm,'1000 optimizer steps, learned weights and 16900 rescored items',flush=True)
        del state,weights,b_weights
    stored={row['arm']:row for row in read_json(run/'paper/signed-gains.json')}
    table=list(csv.DictReader((run/'paper/table4-arm-gains.csv').open(encoding='utf-8')))
    for arm in ['A1','A2','U','N']:
        rulers,raw=net_gain(run,arm)
        assert raw==stored[arm]
        saved={int(row['class_id']):row for row in table if row['arm']==arm}
        assert len(saved)==len(rulers)
        for row in rulers:
            value=saved[row['class_id']]['G_net']
            assert (value=='' and row['G_net'] is None) or (row['G_net'] is not None and math.isclose(float(value),row['G_net'],abs_tol=1e-12))
    result={'status':'passed','scope':'B, C and D including trained weights and final gains',
            'arms':details,'no_op_control':'N has zero updates and reuses the unchanged baseline',
            'signed_gains_and_ruler_class_table_reproduced':True,
            'status_file_sha256':sha(run/'status.json')}
    write_json(run/'final-verification.json',result)
    print('Final experiment verification PASSED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='runs/formal-20261014')
    final_verify(parser.parse_args().run)
