"""CPU-only full EOS-cache audit and independent1000-update training-run audit."""
import argparse
from collections import Counter
import gc
import math
from pathlib import Path
import torch
from safetensors.torch import load_file
from experiment_common import ROOT, read_json, read_rows, sha, write_json
from route_b_staged import OUT, CACHE, arm_rows, SEEDS
from train_distill import cache_name


def cache_audit():
    torch.set_num_threads(2)
    protocol = read_json(OUT/'protocol.json')
    frozen = read_json(OUT/'phase2-frozen.json')
    assert frozen['protocol_sha256'] == sha(OUT/'protocol.json')
    assert frozen['cache_sha256'] == sha(CACHE/'complete.json')
    assert protocol['source_sha256'] == sha(ROOT/'tools/route_b_staged.py')
    rows = read_rows(ROOT/'data/training/pool.jsonl')
    assert len(rows)==3545 and sha(ROOT/'data/training/pool.jsonl')==protocol['pool_sha256']
    inventory = read_json(CACHE/'complete.json')
    assert inventory['items']==3545 and len(inventory['files'])==3545
    endings = read_json(ROOT/'.model_cache/qwen7b/generation_config.json')['eos_token_id']
    endings = set(endings if isinstance(endings,list) else [endings])
    images = {}
    counts = Counter()
    for row in rows:
        path = CACHE/cache_name(row['item_id'])
        assert sha(path)==inventory['files'][path.name]
        x = torch.load(path,map_location='cpu',weights_only=True)
        assert x['item_id']==row['item_id'] and x['response'].strip()
        assert x['finish_reason']=='eos' and int(x['generation_ids'][-1]) in endings
        assert 0<x['generation_ids'].numel()<=128
        targets = x['targets'].reshape(-1)
        assert x['top_indices'].shape==x['top_logits'].shape==(targets.numel(),50)
        assert x['log_normalizer'].shape==(targets.numel(),)
        assert torch.isfinite(x['top_logits']).all() and torch.isfinite(x['log_normalizer']).all()
        assert torch.equal(x['input_ids'][:,x['prompt_length']:].reshape(-1),targets)
        assert x['input_ids'].shape[1]<=2048 and 0<targets.numel()<=128
        if row['image_path'] not in images:images[row['image_path']]=sha(ROOT/row['image_path'])
        assert x['image_sha256']==images[row['image_path']]
        counts[row['axis']]+=1
    result = {'status':'passed','rows':3545,'axis_counts':dict(counts),'raw_eos_verified':True,
              'tensor_alignment_and_finite_top50':True,'no_truncation':True,
              'frozen_config_sha256':sha(OUT/'phase2-frozen.json'),'cache_inventory_sha256':sha(CACHE/'complete.json')}
    write_json(OUT/'full-eos-cache-independent-verification.json',result)
    print(result,flush=True)


def training_audit(arm,seed):
    folder = OUT/'formal'/f'{arm}-seed{seed}'
    marker = read_json(folder/'complete.json')
    metadata = read_json(folder/'train.meta.json')
    assert metadata['protocol_sha256']==sha(OUT/'protocol.json')
    assert metadata['cache_inventory_sha256']==sha(CACHE/'complete.json')
    planned = arm_rows('formal',arm,seed)
    assert metadata['sampling_ids']==[r['item_id'] for r in planned] and len(planned)==600
    losses = read_rows(folder/'loss.jsonl')
    assert marker['actual_updates']==len(losses)==1000
    assert marker['loss_sha256']==sha(folder/'loss.jsonl')
    assert marker['adapter_sha256']==sha(folder/'adapter/adapter_model.safetensors')
    assert [r['step'] for r in losses]==list(range(1,1001))
    for i,r in enumerate(losses):
        assert r['item_id']==planned[i%len(planned)]['item_id'] and r['axis']==planned[i%len(planned)]['axis']
        assert r['actual_optimizer_update'] and r['grad_norm']>0
        assert all(math.isfinite(r[k]) for k in ('loss','ce_gt','ce_teacher','grad_norm','seconds'))
        assert abs(r['loss']-(.5*r['ce_gt']+.5*r['ce_teacher']))<1e-5
    state = torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
    assert state['step']==1000 and state['meta']==metadata
    optimizer_steps = sorted({int(v['step']) for v in state['optimizer']['state'].values()})
    assert optimizer_steps==[1000]
    exported = load_file(str(folder/'adapter/adapter_model.safetensors'))
    assert set(exported)==set(state['adapter'])
    assert all(torch.equal(v,state['adapter'][k]) and torch.isfinite(v).all() for k,v in exported.items())
    assert all(torch.count_nonzero(v)>0 for k,v in exported.items() if 'lora_B' in k)
    peak = max(r['reserved_peak_bytes'] for r in losses)
    result = {'status':'passed','arm':arm,'seed':seed,'real_updates':1000,
              'optimizer_step_values':optimizer_steps,'axis_update_counts':dict(Counter(r['axis'] for r in losses)),
              'dual_ce_finite_nonzero_gradients':True,'exact_optimizer_adapter_export_match':True,
              'peak_reserved_bytes':peak,'peak_reserved_GB':peak/1e9,'peak_reserved_GiB':peak/1024**3,
              'training_step_seconds':sum(r['seconds'] for r in losses),'loss_sha256':sha(folder/'loss.jsonl'),
              'scope':'Step processing time excludes model loading, checkpoint I/O and backup'}
    del state,exported
    gc.collect()
    write_json(folder/'independent-training-verification.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',action='store_true');p.add_argument('--arm',choices=['A1','A2','U']);p.add_argument('--seed',type=int,choices=SEEDS)
    a=p.parse_args()
    if a.cache:cache_audit()
    if a.arm:training_audit(a.arm,a.seed)
