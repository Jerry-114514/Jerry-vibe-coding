"""Fixed1000-update independent coordinate/supervision intervention."""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time

from experiment_common import ROOT,read_json,read_rows,write_json,write_rows,sha
from prepare_9b_repair import OUT,SEEDS,now
from repair_9b_codec import combined_loss


def train(condition,seed,until=1000):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    from gpu_lock import gpu_lock
    from train_distill import response_forward
    import preflight_9b_full as shared
    from route_b_9b_loop import model_backend,gpu_inputs
    from run_9b_repair import admission,phase_budget
    from run_9b_native_frontend_v41 import preserve_tail
    assert condition in ('P-G','N-D','N-G') and seed in SEEDS and until in (50,1000)
    plan=admission();folder=OUT/'training'/f'{condition}-seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():verify_training(condition,seed);return
    path=OUT/'training-inputs'/f'{condition}-seed{seed}.jsonl';rows=read_rows(path)
    meta={'protocol_sha256':sha(OUT/'protocol.json'),'condition':condition,'seed':seed,
          'updates':1000,'input_sha256':sha(path),'sampling_ids':[r['item_id'] for r in rows]}
    if (folder/'train.meta.json').exists():assert read_json(folder/'train.meta.json')==meta
    else:write_json(folder/'train.meta.json',meta)
    with gpu_lock():
        end=phase_budget('training',create=True)
        torch.set_num_threads(2);torch.manual_seed(seed);random.seed(seed)
        torch.cuda.reset_peak_memory_stats()
        processor,base=model_backend();base.config.use_cache=False
        config=plan['training']
        model=get_peft_model(base,LoraConfig(r=config['lora_r'],lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],target_modules=config['lora_targets'],bias='none',task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads();model.train()
        params=[p for p in model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in params)==config['expected_trainable_parameters']
        assert all('lora_' in n and 'visual' not in n for n,p in model.named_parameters() if p.requires_grad)
        optimizer=torch.optim.AdamW(params,lr=config['lr'],weight_decay=config['weight_decay'])
        checkpoint=folder/'resume.pt';ledger=folder/'loss.jsonl';start=0
        if checkpoint.exists():
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert state['meta']==meta
            set_peft_model_state_dict(model,state['adapter']);optimizer.load_state_dict(state['optimizer'])
            start=state['step'];assert {int(v['step']) for v in optimizer.state.values()}=={start}
            torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state_all(state['rng_cuda']);random.setstate(state['rng_python'])
            proof={'restored_step':start,'optimizer_restored':True,'RNG_restored':True,
                   'checkpoint_sha256':sha(checkpoint),'new_pid':os.getpid(),'timestamp_utc':now()}
            with (folder/'resume-events.jsonl').open('a') as f:f.write(json.dumps(proof)+'\n')
            write_json(folder/'resume-verification.json',proof)
            del state
        preserve_tail(ledger)
        previous=read_rows(ledger) if ledger.exists() else []
        committed=[r for r in previous if r['step']<=start]
        assert [r['step'] for r in committed]==list(range(1,start+1))
        if len(previous)!=len(committed):write_rows(folder/f'discarded-uncommitted-{time.time_ns()}.jsonl',previous[len(committed):])
        write_rows(ledger,committed)
        for step in range(start+1,until+1):
            assert time.time()<end and shutil.disk_usage(OUT).free>=plan['minimum_free_bytes']
            row=rows[(step-1)%len(rows)]
            assert sha(ROOT/row['image_path'])==row['source_image_sha256']
            tick=time.monotonic();prompt=shared.prompt_inputs(processor,row);length=prompt['input_ids'].shape[1]
            assert length==row['prompt_tokens'] and prompt['image_grid_thw'][0].tolist()==row['processor_grid_thw']
            assert hashlib.sha256(prompt['input_ids'].numpy().tobytes()).hexdigest()==row['prompt_ids_sha256']
            gt_ids=shared.targets(processor,row['gt_target']);assert gt_ids==row['gt_target_ids']
            gt=gpu_inputs(shared.suffix_inputs(prompt,gt_ids))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,labels=response_forward(model,gt,length,base_model=base)
                ce_gt=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1))
                ce_teacher=None
                if condition.endswith('-D'):
                    ids=shared.targets(processor,row['teacher_target']);assert ids==row['teacher_target_ids']
                    taught=gpu_inputs(shared.suffix_inputs(prompt,ids))
                    logits_t,labels_t=response_forward(model,taught,length,base_model=base)
                    ce_teacher=F.cross_entropy(logits_t.float().reshape(-1,logits_t.shape[-1]),labels_t.reshape(-1))
                loss=combined_loss(ce_gt,ce_teacher,condition)
            assert torch.isfinite(loss)
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,config['clip_grad_norm'])
            assert torch.isfinite(grad) and grad>0
            optimizer.step();torch.cuda.synchronize()
            peak=torch.cuda.max_memory_reserved();assert peak<=plan['reserved_bytes_cap']
            record={'step':step,'condition':condition,'seed':seed,'item_id':row['item_id'],'axis':row['axis'],
                'loss':float(loss.detach()),'ce_gt':float(ce_gt.detach()),
                'ce_teacher':float(ce_teacher.detach()) if ce_teacher is not None else None,
                'teacher_forward_computed':ce_teacher is not None,'grad_norm':float(grad),
                'seconds':time.monotonic()-tick,'reserved_peak_bytes':peak,
                'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'actual_optimizer_update':True}
            with ledger.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            write_json(OUT/'status.json',{'phase':'training','condition':condition,'seed':seed,
                'step':step,'pid':os.getpid(),'timestamp_utc':now()})
            if step%50==0 or step==until:
                state={'meta':meta,'step':step,'adapter':{k:v.detach().cpu() for k,v in get_peft_model_state_dict(model).items()},
                    'optimizer':optimizer.state_dict(),'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
                    'rng_python':random.getstate()}
                temp=checkpoint.with_suffix('.tmp');torch.save(state,temp);temp.replace(checkpoint);del state
                print('repair',condition,seed,step,record['loss'],flush=True)
            del logits,labels,loss,ce_gt,gt,prompt
            if ce_teacher is not None:del logits_t,labels_t,taught
            del ce_teacher
        if until==50:
            marker=folder/'interruption-at50.json'
            if not marker.exists():write_json(marker,{'step':50,'pid':os.getpid(),'checkpoint_sha256':sha(checkpoint),'timestamp_utc':now()})
            return
        model.save_pretrained(folder/'adapter',safe_serialization=True)
        assert {int(v['step']) for v in optimizer.state.values()}=={1000}
        write_json(folder/'complete.json',{'condition':condition,'seed':seed,'actual_updates':1000,
            'adapter_sha256':sha(folder/'adapter/adapter_model.safetensors'),'loss_sha256':sha(ledger),
            'resume_sha256':sha(checkpoint),'optimizer_step_values':[1000],
            'reserved_peak_bytes':torch.cuda.max_memory_reserved(),'timestamp_utc':now()})


def verify_training(condition,seed):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    folder=OUT/'training'/f'{condition}-seed{seed}';marker=read_json(folder/'complete.json')
    rows=read_rows(folder/'loss.jsonl');meta=read_json(folder/'train.meta.json')
    assert [r['step'] for r in rows]==list(range(1,1001)) and marker['actual_updates']==1000
    assert meta['condition']==condition and meta['seed']==seed
    assert sha(folder/'loss.jsonl')==marker['loss_sha256']
    for row in rows:
        assert row['actual_optimizer_update'] and row['grad_norm']>0
        assert all(math.isfinite(row[k]) for k in ('loss','ce_gt','grad_norm'))
        assert row['item_id']==meta['sampling_ids'][(row['step']-1)%600]
        assert row['teacher_forward_computed']==condition.endswith('-D')
        if row['ce_teacher'] is not None:assert math.isfinite(row['ce_teacher'])
    assert sha(folder/'adapter/adapter_model.safetensors')==marker['adapter_sha256']
    assert sha(folder/'resume.pt')==marker['resume_sha256']
    state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
    assert state['step']==1000 and state['meta']==meta
    assert {int(v['step']) for v in state['optimizer']['state'].values()}=={1000}
    exported=load_file(str(folder/'adapter/adapter_model.safetensors'),device='cpu')
    assert set(exported)==set(state['adapter'])
    assert all(torch.equal(v,state['adapter'][k]) and torch.isfinite(v).all() for k,v in exported.items())
    assert all(torch.count_nonzero(v)>0 for k,v in exported.items() if 'lora_B' in k)
    if condition=='P-G' and seed==42:
        stopped=read_json(folder/'interruption-at50.json');events=read_rows(folder/'resume-events.jsonl')
        assert stopped['step']==50 and any(e['restored_step']==50 and e['new_pid']!=stopped['pid'] for e in events)
    result={'status':'passed','actual_updates':1000,'condition':condition,'seed':seed,
        'axes':{a:sum(r['axis']==a for r in rows) for a in ('pope','grounding','consistency','mmstar')},
        'adapter_export_matches_checkpoint':True,'optimizer_states_at1000':True,'loss_finite':True,'timestamp_utc':now()}
    assert set(result['axes'].values())=={250}
    write_json(folder/'verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--until',type=int,default=1000);args=parser.parse_args();train(args.condition,args.seed,args.until)
