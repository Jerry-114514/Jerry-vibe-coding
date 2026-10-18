"""Fixed input evaluation; parse/length failures stay in the denominator."""
import argparse
import gc
import json
import os
import shutil
import time

from experiment_common import ROOT,AXES,read_json,read_rows,write_json,sha,units,score_generation
from prepare_9b_repair import OUT,SEEDS,now
from qwen35_native_frontend import score_native_frontend


def scoring(text,item):
    return score_native_frontend(text,item) if item['axis']=='grounding' else score_generation(text,item)


def manifest(dataset):
    assert dataset in ('development','heldout')
    return read_rows(OUT/f'{dataset}-manifest.jsonl')


def evaluate(condition,seed,dataset):
    import torch
    from gpu_lock import gpu_lock
    from route_b_9b_loop import model_backend,public_prompt_inputs,gpu_inputs
    from qwen35_termination_contract import explicit_eos_ids
    from model_backends import finite_logits_processors
    from run_9b_native_frontend_v41 import preserve_tail
    from run_9b_repair import admission,phase_budget,adapter_for
    from repair_9b_codec import CONDITIONS
    assert condition in (*CONDITIONS,'N') and seed in SEEDS
    assert dataset=='heldout' or condition in ('P-G','N-D','N-G')
    plan=admission();target=manifest(dataset);folder=OUT/'evaluations'/dataset/f'{condition}-seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    adapter=adapter_for(condition,seed)
    meta={'condition':condition,'seed':seed,'dataset':dataset,'protocol_sha256':sha(OUT/'protocol.json'),
        'manifest_sha256':sha(OUT/f'{dataset}-manifest.jsonl'),
        'adapter_sha256':sha(adapter/'adapter_model.safetensors') if adapter else None,
        'eos_token_ids':[248046,248044],'max_new_tokens':128,'thinking':False,
        'failure_policy':'Keep every output and fixed denominator; no prompt or unit switching'}
    if (folder/'eval.meta.json').exists():assert read_json(folder/'eval.meta.json')==meta
    else:write_json(folder/'eval.meta.json',meta)
    ledger=folder/'eval.jsonl';preserve_tail(ledger)
    previous=read_rows(ledger) if ledger.exists() else [];done={r['item_id'] for r in previous}
    assert len(done)==len(previous) and done<={r['item_id'] for r in target}
    if (folder/'scores.json').exists():verify_evaluation(condition,seed,dataset);return
    with gpu_lock():
        end=phase_budget('evaluation',create=True);started=time.monotonic()
        torch.set_num_threads(2);torch.set_grad_enabled(False);torch.cuda.reset_peak_memory_stats()
        processor,model=model_backend(adapter);model.eval().requires_grad_(False)
        assert not any(p.requires_grad for p in model.parameters())
        eos=explicit_eos_ids(processor.tokenizer,model.generation_config)
        assert eos==plan['evaluation']['eos_token_ids']
        for item in target:
            if item['item_id'] in done:continue
            assert time.time()<end and shutil.disk_usage(OUT).free>=plan['minimum_free_bytes']
            tick=time.monotonic()
            with torch.inference_mode():
                cpu=public_prompt_inputs(processor,item)
                if 'frozen_prompt_tokens' in item:assert cpu['input_ids'].shape[1]==item['frozen_prompt_tokens']
                if 'processor_grid_thw' in item:assert cpu['image_grid_thw'][0].tolist()==item['processor_grid_thw']
                batch=gpu_inputs(cpu)
                generated=model.generate(**batch,max_new_tokens=128,do_sample=False,eos_token_id=eos,
                    logits_processor=finite_logits_processors())
                ids=generated[0,batch['input_ids'].shape[1]:];raw=processor.decode(ids,skip_special_tokens=True).strip()
            torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved()
            assert peak<=plan['reserved_bytes_cap']
            row={'item_id':item['item_id'],'axis':item['axis'],'seconds':time.monotonic()-tick,
                'generation_tokens':ids.numel(),'generated_token_ids':ids.tolist(),
                'finish_reason':'eos' if ids.numel() and int(ids[-1]) in eos else 'length',
                'reserved_peak_bytes':peak,'allocated_peak_bytes':torch.cuda.max_memory_allocated(),
                **scoring(raw,item)}
            with ledger.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            done.add(item['item_id'])
            if len(done)%100==0:
                write_json(OUT/'status.json',{'phase':'evaluation','dataset':dataset,'condition':condition,
                    'seed':seed,'rows':len(done),'target':len(target),'pid':os.getpid(),'timestamp_utc':now()})
                print('repair eval',dataset,condition,seed,len(done),'/',len(target),flush=True)
        records=read_rows(ledger);original=units(target,records)
        assert len(records)==len({r['item_id'] for r in records})==len(target)
        assert len(records)==(16900 if dataset=='development' else 960)
        marker={'rows':len(records),'original_units':len(original),'ledger_sha256':sha(ledger),
            'protocol_sha256':sha(OUT/'protocol.json'),'optimizer_loaded':False,'gradients_enabled':False,
            'peak_reserved_bytes':max(r['reserved_peak_bytes'] for r in records),
            'total_generation_seconds':sum(r['seconds'] for r in records),
            'last_process_active_seconds':time.monotonic()-started,'finished_utc':now(),
            'length_capped_outputs':sum(r['finish_reason']=='length' for r in records),
            'parse_failures_by_axis':{a:sum(r['parse_fail_flag'] for r in records if r['axis']==a) for a in AXES},
            'axis_scores':{a:sum(r['correct_bit'] for r in original if r['axis']==a)/sum(r['axis']==a for r in original) for a in AXES}}
        write_json(folder/'scores.json',marker)
        del model,processor;gc.collect();torch.cuda.empty_cache()


def verify_evaluation(condition,seed,dataset):
    from run_9b_repair import adapter_for
    target=manifest(dataset);lookup={r['item_id']:r for r in target}
    folder=OUT/'evaluations'/dataset/f'{condition}-seed{seed}'
    marker=read_json(folder/'scores.json');meta=read_json(folder/'eval.meta.json');rows=read_rows(folder/'eval.jsonl')
    assert sha(folder/'eval.jsonl')==marker['ledger_sha256']
    assert len(rows)==len({r['item_id'] for r in rows})==len(target)==marker['rows']
    assert {r['item_id'] for r in rows}==set(lookup)
    assert meta['protocol_sha256']==marker['protocol_sha256']==sha(OUT/'protocol.json')
    assert not marker['optimizer_loaded'] and not marker['gradients_enabled']
    adapter=adapter_for(condition,seed)
    assert meta['adapter_sha256']==(sha(adapter/'adapter_model.safetensors') if adapter else None)
    for row in rows:
        scored=scoring(row['raw_generation'],lookup[row['item_id']])
        assert all(row[k]==v for k,v in scored.items())
        assert row['generation_tokens']==len(row['generated_token_ids'])<=128
        assert row['finish_reason']==('eos' if row['generated_token_ids'] and row['generated_token_ids'][-1] in (248046,248044) else 'length')
    original=units(target,rows)
    assert marker['original_units']==len(original)==(13700 if dataset=='development' else 480)
    expected={a:sum(r['correct_bit'] for r in original if r['axis']==a)/sum(r['axis']==a for r in original) for a in AXES}
    assert expected==marker['axis_scores']
    write_json(folder/'verification.json',{'status':'passed','rows':len(rows),'original_units':len(original),
        'ledger_sha256':marker['ledger_sha256'],'independent_rescoring':True,'timestamp_utc':now()})
    return marker,{(r['axis'],r['dataset'],r['unit']):r for r in original}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--dataset',choices=['development','heldout'],required=True);args=parser.parse_args()
    evaluate(args.condition,args.seed,args.dataset)
