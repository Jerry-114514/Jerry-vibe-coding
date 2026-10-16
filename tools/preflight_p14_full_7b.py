"""Independent P1-4 full seven-module LoRA + dual-CE memory preflight.

Does not modify or launch existing experiments. Public training inputs only.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

from PIL import Image
from experiment_common import ROOT, AXES, read_json, read_rows, write_json, write_rows, sha, parse_answer

OUT=ROOT/'runs/protocol-v2/calibration/p14-full-7b-v2'
MODULES=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
SEED=2026101603


def prepare():
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(ROOT/'.model_cache/qwen7b'),local_files_only=True,use_fast=False)
    pool=read_rows(ROOT/'data/training/pool.jsonl')
    groups=defaultdict(list)
    for row in pool: groups[(row['axis'],row['bootstrap_unit'])].append(row)
    selected=[]
    for axis in AXES:
        candidates=[max(rows,key=lambda r:len(processor.tokenizer.encode(r['question'])))
                    for (a,_),rows in groups.items() if a==axis]
        longest=max(candidates,key=lambda r:len(processor.tokenizer.encode(r['question'])))
        remaining=sorted([r for r in candidates if r['item_id']!=longest['item_id']],
                         key=lambda r:hashlib.sha256(('p14-full-memory:'+r['bootstrap_unit']).encode()).hexdigest())
        selected.extend([longest]+remaining[:24])
    if len(selected)!=100 or len({r['bootstrap_unit'] for r in selected})!=100:
        raise ValueError('Expected 100 distinct training units')
    random.Random(SEED).shuffle(selected)
    config=read_json(ROOT/'configs/api_teacher.json')
    if config['model']!='qwen3.5-122b-a10b':raise ValueError('Wrong assigned teacher')
    protocol={'schema':'arp.p14-full-memory-preflight.v2','engineering_only':True,'formal_training':False,
              'student':'qwen7b','teacher':config['model'],'steps':100,'seed':SEED,
              'lora':{'r':16,'alpha':32,'dropout':0,'target_modules':MODULES,'scope':'language attention and MLP only'},
              'loss':{'ground_truth_ce':.5,'teacher_response_ce':.5,'schedule':'Both forward graphs retained, one combined backward and one optimizer step'},
              'optimizer':{'name':'AdamW','lr':1e-4,'weight_decay':0,'max_grad_norm':1.0},
              'bf16':True,'gradient_checkpointing':True,'use_cache':False,'micro_batch':1,'gradient_accumulation':1,
              'max_total_tokens':2048,'max_visual_tokens':1280,'max_response_tokens':128,
              'teacher_max_requests':100,'teacher_max_cumulative_tokens':1000000,
              'stress':'After100 real updates, one synthetic shape-only dual-CE update:2048 total/128 response tokens; separate from real training ledger',
              'grounding':'Student explicit processed-image pixels; GT original pixels and teacher explicit normalized1000 converted using actual processor grid',
              'threshold_unit':'GB = 1000000000 bytes; GiB also reported',
              'thresholds_bytes':[20000000000,22000000000],
              'pool_sha256':sha(ROOT/'data/training/pool.jsonl'),
              'model_revision':read_json(ROOT/'configs/models.lock.json')['models']['qwen7b']['revision'],
              'source_sha256':sha(Path(__file__))}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        if read_json(OUT/'protocol.json')!=protocol or read_rows(OUT/'manifest.jsonl')!=selected:
            raise ValueError('Frozen preflight changed')
    else: write_json(OUT/'protocol.json',protocol);write_rows(OUT/'manifest.jsonl',selected)
    maximum=max(len(processor.tokenizer.encode(r['question'])) for r in pool)
    write_json(OUT/'source-coverage.json',{'source_rows':len(pool),'maximum_source_question_tokens':maximum,
               'selected_original_units':100,'units_per_axis':25,'longest_question_per_axis_included':True,
               'all_inputs_from_approved_public_training_pool':True,'evaluation_images_used':False})
    return processor,selected,config


def make_inputs(processor,item,response=None):
    path=(ROOT/item['image_path']).resolve()
    if not path.is_relative_to((ROOT/'data/training').resolve()):raise ValueError('Not a training image')
    with Image.open(path) as image: rgb=image.convert('RGB')
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':item['question']}]}]
    if response is not None:messages.append({'role':'assistant','content':[{'type':'text','text':response}]})
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=response is None,add_vision_id=True)
    inputs=processor(text=[text],images=[rgb],return_tensors='pt',images_kwargs={
        'size':{'shortest_edge':256*28*28,'longest_edge':1280*28*28},
        'min_pixels':256*28*28,'max_pixels':1280*28*28})
    visual=int(inputs['image_grid_thw'].prod(dim=1).sum().item()//4)
    if visual>1280:raise ValueError('Visual token cap exceeded')
    return inputs


def student_item(processor,row):
    item=dict(row)
    if item['axis']=='grounding':
        from run_processed_grounding import PROMPT
        grid=make_inputs(processor,item)['image_grid_thw'][0]
        patch=processor.image_processor.patch_size
        height,width=int(grid[1]*patch),int(grid[2]*patch)
        expression=row['question'].split('Locate the object referred to by: ',1)[1].split('. Return only',1)[0]
        item.update(question=PROMPT.format(expression=expression,width=width,height=height),
                    processed_width=width,processed_height=height)
    return item


def target_text(row,item,raw=None):
    if row['axis']!='grounding':return str(row['answer_key']) if raw is None else raw.strip()
    if raw is None:
        box=row['answer_key'];xscale=item['processed_width']/row['width'];yscale=item['processed_height']/row['height']
    else:
        box=parse_answer(raw,{'answer_type':'bbox','width':1000,'height':1000})
        if box is None:raise ValueError('Teacher normalized bbox is invalid')
        xscale=item['processed_width']/1000;yscale=item['processed_height']/1000
    return json.dumps([box[0]*xscale,box[1]*yscale,box[2]*xscale,box[3]*yscale])


def cache_teacher(rows,config,state):
    import requests
    from run_teacher_role_calibration import content
    ledger=OUT/'teacher.jsonl';inflight=OUT/'api-inflight.json'
    previous=read_rows(ledger) if ledger.exists() else []
    cached={r['item_id']:r for r in previous}
    if len(cached)!=len(previous):raise ValueError('Duplicate teacher cache')
    if inflight.exists():
        if read_json(inflight)['item_id'] not in cached:raise RuntimeError('Ambiguous paid request; no automatic retry')
        inflight.unlink()
    session=requests.Session();session.trust_env=False
    session.headers['Authorization']='Bearer '+os.environ[config['api_key_env']]
    base=os.environ.get('base_url') or config['base_url']
    if base.rstrip('/')!=config['base_url'].rstrip('/'):raise ValueError('API Base URL differs')
    used=sum(r['usage']['total_tokens'] for r in previous)
    for row in rows:
        if row['item_id'] in cached:continue
        if used+16384>1000000:raise RuntimeError('Teacher token budget reached')
        prompt=dict(row)
        if row['axis']=='grounding':
            expression=row['question'].split('Locate the object referred to by: ',1)[1].split('. Return only',1)[0]
            prompt['question']='Locate the object referred to by: '+expression+'. Return only a JSON array containing one object with bbox_2d and label. Normalize [x_min,y_min,x_max,y_max] coordinates to [0,1000] relative to the full image.'
        blocks,hashes=content(prompt)
        write_json(inflight,{'item_id':row['item_id'],'started_utc':datetime.now(timezone.utc).isoformat()})
        response=session.post(base.rstrip('/')+'/chat/completions',json={'model':config['model'],'max_tokens':128,
                              'temperature':0,'enable_thinking':False,'messages':[{'role':'user','content':blocks}]},timeout=120)
        if response.status_code!=200:
            write_json(OUT/'api-failure.json',{'http_status':response.status_code,'item_id':row['item_id']})
            raise RuntimeError('Teacher request failed; no paid retry')
        body=response.json();choice=body['choices'][0]
        entry={'item_id':row['item_id'],'axis':row['axis'],'raw_generation':choice['message'].get('content') or '',
               'returned_model':body.get('model'),'finish_reason':choice.get('finish_reason'),'usage':body['usage'],
               'image_hashes':hashes,'request_id':body.get('id'),'actual_prompt':prompt['question']}
        with ledger.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(entry,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
        cached[row['item_id']]=entry;used+=entry['usage']['total_tokens'];inflight.unlink()
        state.update(status='caching_teacher',cached_teacher_responses=len(cached));write_json(OUT/'status.json',state)
    for entry in cached.values():
        if entry['returned_model']!=config['model'] or not entry['raw_generation'] or entry['finish_reason']!='stop':
            raise ValueError('Teacher cache contains wrong model, empty or truncated answer')
    return cached


def training(processor,rows,cached,state):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
    from protocol_v2_backend import QwenV2Backend
    from train_distill import response_forward
    from gpu_lock import gpu_lock
    from precision_bridge import memory_worker
    prepared=[]
    for row in rows:
        item=student_item(processor,row)
        prompt=make_inputs(processor,item)
        length=prompt['input_ids'].shape[1]
        prepared.append((item,target_text(row,item),target_text(row,item,cached[row['item_id']]['raw_generation']),length))
        for text in prepared[-1][1:3]:
            inputs=make_inputs(processor,item,text)
            if not torch.equal(inputs['input_ids'][:,:length],prompt['input_ids']):raise ValueError('Assistant token boundary changed')
            if inputs['input_ids'].shape[1]>2048 or inputs['input_ids'].shape[1]-length>128:
                raise ValueError('Full training input exceeds frozen limits; do not silently truncate')
    write_json(OUT/'input-verification.json',{'status':'passed','rows':100,'both_response_streams_checked':True})
    with gpu_lock():
        deadline=time.monotonic()+45
        while torch.cuda.mem_get_info()[0]<23*1024**3 and time.monotonic()<deadline:time.sleep(1)
        if torch.cuda.mem_get_info()[0]<23*1024**3:raise RuntimeError('GPU not idle for isolated memory measurement')
        torch.cuda.set_per_process_memory_fraction(.98)
        torch.cuda.reset_peak_memory_stats();torch.manual_seed(SEED)
        stop=threading.Event();monitor=threading.Thread(target=memory_worker,args=(stop,OUT/'nvidia-smi.jsonl','p14-full-7b'),daemon=True);monitor.start()
        try:
            backend=QwenV2Backend('qwen7b');base=backend.model;base.config.use_cache=False
            model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0,target_modules=r'(?:model\.layers|model\.language_model\.layers)\.\d+\.(?:self_attn|mlp)\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)',bias='none',task_type='CAUSAL_LM'))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
            names=[n for n,p in model.named_parameters() if p.requires_grad]
            if not names or any('visual' in n or 'lora_' not in n for n in names):raise ValueError('Unexpected trainable base or vision weights')
            if not any('.mlp.' in n for n in names) or not any('.self_attn.' in n for n in names):raise ValueError('Missing attention/MLP adapters')
            params=[p for p in model.parameters() if p.requires_grad]
            write_json(OUT/'trainable-parameters.json',{'count':sum(p.numel() for p in params),'names':names,'all_base_and_vision_weights_frozen':True})
            optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0)

            def cuda(inputs):
                return {k:v.to('cuda',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k,v in inputs.items()}

            def update(gt,teacher,length):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,targets=response_forward(model,gt,length,base_model=base)
                    ce_gt=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1))
                    logits_t,targets_t=response_forward(model,teacher,length,base_model=base)
                    ce_teacher=F.cross_entropy(logits_t.float().reshape(-1,logits_t.shape[-1]),targets_t.reshape(-1))
                    loss=.5*ce_gt+.5*ce_teacher
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite dual CE')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradients')
                optimizer.step();torch.cuda.synchronize()
                return {'loss':float(loss.detach()),'ce_gt':float(ce_gt.detach()),'ce_teacher':float(ce_teacher.detach()),'grad_norm':float(norm),
                        'gt_sequence_tokens':int(gt['input_ids'].shape[1]),'teacher_sequence_tokens':int(teacher['input_ids'].shape[1]),
                        'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}

            started=time.monotonic()
            for step,(item,gt_text,teacher_text,length) in enumerate(prepared,1):
                gt=cuda(make_inputs(processor,item,gt_text));teacher=cuda(make_inputs(processor,item,teacher_text))
                entry={'step':step,'item_id':item['item_id'],'axis':item['axis'],'actual_optimizer_update':True,**update(gt,teacher,length)}
                del gt,teacher
                with (OUT/'loss.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n');stream.flush()
                state.update(status='training_preflight',actual_updates=step);write_json(OUT/'status.json',state)
                if step%10==0:print('Full P1-4 memory preflight',step,'/100',entry['loss'],flush=True)
            real_peak=torch.cuda.max_memory_reserved()
            adapter={k:v.detach().cpu() for k,v in get_peft_model_state_dict(model).items()}
            nonzero={k:int(torch.count_nonzero(v)) for k,v in adapter.items() if 'lora_B' in k}
            if not nonzero or not all(nonzero.values()):raise ValueError('Some LoRA B modules never updated')
            torch.save(adapter,OUT/'adapter-after-100.pt')
            # Shape stress only: retain image tokens, fill text to the registered upper bound.
            worst=max(prepared,key=lambda x:int(make_inputs(processor,x[0])['image_grid_thw'].prod().item()))[0]
            stress=make_inputs(processor,worst)
            prompt_ids=stress['input_ids'];filler=int(processor.tokenizer.encode('yes',add_special_tokens=False)[0])
            padding=2048-prompt_ids.shape[1]
            if padding<128:raise ValueError('No room for registered stress response')
            stress['input_ids']=torch.cat([prompt_ids,torch.full((1,padding),filler,dtype=prompt_ids.dtype)],dim=1)
            stress['attention_mask']=torch.ones_like(stress['input_ids'])
            inputs=cuda(stress);stress_result=update(inputs,inputs,1920)
            write_json(OUT/'shape-stress.json',{'synthetic_engineering_only':True,'optimizer_updates':1,'response_tokens':128,**stress_result})
            peak=torch.cuda.max_memory_reserved();allocated=torch.cuda.max_memory_allocated()
            decision='freeze_configuration' if peak<=20000000000 else 'offer_memory_reduction_options' if peak<=22000000000 else 'exceeds_requested_4090_budget'
            marker={'status':'complete','engineering_only':True,'formal_training':False,'actual_real_updates':100,'synthetic_stress_updates':1,
                    'loss_kind':'.5 GT-CE + .5 122B-response-CE','peak_reserved_bytes':peak,'peak_reserved_GB':peak/1e9,
                    'peak_reserved_GiB':peak/1024**3,'peak_allocated_bytes':allocated,'real_updates_peak_reserved_bytes':real_peak,
                    'memory_decision':decision,'formal_training_allowed':False,'formal_blocker':'Stage C and exact-checkpoint API bridge not passed; existing experiment code unchanged',
                    'loss_ledger_sha256':sha(OUT/'loss.jsonl'),'teacher_ledger_sha256':sha(OUT/'teacher.jsonl'),
                    'adapter_sha256':sha(OUT/'adapter-after-100.pt'),'all_attention_and_mlp_lora_B_updated':True,
                    'elapsed_training_seconds':time.monotonic()-started,'completed_utc':datetime.now(timezone.utc).isoformat()}
            if decision=='freeze_configuration':write_json(OUT/'frozen-7b-config.json',read_json(OUT/'protocol.json'))
            write_json(OUT/'complete.json',marker);state.update(marker);print(json.dumps(marker),flush=True)
        finally:
            stop.set();monitor.join(timeout=6)


def worker():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'preparing','worker_script':Path(__file__).name,'pid':os.getpid(),'formal_training':False}
        try:
            if (OUT/'loss.jsonl').exists() or (OUT/'complete.json').exists():raise ValueError('Never restart a partially trained memory preflight in place')
            write_json(OUT/'status.json',state)
            processor,rows,config=prepare();cached=cache_teacher(rows,config,state);training(processor,rows,cached,state)
        except Exception as error:
            state.update(status='failed',error_type=type(error).__name__)
            try:
                import torch
                state.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
            except Exception:pass
            raise
        finally:
            write_json(OUT/'status.json',state)
            if state.get('status')=='complete':
                subprocess.run([sys.executable,str(ROOT/'tools/verify_p14_full_preflight.py')],check=True,cwd=ROOT)
            if os.environ.get('GH_TOKEN'):
                from publish_artifacts import publish
                publish(OUT,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker()
    else:
        token=sys.stdin.readline().strip()
        if not token or not os.environ.get('DASHSCOPE_API_KEY'):raise SystemExit('API and backup credentials required')
        if '.arp-p14-full-memory' in subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout:raise SystemExit('Preflight already running')
        OUT.mkdir(parents=True,exist_ok=True)
        command='cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/preflight_p14_full_7b.py --worker >> '+str(OUT/'console.log')+' 2>&1'
        subprocess.run(['screen','-dmS','arp-p14-full-memory','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Started independent full P1-4 7B preflight; existing experiments remain stopped')
