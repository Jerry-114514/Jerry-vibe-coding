"""100 real BF16 updates on a small student, never a formal experimental arm."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

from experiment_common import AXES, ROOT, read_json, read_rows, sha, write_json, write_rows

OUT = ROOT/'runs/protocol-v2/calibration/training-memory-3b'
SEED = 2026101601
STEPS = 100

def select_rows():
    grouped = defaultdict(list)
    for row in read_rows(ROOT/'data/training/pool.jsonl'):
        grouped[(row['axis'],row['bootstrap_unit'])].append(row)
    selected = []
    for axis in AXES:
        groups = sorted([v for (a,_),v in grouped.items() if a==axis],
                        key=lambda rows: hashlib.sha256(('v2-memory-pilot:'+rows[0]['bootstrap_unit']).encode()).hexdigest())
        for group in groups[:25]:
            selected.append(sorted(group,key=lambda r:r['item_id'])[0])
    if len(selected)!=100 or len({(r['axis'],r['bootstrap_unit']) for r in selected})!=100:
        raise ValueError('Need 25 distinct training units per axis')
    random.Random(SEED).shuffle(selected)
    return selected

def pilot(student='qwen3b',loss_kind='ce_kd'):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from gpu_lock import gpu_lock
    from model_backends import Backend
    from train_distill import cache_name, response_forward, sparse_kl
    from precision_bridge import memory_worker
    import threading
    OUT.mkdir(parents=True,exist_ok=True)
    rows=select_rows()
    teacher=ROOT/'runs/formal-20261014/teacher-cache'
    teacher_meta=read_json(teacher/'cache.meta.json')
    if teacher_meta['manifest_sha256']!=sha(ROOT/'data/training/pool.jsonl'):
        raise ValueError('Frozen teacher pool mismatch')
    completed=read_json(teacher/'complete.json')
    for row in rows:
        if sha(teacher/cache_name(row['item_id']))!=completed['files'][cache_name(row['item_id'])]:
            raise ValueError('Teacher cache checksum mismatch')
    write_rows(OUT/'manifest.jsonl',rows)
    memory_cap_gib=12 if student=='qwen3b' else 23
    protocol={'scope':'Engineering pilot only; no formal C/D gate opened',
              'steps':STEPS,'seed':SEED,'student':student,'student_precision':'bf16',
              'teacher_precision':teacher_meta['inference_backend_version'],
              'teacher_cache_sha256':sha(teacher/'cache.meta.json'),
              'manifest_sha256':sha(OUT/'manifest.jsonl'),
              'max_torch_reserved_gib':memory_cap_gib,'confirmation':False,
              'lora':{'r':16,'alpha':32,'lr':1e-4},
              'loss':'.5 CE + .5 top50-tail KL' if loss_kind=='ce_kd' else 'Teacher-response CE; no logits used'}
    if (OUT/'protocol.json').exists() and read_json(OUT/'protocol.json')!=protocol:
        raise ValueError('Pilot resume protocol changed')
    write_json(OUT/'protocol.json',protocol)
    if (OUT/'complete.json').exists():
        print('Training-memory pilot already complete',flush=True)
        return
    write_json(OUT/'status.json',{'status':'queued','waiting_for':'shared GPU lock'})
    with gpu_lock():
        total=torch.cuda.get_device_properties(0).total_memory
        free,_=torch.cuda.mem_get_info()
        deadline=time.monotonic()+45
        # The previous Python process may release flock shortly before its CUDA
        # context is destroyed. Wait for that handoff instead of declaring failure.
        while free<21*1024**3 and time.monotonic()<deadline:
            time.sleep(1)
            free,_=torch.cuda.mem_get_info()
        if free<21*1024**3:
            raise RuntimeError('Training pilot requires the GPU to be idle')
        torch.cuda.set_per_process_memory_fraction(memory_cap_gib*1024**3/total)
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(SEED)
        backend=Backend(student,precision='bf16')
        if backend.tokenizer.get_vocab()!=read_json(teacher/'tokenizer.json')['vocab']:
            raise ValueError('Teacher/student vocabulary ID alignment failed')
        base=backend.model
        base.config.use_cache=False
        model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0,
                   target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        model.train()
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0)
        checkpoint=OUT/'resume.pt'
        start=0
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            set_peft_model_state_dict(model,saved['adapter'])
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
            start=saved['step']
        history=read_rows(OUT/'loss.jsonl') if (OUT/'loss.jsonl').exists() else []
        write_rows(OUT/'loss.jsonl',[r for r in history if r['step']<=start])
        stop=threading.Event()
        worker=threading.Thread(target=memory_worker,args=(stop,OUT/'nvidia-smi.jsonl',student+'-training'),daemon=True)
        worker.start()
        tick=time.monotonic()
        try:
            for step in range(start,STEPS):
                row=rows[step%len(rows)]
                cache=torch.load(teacher/cache_name(row['item_id']),map_location='cpu',weights_only=True)
                if sha(ROOT/row['image_path'])!=cache['image_sha256']:
                    raise ValueError('Training image checksum mismatch')
                inputs=backend.inputs(row,cache['response'])
                if not torch.equal(inputs['input_ids'].cpu(),cache['input_ids']):
                    raise ValueError('Teacher/student input IDs or assistant boundary differ')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,targets=response_forward(model,inputs,cache['prompt_length'],base_model=base)
                    ce=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1))
                    kd=sparse_kl(logits,cache) if loss_kind=='ce_kd' else logits.new_zeros((),dtype=torch.float32)
                    loss=.5*ce+.5*kd if loss_kind=='ce_kd' else ce
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite BF16 loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Non-finite BF16 gradients')
                optimizer.step()
                entry={'step':step+1,'item_id':row['item_id'],'axis':row['axis'],
                       'loss':float(loss.detach()),'ce':float(ce.detach()),'kd':float(kd.detach()),
                       'grad_norm':float(norm),'actual_optimizer_update':True,
                       'supervised_tokens':int(cache['targets'].numel())}
                with (OUT/'loss.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(entry,allow_nan=False)+'\n')
                if (step+1)%10==0:
                    print('BF16 training memory pilot:',step+1,'/',STEPS,entry['loss'],flush=True)
                    write_json(OUT/'status.json',{'status':'running','actual_updates':step+1})
                if (step+1)%25==0:
                    temporary=checkpoint.with_suffix('.tmp')
                    torch.save({'adapter':get_peft_model_state_dict(model),'optimizer':optimizer.state_dict(),
                                'step':step+1,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temporary)
                    temporary.replace(checkpoint)
            state=get_peft_model_state_dict(model)
            nonzero_b={k:int(torch.count_nonzero(v)) for k,v in state.items() if 'lora_B' in k}
            if not nonzero_b or not all(nonzero_b.values()):
                raise RuntimeError('Pilot did not update the zero-initialized LoRA B matrices')
            model.save_pretrained(OUT/'adapter',safe_serialization=True)
            evidence={'status':'complete','engineering_pilot_only':True,'formal_training':False,
                      'student':student,'loss_kind':loss_kind,
                      'actual_updates':STEPS,'seed':SEED,'nonzero_lora_B':nonzero_b,
                      'elapsed_seconds_this_invocation':time.monotonic()-tick,
                      'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                      'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                      'completed_utc':datetime.now(timezone.utc).isoformat()}
            write_json(OUT/'complete.json',evidence)
            write_json(OUT/'status.json',{'status':'complete','actual_updates':STEPS,'formal_training':False})
            print(evidence,flush=True)
        except Exception as exc:
            write_json(OUT/'status.json',{'status':'failed','error':type(exc).__name__,'message':str(exc)})
            raise
        finally:
            stop.set()
            worker.join(timeout=6)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student',choices=['qwen3b','qwen7b'],default='qwen3b')
    parser.add_argument('--loss',choices=['ce_kd','ce'],default='ce_kd')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--auto-publish',action='store_true')
    args=parser.parse_args()
    OUT=ROOT/('runs/protocol-v2/calibration/training-memory-'+('3b' if args.student=='qwen3b' else '7b'))
    if args.detach:
        session='arp-v2-train-'+args.student
        sessions=subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout
        if '.'+session in sessions:
            raise RuntimeError('Training pilot is already running')
        token=sys.stdin.readline().strip() if args.auto_publish else None
        if args.auto_publish and not token:
            raise RuntimeError('Backup credential required on stdin')
        command=[sys.executable,'-u',str(ROOT/'tools/training_memory_pilot.py'),
                 '--student',args.student,'--loss',args.loss]
        if args.auto_publish:
            command.append('--auto-publish')
        environment={**os.environ,**({'GH_TOKEN':token} if token else {})}
        logfile=ROOT/'runs/protocol-v2'/('train-'+args.student+'-console.log')
        shell='source /etc/network_turbo >/dev/null 2>&1; cd '+shlex.quote(str(ROOT))+'; exec '+shlex.join(command)+' > '+shlex.quote(str(logfile))+' 2>&1'
        subprocess.run(['screen','-dmS',session,'bash','-lc',shell],env=environment,check=True)
        print('Queued',session,'with shared GPU lock and memory cap',flush=True)
        sys.exit(0)
    try:
        pilot(args.student,args.loss)
        subprocess.run([sys.executable,str(ROOT/'tools/verify_training_pilot.py'),'--student',args.student],check=True,cwd=ROOT)
        if args.auto_publish:
            from run_cached_pilots import publish_completed
            publish_completed()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        previous=read_json(OUT/'status.json') if (OUT/'status.json').exists() else {}
        write_json(OUT/'status.json',{**previous,'status':'failed','error':type(exc).__name__,
                                     'message':str(exc),'stopped_utc':datetime.now(timezone.utc).isoformat()})
        raise
