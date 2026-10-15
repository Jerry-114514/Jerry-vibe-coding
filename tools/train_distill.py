"""D2/D3: separate teacher cache, top-50+tail KL, step-matched LoRA arms."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import time

from experiment_common import ROOT, read_json, read_rows, sha, write_json, write_rows
from model_backends import Backend, BACKEND_VERSION


def cache_name(item_id):
    return hashlib.sha256(item_id.encode()).hexdigest()+'.pt'


def response_forward(model, inputs, prompt_length, base_model=None):
    """Only materialize logits needed for assistant targets, not vision tokens."""
    length = inputs['input_ids'].shape[1]
    count = length-prompt_length
    if count <= 0:
        raise ValueError('No response tokens')
    signature = inspect.signature((base_model or model).forward)
    kwargs = {}
    if 'logits_to_keep' in signature.parameters:
        kwargs['logits_to_keep'] = count+1
    elif 'num_logits_to_keep' in signature.parameters:
        kwargs['num_logits_to_keep'] = count+1
    outputs = model(**inputs, use_cache=False, **kwargs)
    logits = outputs.logits[:, -(count+1):-1]
    targets = inputs['input_ids'][:, prompt_length:]
    if logits.shape[1] != targets.shape[1]:
        raise ValueError('Causal response alignment failed')
    return logits, targets


def cached_teacher(manifest, destination, limit=None):
    import torch
    rows = read_rows(manifest)
    if limit:
        rows = rows[:limit]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    protocol = read_json(ROOT/'configs/experiment.json')
    lock = {'manifest_sha256': sha(manifest), 'top_k': protocol['training']['top_k'],
            'inference_backend_version': BACKEND_VERSION,
            'temperature': protocol['training']['kd_temperature'], 'teacher': 'qwen7b',
            'model_revision': read_json(ROOT/'configs/models.lock.json')['models']['qwen7b']['revision'],
            'limit': limit}
    meta = destination/'cache.meta.json'
    if meta.exists() and read_json(meta) != lock:
        raise ValueError('Teacher cache protocol mismatch')
    write_json(meta, lock)
    teacher = Backend('qwen7b')
    write_json(destination/'tokenizer.json', {'vocab': teacher.tokenizer.get_vocab(),
                                            'special_tokens_map': teacher.tokenizer.special_tokens_map})
    for index, item in enumerate(rows):
        path = destination/cache_name(item['item_id'])
        if path.exists():
            continue
        response = teacher.generate(item)
        inputs = teacher.inputs(item, response)
        prompt = teacher.inputs(item)['input_ids']
        p = prompt.shape[1]
        if not torch.equal(inputs['input_ids'][:, :p], prompt):
            raise ValueError('Teacher prompt/response tokenization boundary changed')
        with torch.inference_mode():
            logits, targets = response_forward(teacher.model, inputs, p)
            logits = logits[0].float()/protocol['training']['kd_temperature']
            top_values, top_indices = logits.topk(protocol['training']['top_k'], dim=-1)
            log_norm = torch.logsumexp(logits, dim=-1)
        content = {'item_id': item['item_id'], 'response': response, 'prompt_length': p,
                   'input_ids': inputs['input_ids'].cpu(), 'targets': targets.cpu(),
                   'top_indices': top_indices.cpu().to(torch.int32), 'top_logits': top_values.cpu(),
                   'log_normalizer': log_norm.cpu(), 'image_sha256': sha(ROOT/item['image_path'])}
        temp = path.with_suffix('.tmp')
        torch.save(content, temp)
        temp.replace(path)
        if index % 100 == 0 or limit:
            print(f'teacher {index+1}/{len(rows)} {item["item_id"]}', flush=True)
    write_json(destination/'complete.json', {'items':len(rows), 'manifest_sha256':sha(manifest),
                                            'files':{cache_name(x['item_id']):sha(destination/cache_name(x['item_id'])) for x in rows}})


def sparse_kl(student_logits, cache, temperature=1.0):
    """Top-50 probability masses plus the omitted vocabulary as one tail bucket."""
    import torch
    log_student = torch.log_softmax(student_logits[0].float()/temperature, dim=-1)
    indices = cache['top_indices'].to(log_student.device, torch.int64)
    log_teacher = (cache['top_logits']-cache['log_normalizer'][:, None]).to(log_student.device)
    p_teacher = log_teacher.exp()
    log_q_top = log_student.gather(-1, indices)
    p_tail = (1-p_teacher.sum(dim=-1)).clamp_min(1e-7)
    q_tail = (1-log_q_top.exp().sum(dim=-1)).clamp_min(1e-7)
    return ((p_teacher*(log_teacher-log_q_top)).sum(dim=-1)+p_tail*(p_tail.log()-q_tail.log())).mean()*temperature**2


def train_arm(arm, manifest, cache_path, output, steps=None, smoke=False, run=None):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not smoke:
        from analyze_rulers import verify_go
        verify_go(run or ROOT/'runs/formal')
    elif not output.resolve().is_relative_to((ROOT/'runs/preflight').resolve()):
        raise ValueError('Smoke adapters must remain under runs/preflight')
    params = read_json(ROOT/'configs/experiment.json')['training']
    steps = steps or params['steps']
    if not smoke and steps != params['steps']:
        raise ValueError('Formal arms must have the preregistered step budget')
    torch.manual_seed(params['seed'])
    random.seed(params['seed'])
    lock = {'arm':arm, 'manifest_sha256':sha(manifest), 'cache_meta_sha256':sha(Path(cache_path)/'cache.meta.json'),
            'protocol':params, 'steps':steps, 'smoke':smoke,
            'student_model_revision':read_json(ROOT/'configs/models.lock.json')['models']['qwen3b']['revision']}
    if (output/'train.meta.json').exists() and read_json(output/'train.meta.json') != lock:
        raise ValueError('Training resume protocol mismatch')
    write_json(output/'train.meta.json', lock)
    if (output/'complete.json').exists():
        return
    if arm == 'N':
        write_json(output/'complete.json', {'budget_steps':steps, 'effective_updates':0, 'no_op':True, 'smoke':smoke})
        return
    rows = read_rows(manifest)
    if not rows:
        raise ValueError('Empty training arm')
    backend = Backend('qwen3b')
    vocab = read_json(Path(cache_path)/'tokenizer.json')['vocab']
    if backend.tokenizer.get_vocab() != vocab:
        raise ValueError('Teacher/student tokenizer vocabularies differ; logits are not aligned')
    base_model = backend.model
    if any(str(device) not in ('0','cuda:0','cuda') for device in getattr(base_model,'hf_device_map',{}).values()):
        raise RuntimeError('Training requires the 3B base entirely on GPU')
    base_model.config.use_cache = False
    model = get_peft_model(base_model, LoraConfig(r=params['r'], lora_alpha=params['alpha'], lora_dropout=0.0,
                           target_modules=['q_proj','k_proj','v_proj','o_proj'], bias='none', task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=params['lr'], weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda')
    checkpoint = output/'resume.pt'
    start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, saved['adapter'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        start = saved['step']
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    order = list(range(len(rows)))
    random.Random(params['seed']).shuffle(order)
    before = [p.detach().clone() for p in trainable] if smoke else None
    log_path = output/'loss.jsonl'
    if start and log_path.exists():
        write_rows(log_path, [x for x in read_rows(log_path) if x['step'] <= start])
    elif not start and log_path.exists():
        log_path.write_text('', encoding='utf-8')
    with log_path.open('a', encoding='utf-8') as log:
        for step in range(start, steps):
            item = rows[order[step % len(order)]]
            cache = torch.load(Path(cache_path)/cache_name(item['item_id']), map_location='cpu', weights_only=True)
            if sha(ROOT/item['image_path']) != cache['image_sha256']:
                raise ValueError('Training image changed after teacher caching')
            inputs = backend.inputs(item, cache['response'])
            if not torch.equal(inputs['input_ids'].cpu(), cache['input_ids']):
                raise ValueError('Teacher/student inputs differ')
            # AMP's initial scale can overflow even with finite forward values.
            # Retry the same logical step at a lower scale; do not count skipped
            # updates against the fixed 1000 actual-optimizer-update budget.
            for amp_retry in range(20):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits, targets = response_forward(model, inputs, cache['prompt_length'], base_model=base_model)
                    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
                    kd = sparse_kl(logits, cache, params['kd_temperature'])
                    loss = (1-params['kd_weight'])*ce+params['kd_weight']*kd
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if torch.isfinite(norm):
                    scaler.step(optimizer)
                    scaler.update()
                    break
                new_scale=scaler.get_scale()/2
                scaler.update(new_scale=new_scale)
                print(f'{arm}: AMP overflow, retry step {step+1} with scale {new_scale}',flush=True)
            else:
                raise FloatingPointError('Gradients remain non-finite after 20 AMP scale reductions')
            entry = {'step':step+1, 'loss':float(loss.detach()), 'ce':float(ce.detach()),
                     'kd':float(kd.detach()), 'grad_norm':float(norm), 'item_id':item['item_id'],
                     'amp_retries':amp_retry,'amp_scale':scaler.get_scale()}
            log.write(json.dumps(entry)+'\n')
            log.flush()
            if smoke or (step+1)%100 == 0:
                print(arm, entry, flush=True)
            if (step+1)%100 == 0 or step+1 == steps:
                from peft import get_peft_model_state_dict
                saved = {'adapter':get_peft_model_state_dict(model), 'optimizer':optimizer.state_dict(),
                         'scaler':scaler.state_dict(), 'step':step+1,
                         'rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all()}
                temp = checkpoint.with_suffix('.tmp')
                torch.save(saved,temp)
                temp.replace(checkpoint)
    changed = any(not torch.equal(a,b) for a,b in zip(before,trainable)) if smoke else None
    if smoke and not changed:
        raise AssertionError('LoRA optimizer did not change any trainable parameter')
    model.save_pretrained(output/'adapter', safe_serialization=True)
    backend.processor.save_pretrained(output/'adapter')
    write_json(output/'complete.json', {'steps':steps, 'effective_updates':steps, 'smoke':smoke,
                                        'parameters_changed':changed, 'trainable_parameters':sum(p.numel() for p in trainable),
                                        'peak_vram_bytes':torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['cache','train'])
    p.add_argument('--manifest',required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--output')
    p.add_argument('--arm',choices=['A1','A2','U','N'],default='A1')
    p.add_argument('--steps',type=int)
    p.add_argument('--limit',type=int)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--run')
    a=p.parse_args()
    from gpu_lock import gpu_lock
    with gpu_lock():
        if a.action=='cache':
            cached_teacher(a.manifest,a.cache,a.limit)
        else:
            train_arm(a.arm,a.manifest,a.cache,a.output,a.steps,a.smoke,a.run)
