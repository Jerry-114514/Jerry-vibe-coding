"""Independently verify real updates, finite weights and the saved optimizer state."""
import math
import argparse
import torch
from safetensors.torch import load_file
from experiment_common import ROOT, read_rows, sha, write_json

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--student',choices=['qwen3b','qwen7b'],default='qwen3b')
    args=parser.parse_args()
    path=ROOT/('runs/protocol-v2/calibration/training-memory-'+('3b' if args.student=='qwen3b' else '7b'))
    log=read_rows(path/'loss.jsonl')
    saved=torch.load(path/'resume.pt',map_location='cpu',weights_only=True)
    adapter=load_file(str(path/'adapter/adapter_model.safetensors'))
    checks={
        '100_sequential_actual_updates': [r['step'] for r in log]==list(range(1,101)) and all(r['actual_optimizer_update'] for r in log),
        'finite_losses_and_gradients': all(math.isfinite(r[k]) for r in log for k in ['loss','ce','kd','grad_norm']),
        'resume_step_100': saved['step']==100,
        'adapter_equals_resume': set(adapter)==set(saved['adapter']) and all(torch.equal(v,saved['adapter'][k]) for k,v in adapter.items()),
        'all_weights_finite': all(torch.isfinite(v).all().item() for v in adapter.values()),
        'nonzero_lora_B': all(torch.count_nonzero(v).item()>0 for k,v in adapter.items() if 'lora_B' in k),
    }
    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,
            'axis_updates':{a:sum(r['axis']==a for r in log) for a in sorted({r['axis'] for r in log})},
            'loss_sha256':sha(path/'loss.jsonl'),'formal_training':False}
    write_json(path/'verification.json',result)
    print(result,flush=True)
    if not all(checks.values()):
        raise RuntimeError('Independent pilot verification failed')
