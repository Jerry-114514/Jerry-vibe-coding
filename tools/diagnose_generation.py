"""Locate the first nonfinite layer on the next unfinished formal input."""
import argparse
import torch
from experiment_common import ROOT, read_rows, write_json
from model_backends import Backend

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', default='qwen3b')
p.add_argument('--run', default='runs/formal-20261014')
p.add_argument('--vision-fp32', action='store_true')
a = p.parse_args()
ledger = ROOT/a.run/'ledger'/f'{a.checkpoint}-000.jsonl'
done = {r['item_id'] for r in read_rows(ledger)}
item = next(r for r in read_rows(ROOT/'data/manifest.jsonl') if r['item_id'] not in done)
backend = Backend(a.checkpoint)
if a.vision_fp32:
    backend.model.visual.float()
bad = []
handles = []
def hook(name):
    def inspect(module, args, output):
        values = output if isinstance(output, tuple) else (output,)
        for value in values:
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all() and len(bad)<16:
                bad.append({'module':name,'dtype':str(value.dtype),'shape':list(value.shape)})
    return inspect
for name,module in backend.model.named_modules():
    if not list(module.children()):
        handles.append(module.register_forward_hook(hook(name)))
result={'checkpoint':a.checkpoint,'item_id':item['item_id'],'image_path':item['image_path']}
try:
    result['response']=backend.generate(item)
except Exception as error:
    result['error']=repr(error)
finally:
    result['first_nonfinite_modules']=bad
    for handle in handles:
        handle.remove()
    write_json(ROOT/'runs/preflight'/f'{a.checkpoint}-formal-numerics.json',result)
    print(result,flush=True)
