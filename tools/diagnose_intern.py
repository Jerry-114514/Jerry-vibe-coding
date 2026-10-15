"""Diagnose numerical health without changing any inference ledger."""
import argparse
import torch
from experiment_common import ROOT,read_rows,write_json
from model_backends import Backend

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default='intern2b')
p.add_argument('--sdpa',action='store_true')
a=p.parse_args()
backend=Backend(a.checkpoint)
if a.sdpa:
    backend.model.language_model.config._attn_implementation='sdpa'
item=read_rows(ROOT/'data/manifest.jsonl')[0]
pixels=backend.intern_pixels(item)
bad=[]
hooks=[]
def hook(name):
    def inspect(module,args,output):
        value=output[0] if isinstance(output,tuple) else output
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all() and len(bad)<12:
            bad.append({'module':name,'dtype':str(value.dtype),'shape':list(value.shape)})
    return inspect
for name,module in backend.model.named_modules():
    if not list(module.children()):
        hooks.append(module.register_forward_hook(hook(name)))
with torch.inference_mode():
    features=backend.model.extract_feature(pixels)
    result={'checkpoint':a.checkpoint,'pixels_finite':bool(torch.isfinite(pixels).all()),
            'features_finite':bool(torch.isfinite(features).all()),'first_nonfinite_modules':bad}
    print(result,flush=True)
    if result['features_finite']:
        response=backend.model.chat(backend.tokenizer,pixels,'<image>\n'+item['question'],
                                    {'max_new_tokens':4,'do_sample':False,'num_beams':1})
        result.update(response=response,first_nonfinite_modules=bad)
        print(result,flush=True)
for handle in hooks:
    handle.remove()
write_json(ROOT/'runs/preflight'/f'{a.checkpoint}-numerics.json',result)
