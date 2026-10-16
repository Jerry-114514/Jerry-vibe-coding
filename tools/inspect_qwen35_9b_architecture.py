"""CPU/meta-only LoRA target inventory; never occupy the live7B GPU."""
from collections import Counter
import json
from accelerate import init_empty_weights
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
from torch import nn
from experiment_common import ROOT, write_json

out = ROOT/'runs/qwen35-9b-infrastructure-20261016'
config = AutoConfig.from_pretrained(str(ROOT/'.model_cache/qwen35_9b'),local_files_only=True)
with init_empty_weights():
    model = Qwen3_5ForConditionalGeneration(config)
selected = []
for name,module in model.named_modules():
    if isinstance(module,nn.Linear) and '.layers.' in name and 'visual' not in name and (
            '.self_attn.' in name or '.linear_attn.' in name or '.mlp.' in name):
        selected.append({'name':name,'in':module.in_features,'out':module.out_features,
                         'r16_trainable_parameters':16*(module.in_features+module.out_features)})
assert any('.linear_attn.' in r['name'] for r in selected)
assert not any('visual' in r['name'] for r in selected)
parameters = sum(r['r16_trainable_parameters'] for r in selected)
result = {'status':'passed_cpu_inventory','architecture':config.architectures,
          'layer_type_counts':dict(Counter(config.text_config.layer_types)),
          'lora_target_inventory':selected,'r16_parameters_if_all_listed_targets_used':parameters,
          'fp32_lora_and_two_adam_moments_bytes':parameters*12,
          'training_preflight_complete':False,'warning':'7B q/k/v/o-only attention list omits9B linear-attention projections; freeze9B scope separately'}
write_json(out/'architecture-inventory.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='lora_target_inventory'}),flush=True)
