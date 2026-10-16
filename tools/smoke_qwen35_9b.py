"""Small9B BF16 vision/CUDA smoke, after7B minimal acceptance and underGPU lock."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import time
from PIL import Image
import torch
import transformers
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from experiment_common import ROOT, read_rows, read_json, write_json
from gpu_lock import gpu_lock

out = ROOT/'runs/qwen35-9b-infrastructure-20261016'
path = ROOT/'.model_cache/qwen35_9b'
assert read_json(ROOT/'runs/route-b-staged-7b-20261016/minimal-acceptance.json')['status'] == 'passed'
with gpu_lock():
    started = time.monotonic()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(path),local_files_only=True,
                dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
    processor = AutoProcessor.from_pretrained(str(path),local_files_only=True)
    row = next(r for r in read_rows(ROOT/'data/training/pool.jsonl') if r['axis'] == 'pope')
    with Image.open(ROOT/row['image_path']) as source:
        image = source.convert('RGB')
    messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}]
    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    batch = processor(text=[text],images=[image],return_tensors='pt',images_kwargs={
                        'min_pixels':256*32*32,'max_pixels':1280*32*32})
    assert batch['input_ids'].shape[1] <= 2048
    batch = {k:v.to('cuda',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k,v in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        logits = model(**batch,use_cache=False,logits_to_keep=1).logits
        assert torch.isfinite(logits).all()
        generated = model.generate(**batch,max_new_tokens=16,do_sample=False)
    torch.cuda.synchronize()
    response = processor.decode(generated[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
    result = {'status':'passed','repo':'Qwen/Qwen3.5-9B','revision':read_json(out/'model.lock.json')['revision'],
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'transformers':transformers.__version__,
              'input_tokens':int(batch['input_ids'].shape[1]),'response':response,'finite_vision_logits':True,
              'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'elapsed_seconds':time.monotonic()-started,'training_memory_preflight_complete':False}
    assert result['peak_reserved_bytes'] < 30_000_000_000
    write_json(out/'gpu-smoke.json',result)
    print(result,flush=True)
