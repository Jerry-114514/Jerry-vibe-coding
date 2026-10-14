"""Official HF backends, immutable revisions, fp16 and greedy decoding."""
import os
from pathlib import Path
import time

from experiment_common import MODELS, ROOT, read_json, write_json


def model_path(key):
    from huggingface_hub import snapshot_download
    lock = read_json(ROOT / 'configs/models.lock.json')
    spec = lock['models'][key]
    system_keys = {'intern2b', 'intern8b', 'llava05b'}
    cache_root = Path('/root/arp-model-cache') if key in system_keys else ROOT / '.model_cache'
    destination = cache_root / key
    destination.mkdir(parents=True, exist_ok=True)
    import shutil
    # Refuse unbounded downloads: preserve at least 5 GiB for run artifacts.
    existing = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file() and p.suffix == '.safetensors')
    needed = max(0, spec['weight_bytes']-existing)
    if shutil.disk_usage(destination).free < needed+5*1024**3:
        raise RuntimeError('Insufficient space for '+key+' plus 5 GiB artifact reserve')
    from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata
    from download_ranges import download_ranges
    info = HfApi(endpoint='https://hf-mirror.com').model_info(spec['repo_id'], revision=spec['revision'], files_metadata=True)
    import requests, hashlib
    for file in info.siblings:
        if file.rfilename.endswith(('.json','.txt','.model','.tiktoken','.py','.jinja')):
            target = destination/file.rfilename
            if target.exists():
                continue
            session = requests.Session()
            session.trust_env = False
            source = f'https://hf-mirror.com/{spec["repo_id"]}/resolve/{spec["revision"]}/{file.rfilename}'
            for attempt in range(4):
                try:
                    response = session.get(source, timeout=(30,120))
                    response.raise_for_status()
                    content = response.content
                    blob = getattr(file,'blob_id',None)
                    if blob and len(blob)==40 and hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()!=blob:
                        raise ValueError('Pinned model file Git SHA mismatch: '+file.rfilename)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(content)
                    break
                except requests.RequestException:
                    if attempt==3:
                        raise
                    time.sleep(2**attempt)
    weights_meta = []
    for file in info.siblings:
        if file.rfilename.endswith('.safetensors'):
            digest = file.lfs.get('sha256') if isinstance(file.lfs,dict) else getattr(file.lfs,'sha256',None)
            url = hf_hub_url(spec['repo_id'],file.rfilename,revision=spec['revision'])
            size = file.size
            if size is None or digest is None:
                metadata = get_hf_file_metadata(url)
                size = metadata.size
                digest = metadata.etag if len(metadata.etag or '') == 64 else digest
            if size is None or digest is None:
                raise RuntimeError('Cannot verify model weight size and SHA256')
            ms_repo = ('AI-ModelScope/' + spec['repo_id'].split('/',1)[1]) if spec['backend']=='llava' else spec['repo_id']
            alternate = f'https://modelscope.cn/api/v1/models/{ms_repo}/repo?Revision=master&FilePath={file.rfilename}'
            try:
                download_ranges(alternate,destination/file.rfilename,size,digest,direct=True)
                source = 'modelscope; content SHA256 matched pinned HuggingFace weight'
            except RuntimeError:
                download_ranges(url,destination/file.rfilename,size,digest)
                source = 'huggingface pinned revision'
            weights_meta.append({'file':file.rfilename,'bytes':size,'sha256':digest,'source':source})
    if not weights_meta:
        raise RuntimeError('No safetensors weights found')
    write_json(destination/'weights.verified.json', {'revision':spec['revision'],'files':weights_meta})
    return str(destination)


def freeze_models():
    from huggingface_hub import HfApi
    api = HfApi()
    path = ROOT / 'configs/models.lock.json'
    if path.exists():
        return read_json(path)
    output = {'endpoint': api.endpoint, 'models': {}}
    for key, (repo_id, backend) in MODELS.items():
        info = api.model_info(repo_id, files_metadata=True)
        weights = [s for s in info.siblings if s.rfilename.endswith('.safetensors')]
        output['models'][key] = {'repo_id': repo_id, 'backend': backend, 'revision': info.sha,
                                'weight_bytes': sum(s.size or 0 for s in weights)}
    write_json(path, output)
    return output


class Backend:
    def __init__(self, key, adapter=None):
        import torch
        from transformers import (AutoModel, AutoProcessor, AutoTokenizer,
                                  Qwen2_5_VLForConditionalGeneration,
                                  LlavaOnevisionForConditionalGeneration)
        self.torch = torch
        self.key = key
        self.kind = MODELS[key][1]
        path = model_path(key)
        kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True,
                      device_map='auto', max_memory={0: '21GiB', 'cpu': '80GiB'},
                      offload_folder=str(ROOT / '.offload' / key))
        if self.kind == 'intern':
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True, use_flash_attn=False, **kwargs).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
            self.processor = None
        else:
            cls = Qwen2_5_VLForConditionalGeneration if self.kind == 'qwen' else LlavaOnevisionForConditionalGeneration
            self.model = cls.from_pretrained(path, attn_implementation='sdpa', **kwargs).eval()
            proc_args = dict(min_pixels=256*28*28, max_pixels=1280*28*28) if self.kind == 'qwen' else {}
            self.processor = AutoProcessor.from_pretrained(path, **proc_args)
            self.tokenizer = self.processor.tokenizer
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.device = self.model.get_input_embeddings().weight.device

    def inputs(self, item, response=None):
        from PIL import Image
        image = Image.open(ROOT / item['image_path']).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': item['question']}]}]
        if response is not None:
            messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': response}]})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=response is None)
        inputs = self.processor(text=[text], images=[image], return_tensors='pt')
        return {k: v.to(self.device, dtype=self.torch.float16) if v.is_floating_point() else v.to(self.device)
                for k, v in inputs.items()}

    def intern_pixels(self, item):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from PIL import Image
        image = Image.open(ROOT / item['image_path']).convert('RGB')
        width, height = image.size
        ratios = sorted({(i, j) for i in range(1, 13) for j in range(1, 13) if i*j <= 12}, key=lambda r: (r[0]*r[1], r))
        best, difference = (1, 1), float('inf')
        for ratio in ratios:
            d = abs(width/height - ratio[0]/ratio[1])
            if d < difference or (d == difference and width*height > .5*448**2*ratio[0]*ratio[1]):
                best, difference = ratio, d
        resized = image.resize((448*best[0], 448*best[1]))
        tiles = [resized.crop((x*448, y*448, (x+1)*448, (y+1)*448))
                 for y in range(best[1]) for x in range(best[0])]
        if len(tiles) > 1:
            tiles.append(image.resize((448, 448)))
        transform = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
                               T.Normalize((.485, .456, .406), (.229, .224, .225))])
        device = next(self.model.vision_model.parameters()).device
        return self.torch.stack([transform(tile) for tile in tiles]).to(device, self.torch.float16)

    def generate(self, item):
        with self.torch.inference_mode():
            if self.kind == 'intern':
                return self.model.chat(self.tokenizer, self.intern_pixels(item), '<image>\n' + item['question'],
                                       {'max_new_tokens': 128, 'do_sample': False, 'num_beams': 1})
            inputs = self.inputs(item)
            out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False, num_beams=1)
            return self.processor.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
