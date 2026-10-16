"""Cached all-BF16 Qwen backend with explicit ordered multi-image inputs."""
from PIL import Image
from experiment_common import ROOT, read_json, sha
from model_backends import finite_logits_processors

BACKEND_VERSION = 'qwen-bf16-labelled-images-explicit-size-total1280.v3'


def processor_inputs(processor, item):
    paths = item['image_paths']
    if not 1 <= len(paths) <= 7:
        raise ValueError('Registered input supports one to seven ordered images')
    images = []
    for relative in paths:
        path = (ROOT / relative).resolve()
        if not path.is_relative_to((ROOT / 'data').resolve()):
            raise ValueError('Image outside project data')
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    maximum = 1280 // len(images)
    # Leave room for the processor's patch-grid rounding when upscaling small
    # images; equal min/max pixel areas can exceed max after ceil rounding.
    minimum = min(256, max(4, maximum//2))
    messages = [{'role':'user','content':[{'type':'image'} for _ in images] +
                 [{'type':'text','text':item['question']}]}]
    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,add_vision_id=True)
    inputs = processor(text=[text],images=images,return_tensors='pt',
                       images_kwargs={'size':{'shortest_edge':minimum*28*28,'longest_edge':maximum*28*28},
                                      'min_pixels':minimum*28*28,'max_pixels':maximum*28*28})
    if len(inputs['image_grid_thw']) != len(images):
        raise ValueError('Processor silently changed the number of supplied images')
    visual_tokens = int(inputs['image_grid_thw'].prod(dim=1).sum().item() // 4)
    if visual_tokens > 1280:
        raise ValueError('Actual visual tokens exceed frozen aggregate budget')
    return inputs, {'image_count':len(images),'image_grid_thw':inputs['image_grid_thw'].tolist(),
                    'visual_tokens':visual_tokens,'input_tokens':int(inputs['input_ids'].shape[1])}


class QwenV2Backend:
    def __init__(self,key):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if key not in ('qwen3b','qwen7b'):
            raise ValueError('This registered backend is only for cached Qwen2.5-VL 3B/7B')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 GPU support required')
        path = ROOT / '.model_cache' / key
        locked = read_json(ROOT / 'configs/models.lock.json')['models'][key]
        verified = read_json(path / 'weights.verified.json')
        if verified['revision'] != locked['revision']:
            raise ValueError('Cached model revision differs')
        for entry in verified['files']:
            weight = path / entry['file']
            if weight.stat().st_size != entry['bytes'] or sha(weight) != entry['sha256']:
                raise ValueError('Cached model weight checksum differs')
        self.torch = torch
        self.key = key
        self.backend_version = BACKEND_VERSION
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(path),local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':0},
            attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
        self.processor = AutoProcessor.from_pretrained(str(path),local_files_only=True,use_fast=False,
                                                       min_pixels=256*28*28,max_pixels=1280*28*28)
        self.last_usage = {}

    def generate(self,item):
        with self.torch.inference_mode():
            inputs, usage = processor_inputs(self.processor,item)
            inputs = {key:value.to('cuda',dtype=self.torch.bfloat16) if value.is_floating_point()
                      else value.to('cuda') for key,value in inputs.items()}
            output = self.model.generate(**inputs,max_new_tokens=128,do_sample=False,num_beams=1,
                                         logits_processor=finite_logits_processors())
            self.last_usage = {**usage,'output_tokens':int(output.shape[1]-inputs['input_ids'].shape[1])}
            return self.processor.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
