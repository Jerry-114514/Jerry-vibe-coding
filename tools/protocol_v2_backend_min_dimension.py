"""Same v3 processor contract; deterministically upscale only sub-28px image sides."""
from PIL import Image
from experiment_common import ROOT
from protocol_v2_backend import QwenV2Backend

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
            rgb=image.convert('RGB')
            if min(rgb.size)<28:
                import math
                scale=28/min(rgb.size)
                rgb=rgb.resize((math.ceil(rgb.width*scale),math.ceil(rgb.height*scale)),Image.Resampling.LANCZOS)
            images.append(rgb)
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

