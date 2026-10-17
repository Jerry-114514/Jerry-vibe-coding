"""One explicit processor-pixel contract for7B/9B, never infer units from GT."""
from experiment_common import ROOT,parse_answer,score_generation
from scoring import box_iou


def dimensions(processor,item,model):
    from PIL import Image
    path=(ROOT/item['image_path']).resolve();assert path.is_relative_to((ROOT/'data').resolve())
    with Image.open(path) as source:
        assert source.size==(item['width'],item['height'])
        image=source.convert('RGB')
    if model=='qwen9b':
        if min(image.size)<32:
            scale=32/min(image.size)
            image=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
        kwargs={'min_pixels':256*32*32,'max_pixels':1280*32*32}
    else:
        assert model=='qwen7b'
        kwargs={'size':{'shortest_edge':256*28*28,'longest_edge':1280*28*28},'min_pixels':256*28*28,'max_pixels':1280*28*28}
    result=processor.image_processor(images=[image],return_tensors='pt',**kwargs)
    grid=result['image_grid_thw'][0].tolist();patch=processor.image_processor.patch_size
    assert grid[0]==1 and grid[1]*grid[2]//4<=1280
    return {'processor_width':grid[2]*patch,'processor_height':grid[1]*patch,'processor_grid_thw':grid,'processor_patch_size':patch}


def item_with_contract(item,size):
    row=dict(item)
    if row['axis']!='grounding':return row
    suffix=f'in original-image pixel coordinates. The image is {row["width"]} pixels wide and {row["height"]} pixels high.'
    assert row['question'].endswith(suffix),'Unknown source coordinate prompt'
    row.update(size)
    row['question']=row['question'][:-len(suffix)]+f'in processor-resized-image pixel coordinates. The supplied image is {size["processor_width"]} pixels wide and {size["processor_height"]} pixels high. Return only the four numbers.'
    return row


def score(text,item):
    if item['axis']!='grounding':return score_generation(text,item)
    width,height=item['processor_width'],item['processor_height']
    assert width>0 and height>0
    box=parse_answer(text,{'answer_type':'bbox','width':width,'height':height})
    pixel=[v*(item['width']/width if i%2==0 else item['height']/height) for i,v in enumerate(box)] if box is not None else None
    overlap=box_iou(pixel,item['answer_key']) if pixel is not None else None
    return {'raw_generation':text,'parsed_processor_pixels':box,'parsed_answer':pixel,
            'coordinate_system':'processor_pixels_to_original_pixels','iou':overlap,
            'parse_fail_flag':int(box is None),'correct_bit':int(overlap>=.5) if overlap is not None else 0}


def assert_runtime_grid(batch,item):
    if item['axis']=='grounding':
        actual=batch['image_grid_thw'][0].tolist()
        assert actual==item['processor_grid_thw'],'Actual encoded image grid differs from frozen coordinate dimensions'


def cpu_7b_inputs(processor,row):
    from PIL import Image
    with Image.open(ROOT/row['image_path']) as source:image=source.convert('RGB')
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,add_vision_id=True)
    return processor(text=[text],images=[image],return_tensors='pt',images_kwargs={
        'size':{'shortest_edge':256*28*28,'longest_edge':1280*28*28},'min_pixels':256*28*28,'max_pixels':1280*28*28})
