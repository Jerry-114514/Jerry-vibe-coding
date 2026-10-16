"""CPU-only full benchmark preprocessing audit before the scheduled GPU eval."""
from collections import Counter,defaultdict
import json
import os
import statistics
import time
from PIL import Image
from experiment_common import ROOT, AXES, read_json, read_rows, sha, write_json


def main():
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    out=ROOT/'runs/route-b-staged-7b-20261016'
    processor=AutoProcessor.from_pretrained(str(ROOT/'.model_cache/qwen7b'),local_files_only=True,use_fast=False)
    manifest=ROOT/'data/manifest.jsonl'
    rows=read_rows(manifest)
    lock=read_json(ROOT/'data/experiment.lock.json')
    assert sha(manifest)==lock['files']['data/manifest.jsonl']
    assert len(rows)==16900 and len({r['item_id'] for r in rows})==16900
    assets={r['path']:r for r in read_rows(ROOT/'data/assets.sha256.jsonl')}
    assert sha(ROOT/'data/assets.sha256.jsonl')==lock['files']['data/assets.sha256.jsonl']
    training=read_json(ROOT/'data/training/pool.lock.json')
    assert training['evaluation_manifest_sha256']==sha(manifest)
    assert training['manifest_sha256']==sha(ROOT/'data/training/pool.jsonl')
    assert training['image_byte_and_decoded_pixel_overlap']==0
    metrics=defaultdict(list);checked={};groups=Counter();violations=[]
    started=time.monotonic()
    for i,row in enumerate(rows,1):
        relative=row['image_path']
        image=ROOT/relative
        if relative not in checked:
            assert image.stat().st_size==assets[relative]['bytes'] and sha(image)==assets[relative]['sha256']
            checked[relative]=assets[relative]['sha256']
        with Image.open(image) as source:
            assert source.size==(row['width'],row['height'])
            rgb=source.convert('RGB')
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,add_vision_id=True)
        x=processor(text=[text],images=[rgb],return_tensors='pt',images_kwargs={
           'size':{'shortest_edge':256*28*28,'longest_edge':1280*28*28},
           'min_pixels':256*28*28,'max_pixels':1280*28*28})
        tokens=int(x['input_ids'].shape[1]);visual=int(x['image_grid_thw'].prod(dim=1).sum().item()//4)
        metrics[row['axis']].append(tokens)
        groups[(row['axis'],row['dataset'],row['bootstrap_unit'])]+=1
        if tokens>2048 or visual>1280:
            violations.append({'item_id':row['item_id'],'axis':row['axis'],'prompt_tokens':tokens,'visual_tokens':visual})
        if i%1000==0:
            write_json(out/'eval-input-preflight-progress.json',{'rows':i,'expected':16900,'violations':len(violations),'pid':os.getpid()})
    assert all(count==5 for (axis,_,_),count in groups.items() if axis=='consistency')
    assert not torch.cuda.is_initialized()
    result={'status':'passed' if not violations else 'input_budget_review_required','rows':16900,
            'unique_images_verified':len(checked),'original_units':len(groups),
            'axis_prompt_counts':dict(Counter(r['axis'] for r in rows)),
            'prompt_tokens':{a:{'min':min(v),'max':max(v),'mean':statistics.mean(v)} for a,v in metrics.items()},
            'budget_violations':violations,'cpu_only':True,'no_cuda_context':True,
            'manifest_sha256':sha(manifest),'asset_manifest_sha256':sha(ROOT/'data/assets.sha256.jsonl'),
            'elapsed_seconds':time.monotonic()-started,'no_evaluation_predictions_generated':True}
    write_json(out/'eval-input-independent-preflight.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
