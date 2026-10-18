"""Deterministic non-score-selected geometry review; anonymous model labels."""
import hashlib
import random
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from experiment_common import ROOT,read_rows,write_json,sha
from prepare_9b_repair import OLD_EVAL,OUT,now


def visualize():
    target=OUT/'visual-review';target.mkdir(exist_ok=True)
    manifest=[r for r in read_rows(OLD_EVAL/'formal-manifest.jsonl') if r['axis']=='grounding']
    models=('N-seed42','U-seed42','U-seed137')
    predictions={name:{r['item_id']:r for r in read_rows(OLD_EVAL/'formal'/name/'eval.jsonl')} for name in models}
    chosen=[];seen=set()
    for dataset in sorted({r['dataset'] for r in manifest}):
        candidates=sorted((r for r in manifest if r['dataset']==dataset),
            key=lambda r:hashlib.sha256(('2026101805:'+r['item_id']).encode()).hexdigest())
        count=0
        for item in candidates:
            if item['image_group'] in seen:continue
            chosen.append(item);seen.add(item['image_group']);count+=1
            if count==4:break
        assert count==4
    fig,axes=plt.subplots(4,3,figsize=(15,18),constrained_layout=True)
    assignment=[];colors={'A':'#d82d2d','B':'#257dc0','C':'#31a354'}
    for i,(item,ax) in enumerate(zip(chosen,axes.flat),start=1):
        with Image.open(ROOT/item['image_path']) as image:ax.imshow(image.convert('RGB'))
        gt=item['answer_key'];ax.add_patch(Rectangle((gt[0],gt[1]),gt[2]-gt[0],gt[3]-gt[1],fill=False,edgecolor='black',linewidth=3,label='GT'))
        order=list(models);random.Random('2026101805:'+item['item_id']).shuffle(order)
        mapping={}
        for label,name in zip('ABC',order):
            mapping[label]=name;box=predictions[name][item['item_id']]['parsed_answer']
            if box is not None:ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor=colors[label],linewidth=1.8,linestyle='--',label=label))
        ax.set_title(f"{i}. {item['dataset']}",fontsize=11)
        question=item['question'].split(' Return only')[0]
        ax.set_xlabel('\n'.join(textwrap.wrap(question,58)),fontsize=8)
        ax.set_xticks([]);ax.set_yticks([]);ax.legend(loc='upper right',fontsize=7)
        assignment.append({'panel':i,'item_id':item['item_id'],'image_group':item['image_group'],
            'model_key':mapping,'selection_not_based_on_IoU':True})
    fig.suptitle('Fixed non-score-selected sample: GT black; model identity masked A/B/C per panel',fontsize=14)
    fig.savefig(target/'anonymous-contact-sheet.png',dpi=130);plt.close(fig)
    write_json(target/'selection-and-unblinding-key.json',{'timestamp_utc':now(),'source_sha256':sha(__file__),
        'selection':'4 per dataset, hash seed2026101805, distinct image groups, no score/parse filtering',
        'rows':assignment,'human_ARP_blind_review_complete':False,
        'no_prompt_change_no_new_inference':True})
    print('Prepared12 anonymous panels; no new inference',flush=True)


if __name__=='__main__':visualize()
