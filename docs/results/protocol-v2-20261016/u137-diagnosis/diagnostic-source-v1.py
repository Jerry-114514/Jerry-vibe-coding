"""Post-result CPU diagnosis: fixed global hypotheses, never choose per-row units."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from statistics import mean, median

from experiment_common import ROOT, read_json, read_rows, sha, write_json
from scoring import box_iou

BASE = ROOT / 'runs/route-b-9b-122b-20261017'
STUDENT = BASE / 'student-experiment'
EVAL = STUDENT / 'native-frontend-evaluation-v41'
OUT = EVAL / 'u137-diagnostic-20261018'
MODELS = ('N-seed42', 'A1-seed42', 'A2-seed42', 'U-seed42', 'A2-seed137', 'U-seed137')
HYPOTHESES = ('registered_native1000', 'hypothetical_original_pixels', 'hypothetical_processor_pixels')


def transform(box, item, hypothesis):
    if box is None:
        return None
    if hypothesis == 'registered_native1000':
        return [v * (item['width'] if i % 2 == 0 else item['height']) / 1000 for i, v in enumerate(box)]
    if hypothesis == 'hypothetical_original_pixels':
        return list(box)
    if hypothesis == 'hypothetical_processor_pixels':
        return [v * (item['width']/item['processor_width'] if i % 2 == 0 else item['height']/item['processor_height']) for i, v in enumerate(box)]
    raise ValueError(hypothesis)


def prepare():
    sources = {name:sha(ROOT/'tools'/name) for name in ('diagnose_u137_geometry.py', 'qwen35_native_frontend.py', 'experiment_common.py', 'scoring.py')}
    inputs = {'protocol':sha(EVAL/'protocol.json'), 'manifest':sha(EVAL/'formal-manifest.jsonl')}
    for name in MODELS:
        for file in ('eval.jsonl','scores.json'):
            inputs[name+'/'+file] = sha(EVAL/'formal'/name/file)
    path = OUT/'diagnostic-plan.json'
    if path.exists():
        plan = read_json(path)
        assert plan['source_sha256'] == sources and plan['input_sha256'] == inputs
        return plan
    plan = {'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Exploratory post-result diagnosis; not a repaired primary score or confirmation',
        'source_sha256':sources,'input_sha256':inputs,'models':MODELS,
        'hypotheses':HYPOTHESES,'no_per_item_best_units':True,'no_clipping_or_GT_fit':True,
        'same_three_global_maps_for_all_six_models':True,
        'official_score_remains_native1000':True,'new_predictions':0,
        'training_checks':'1000 contiguous real updates, sampling metadata, dualCE/grad finite, hashes, axis exposure, target conflict, saved adapter finite'}
    write_json(path,plan)
    return plan


def aggregate(rows):
    ious = [r['iou'] for r in rows]
    return {'rows':len(rows),'correct':sum(v>=.5 for v in ious),
        'accuracy':sum(v>=.5 for v in ious)/len(rows),
        'mean_iou':mean(ious),'median_iou':median(ious),
        'within_image_bounds':sum(r['within_bounds'] for r in rows)}


def geometry():
    from qwen35_native_frontend import score_native_frontend
    from run_9b_native_frontend_v41 import verify_evaluation
    plan = prepare()
    manifest = {r['item_id']:r for r in read_rows(EVAL/'formal-manifest.jsonl')}
    results = {}
    for name in MODELS:
        arm, seed = name.split('-seed')
        marker,_ = verify_evaluation(arm,int(seed))
        ledger = read_rows(EVAL/'formal'/name/'eval.jsonl')
        grounding = [r for r in ledger if r['axis']=='grounding']
        assert len(grounding)==2400
        maps = {}
        for hypothesis in HYPOTHESES:
            records=[]
            for row in grounding:
                item=manifest[row['item_id']]
                box=transform(row['parsed_native1000'],item,hypothesis)
                overlap=box_iou(box,item['answer_key']) if box is not None else 0.
                # This diagnostic algebraically cancels the registered processor chain.
                if hypothesis=='registered_native1000':
                    assert abs(overlap-(row['iou'] or 0.))<1e-10
                records.append({'item_id':row['item_id'],'dataset':item['dataset'],
                    'size_bin':'under512' if max(item['width'],item['height'])<512 else ('512to999' if max(item['width'],item['height'])<1000 else '1000plus'),
                    'aspect':'portrait' if item['height']>item['width'] else ('landscape' if item['width']>item['height'] else 'square'),
                    'iou':overlap,'within_bounds':bool(box is not None and 0<=box[0]<box[2]<=item['width'] and 0<=box[1]<box[3]<=item['height'])})
            groups={}
            for field in ('dataset','size_bin','aspect'):
                groups[field]={key:aggregate([r for r in records if r[field]==key]) for key in sorted({r[field] for r in records})}
            maps[hypothesis]={**aggregate(records),'strata':groups}
        boxes=[tuple(r['parsed_native1000']) for r in grounding if r['parsed_native1000'] is not None]
        results[name]={'official_axis_scores':marker['axis_scores'],
            'parse_failures':sum(r['parse_fail_flag'] for r in grounding),
            'length_capped':sum(r['finish_reason']=='length' for r in grounding),
            'unique_boxes':len(set(boxes)),
            'most_repeated_boxes':Counter(boxes).most_common(5),
            'coordinate_medians':[median(b[i] for b in boxes) for i in range(4)],
            'maps':maps}
    write_json(OUT/'geometry-results.json',{'plan_sha256':sha(OUT/'diagnostic-plan.json'),
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'official_results_unchanged':True,
        'new_predictions':0,'results':results})
    print('GEOMETRY', {name:{h:round(v['accuracy'],6) for h,v in row['maps'].items()} for name,row in results.items()},flush=True)


def training():
    import torch
    from safetensors.torch import load_file
    import preflight_9b_full as shared
    torch.set_num_threads(1)
    prepare()
    pool={r['item_id']:r for r in read_rows(BASE/'eligible-pool.jsonl')}
    teachers={r['item_id']:r for r in read_rows(BASE/'teacher.jsonl')}
    protocol=read_json(EVAL/'protocol.json')
    results={}
    for name in ('U-seed42','U-seed137'):
        folder=STUDENT/'formal'/name
        marker=read_json(folder/'complete.json');meta=read_json(folder/'train.meta.json')
        ledger=read_rows(folder/'loss.jsonl');adapter=folder/'adapter/adapter_model.safetensors'
        assert marker['seed']==meta['seed']==int(name.split('seed')[1])
        assert marker['arm']==meta['arm']=='U'
        assert marker['actual_updates']==len(ledger)==1000
        assert [r['step'] for r in ledger]==list(range(1,1001))
        assert sha(folder/'loss.jsonl')==marker['loss_sha256']
        assert sha(adapter)==marker['adapter_sha256']==protocol['adapters_sha256'][name]
        assert all(r['actual_optimizer_update'] and r['grad_norm']>0 and all(math.isfinite(r[k]) for k in ('loss','ce_gt','ce_teacher','grad_norm')) for r in ledger)
        ids=meta['sampling_ids'];assert all(r['item_id']==ids[(r['step']-1)%len(ids)] for r in ledger)
        model_tensors=load_file(str(adapter),device='cpu')
        tensor_summary={}
        for category in ('lora_A','lora_B'):
            selected=[v for k,v in model_tensors.items() if category in k]
            assert selected and all(torch.isfinite(v).all() for v in selected)
            tensor_summary[category]={'tensors':len(selected),'norm':sum(v.float().square().sum().item() for v in selected)**.5,
                                      'max_abs':max(v.float().abs().max().item() for v in selected)}
        del model_tensors
        axes={}
        for axis in ('pope','grounding','consistency','mmstar'):
            group=[r for r in ledger if r['axis']==axis]
            distinct=list(dict.fromkeys(r['item_id'] for r in group))
            examples=[pool[k] for k in distinct]
            conflicts=sum(shared.answer_text(pool[k])!=shared.answer_text(pool[k],teachers[k]) for k in distinct)
            numeric_mismatch=[]
            if axis=='grounding':
                for k in distinct:
                    gt=json.loads(shared.answer_text(pool[k]));teacher=json.loads(shared.answer_text(pool[k],teachers[k]))
                    numeric_mismatch.append(max(abs(a-b) for a,b in zip(gt,teacher)))
            axes[axis]={'updates':len(group),'unique_examples':len(distinct),
                'unique_image_paths':len({r['image_path'] for r in examples}),
                'mean_loss':mean(r['loss'] for r in group),'last25_mean_loss':mean(r['loss'] for r in group[-25:]),
                'mean_ce_gt':mean(r['ce_gt'] for r in group),'mean_ce_teacher':mean(r['ce_teacher'] for r in group),
                'max_grad_before_clip':max(r['grad_norm'] for r in group),'preclip_grad_over1':sum(r['grad_norm']>1 for r in group),
                'gt_teacher_target_text_mismatches':conflicts,
                'grounding_numeric_maxdiff_median':median(numeric_mismatch) if numeric_mismatch else None}
        worst=sorted(ledger,key=lambda r:r['grad_norm'],reverse=True)[:5]
        results[name]={'hashes_match':True,'real_updates_contiguous_and_finite':True,
            'sampling_matches_metadata':True,'sample_pool_size':len(ids),'sampling_ids_sha256':sha(folder/'train.meta.json'),
            'axes':axes,'adapter_tensor_summary':tensor_summary,'largest_preclip_gradients':worst,
            'resume_record':read_json(folder/'resume-verification.json') if (folder/'resume-verification.json').exists() else None}
    assert not torch.cuda.is_initialized()
    write_json(OUT/'training-results.json',{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'plan_sha256':sha(OUT/'diagnostic-plan.json'),'cuda_initialized':False,'results':results})
    print('TRAINING',json.dumps(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','geometry','training']);args=parser.parse_args()
    if args.action=='prepare':
        prepare();print('Post-result diagnostic hypotheses frozen; no primary score changes')
    elif args.action=='geometry':geometry()
    else:training()
