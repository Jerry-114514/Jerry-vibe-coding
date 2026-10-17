"""User-amended9B evaluation: explicit normalized1000 prompt, pixel IoU.

Old pixel outputs remain immutable. Minimum calibration generates only100
grounding prompts per model; the other700 unchanged outputs are reused.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

from experiment_common import ROOT, AXES, read_json, read_rows, write_json, write_rows, sha, units, parse_answer, score_generation
from scoring import box_iou
from route_b_9b_loop import OUT, BASE, SEEDS, MINIMAL_MANIFEST, freeze

NATIVE = OUT / 'native-evaluation-v2'
SOURCES = ('evaluate_9b_native.py', 'run_9b_native_evaluation.py', 'analyze_9b_native.py', 'resilient_9b_delivery.py', 'verify_9b_recovery_availability.py')


def native_item(item):
    row = dict(item)
    if row['axis'] == 'grounding':
        suffix = f'in original-image pixel coordinates. The image is {row["width"]} pixels wide and {row["height"]} pixels high.'
        assert row['question'].endswith(suffix), 'Unknown pixel prompt contract; do not guess a replacement'
        row['question'] = row['question'][:-len(suffix)] + 'in normalized coordinates from 0 to 1000, independent of the original image dimensions. Return only the four numbers.'
    return row


def score_native(text, item):
    if item['axis'] != 'grounding':
        return score_generation(text, item)
    box = parse_answer(text, {'answer_type': 'bbox', 'width': 1000, 'height': 1000})
    pixel = [v*(item['width'] if i % 2 == 0 else item['height'])/1000 for i, v in enumerate(box)] if box is not None else None
    overlap = box_iou(pixel, item['answer_key']) if pixel is not None else None
    return {'raw_generation': text, 'parsed_native1000': box, 'parsed_answer': pixel,
            'coordinate_system': 'normalized1000_to_original_pixels', 'iou': overlap,
            'parse_fail_flag': int(box is None), 'correct_bit': int(overlap >= .5) if overlap is not None else 0}


def prepare():
    path = NATIVE / 'protocol.json'
    if path.exists():
        plan = read_json(path)
        for name, digest in plan['source_sha256'].items():
            assert sha(ROOT / 'tools' / name) == digest
        for phase, entry in plan['inputs'].items():
            assert sha(NATIVE / (phase+'-manifest.jsonl')) == entry['native_sha256']
            source = MINIMAL_MANIFEST if phase == 'minimal' else ROOT / 'data/manifest.jsonl'
            assert sha(source) == entry['original_sha256']
        return plan
    from transformers import AutoProcessor
    from route_b_9b_loop import public_prompt_inputs
    import preflight_9b_full as shared
    __import__('torch').set_num_threads(2)
    processor = AutoProcessor.from_pretrained(str(shared.MODEL), local_files_only=True)
    NATIVE.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for phase, source, expected in [('minimal', MINIMAL_MANIFEST, 800), ('formal', ROOT / 'data/manifest.jsonl', 16900)]:
        original = read_rows(source)
        assert len(original) == len({r['item_id'] for r in original}) == expected
        native = [native_item(r) for r in original]
        lengths = []
        for row in native:
            if row['axis'] == 'grounding':
                batch = public_prompt_inputs(processor, row)
                lengths.append(int(batch['input_ids'].shape[1]))
        target = NATIVE / (phase+'-manifest.jsonl')
        write_rows(target, native)
        inputs[phase] = {'rows': expected, 'original_sha256': sha(source), 'native_sha256': sha(target),
                         'grounding_rows': len(lengths), 'native_grounding_max_prompt_tokens': max(lengths),
                         'unchanged_non_grounding_rows': expected-len(lengths)}
    plan = {'schema': 'independent.9b.native1000.evaluation.v2', 'user_requested_amendment': True,
            'registered_after_old_pixel_minimum_outputs': True, 'timestamp': datetime.now(timezone.utc).isoformat(),
            'student_protocol_sha256': sha(OUT / 'protocol.json'), 'training_plan_sha256': sha(OUT / 'formal-plan.json'),
            'source_sha256': {n: sha(ROOT / 'tools' / n) for n in SOURCES}, 'inputs': inputs,
            'coordinate_contract': 'All models output normalized0..1000; x*width/1000,y*height/1000; original-pixel GT and IoU>=.5; no GT-driven unit selection',
            'training_unchanged': True, 'training_target_coordinate_contract': 'original-pixel targets, as already frozen',
            'extra_calibration_generations': 400, 'calibration_active_wall_seconds_cap': 900,
            'N_grounding_minimum_accuracy': .80, 'N_grounding_maximum_parse_failures': 5,
            'calibration_reuse': '700 unchanged non-grounding outputs per minimum model; retain source ledger SHA',
            'formal_models': 16, 'formal_generations': 270400, 'formal_evaluation_wall_hours_cap': 40,
            'max_new_tokens': 128, 'max_total_tokens': 2048, 'max_visual_tokens': 1280,
            'statistics': 'Original registered10000 paired seed/image bootstrap,12 Bonferroni comparisons,seed2026101703',
            'scope': 'User-amended measurement calibration; independent extension, not ARP confirmation or capacity-only effect'}
    assert not __import__('torch').cuda.is_initialized(), 'Preparation must remain CPU only'
    freeze(path, plan)
    return plan


def evaluate(phase, arm, seed, locked=False, deadline=None):
    import torch
    from gpu_lock import gpu_lock
    from route_b_9b_loop import admission, model_backend, public_prompt_inputs, gpu_inputs
    from model_backends import finite_logits_processors
    admission()
    plan = prepare()
    if phase == 'formal':
        assert read_json(NATIVE / 'minimal-acceptance.json')['status'] == 'passed'
        assert all(read_json(OUT/'formal'/f'{a}-seed{s}'/'complete.json')['actual_updates']==1000 for s in SEEDS for a in ('A1','A2','U'))
    manifest_path = NATIVE / (phase+'-manifest.jsonl')
    manifest = read_rows(manifest_path)
    folder = NATIVE / phase / f'{arm}-seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    original_folder = OUT / phase / f'{arm}-seed{seed}'
    adapter = None if arm == 'N' else original_folder / 'adapter'
    if adapter is not None:
        assert sha(adapter/'adapter_model.safetensors') == read_json(original_folder/'complete.json')['adapter_sha256']
    meta = {'manifest_sha256': sha(manifest_path), 'measurement_protocol_sha256': sha(NATIVE/'protocol.json'),
            'student_protocol_sha256': plan['student_protocol_sha256'], 'arm': arm, 'seed': seed,
            'thinking': False, 'max_new_tokens': 128, 'reuse_non_grounding': phase=='minimal'}
    if phase == 'minimal':
        meta['reused_legacy_ledger_sha256'] = sha(original_folder/'eval.jsonl')
    freeze(folder/'eval.meta.json', meta)
    ledger = folder/'eval.jsonl'
    if phase == 'minimal' and not ledger.exists():
        old = read_rows(original_folder/'eval.jsonl')
        assert len(old)==800 and sha(original_folder/'eval.jsonl') == read_json(original_folder/'scores.json')['ledger_sha256']
        reused = [{**r, 'reused_unchanged_non_grounding': True} for r in old if r['axis']!='grounding']
        assert len(reused)==700
        write_rows(ledger, reused)
    records = read_rows(ledger) if ledger.exists() else []
    done = {r['item_id'] for r in records}
    assert len(done)==len(records) and done <= {r['item_id'] for r in manifest}
    if (folder/'scores.json').exists():
        return verify_evaluation(arm, seed, phase)[0]
    def infer():
        began = time.monotonic()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        processor, model = model_backend(adapter)
        model.eval().requires_grad_(False)
        assert not any(p.requires_grad for p in model.parameters())
        eos=model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos])
        for row in manifest:
            if row['item_id'] in done: continue
            if deadline is not None: assert time.monotonic()<deadline, 'Fixed calibration active-wall budget exhausted'
            assert shutil.disk_usage(OUT).free >= 5*1024**3
            started=time.monotonic()
            with torch.inference_mode():
                batch=gpu_inputs(public_prompt_inputs(processor,row))
                generated=model.generate(**batch,max_new_tokens=128,do_sample=False,logits_processor=finite_logits_processors())
                ids=generated[0,batch['input_ids'].shape[1]:]
                raw=processor.decode(ids,skip_special_tokens=True).strip()
            torch.cuda.synchronize()
            record={'item_id':row['item_id'],'axis':row['axis'],'seconds':time.monotonic()-started,
                    'finish_reason':'eos' if ids.numel() and int(ids[-1]) in eos else 'length',
                    'generation_tokens':ids.numel(),**score_native(raw,row)}
            with ledger.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            done.add(row['item_id'])
            assert torch.cuda.max_memory_reserved()<=30_000_000_000
            if len(done)%100==0: print('native9B',phase,arm,seed,len(done),'/',len(manifest),flush=True)
        rows=read_rows(ledger); original=units(manifest,rows)
        scores={a:sum(r['correct_bit'] for r in original if r['axis']==a)/sum(r['axis']==a for r in original) for a in AXES}
        result={'rows':len(rows),'original_units':len(original),'ledger_sha256':sha(ledger),'axis_scores':scores,
                'adapter_reloaded_for_inference':arm!='N','peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'optimizer_loaded':False,'gradients_enabled':False,
                'measurement_protocol_sha256':sha(NATIVE/'protocol.json'),
                'total_generation_seconds':sum(r['seconds'] for r in rows),
                'new_generation_seconds':sum(r['seconds'] for r in rows if not r.get('reused_unchanged_non_grounding')),
                'length_capped_outputs':sum(r['finish_reason']=='length' for r in rows),
                'active_process_seconds':time.monotonic()-began}
        write_json(folder/'scores.json',result)
        del processor,model
        gc.collect();torch.cuda.empty_cache()
        return result
    if locked:
        return infer()
    with gpu_lock():
        return infer()


def verify_evaluation(arm, seed, phase='formal'):
    folder=NATIVE/phase/f'{arm}-seed{seed}'
    manifest=read_rows(NATIVE/(phase+'-manifest.jsonl'));lookup={r['item_id']:r for r in manifest}
    rows=read_rows(folder/'eval.jsonl');marker=read_json(folder/'scores.json')
    assert len(rows)==len({r['item_id'] for r in rows})==len(manifest)==marker['rows']
    assert {r['item_id'] for r in rows}==set(lookup) and sha(folder/'eval.jsonl')==marker['ledger_sha256']
    assert marker['measurement_protocol_sha256']==sha(NATIVE/'protocol.json')
    assert read_json(folder/'eval.meta.json')['manifest_sha256']==sha(NATIVE/(phase+'-manifest.jsonl'))
    for row in rows:
        computed=score_native(row['raw_generation'],lookup[row['item_id']])
        assert all(computed[k]==row[k] for k in ('parsed_answer','correct_bit','parse_fail_flag','iou'))
        if row['axis']=='grounding':assert computed['parsed_native1000']==row['parsed_native1000']
    original=units(manifest,rows)
    assert marker['axis_scores']=={a:sum(r['correct_bit'] for r in original if r['axis']==a)/sum(r['axis']==a for r in original) for a in AXES}
    assert marker['original_units']==len(original) and marker['optimizer_loaded'] is False and marker['gradients_enabled'] is False
    assert arm=='N' or marker['adapter_reloaded_for_inference']
    return marker,{(r['axis'],r['dataset'],r['unit']):r for r in original}


def verify_formal(return_evaluations=False):
    from verify_9b_student_outputs import verify_training
    for seed in SEEDS:
        for arm in ('A1','A2','U'):verify_training(arm,seed,allow_offloaded=True)
    models=[('N',42)]+[(a,s) for s in SEEDS for a in ('A1','A2','U')]
    evaluations={(a,s):verify_evaluation(a,s) for a,s in models}
    result={'status':'passed','training_runs':15,'updates_per_run':1000,'evaluation_models':16,'prompts_per_model':16900,
            'measurement_protocol_sha256':sha(NATIVE/'protocol.json'),'coordinate_contract':'explicit_normalized1000_then_pixel_iou'}
    write_json(NATIVE/'formal-independent-verification.json',result)
    return (result,evaluations) if return_evaluations else result


def calibrate():
    plan=prepare()
    from gpu_lock import gpu_lock
    with gpu_lock():
        path=NATIVE/'calibration-active-wall-budget.json'
        if path.exists():
            end=datetime.fromisoformat(read_json(path)['end_utc']).timestamp()
            remaining=end-datetime.now(timezone.utc).timestamp()
        else:
            start=datetime.now(timezone.utc).timestamp();remaining=900
            write_json(path,{'start_utc':datetime.fromtimestamp(start,timezone.utc).isoformat(),
                            'end_utc':datetime.fromtimestamp(start+900,timezone.utc).isoformat(),
                            'extra_generation_budget':400,'user_measurement_amendment':True,'never_reset_on_restart':True})
        assert remaining>0
        deadline=time.monotonic()+remaining
        for arm in ('N','A1','A2','U'):
            result=evaluate('minimal',arm,42,locked=True,deadline=deadline)
            if arm=='N':
                ground=[r for r in read_rows(NATIVE/'minimal/N-seed42/eval.jsonl') if r['axis']=='grounding']
                assert result['axis_scores']['grounding']>=plan['N_grounding_minimum_accuracy'], 'N native calibration below fixed80% sanity threshold; inspect without retries'
                assert sum(r['parse_fail_flag'] for r in ground)<=plan['N_grounding_maximum_parse_failures']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','calibrate','eval','verify'])
    parser.add_argument('--phase',choices=['minimal','formal'],default='formal')
    parser.add_argument('--arm',choices=['N','A1','A2','U'],default='N');parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare()),flush=True)
    elif args.action=='calibrate':calibrate()
    elif args.action=='verify':print(json.dumps(verify_formal()),flush=True)
    else:evaluate(args.phase,args.arm,args.seed)
