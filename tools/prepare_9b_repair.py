"""CPU preparation for a fixed two-seed 2x2 representation/supervision study."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import random
import shutil

from experiment_common import ROOT, AXES, read_json, read_rows, write_json, write_rows, sha
from repair_9b_codec import CONDITIONS, training_example

OUT = ROOT/'runs/9b-u137-factorial-20261018'
OLD = ROOT/'runs/route-b-9b-122b-20261017/student-experiment'
OLD_EVAL = OLD/'native-frontend-evaluation-v41'
BASE = OLD.parent
SEEDS = (42,137)
SOURCE = ROOT/'runs/protocol-v2-data/clean-role-manifests-v4'
SOURCE_FILES = ('prepare_9b_repair.py','repair_9b_codec.py','train_9b_repair.py',
    'evaluate_9b_repair.py','run_9b_repair.py','report_9b_repair.py','verify_9b_repair_environment.py','verify_9b_repair_holdout.py',
    'route_b_9b_loop.py','preflight_9b_full.py','train_distill.py','gpu_lock.py',
    'qwen35_native_frontend.py','qwen35_termination_contract.py','model_backends.py',
    'experiment_common.py','scoring.py','run_9b_native_frontend_v41.py',
    'independent_manifest_v2.py','freeze_independent_candidates.py','publish_artifacts.py',
    'run_9b_formal_when_ready.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def check_data():
    state=read_json(OUT/'data-preparation.json')
    for relative,digest in state['files_sha256'].items():
        assert sha(ROOT/relative)==digest,relative
    return state


def prepare_data():
    if (OUT/'data-preparation.json').exists():
        return check_data()
    import torch
    from transformers import AutoProcessor
    import preflight_9b_full as shared
    from route_b_9b_loop import arm_rows,public_prompt_inputs
    from qwen35_native_frontend import native_prompt
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    pool=read_rows(BASE/'eligible-pool.jsonl')
    teachers={r['item_id']:r for r in read_rows(BASE/'teacher.jsonl')}
    assert {r['item_id'] for r in pool}<={r['item_id'] for r in read_rows(ROOT/'data/training/pool.jsonl')}
    processor=AutoProcessor.from_pretrained(str(shared.MODEL),local_files_only=True)
    generated=[];historical={};input_cache={};image_hashes={}
    for seed in SEEDS:
        rows=arm_rows(pool,'formal','U',seed)
        source=OLD/'formal'/f'U-seed{seed}'
        meta=read_json(source/'train.meta.json');done=read_json(source/'complete.json')
        assert [r['item_id'] for r in rows]==meta['sampling_ids'] and len(rows)==600
        assert done['actual_updates']==1000 and sha(source/'loss.jsonl')==done['loss_sha256']
        assert sha(source/'adapter/adapter_model.safetensors')==done['adapter_sha256']
        historical[f'P-D-seed{seed}']={'source':str(source.relative_to(ROOT)),
            'adapter_sha256':done['adapter_sha256'],'loss_sha256':done['loss_sha256'],
            'meta_sha256':sha(source/'train.meta.json'),'source_sha256':sha(ROOT/'tools/route_b_9b_loop.py')}
        for condition in CONDITIONS:
            examples=[]
            for index,row in enumerate(rows):
                teacher=teachers[row['item_id']]
                if row['image_path'] not in image_hashes:
                    image_hashes[row['image_path']]=sha(ROOT/row['image_path'])
                assert any(x['sha256']==image_hashes[row['image_path']] for x in teacher['image_hashes'])
                example=training_example(row,teacher,condition)
                # Assert the reuse condition is byte-identical to historical targets.
                if condition=='P-D':
                    assert example['question']==row['question']
                    assert example['gt_target']==shared.answer_text(row)
                    assert example['teacher_target']==shared.answer_text(row,teacher)
                key=(row['item_id'],example['question'])
                if key not in input_cache:
                    batch=shared.prompt_inputs(processor,example)
                    input_cache[key]={'prompt_tokens':int(batch['input_ids'].shape[1]),
                        'prompt_ids_sha256':hashlib.sha256(batch['input_ids'].numpy().tobytes()).hexdigest(),
                        'processor_grid_thw':batch['image_grid_thw'][0].tolist()}
                gt=shared.targets(processor,example['gt_target']);taught=shared.targets(processor,example['teacher_target'])
                assert input_cache[key]['prompt_tokens']+max(len(gt),len(taught))<=2048
                examples.append({**example,**input_cache[key],'gt_target_ids':gt,
                    'teacher_target_ids':taught,'source_image_sha256':image_hashes[row['image_path']]})
                if index and index%200==0:print('CPU train inputs',condition,seed,index,flush=True)
            path=OUT/'training-inputs'/f'{condition}-seed{seed}.jsonl';write_rows(path,examples);generated.append(path)
    confirmation=prepare_confirmation(processor)
    generated.extend(confirmation)
    development=OUT/'development-manifest.jsonl'
    shutil.copyfile(OLD_EVAL/'formal-manifest.jsonl',development);generated.append(development)
    jobs=[{'condition':c,'seed':seed,'id':f'{c}-seed{seed}'} for seed in SEEDS for c in CONDITIONS if c!='P-D']
    random.Random(2026101802).shuffle(jobs)
    path=OUT/'training-order.json';write_json(path,{'seed':2026101802,'jobs':jobs});generated.append(path)
    path=OUT/'historical-reuse.json';write_json(path,historical);generated.append(path)
    generated.extend([BASE/'eligible-pool.jsonl',BASE/'teacher.jsonl',OLD/'protocol.json',
        ROOT/'configs/route_b_9b_122b.plan.json',OLD_EVAL/'protocol.json',
        SOURCE/'complete.json',SOURCE/'verification.json',SOURCE/'confirm.jsonl',
        ROOT/'data/protocol-v2/historical-image-fingerprints.jsonl'])
    assert not torch.cuda.is_initialized()
    state={'status':'CPU_prepared_no_GPU','timestamp_utc':now(),
        'training_manifests':8,'new_training_runs':6,'reused_runs':2,
        'new_updates':6000,'per_run_updates':1000,'historical_PD_byte_identity':True,
        'source_image_hashes':image_hashes,
        'files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in generated}}
    write_json(OUT/'data-preparation.json',state)
    return state


def prepare_confirmation(processor):
    from PIL import Image
    import numpy as np
    from freeze_independent_candidates import HashTree
    from independent_manifest_v2 import select_axis
    from qwen35_native_frontend import native_prompt
    from route_b_9b_loop import public_prompt_inputs
    complete=read_json(SOURCE/'complete.json')
    assert read_json(SOURCE/'verification.json')['status']=='passed'
    assert sha(SOURCE/'confirm.jsonl')==complete['manifests']['confirm']['sha256']
    groups=defaultdict(list)
    for row in read_rows(SOURCE/'confirm.jsonl'):
        groups[(row['axis'],row['bootstrap_unit'])].append(row)
    chosen=[];counts={}
    for axis in AXES:
        eligible=[]
        for (a,_),group in groups.items():
            if a!=axis:continue
            if any(len(r['image_paths'])!=1 for r in group):continue
            if axis in ('consistency','mmstar') and any(not r.get('options') or
                    not set(r['options'])<=set('ABCD') or r['answer_key'] not in r['options'] for r in group):continue
            assert len(group)==(5 if axis=='consistency' else 1)
            eligible.append(group[0])
        counts[axis]=len(eligible)
        selected=select_axis(eligible,120)
        for representative in selected:
            chosen.extend(sorted(groups[(axis,representative['bootstrap_unit'])],key=lambda r:r['item_id']))
    assert len(chosen)==960 and len({r['item_id'] for r in chosen})==960
    # Membership-only replay search: never inspect previous correctness or scores.
    ids={r['item_id'] for r in chosen};scanned=[]
    for filename in ('eval.jsonl','ledger.jsonl','teacher.jsonl'):
        for path in sorted((ROOT/'runs').rglob(filename)):
            if path.is_relative_to(OUT):continue
            for row in read_rows(path):
                assert row.get('item_id') not in ids, 'Heldout ID was used in prior inference: '+str(path)
            scanned.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size})
    metadata={r['path']:r for r in read_rows(SOURCE/'selected-image-assets.jsonl')}
    history=read_rows(ROOT/'data/protocol-v2/historical-image-fingerprints.jsonl')
    historical_pixels={r['pixel_sha256'] for r in history};tree=HashTree()
    for row in history:tree.add(int(row['dhash64'],16),row['pixel_sha256'])
    checked={};result=[]
    for row in chosen:
        relative=row['image_paths'][0]
        if relative not in checked:
            content=(ROOT/relative).read_bytes()
            with Image.open(io.BytesIO(content)) as image:rgb=image.convert('RGB')
            pixels=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            grey=np.array(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS));differences=grey[:,1:]>grey[:,:-1]
            dhash=sum(int(bit)<<i for i,bit in enumerate(differences.flat))
            facts={'source_sha256':hashlib.sha256(content).hexdigest(),'pixel_sha256':pixels,
                   'width':rgb.width,'height':rgb.height,'dhash64':f'{dhash:016x}','bytes':len(content)}
            assert all(facts[k]==metadata[relative][k] for k in facts)
            assert pixels not in historical_pixels and not tree.near(dhash,4)
            checked[relative]=facts
        facts=checked[relative]
        item={**row,'image_path':relative,'width':facts['width'],'height':facts['height'],
            'repair_holdout':True,'independent_confirmation':False,
            'holdout_scope':'fresh single-image public-label follow-up, not original ARP confirmation'}
        if item['answer_type']=='option_letter_v2':item['answer_type']='option_letter'
        item=native_prompt(item)
        batch=public_prompt_inputs(processor,item)
        grid=batch['image_grid_thw'][0].tolist();patch=processor.image_processor.patch_size
        item.update(processor_grid_thw=grid,processor_patch_size=patch,
            processor_width=grid[2]*patch,processor_height=grid[1]*patch,
            frozen_prompt_tokens=int(batch['input_ids'].shape[1]))
        result.append(item)
    path=OUT/'heldout-manifest.jsonl';write_rows(path,result)
    proof=OUT/'heldout-selection.json'
    write_json(proof,{'timestamp_utc':now(),'original_units_per_axis':120,'prompt_rows':960,
        'eligible_units':counts,'single_image_and_AtoD_choice_subset_explicit':True,
        'reasoning_axis_is_MMMU_proxy_not_MMStar':True,'selection_rule':'existing fixed source stratified deterministic selection; no outcome filtering',
        'all_selected_image_bytes_and_RGB_verified':True,'historical_exact_or_dhash_radius4_matches':0,
        'transitive_isolation_inherited_from_verified_clean_v4':True,
        'prior_inference_membership_scan':scanned,'selected_images':checked,
        'original_ARP_human_grounding_review_complete':False,'original_ARP_training_allowed':False})
    return [path,proof]


def freeze_protocol():
    state=check_data()
    old=read_json(OLD_EVAL/'protocol.json')
    for name,digest in old['source_sha256'].items():assert sha(ROOT/'tools'/name)==digest,('Frozen v4.1 source changed',name)
    assert read_json(OUT/'heldout-novelty-verification.json')['status']=='passed'
    path=OUT/'protocol.json';sources={name:sha(ROOT/'tools'/name) for name in SOURCE_FILES}
    if path.exists():
        plan=read_json(path);assert plan['source_sha256']==sources
        assert plan['data_preparation_sha256']==sha(OUT/'data-preparation.json')
        assert plan['environment_lock_sha256']==sha(OUT/'environment-lock.json')
        assert plan['heldout_novelty_sha256']==sha(OUT/'heldout-novelty-verification.json')
        return plan
    config=read_json(ROOT/'configs/route_b_9b_122b.plan.json')
    plan={'schema':'independent.9b.u137.factorial.v1','timestamp_utc':now(),
        'authorization':'User2026-10-18: implement the U137 diagnosis and 2x2 repair plan',
        'source_sha256':sources,'data_preparation_sha256':sha(OUT/'data-preparation.json'),
        'environment_lock_sha256':sha(OUT/'environment-lock.json'),
        'heldout_novelty_sha256':sha(OUT/'heldout-novelty-verification.json'),
        'student':config['student'],'revision':config['revision'],'training':config['training'],
        'analysis_packages':{n:importlib.metadata.version(n) for n in ('numpy','matplotlib')},
        'conditions':list(CONDITIONS),'seeds':list(SEEDS),'new_training_runs':6,
        'updates_per_run':1000,'reuse_historical_pixel_dual_CE_runs':2,
        'sampling_identical_within_seed':True,'GT_only_loss_scale':1.,
        'normalized_representation':'Apply v/axis_size*1000 to frozen legacy GT/teacher pixel targets; same json.dumps, no added rounding, clipping or label filtering. Tokenization/length may change as part of representation intervention.',
        'teacher_cache_only_no_API':True,'old16_results_and_budgets_unchanged':True,
        'wait_for_old16_complete_verified_delivered':True,
        'training_wall_seconds_cap':6*3600,'evaluation_wall_seconds_cap':24*3600,
        'budget_clock_starts_after_old_queue_and_GPU_lock':True,'never_reset_on_restart':True,
        'price_CNY_per_hour':2.7,'nominal_phase_cap_CNY':81.,
        'reserved_bytes_cap':30000000000,'minimum_free_bytes':5*1024**3,
        'evaluation':{'development_rows_per_new_model':16900,'new_development_models':6,
            'heldout_rows_per_model':960,'heldout_models':9,
            'total_new_prediction_cap':6*16900+9*960,'eos_token_ids':[248046,248044],
            'max_new_tokens':128,'thinking':False,
            'primary_native_coordinate_contract_unchanged':True,
            'failures_count_as_outcomes_no_accuracy_based_restarts':True},
        'resume_test':'P-G seed42: save/exit50; new process resumes51..1000; no extra updates',
        'endpoints':'fixed last1000, no evaluation-based checkpoint choice',
        'statistics':{'replicates':10000,'seed':2026101804,'comparisons':'coordinate main effect, teacher main effect, interaction x4 axes x2 sets','family_size':24,
            'two_seeds_exploratory_not_five_seed_confirmation':True},
        'stops':['hash mismatch','nonfinite or zero train gradient','reserved>30GB','disk<5GiB','input/target training cap exceeded','duplicate/coverage error','absolute budget deadline'],
        'scope':'Post-result mechanism follow-up, not a correction of U137 score or original ARP gate; heldout is restricted single-image public-label subset; no promised accuracy recovery'}
    write_json(path,plan)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['data','freeze','check']);args=parser.parse_args()
    result=prepare_data() if args.action=='data' else (freeze_protocol() if args.action=='freeze' else check_data())
    print(json.dumps({k:v for k,v in result.items() if k in ('status','schema','timestamp_utc','new_training_runs')}),flush=True)
