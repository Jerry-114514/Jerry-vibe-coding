"""Freeze only the verified P1-4 numerical recipe; never grant stage-C approval."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from experiment_common import ROOT,read_json,read_rows,sha,write_json


def main():
    run=ROOT/'runs/protocol-v2/calibration/p14-full-7b-5090-v1'
    original=ROOT/'runs/protocol-v2/calibration/p14-full-7b-v2'
    done=read_json(run/'complete.json');protocol=read_json(run/'protocol.json')
    if read_json(run/'verification.json')['status']!='passed' or done['peak_reserved_bytes']>26000000000:
        raise ValueError('User26GB hardware acceptance failed')
    if done['loss_ledger_sha256']!=sha(run/'loss.jsonl') or done['teacher_ledger_sha256']!=sha(run/'teacher.jsonl'):
        raise ValueError('Preflight artifacts changed')
    prior=read_json(original/'protocol.json')
    keys=['student','teacher','steps','seed','lora','loss','optimizer','bf16','gradient_checkpointing','use_cache',
          'micro_batch','gradient_accumulation','max_total_tokens','max_visual_tokens','max_response_tokens','grounding','pool_sha256','model_revision']
    if any(protocol[k]!=prior[k] for k in keys):raise ValueError('5090 recipe differs from4090')
    if sha(run/'manifest.jsonl')!=sha(original/'manifest.jsonl'):raise ValueError('Different preflight units')
    times=subprocess.run(['stat','--printf','%w\n%y\n',str(run/'loss.jsonl')],capture_output=True,text=True,check=True).stdout.splitlines()
    birth=datetime.fromisoformat(times[0]);last=datetime.fromisoformat(times[1])
    steady=(last-birth).total_seconds()/99
    conservative=done['elapsed_training_seconds']/100
    gate=read_json(ROOT/'runs/protocol-v2/paired-gate-development/gate.json')
    local7=ROOT/'runs/protocol-v2/audit/qwen-clean-v2/qwen7b/ledger.jsonl'
    api=ROOT/'runs/protocol-v2/audit/api-qwen3-vl-plus-v1/ledger.jsonl'
    teacher_pool=read_rows(ROOT/'data/training/pool.jsonl')
    recipe={'schema':'arp.p14.7b.5090.frozen-numerical-recipe.v1',
            'freeze_scope':'Numerical/training recipe only; stageC approval, arm axes/manifests and main runner remain separate prerequisites',
            'student':{'id':'Qwen/Qwen2.5-VL-7B-Instruct','revision':protocol['model_revision']},
            'teacher':{'model':'qwen3.5-122b-a10b','method':'offline sequence-response CE','immutable_API_revision_confirmed':False},
            **{k:protocol[k] for k in ['lora','loss','optimizer','bf16','gradient_checkpointing','use_cache','micro_batch','gradient_accumulation','max_total_tokens','max_visual_tokens','max_response_tokens','grounding']},
            'steps_per_arm':1000,'seeds':[42,137,256,512,1024],'arms':['A1','A2','U'],'trained_runs':15,
            'baseline_N':'Untrained7B, no repeated training',
            'training_pool_sha256':protocol['pool_sha256'],'training_pool_prompts':len(teacher_pool),
            'hardware':{'gpu':done['gpu_name'],'preflight_reserved_bytes':done['peak_reserved_bytes'],
                        'user_maximum_reserved_bytes':26000000000,'passed':True,
                        'nominal32GB_headroom_bytes':32000000000-done['peak_reserved_bytes'],
                        'CUDA_visible_headroom_bytes':done['headroom_bytes']},
            'preflight_complete_sha256':sha(run/'complete.json'),'preflight_protocol_sha256':sha(run/'protocol.json'),
            'preflight_manifest_sha256':sha(run/'manifest.jsonl')}
    path=ROOT/'configs/p14_7b_5090.frozen.json'
    encoded=(json.dumps(recipe,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    if path.exists() and path.read_bytes()!=encoded:raise ValueError('Do not overwrite a different frozen recipe')
    path.write_bytes(encoded)
    digest=hashlib.sha256(encoded).hexdigest()
    timing={'method':'(last loss-file write minus first creation)/99, steps2..100; includes CPU input preparation and per-step logging',
            'file_birth':times[0],'file_last_write':times[1],'mean_seconds_per_step':steady,
            'conservative_seconds_per_step':conservative,
            'conservative_note':'Whole measured100-update block includes one synthetic stress update and adapter checks/save',
            'per1000_step_minutes':[steady*1000/60,conservative*1000/60],
            'fifteen_run_hours':[steady*15000/3600,conservative*15000/3600],
            'planning_hours_including_load_save':[3,4],
            'quote_CNY_per_day_user_provided':64.77,'requested_cost_CNY_per_hour':2.70,
            'fifteen_run_compute_cost_CNY':[steady*15000/3600*2.70,conservative*15000/3600*2.70],
            'planning_GPU_cost_CNY':[8.1,10.8],
            'billing_note':'Hourly resource estimate; if billing is a daily package, budget64.77CNY for one day; not confirmation of billing granularity',
            'excludes':['122B main teacher API generation','4090 evaluation','paid storage'],
            'uncertainty':'Balanced100-item sample; A1/A2 single-axis length distributions may differ. Re-estimate from first20 scheduled steps without changing step/seed budget'}
    readiness={'checked_utc':datetime.now(timezone.utc).isoformat(),'frozen_config_path':str(path.relative_to(ROOT)),
               'frozen_config_sha256':digest,'hardware_and_numerical_recipe_frozen':True,'formal_training_allowed':False,
               'current_C_decision':gate['decision'],'C_is_formal':gate['formal'],
               'local7b_audit_prompts':len(read_rows(local7)),'audit_expected_prompts_per_model':9600,
               'commercial_audit_prompts':len(read_rows(api)),
               'same_checkpoint_API_bridge_passed':False,'grounding_blind_review_complete':False,
               'main_teacher_cache_complete':False,'cached_teacher_prompts':100,'teacher_pool_prompts':len(teacher_pool),
               'axis_arm_manifests_frozen':False,'formal7b_dualCE_runner_implemented_and_accepted':False,
               'independent_distillation_possible':'Technically yes after separate preregistration and preparation; cannot replace original ARP C-dependent axis-selection claim',
               'original_experiments_resumed':False}
    write_json(run/'timing-and-budget.json',timing);write_json(run/'main-readiness.json',readiness)
    write_json(run/'configuration-freeze.json',{'path':str(path.relative_to(ROOT)),'sha256':digest,'GB26_memory_gate_passed':True,
               'same_configuration_as_4090_verified':True,'frozen_config_bytes':len(encoded),'formal_training_allowed':False})
    (run/'frozen-config.sha256').write_text(digest+'  configs/p14_7b_5090.frozen.json\n')
    print(json.dumps({'freeze_sha256':digest,'timing':timing,'readiness':readiness},ensure_ascii=False))


if __name__=='__main__':main()
