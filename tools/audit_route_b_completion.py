"""Full objective completion audit. Never equate a partial run with success."""
import argparse
import json
import math
from pathlib import Path
import subprocess
from PIL import Image
from experiment_common import ROOT, AXES, read_json, sha, write_json
from route_b_staged import OUT, CACHE, SEEDS


def readiness():
    required = [OUT/'minimal-independent-verification.json',OUT/'full-eos-cache-independent-verification.json',
                OUT/'eval-input-independent-preflight.json',OUT/'stage3-complete.json',
                OUT/'formal-independent-verification.json',OUT/'formal-paired-analysis.json',
                OUT/'completion-reports/training-report-delivered.json',OUT/'completion-reports/final-report-delivered.json',
                OUT/'completion-reports/multiplicity/multiplicity-results.json',
                OUT/'completion-reports/multiplicity/delivered.json',
                ROOT/'runs/qwen35-9b-infrastructure-20261016/infrastructure-complete.json']
    required += [OUT/'formal'/f'{arm}-seed{seed}'/name for seed in SEEDS for arm in ('A1','A2','U')
                 for name in ('complete.json','resume.pt','adapter/adapter_model.safetensors','scores.json')]
    required += [OUT/'formal/N-seed42/scores.json']
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    return {'status':'ready_for_full_audit' if not missing else 'incomplete','missing':missing}


def existing_code_unchanged():
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','98f61ae','tools'],cwd=ROOT,text=True).splitlines()
    checked=[]
    for name in paths:
        if not name.endswith('.py'):continue
        original=subprocess.check_output(['git','show','98f61ae:'+name],cwd=ROOT)
        actual=(ROOT/name).read_bytes()
        assert original.replace(b'\r\n',b'\n')==actual.replace(b'\r\n',b'\n'),'Existing experiment source changed: '+name
        checked.append(name)
    return checked


def verify():
    ready=readiness()
    if ready['missing']:return ready
    from audit_route_b_formal_inputs import cache_audit,training_audit
    from verify_route_b_stage_outputs import verify as verify_outputs
    cache_audit()
    for seed in SEEDS:
        for arm in ('A1','A2','U'):
            training_audit(arm,seed)
    verify_outputs('formal')
    from audit_route_b_evaluation_scores import audit as verify_evaluation_aggregates
    verify_evaluation_aggregates(require_all=True)
    expected={f'{a}-seed{s}' for s in SEEDS for a in ('A1','A2','U')}|{'N-seed42'}
    actual={p.name for p in (OUT/'formal').iterdir() if p.is_dir()}
    assert actual==expected
    assert not (OUT/'formal/N-seed42/resume.pt').exists() and not (OUT/'formal/N-seed42/adapter').exists()
    assert read_json(OUT/'status.json')['phase']=='all_four_stages_complete'
    assert read_json(OUT/'stage3-complete.json')['updates']==15000
    assert read_json(OUT/'minimal-independent-verification.json')['status']=='passed'
    assert read_json(OUT/'eval-input-independent-preflight.json')['status']=='passed'
    eval_folder=OUT/'completion-reports/evaluation'
    result=read_json(eval_folder/'evaluation-results.json')
    assert result['status']=='all_evaluations_complete' and result['completed_models']==16
    assert result['processed_prompts']==270400 and result['remaining_estimated_hours']==0
    assert set(result['paired_contrasts'])=={'A1-A2','A1-U','A1-N'}
    for label,contrast in result['paired_contrasts'].items():
        assert contrast['status']=='available' and contrast['matched_seeds']==SEEDS
        assert contrast['training_seed_uncertainty_included'] and set(contrast['axes'])==set(AXES)
        left,right=label.split('-')
        for axis,value in contrast['axes'].items():
            expected_delta=sum(result['scores'][f'{left}-seed{s}']['axis_scores'][axis]-
                    result['scores'][f'{right}-seed{42 if right=="N" else s}']['axis_scores'][axis] for s in SEEDS)/5
            assert abs(value['difference']-expected_delta)<1e-12
            lo,hi=value['ci95']
            assert all(math.isfinite(v) for v in (lo,hi,value['difference'])) and -1<=lo<=hi<=1
    plan=read_json(OUT/'requested-report-plan.json')
    assert plan['observer_source_sha256']==sha(ROOT/'tools/route_b_completion_reports.py')
    assert plan['bootstrap_replicates']==2000
    corrected_folder=OUT/'completion-reports/multiplicity'
    corrected=read_json(corrected_folder/'multiplicity-results.json')
    correction_plan=read_json(corrected_folder/'reporting-amendment.json')
    assert corrected['status']=='all16_evaluations_complete' and len(corrected['scores'])==16
    assert corrected['family_size']==12 and corrected['bootstrap_replicates']==10000
    assert corrected['matched_seeds']==SEEDS and set(corrected['paired_contrasts'])=={'A1-A2','A1-U','A1-N'}
    assert correction_plan['source_sha256']==sha(ROOT/'tools/report_route_b_multiplicity.py')
    assert corrected['amendment_sha256']==sha(corrected_folder/'reporting-amendment.json')
    for label,axes in corrected['paired_contrasts'].items():
        assert set(axes)==set(AXES)
        left,right=label.split('-')
        for axis,value in axes.items():
            expected_delta=sum(corrected['scores'][f'{left}-seed{s}'][axis]-
                    corrected['scores'][f'{right}-seed{42 if right=="N" else s}'][axis] for s in SEEDS)/5
            assert abs(value['difference']-expected_delta)<1e-12
            lo,hi=value['pointwise_ci95']
            adjusted_lo,adjusted_hi=value['bonferroni_familywise_ci95']
            assert all(math.isfinite(v) for v in (lo,hi,adjusted_lo,adjusted_hi))
            assert -1<=adjusted_lo<=lo<=hi<=adjusted_hi<=1
    with Image.open(OUT/'completion-reports/training/loss-curves.png') as image:
        image.verify()
    nine=ROOT/'runs/qwen35-9b-infrastructure-20261016'
    smoke=read_json(nine/'gpu-smoke.json')
    assert smoke['status']=='passed' and smoke['finite_vision_logits']
    assert smoke['revision']==read_json(nine/'model.lock.json')['revision']
    assert read_json(nine/'environment-ready.json')['old_environment_unchanged']
    original_gate=read_json(ROOT/'runs/protocol-v2/paired-gate-development/gate.json')
    assert original_gate['training_allowed'] is False
    proof={'status':'passed','scope':'Full independent routeB objective, not originalARP stageC',
           'training_runs':15,'updates_per_run':1000,'evaluation_models':16,'prompts_per_model':16900,
           'requested_paired_contrasts':['A1-A2','A1-U','A1-N'],'all_five_seeds_verified':True,
           'multiplicity_family_size':12,'multiplicity_correction':'Bonferroni percentile bootstrap, nominal approximate coverage',
           'full_eos_rows':3545,'9b_gpu_smoke_passed':True,'original_arp_gate_unpassed':True,
           'existing_python_sources_unchanged':existing_code_unchanged(),
           'formal_config_sha256':sha(OUT/'phase2-frozen.json'),
           'github_delivery_audit_required_separately':True}
    write_json(OUT/'objective-content-verification.json',proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--source-only',action='store_true');a=p.parse_args()
    value={'status':'passed','unchanged_existing_python_sources':len(existing_code_unchanged())} if a.source_only else verify() if a.verify else readiness()
    print(json.dumps(value,ensure_ascii=False),flush=True)
