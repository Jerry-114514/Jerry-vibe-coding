"""Register user-requested reporting contrasts before any formal evaluation."""
from datetime import datetime, timezone
from experiment_common import ROOT, write_json, read_json, sha

out = ROOT/'runs/route-b-staged-7b-20261016'
path = out/'requested-report-plan.json'
content = {'contrasts':['A1-A2','A1-U','A1-N'],'axes':['pope','grounding','consistency','mmstar'],
           'interval':'Pointwise descriptive95% paired seed x source-image-cluster bootstrap',
           'bootstrap_replicates':2000,'unmatched_evaluations':'pending, never imputed',
           'single_seed_limit':'No estimate of training seed uncertainty with onlyone matched seed',
           'minimal_loop_results_already_observed':True,'formal_evaluation_results_already_observed':False,
           'extra_training_budget':0,'scope':'Independent exploratory experiment, notARP confirmation',
           'observer_source_sha256':sha(ROOT/'tools/route_b_completion_reports.py')}
if path.exists():
    previous = read_json(path)
    assert all(previous.get(k)==v for k,v in content.items())
else:
    content['registered_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(path,content)
print('Report plan SHA256:',sha(path))
