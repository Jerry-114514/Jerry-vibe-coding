"""Record the authorized scheduling override without changing experiment code."""
from datetime import datetime, timezone
import shutil
from experiment_common import ROOT, read_json, write_json

old = ROOT/'runs/route-b-7b-20261016'
new = ROOT/'runs/route-b-staged-7b-20261016'
new.mkdir(parents=True,exist_ok=True)
if (old/'status.json').exists() and not (old/'status.before-staged.json').exists():
    shutil.copyfile(old/'status.json',old/'status.before-staged.json')
write_json(old/'status.json',{'phase':'superseded_by_user_stage1_priority',
           'replacement':str(new.relative_to(ROOT)),'original_gpu_worker_stopped':True,
           'completed_cache_files_retained':1162,'timestamp':datetime.now(timezone.utc).isoformat()})
shutil.copyfile(old/'migration-verification.json',new/'migration-verification.json')
shutil.copyfile(old/'cache-audit-complete.json',new/'legacy-cache-audit-complete.json')
write_json(new/'resource-override.json',{'authorized_by_user':True,'training_gpu':'RTX5090',
           'evaluation_gpu':'RTX5090','original4090':'released','hourly_CNY':2.70,
           'evaluation_gpu_hours_cap':30,'evaluation_CNY_cap':81,
           'stage_order':['minimal500_cache_train_eval_acceptance','full3545_cache_and_freeze','15trainings','5090eval_and_report']})
print('Historical run superseded; migration evidence attached to staged run')
