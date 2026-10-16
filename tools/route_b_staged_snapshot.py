"""Small, read-only snapshot of the active four-stage route-B run."""
import json
import subprocess
from experiment_common import ROOT, read_json
from experiment_common import read_rows

out = ROOT/'runs/route-b-staged-7b-20261016'
for name in ('status.json','failure.json','minimal-acceptance.json','stage3-complete.json'):
    p = out/name
    if p.exists():
        print(name, json.dumps(read_json(p),ensure_ascii=False))
for p in sorted((out/'minimal').glob('*/scores.json')):
    print(str(p.parent.relative_to(out)),json.dumps(read_json(p),ensure_ascii=False))
for p in sorted((out/'minimal').glob('*/eval.jsonl')):
    rows = read_rows(p)
    print('eval_partial',p.parent.name,'rows',len(rows),'mean_seconds',sum(r['seconds'] for r in rows)/len(rows) if rows else None,
          'last_seconds',rows[-1]['seconds'] if rows else None)
log = out/'console.log'
if log.exists():
    print('console_tail',log.read_text(errors='replace')[-2000:])
print(subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip())
