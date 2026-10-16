"""Compact read-only snapshot; no credentials or raw training images printed."""
from collections import Counter
import json
from pathlib import Path
import subprocess
from experiment_common import ROOT, read_json, read_rows

out = ROOT / 'runs/route-b-7b-20261016'
for filename in ['status.json', 'failure.json', 'minimal-acceptance.json']:
    path = out / filename
    if path.exists():
        print(filename, json.dumps(read_json(path), ensure_ascii=False))
log = out / 'console.log'
if log.exists():
    print('console_tail', log.read_text(errors='replace')[-3500:])
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
cache_files = list((out / 'teacher-cache-bf16-eos').glob('*.pt'))
if len(cache_files) > 2:
    stamps = sorted(p.stat().st_mtime for p in cache_files)
    interval = (stamps[-1] - stamps[0]) / (len(stamps) - 1)
    print('cache_throughput', {'files': len(stamps), 'mean_seconds_per_row': interval,
          'remaining_seconds_at_current_mix': (3545 - len(stamps))*interval})
for label, folder in [('local7b', 'runs/protocol-v2'), ('api', 'runs/protocol-v2')]:
    if label == 'api':
        break
    candidates = [p for p in (ROOT / folder).rglob('*.jsonl') if p.name == 'ledger.jsonl' and
                  ('qwen7b' in p.parts or any('api-qwen3-vl-plus' in x for x in p.parts))]
    for p in candidates[:20]:
        try:
            rows = read_rows(p)
            seconds = [r.get('elapsed_seconds', r.get('seconds')) for r in rows]
            seconds = [x for x in seconds if isinstance(x, (float, int))]
            print('ledger', str(p.relative_to(ROOT)), len(rows),
                  'mean_seconds', sum(seconds)/len(seconds) if seconds else None)
            if rows and isinstance(rows[0].get('usage'), dict):
                print('average_usage', {k: sum(r.get('usage', {}).get(k, 0) for r in rows)/len(rows)
                      for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
        except Exception as e:
            print(type(e).__name__, str(p.relative_to(ROOT)))
