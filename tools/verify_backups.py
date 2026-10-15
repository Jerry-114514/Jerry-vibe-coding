"""Compare completed experiment artifacts with fresh GitHub release inventories."""
import hashlib
import json
from pathlib import Path
import sys
import requests
from experiment_common import ROOT, read_json, sha, write_json

token=sys.stdin.readline().strip()
if not token:
    raise RuntimeError('GitHub credential required on stdin')
session=requests.Session()
session.trust_env=False
session.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
response=session.get('https://api.github.com/repos/Jerry-114514/Jerry-vibe-coding/releases/tags/arp-formal-20261014',timeout=60)
response.raise_for_status()
release=response.json()
assets={row['name']:row for row in release['assets']}
reports=list((ROOT/'runs/github-snapshots').glob('*/github-storage.json'))
reports+=list((ROOT/'runs/bc-verification-20261015').glob('github-storage.json'))
digests=set()
verified=[]
for report in reports:
    for record in read_json(report)['assets']:
        asset=assets.get(record['name'])
        if not asset or asset['size']!=record['bytes']:
            raise ValueError('GitHub asset absent or incomplete: '+record['name'])
        if asset.get('digest')!='sha256:'+record['sha256']:
            raise ValueError('GitHub server digest absent or mismatched: '+record['name'])
        verified.append(record['name'])
        if record['name'].endswith('-inventory.json'):
            response=session.get(asset['url'],headers={'Accept':'application/octet-stream'},timeout=(20,60))
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest()!=record['sha256']:
                raise ValueError('Downloaded inventory checksum mismatch')
            digests.update(row['sha256'] for row in response.json()['files'])
            print('Verified GitHub inventory:',record['name'],flush=True)
run=ROOT/'runs/formal-20261014'
required=[]
for pattern in ['ledger/*','analysis/*','analysis-cluster/*','paper/*','training/*']:
    required += [p for p in run.glob(pattern) if p.is_file()]
required += [run/'bc-verification.json']
for folder in [run/'teacher-cache',*(run/'arms').glob('*')]:
    if (folder/'complete.json').exists():
        required += [p for p in folder.rglob('*') if p.is_file()]
for marker in (run/'post').glob('*/*.complete.json'):
    ledger=marker.with_suffix('').with_suffix('.jsonl')
    required += [ledger,ledger.with_suffix('.meta.json'),marker]
required += [p for p in (ROOT/'data/training').rglob('*') if p.is_file() and 'raw' not in p.relative_to(ROOT/'data/training').parts]
required += [ROOT/'data/manifest.jsonl',ROOT/'data/experiment.lock.json',ROOT/'data/baselines/per_item.jsonl']
missing=[str(p.relative_to(ROOT)) for p in required if sha(p) not in digests]
result={'status':'passed' if not missing else 'missing','release_url':release['html_url'],
        'verified_github_assets':verified,'required_completed_files':len(required),
        'missing_files':missing,'live_partial_results_excluded':True,
        'verification':'Fresh GitHub asset size and server SHA256 plus downloaded inventory SHA256; completed file contents matched to inventories'}
write_json(run/'github-backup-verification.json',result)
print(json.dumps(result,ensure_ascii=False),flush=True)
