"""Copy only verified final small reports/figures locally; raw data stay remote."""
import base64
import hashlib
import json
from pathlib import Path
import shlex

import autodl_new_instance as connection

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'docs/results/protocol-v2-20261016/route-b-final'
OUT = 'runs/route-b-staged-7b-20261016'
FILES = {
    **{name: f'{OUT}/completion-reports/final-figures/{name}' for name in (
        'final-summary.md', 'four-axis-scores.png', 'four-axis-scores.pdf',
        'paired-intervals.png', 'paired-intervals.pdf', 'figure-data.json', 'delivered.json')},
    **{name: f'{OUT}/completion-reports/multiplicity/{name}' for name in (
        'multiplicity-results.json', 'multiplicity-report.md', 'reporting-amendment.json')},
    **{name: f'{OUT}/completion-reports/evaluation/{name}' for name in (
        'evaluation-results.json', 'evaluation-report.md')},
    'content-verification.json': f'{OUT}/objective-content-verification.json',
    'delivery-verification.json': f'{OUT}/final-delivery-verification.json',
    'finalization.json': f'{OUT}/finalization/complete.json',
    'evaluation-verification.json': f'{OUT}/evaluation-aggregate-independent-verification.json',
    **{f'{name}-delivery.json': f'{OUT}/evaluation-delivery-checks/{name}/complete.json'
       for name in ('A1-seed1024', 'A2-seed1024', 'U-seed1024')},
}


def collect():
    code = '''import base64, hashlib, json
from pathlib import Path
root = Path('/root/autodl-tmp/arp')
out = root / %r
assert json.loads((out/'finalization/complete.json').read_text())['status'] == 'full_7b_content_and_delivery_verified'
assert json.loads((out/'completion-reports/final-figures/delivered.json').read_text())['status'] == 'uploaded_and_fresh_server_verified'
records = {}
for name, relative in %r.items():
    data = (root / relative).read_bytes()
    assert len(data) < 8*1024**2
    records[name] = {'remote_path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'base64': base64.b64encode(data).decode()}
assert sum(record['bytes'] for record in records.values()) < 32*1024**2
print(json.dumps(records))
''' % (OUT, FILES)
    result = connection.transport.run('/root/autodl-tmp/arp-venv/bin/python -c ' + shlex.quote(code),
                                      capture_output=True)
    records = json.loads(result.stdout)
    assert set(records) == set(FILES)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, record in records.items():
        data = base64.b64decode(record['base64'], validate=True)
        assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
        (DESTINATION / name).write_bytes(data)
        manifest[name] = {k: v for k, v in record.items() if k != 'base64'}
    (DESTINATION / 'local-copy-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'verified_small_reports_copied', 'files': len(manifest),
                      'bytes': sum(r['bytes'] for r in manifest.values()), 'destination': str(DESTINATION)}))


if __name__ == '__main__':
    collect()
