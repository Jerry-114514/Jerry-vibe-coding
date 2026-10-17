"""Direct AutoDL-to-Release archive on eligible teacher completion or failure.

Exports teacher records and diagnostics only; never weights/7B/9B checkpoints.
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time

from experiment_common import ROOT, read_json, write_json
from publish_artifacts import publish

BASE = ROOT / 'runs/route-b-9b-122b-20261017'


def watch():
    import fcntl
    with (BASE / 'teacher-archive-worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 12 * 3600
        while time.monotonic() < deadline:
            complete = (BASE / 'teacher-eligible-complete.json').exists()
            stopped = (BASE / 'resume-v3-stopped.json').exists()
            if complete or stopped:
                if (BASE / 'teacher-delivery.json').exists():
                    return
                assert shutil.disk_usage(BASE).free >= 5 * 1024 ** 3, 'Disk stop before teacher archive'
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                target = ROOT / 'runs/9b-teacher-delivery-20261017' / stamp
                source_names = ('teacher-protocol.json', 'teacher.jsonl', 'teacher-status.json',
                                'teacher-api-failure.json', 'teacher-stopped.json', 'teacher-api-inflight.json',
                                'resume-v2-stopped.json', 'resume-v3-stopped.json',
                                'eligible-pool.jsonl', 'quarantined-pool.jsonl', 'teacher-eligible-complete.json')
                sources = [BASE / name for name in source_names if (BASE / name).is_file()]
                for directory in ('resume-v2', 'resume-v3', 'api400-investigation'):
                    sources += [p for p in (BASE / directory).rglob('*')
                                if p.is_file() and p.suffix in ('.json', '.jsonl')]
                for source in sources:
                    if source.name in ('github-inventory.json', 'github-storage.json', 'teacher-delivery.json'):
                        continue
                    destination = target / 'files' / source.relative_to(ROOT)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # Completed raw records are immutable. Hard links avoid an
                    # extra image-bearing request copy on the small data disk.
                    os.link(source, destination)
                write_json(target / 'scope.json', {'eligible_cache_complete': complete,
                           'failure_observed': stopped, 'source': str(BASE.relative_to(ROOT)),
                           'no_base_model_or_optimizer_files': True,
                           'note': 'Provider-refused source groups are quarantined; no claim of all3545 teacher responses'})
                receipt = publish(target, 'arp-9b-122b-20261017', token=os.environ['GH_TOKEN'])
                write_json(BASE / 'teacher-delivery.json', {**receipt, 'snapshot': str(target.relative_to(ROOT)),
                           'eligible_cache_complete': complete})
                return
            time.sleep(30)
        raise RuntimeError('Teacher archive deadline reached; inspect live cache process')


if __name__ == '__main__':
    watch()
