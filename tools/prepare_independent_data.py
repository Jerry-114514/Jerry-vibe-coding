"""Acquire public independent-evaluation sources on AutoDL with official LFS hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import requests
from experiment_common import ROOT, read_json, sha, write_json

OUT = ROOT / 'runs/protocol-v2-data/source-download'
DATA = ROOT / 'data/protocol-v2/raw'
MUTEX = threading.Lock()


def fetch_file(dataset, entry, progress):
    destination = DATA / dataset['repo'].replace('/', '--') / entry['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == entry['bytes'] and sha(destination) == entry['sha256']:
            return {'repo': dataset['repo'], **entry, 'local_path': str(destination.relative_to(ROOT)), 'verified': True}
        raise ValueError('Existing final source file fails the frozen digest')
    partial = destination.with_suffix(destination.suffix + '.part')
    if partial.exists() and partial.stat().st_size == entry['bytes'] and sha(partial) == entry['sha256']:
        partial.replace(destination)
        return {'repo': dataset['repo'], **entry, 'local_path': str(destination.relative_to(ROOT)), 'verified': True}
    for attempt in range(6):
        host = 'https://hf-mirror.com' if attempt % 2 == 0 else 'https://huggingface.co'
        url = host + '/datasets/' + dataset['repo'] + '/resolve/' + dataset['revision'] + '/' + entry['path']
        if shutil.disk_usage(DATA).free < 8 * 1024**3:
            raise RuntimeError('Source download stopped below eight GiB free disk')
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(url, headers={'Range': 'bytes=' + str(offset) + '-'} if offset else {},
                              params={'download': 'true', 'arp_download': str(int(time.time()))},
                              stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                resume = offset > 0 and response.status_code == 206
                if resume and not response.headers.get('Content-Range', '').startswith('bytes ' + str(offset) + '-'):
                    raise ValueError('Server returned inconsistent range')
                with partial.open('ab' if resume else 'wb') as stream:
                    last = time.monotonic()
                    for block in response.iter_content(4 * 1024**2):
                        if not block:
                            continue
                        stream.write(block)
                        if stream.tell() > entry['bytes']:
                            raise ValueError('Downloaded source exceeds official size')
                        if time.monotonic() - last > 15:
                            progress()
                            last = time.monotonic()
            if partial.stat().st_size != entry['bytes'] or sha(partial) != entry['sha256']:
                raise ValueError('Source bytes do not match official pinned LFS digest')
            partial.replace(destination)
            return {'repo': dataset['repo'], **entry, 'local_path': str(destination.relative_to(ROOT)),
                    'verified': True, 'download_host': host}
        except requests.RequestException:
            if attempt == 5:
                raise RuntimeError('Public source network failed after bounded retries') from None
            time.sleep(min(2**attempt, 10))
    raise RuntimeError('Source download did not terminate')


def worker():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    with (OUT / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read_json(ROOT / 'configs/independent_data_sources.json')
        total = sum(entry['bytes'] for dataset in config['datasets'] for entry in dataset['files'])
        if total > 5 * 1024**3:
            raise ValueError('Frozen source-download disk budget exceeded')
        state = {'status': 'running', 'worker_script': Path(__file__).name, 'pid': os.getpid(),
                 'scope': 'Public raw source acquisition only; independent manifests are a later stage',
                 'expected_files': sum(len(dataset['files']) for dataset in config['datasets']),
                 'expected_bytes': total, 'completed_files': 0, 'completed_bytes': 0,
                 'started_utc': datetime.now(timezone.utc).isoformat(), 'formal_training_allowed': False}
        inventory = []
        failures = []

        def progress(entry=None):
            with MUTEX:
                if entry:
                    inventory.append(entry)
                state.update(completed_files=len(inventory), completed_bytes=sum(row['bytes'] for row in inventory),
                             failed_files=len(failures),
                             updated_utc=datetime.now(timezone.utc).isoformat())
                write_json(OUT / 'status.json', state)
                if entry:
                    print('Verified source', len(inventory), '/', state['expected_files'], entry['repo'], entry['path'], flush=True)

        progress()
        try:
            jobs = [(dataset, entry) for dataset in config['datasets'] for entry in dataset['files']]

            def one(dataset, entry):
                try:
                    result = fetch_file(dataset, entry, progress)
                    progress(result)
                    return result
                except Exception as error:
                    with MUTEX:
                        failures.append({'repo': dataset['repo'], 'path': entry['path'], 'error_type': type(error).__name__})
                        write_json(OUT / 'file-failures.json', failures)
                    progress()
                    return None

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(one, dataset, entry) for dataset, entry in jobs]
                for future in as_completed(futures):
                    future.result()
            write_json(OUT / 'verified-source-files.json', sorted(inventory, key=lambda row: (row['repo'], row['path'])))
            if failures:
                raise RuntimeError('Some sources failed; preserve verified files for checksum-safe resume')
            write_json(OUT / 'sources.lock.json', config)
            state.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                         verified_inventory_sha256=sha(OUT / 'verified-source-files.json'))
            write_json(OUT / 'status.json', state)
            write_json(OUT / 'complete.json', state)
        except Exception as error:
            state.update(status='failed', error_type=type(error).__name__, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(OUT / 'status.json', state)
            raise
        if os.environ.get('GH_TOKEN'):
            from publish_artifacts import publish
            try:
                publish(OUT, 'arp-protocol-v2-20261016', token=os.environ['GH_TOKEN'])
            except Exception:
                write_json(OUT / 'backup-failed.json', {'status': 'failed', 'source_acquisition_complete': True})
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker()
    else:
        token = sys.stdin.readline().strip()
        if not token:
            raise SystemExit('Completion-backup credential required on stdin')
        sessions = subprocess.run(['screen', '-ls'], capture_output=True, text=True).stdout
        if '.arp-v2-data-sources' in sessions:
            raise SystemExit('Source-download worker already exists')
        OUT.mkdir(parents=True, exist_ok=True)
        command = ('source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; '
                   'exec /root/autodl-tmp/arp-venv/bin/python -u tools/prepare_independent_data.py --worker >> ' +
                   str(OUT / 'console.log') + ' 2>&1')
        subprocess.run(['screen', '-dmS', 'arp-v2-data-sources', 'bash', '-c', command],
                       env={**os.environ, 'GH_TOKEN': token}, check=True)
        print('Started independent public-source download with official revision/hash verification', flush=True)
