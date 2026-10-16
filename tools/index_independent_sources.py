"""Stream new public sources into a deduplicated, unassigned candidate universe.

This stage preserves multiple images, variable option counts and open answers.
It does not score models, assign confirmation roles, or authorize training.
"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from experiment_common import ROOT, read_json, read_rows, sha, write_json, write_rows

OUT = ROOT / 'runs/protocol-v2-data/candidate-index'
DATA = ROOT / 'data/protocol-v2'


def image_facts(content):
    with Image.open(io.BytesIO(content)) as image:
        rgb = image.convert('RGB')
        pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        grey = np.array(rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
        differences = grey[:, 1:] > grey[:, :-1]
        dhash = sum(int(bit) << index for index, bit in enumerate(differences.flat))
        return {'source_sha256': hashlib.sha256(content).hexdigest(), 'pixel_sha256': pixels,
                'width': rgb.width, 'height': rgb.height, 'dhash64': f'{dhash:016x}', 'bytes': len(content)}


def normalized_options(value):
    options = ast.literal_eval(value) if isinstance(value, str) else value
    if options is None:
        return []
    if not isinstance(options, (list, tuple)) or any(not isinstance(x, str) for x in options):
        raise ValueError('Unexpected public option schema')
    if len(options) > 26:
        raise ValueError('More than 26 choices requires another registered parser')
    return list(options)


def wait_for_sources(state):
    source_status = ROOT / 'runs/protocol-v2-data/source-download/status.json'
    while True:
        status = read_json(source_status)
        if status['status'] == 'complete':
            return
        if status['status'] == 'failed':
            raise RuntimeError('Source acquisition failed; do not index an incomplete universe')
        pid = status.get('pid')
        if not pid or not Path('/proc/' + str(pid)).exists():
            raise RuntimeError('Source worker missing; indexing remains stopped')
        state.update(status='waiting', waiting_for_pid=pid, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(OUT / 'status.json', state)
        time.sleep(20)


def old_image_inventory(progress):
    destination = DATA / 'historical-image-fingerprints.jsonl'
    evaluation = read_rows(ROOT / 'data/manifest.jsonl')
    training = read_rows(ROOT / 'data/training/pool.jsonl')
    paths = sorted({row['image_path'] for row in evaluation + training})
    lock = {'evaluation_manifest_sha256': sha(ROOT / 'data/manifest.jsonl'),
            'training_manifest_sha256': sha(ROOT / 'data/training/pool.jsonl'), 'image_paths': len(paths)}
    meta = destination.with_suffix('.meta.json')
    if destination.exists() and meta.exists():
        prior = read_json(meta)
        if any(prior.get(key) != value for key, value in lock.items()) or prior.get('inventory_sha256') != sha(destination):
            raise ValueError('Historical exclusion inventory changed')
        return read_rows(destination)

    def one(relative):
        return {'path': relative, **image_facts((ROOT / relative).read_bytes())}

    output = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(one, paths):
            output.append(row)
            if len(output) % 200 == 0:
                progress('historical-image-hashes', len(output), len(paths))
    write_rows(destination, output)
    write_json(meta, {**lock, 'inventory_sha256': sha(destination)})
    return output


def build():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    with (OUT / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status': 'running', 'worker_script': Path(__file__).name, 'pid': os.getpid(),
                 'scope': 'Unassigned candidates only; no independent-confirmation claim yet',
                 'started_utc': datetime.now(timezone.utc).isoformat(), 'formal_training_allowed': False}

        def progress(stage, done, total):
            state.update(status='running', stage=stage, completed_items=done, expected_items=total,
                         updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(OUT / 'status.json', state)
            print(stage, done, '/', total, flush=True)

        try:
            wait_for_sources(state)
            config = read_json(ROOT / 'configs/independent_data_sources.json')
            config_sha = sha(ROOT / 'configs/independent_data_sources.json')
            if (OUT / 'complete.json').exists():
                complete = read_json(OUT / 'complete.json')
                if complete['sources_config_sha256'] != config_sha or complete['candidates_sha256'] != sha(DATA / 'candidates.jsonl'):
                    raise ValueError('Completed candidate universe changed')
                print('Candidate universe already verified complete', flush=True)
                return
            old = old_image_inventory(progress)
            old_bytes = {row['source_sha256'] for row in old}
            old_pixels = {row['pixel_sha256'] for row in old}
            candidates, excluded, assets, seen, cache = [], [], {}, set(), {}
            raw_counts = Counter()
            for dataset in config['datasets']:
                ai2d = dataset['repo'].endswith('/ai2d')
                global_row = 0
                for entry in dataset['files']:
                    path = DATA / 'raw' / dataset['repo'].replace('/', '--') / entry['path']
                    if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                        raise ValueError('Raw source no longer matches official digest')
                    parquet = pq.ParquetFile(path)
                    for batch in parquet.iter_batches(batch_size=8):
                        for row in batch.to_pylist():
                            identity = ('ai2d:test:' + str(global_row)) if ai2d else 'mmmu:' + str(row['id'])
                            global_row += 1
                            raw_counts[dataset['repo']] += 1
                            question = row['question']
                            options = normalized_options(row.get('options'))
                            kind = 'multiple-choice' if ai2d else row['question_type']
                            answer = row['answer']
                            if ai2d:
                                if not isinstance(answer, str) or not answer.isdigit() or not 0 <= int(answer) < len(options):
                                    raise ValueError('AI2D answer must be an official zero-based option index')
                                answer = chr(65 + int(answer))
                            elif kind == 'multiple-choice':
                                if answer not in [chr(65+i) for i in range(len(options))]:
                                    raise ValueError('MMMU answer not among variable choices')
                            elif not answer or answer == '?':
                                raise ValueError('MMMU open answer not publicly labeled')
                            images = [row['image']] if ai2d else [row.get('image_' + str(i)) for i in range(1, 8)]
                            images = [image for image in images if image is not None]
                            if not images:
                                raise ValueError('Public question has no images')
                            contents = [image['bytes'] if isinstance(image, dict) else image for image in images]
                            facts = []
                            for content in contents:
                                digest = hashlib.sha256(content).hexdigest()
                                if digest not in cache:
                                    cache[digest] = image_facts(content)
                                facts.append(cache[digest])
                            if any(f['source_sha256'] in old_bytes or f['pixel_sha256'] in old_pixels for f in facts):
                                excluded.append({'item_id': identity, 'reason': 'historical_training_or_evaluation_image_overlap'})
                                continue
                            original = json.dumps([sorted(f['pixel_sha256'] for f in facts), question.strip(), sorted(options)], ensure_ascii=False)
                            if original in seen:
                                excluded.append({'item_id': identity, 'reason': 'duplicate_source_question_and_image'})
                                continue
                            seen.add(original)
                            paths = []
                            for content, fact in zip(contents, facts):
                                relative = 'data/protocol-v2/images/' + fact['source_sha256'] + '.image'
                                destination = ROOT / relative
                                if not destination.exists():
                                    if shutil.disk_usage(DATA).free < 8 * 1024**3:
                                        raise RuntimeError('Indexing stopped below eight GiB free disk')
                                    destination.parent.mkdir(parents=True, exist_ok=True)
                                    destination.write_bytes(content)
                                elif sha(destination) != fact['source_sha256']:
                                    raise ValueError('Existing extracted image differs')
                                assets[relative] = {'path': relative, **fact}
                                paths.append(relative)
                            references = [int(n) for n in re.findall(r'<image (\d+)>', question)]
                            if references and max(references) > len(paths):
                                raise ValueError('Question image references exceed preserved images')
                            candidates.append({'item_id': identity, 'axis': 'consistency' if ai2d else 'mmstar',
                                'dataset': 'ai2d' if ai2d else 'mmmu', 'source_repo': dataset['repo'],
                                'source_revision': dataset['revision'], 'source_file': entry['path'],
                                'source_row': global_row-1, 'split': 'test', 'role': 'unassigned',
                                'question': question, 'question_type': kind, 'options': {chr(65+i): value for i, value in enumerate(options)},
                                'answer_key': answer, 'answer_type': 'option_letter_v2' if kind == 'multiple-choice' else 'mmmu_open',
                                'image_paths': paths, 'image_pixel_sha256': [f['pixel_sha256'] for f in facts],
                                'subject': 'ai2d' if ai2d else entry['path'].split('/')[0],
                                'independent_confirmation': False})
                        progress(dataset['repo'], global_row, None)
            write_rows(DATA / 'candidates.jsonl', candidates)
            write_rows(DATA / 'image-assets.jsonl', sorted(assets.values(), key=lambda row: row['path']))
            write_rows(OUT / 'excluded-items.jsonl', excluded)
            for name in ('candidates.jsonl', 'image-assets.jsonl', 'historical-image-fingerprints.jsonl'):
                shutil.copy2(DATA / name, OUT / name)
            summary = {'status': 'complete', 'sources_config_sha256': config_sha,
                       'raw_rows': dict(raw_counts), 'accepted_candidates': len(candidates),
                       'by_dataset': dict(Counter(row['dataset'] for row in candidates)),
                       'excluded_items': len(excluded), 'image_assets': len(assets),
                       'unique_pixel_images': len({row['pixel_sha256'] for row in assets.values()}),
                       'multi_image_candidates': sum(len(row['image_paths']) > 1 for row in candidates),
                       'open_answer_candidates': sum(row['answer_type'] == 'mmmu_open' for row in candidates),
                       'max_choice_count': max(len(row['options']) for row in candidates),
                       'candidates_sha256': sha(DATA / 'candidates.jsonl'),
                       'image_assets_sha256': sha(DATA / 'image-assets.jsonl'),
                       'historical_image_inventory_sha256': sha(DATA / 'historical-image-fingerprints.jsonl'),
                       'new_train_audit_confirm_roles_frozen': False, 'near_duplicate_human_review': 'pending',
                       'formal_training_allowed': False, 'completed_utc': datetime.now(timezone.utc).isoformat()}
            write_json(OUT / 'complete.json', summary)
            write_json(OUT / 'candidate-summary.json', summary)
            state.update(summary)
        except Exception as error:
            state.update(status='failed', error_type=type(error).__name__)
            raise
        finally:
            write_json(OUT / 'status.json', state)
        if os.environ.get('GH_TOKEN'):
            from publish_artifacts import publish
            publish(OUT, 'arp-protocol-v2-20261016', token=os.environ['GH_TOKEN'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        build()
    else:
        token = sys.stdin.readline().strip()
        if not token:
            raise SystemExit('Completion-backup credential required on stdin')
        sessions = subprocess.run(['screen', '-ls'], capture_output=True, text=True).stdout
        if '.arp-v2-source-index' in sessions:
            raise SystemExit('Candidate indexing already active')
        OUT.mkdir(parents=True, exist_ok=True)
        command = ('source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; '
                   'exec /root/autodl-tmp/arp-venv/bin/python -u tools/index_independent_sources.py --worker >> ' + str(OUT / 'console.log') + ' 2>&1')
        subprocess.run(['screen', '-dmS', 'arp-v2-source-index', 'bash', '-c', command],
                       env={**os.environ, 'GH_TOKEN': token}, check=True)
        print('Started streaming candidate index; waits for verified source acquisition', flush=True)
