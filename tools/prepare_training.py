"""D1: independently held-out task proxies; never train on evaluation items."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

from experiment_common import ROOT, AXES, read_json, read_rows, sha, write_json, write_rows


def rank(text):
    return hashlib.sha256(('20261014:' + str(text)).encode()).hexdigest()


def pixel_hash(content):
    from PIL import Image
    with Image.open(io.BytesIO(content)) as image:
        rgb = image.convert('RGB')
        return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()


def get_file(url, path):
    import requests
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    for attempt in range(4):
        try:
            session = requests.Session()
            session.trust_env = False
            if 'images.cocodataset.org' in url:
                url = url.replace('https://', 'http://', 1)
            response = session.get(url, timeout=(15, 90))
            response.raise_for_status()
            temp = path.with_suffix(path.suffix + '.partial')
            temp.write_bytes(response.content)
            temp.replace(path)
            return path
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def prepare(max_units=600):
    import pyarrow.parquet as pq
    from PIL import Image
    from refer_api import REFER, load_original
    from prepare_phase_a import image_bytes, TEMPLATES
    from huggingface_hub import HfApi, hf_hub_download
    output = ROOT / 'data/training'
    if (output / 'pool.lock.json').exists():
        locked = read_json(output / 'pool.lock.json')
        if max_units != locked['max_units'] or sha(output / 'pool.jsonl') != locked['manifest_sha256']:
            raise ValueError('Training pool is already frozen under another protocol')
        print(locked, flush=True)
        return
    evaluation = read_rows(ROOT / 'data/manifest.jsonl')
    eval_bytes = {x['sha256'] for x in read_rows(ROOT / 'data/assets.sha256.jsonl')}
    eval_pixels = set()
    for relative in {x['image_path'] for x in evaluation}:
        eval_pixels.add(pixel_hash((ROOT / relative).read_bytes()))
    eval_names = {Path(x['image_path']).name for x in evaluation}
    pool, sources = [], []
    download_jobs = {}
    _, instances = load_original(ROOT / 'data/raw/refer/refcoco.zip', 'refcoco', 'unc')
    categories = {x['id']: x['name'] for x in instances['categories']}
    present = defaultdict(set)
    for ann in instances['annotations']:
        present[ann['image_id']].add(ann['category_id'])
    api = REFER(ROOT / 'data/raw/refer_mirror', dataset='refcoco', splitBy='unc')
    candidates = [x for x in api.Imgs.values() if x['file_name'].startswith('COCO_train2014_') and x['file_name'] not in eval_names and present[x['id']]]
    for i, img in enumerate(sorted(candidates, key=lambda x: rank('pope:'+str(x['id'])))[:max_units]):
        positives = sorted(present[img['id']])
        negatives = sorted(set(categories)-present[img['id']])
        if not negatives:
            continue
        positive = i % 2 == 0
        cats = positives if positive else negatives
        cat = cats[int(rank('category:'+str(img['id']))[:8], 16) % len(cats)]
        identity = f'train:pope:{img["id"]}:{cat}'
        relative = 'data/training/images/coco/'+img['file_name']
        pool.append({'axis': 'pope', 'item_id': identity, 'bootstrap_unit': identity,
                     'image_path': relative, 'question': f'Is there a {categories[cat]} in the image? Answer with yes or no only.',
                     'answer_key': 'yes' if positive else 'no', 'answer_type': 'yes_no',
                     'dataset': 'coco-train-object-proxy', 'split': 'train', 'width': img['width'], 'height': img['height']})
        download_jobs[relative] = 'https://images.cocodataset.org/train2014/'+img['file_name']
    for dataset, split in [('refcoco', 'unc'), ('refcoco+', 'unc'), ('refcocog', 'umd')]:
        api = REFER(ROOT / 'data/raw/refer_mirror', dataset=dataset, splitBy=split)
        selected, seen = [], set()
        for rid in sorted(api.getRefIds(split='train'), key=lambda x: rank(dataset+str(x))):
            ref = api.Refs[rid]
            img = api.Imgs[ref['image_id']]
            if img['file_name'] in eval_names or img['id'] in seen:
                continue
            x, y, width, height = api.getRefBox(rid)
            if not (0 <= x < x+width <= img['width'] and 0 <= y < y+height <= img['height']):
                continue
            sent = min(ref['sentences'], key=lambda x: x['sent_id'])
            identity = f'train:grounding:{dataset}:{rid}:{sent["sent_id"]}'
            relative = 'data/training/images/coco/'+img['file_name']
            pool.append({'axis': 'grounding', 'item_id': identity, 'bootstrap_unit': identity,
                         'image_path': relative,
                         'question': f'Locate the object referred to by: {sent["sent"]}. Return only its bounding box as [x_min, y_min, x_max, y_max] in original-image pixel coordinates. The image is {img["width"]} pixels wide and {img["height"]} pixels high.',
                         'answer_key': [x, y, x+width, y+height], 'answer_type': 'bbox_xyxy_pixels',
                         'dataset': dataset, 'split': 'train', 'width': img['width'], 'height': img['height']})
            download_jobs[relative] = 'https://images.cocodataset.org/train2014/'+img['file_name']
            seen.add(img['id'])
            selected.append(rid)
            if len(selected) >= max_units//3:
                break
    # Original questions omitted from A are held out by image bytes AND decoded pixels.
    bench_path = ROOT / 'data/raw/mmbench/dev-00000-of-00001.parquet'
    seen = set()
    bench_units = 0
    for row in sorted(pq.read_table(bench_path).to_pylist(), key=lambda x: rank('bench'+str(x['index']))):
        content = image_bytes(row)
        digest = hashlib.sha256(content).hexdigest()
        if digest in eval_bytes or pixel_hash(content) in eval_pixels:
            continue
        opts = {letter: row[letter] for letter in 'ABCD' if row.get(letter) and row[letter].strip().lower() not in ('nan','none','null')}
        identity_key = (digest, row['question'].strip(), tuple(sorted(opts.values())))
        if identity_key in seen or len(opts) < 2 or row['answer'] not in opts:
            continue
        seen.add(identity_key)
        relative = f'data/training/images/mmbench/{row["index"]}.jpg'
        image_path = ROOT / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(content)
        with Image.open(image_path) as img:
            width, height = img.size
        hint = (row.get('hint') or '').strip()
        context = 'Context: '+hint+'\n' if hint and hint.lower() not in ('nan','none','null') else ''
        choices = '\n'.join(f'{k}. {v}' for k,v in opts.items())
        base = 'train:consistency:'+str(row['index'])
        for p, template in enumerate(TEMPLATES, 1):
            pool.append({'axis': 'consistency', 'item_id': base+f':p{p}', 'bootstrap_unit': base,
                         'image_path': relative, 'question': context+template.format(question=row['question'])+'\n'+choices+'\nReturn only the option letter.',
                         'answer_key': row['answer'], 'answer_type': 'option_letter', 'options': list(opts),
                         'dataset': 'mmbench-heldout', 'split': 'heldout-from-en-dev', 'width': width, 'height': height})
        bench_units += 1
        if bench_units >= max_units:
            break
    api_hf = HfApi()
    dataset_id = 'derek-thomas/ScienceQA'
    spec = api_hf.dataset_info(dataset_id, files_metadata=True)
    sources.append({'dataset': dataset_id, 'revision': spec.sha,
                    'author_endorsed_url': 'https://github.com/lupantech/ScienceQA#-download-the-dataset'})
    science_count = 0
    for sibling in sorted(spec.siblings, key=lambda x: x.rfilename):
        if not sibling.rfilename.endswith('.parquet') or 'train-' not in sibling.rfilename:
            continue
        from huggingface_hub import hf_hub_url
        from download_ranges import download_ranges
        digest = sibling.lfs.get('sha256') if isinstance(sibling.lfs,dict) else getattr(sibling.lfs,'sha256',None)
        path = download_ranges(hf_hub_url(dataset_id,sibling.rfilename,repo_type='dataset',revision=spec.sha),
                               output/'raw/scienceqa'/Path(sibling.rfilename).name,sibling.size,digest)
        source_index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                idx = source_index
                source_index += 1
                if not row.get('image') or len(row['choices']) > 4:
                    continue
                content = image_bytes(row)
                digest = hashlib.sha256(content).hexdigest()
                if digest in eval_bytes or pixel_hash(content) in eval_pixels:
                    continue
                identity = f'train:scienceqa:{Path(sibling.rfilename).stem}:{idx}'
                relative = f'data/training/images/scienceqa/{science_count}.png'
                destination = ROOT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                with Image.open(destination) as img:
                    width, height = img.size
                options = list('ABCD'[:len(row['choices'])])
                choices = '\n'.join(f'{letter}. {text}' for letter, text in zip(options, row['choices']))
                hint = ('Context: '+row['hint']+'\n') if row.get('hint') else ''
                pool.append({'axis': 'mmstar', 'item_id': identity, 'bootstrap_unit': identity, 'image_path': relative,
                             'question': hint+row['question']+'\n'+choices+'\nReturn only the option letter.',
                             'answer_key': options[row['answer']], 'options': options, 'answer_type': 'option_letter',
                             'dataset': 'scienceqa-train-reasoning-proxy', 'split': 'train', 'width': width, 'height': height})
                science_count += 1
                if science_count >= max_units:
                    break
            if science_count >= max_units:
                break
        sources.append({'file': sibling.rfilename, 'sha256': sha(path)})
        if science_count >= max_units:
            break
    def download(job):
        relative, url = job
        return get_file(url, ROOT / relative)
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(download, download_jobs.items()))
    invalid = set()
    assets = []
    for relative in sorted({row['image_path'] for row in pool}):
        path = ROOT / relative
        digest = sha(path)
        pixels = pixel_hash(path.read_bytes())
        if digest in eval_bytes or pixels in eval_pixels:
            invalid.add(relative)
        assets.append({'path': relative, 'sha256': digest, 'pixel_sha256': pixels, 'bytes': path.stat().st_size})
    pool = [row for row in pool if row['image_path'] not in invalid]
    pool.sort(key=lambda x: (AXES.index(x['axis']), rank(x['item_id'])))
    for axis in AXES:
        if len({row['bootstrap_unit'] for row in pool if row['axis']==axis}) < 10:
            raise ValueError('Insufficient disjoint training units: '+axis)
    for row in pool:
        if row['split'] == 'val' or row['item_id'] in {x['item_id'] for x in evaluation}:
            raise ValueError('Evaluation leakage')
    write_rows(output / 'pool.jsonl', pool)
    write_rows(output / 'assets.jsonl', assets)
    write_json(output / 'sources.json', sources)
    report = {'max_units': max_units, 'manifest_sha256': sha(output/'pool.jsonl'),
              'evaluation_manifest_sha256': sha(ROOT/'data/manifest.jsonl'),
              'axes': {axis: {'rows':sum(x['axis']==axis for x in pool),
                              'units':len({x['bootstrap_unit'] for x in pool if x['axis']==axis})} for axis in AXES},
              'image_byte_and_decoded_pixel_overlap': 0, 'excluded_overlap_images': len(invalid),
              'training_sources_user_approved': True, 'mmstar_training_proxy': 'ScienceQA train (not MMStar evaluation)'}
    write_json(output / 'pool.lock.json', report)
    print(report, flush=True)


def arms(run):
    import numpy as np
    run = Path(run)
    from analyze_rulers import verify_go
    gate = verify_go(run)
    summary = read_json(run/'analysis/summary.json')
    pi = summary['families']['qwen']['pi_boot_mean']
    ranking = sorted(AXES, key=lambda x: (-pi[x], AXES.index(x)))
    if abs(pi[ranking[0]]-pi[ranking[1]]) <= summary['protocol']['tie_epsilon']:
        raise ValueError('Top training axis is tied: single-axis selection is undefined')
    pool = read_rows(ROOT/'data/training/pool.jsonl')
    output = run/'training'
    write_rows(output/'A1.jsonl', [x for x in pool if x['axis']==ranking[0]])
    write_rows(output/'A2.jsonl', [x for x in pool if x['axis']==ranking[1]])
    # Balanced task mixture; the trainer cycles each arm under the same step budget.
    size = min(sum(x['axis']==axis for x in pool) for axis in AXES)
    uniform = [x for axis in AXES for x in [y for y in pool if y['axis']==axis][:size]]
    write_rows(output/'U.jsonl', uniform)
    write_rows(output/'N.jsonl', [])
    write_json(output/'arm_selection.json', {'ranking':ranking, 'pi':pi, 'gate':gate,
                                           'pool_sha256':sha(ROOT/'data/training/pool.jsonl'),
                                           'no_op': 'N has no parameter updates; identical baseline ledger is reused.'})
    control=summary['families']['qwen']['homogeneous_floor_control']
    write_json(output/'homogeneous_floor_arm_selection.json', {
        'control':'D5: replace all c_a by c_bar, retaining scores and paired bootstrap draws',
        'actual_top_axis':ranking[0], 'actual_runner_up_axis':ranking[1],
        'homogeneous_control':control,
        'top_axis_changed':control['top_axis']!=ranking[0] if control['arm_selection_defined'] else None,
        'runner_up_axis_changed':control['runner_up_axis']!=ranking[1] if control['arm_selection_defined'] else None,
        'additional_training':False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['pool','arms'])
    p.add_argument('--max-units', type=int, default=600)
    p.add_argument('--run', default=str(ROOT/'runs/formal'))
    a=p.parse_args()
    prepare(a.max_units) if a.action=='pool' else arms(a.run)
