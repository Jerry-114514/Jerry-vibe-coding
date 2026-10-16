"""New public COCO/REFER candidates, excluding all historical image names.

Grounding uses test expressions. POPE-style questions use the COCO annotation
subset preserved by REFER; these are not official POPE benchmark scores.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import requests
from experiment_common import ROOT, read_json, read_rows, sha, write_json, write_rows
from index_independent_sources import image_facts
from refer_api import load_original

OUT = ROOT / 'runs/protocol-v2-data/coco-candidates'
DATA = ROOT / 'data/protocol-v2'
SEED = '20261016:new-coco-candidates:'
SESSIONS = threading.local()


def rank(value):
    return hashlib.sha256((SEED+str(value)).encode()).hexdigest()


def candidates():
    history = read_rows(ROOT/'data/manifest.jsonl') + read_rows(ROOT/'data/training/pool.jsonl')
    excluded_names = {Path(row['image_path']).name for row in history}
    source_locks = {entry['path']: entry['sha256'] for entry in read_json(ROOT/'data/download_sources.json')}
    output, images = [], {}
    sources = []
    for dataset, split_by in [('refcoco','unc'), ('refcoco+','unc'), ('refcocog','umd')]:
        path = ROOT/'data/raw/refer'/(dataset+'.zip')
        if sha(path) != source_locks[str(path.relative_to(ROOT)).replace('\\','/')]:
            raise ValueError('Original public REFER source checksum mismatch')
        refs, instances = load_original(path, dataset, split_by)
        sources.append({'dataset': dataset, 'archive_sha256': sha(path), 'source': 'original_public_REFER_archive'})
        lookup = {row['id']: row for row in instances['images']}
        annotations = {row['id']: row for row in instances['annotations']}
        for ref in sorted(refs, key=lambda ref: rank(dataset+':'+str(ref['ref_id']))):
            if ref['split'] not in ('test','testA','testB'):
                continue
            image = lookup[ref['image_id']]
            if image['file_name'] in excluded_names:
                continue
            sentences = sorted(ref['sentences'], key=lambda sentence: rank(str(sentence['sent_id'])))
            if not sentences:
                raise ValueError('Public reference has no expression')
            expression = sentences[0]['sent']
            x, y, width, height = annotations[ref['ann_id']]['bbox']
            relative = 'data/protocol-v2/coco-images/'+image['file_name']
            images[relative] = image
            identity = 'new:grounding:'+dataset+':'+str(ref['ref_id'])+':'+str(sentences[0]['sent_id'])
            output.append({'axis':'grounding','item_id':identity,'dataset':dataset,'subset':ref['split'],
                           'source_ref_id':ref['ref_id'],'source_sentence_id':sentences[0]['sent_id'],
                           'source_image_id':image['id'],'source_split':ref['split'],'split':'test','role':'unassigned',
                           'question':'Locate the object referred to by: '+expression+
                               '. Return only its bounding box as [x_min, y_min, x_max, y_max] in original-image pixel coordinates. '+
                               f'The image is {image["width"]} pixels wide and {image["height"]} pixels high.',
                           'referring_expression':expression,'answer_key':[x,y,x+width,y+height],
                           'answer_type':'bbox','question_type':'grounding','options':{},
                           'image_paths':[relative],'width':image['width'],'height':image['height'],
                           'independent_confirmation':False})
    _, instances = load_original(ROOT/'data/raw/refer/refcoco.zip','refcoco','unc')
    categories = {row['id']:row['name'] for row in instances['categories']}
    present = defaultdict(set)
    for annotation in instances['annotations']:
        present[annotation['image_id']].add(annotation['category_id'])
    frequency = Counter(category for values in present.values() for category in values)
    cooccurrence = Counter((a,b) for values in present.values() for a in values for b in values if a!=b)
    eligible = [image for image in instances['images'] if image['file_name'] not in excluded_names and present[image['id']]]
    # Prefer already needed test images to reduce downloads; no model outcomes used.
    needed = {image['id'] for image in images.values()}
    eligible = sorted(eligible, key=lambda image: (image['id'] not in needed, rank('pope:'+str(image['id']))))[:2000]
    if len(eligible) < 2000:
        raise ValueError('Insufficient new annotated COCO images for candidate pool')
    for index, image in enumerate(eligible):
        positives = present[image['id']]
        absent = set(categories)-positives
        if not absent:
            raise ValueError('No absent category for POPE-style pair')
        subset = ('random','popular','adversarial')[index % 3]
        positive = min(positives, key=lambda category: rank(f'positive:{image["id"]}:{category}'))
        if subset=='random':
            negative = min(absent, key=lambda category: rank(f'negative:{image["id"]}:{category}'))
        elif subset=='popular':
            negative = min(absent, key=lambda category: (-frequency[category], category))
        else:
            negative = min(absent, key=lambda category: (-sum(cooccurrence[(other,category)] for other in positives), -frequency[category], category))
        relative = 'data/protocol-v2/coco-images/'+image['file_name']
        images[relative] = image
        for category, answer in [(positive,'yes'), (negative,'no')]:
            identity = f'new:pope-style:{subset}:{image["id"]}:{category}:{answer}'
            output.append({'axis':'pope','item_id':identity,'dataset':'coco-refer-pope-style','subset':subset,
                           'source_image_id':image['id'],'source_category_id':category,'split':'constructed','role':'unassigned',
                           'question':f'Is there a {categories[category]} in the image? Answer with yes or no only.',
                           'answer_key':answer,'answer_type':'yes_no','question_type':'yes_no','options':{},
                           'image_paths':[relative],'width':image['width'],'height':image['height'],
                           'independent_confirmation':False})
    return output, images, sources


def fetch(relative, image):
    destination = ROOT/relative
    if destination.exists():
        content = destination.read_bytes()
    else:
        name = image['file_name']
        split = 'train2014' if 'train2014' in name else 'val2014'
        urls = ['http://images.cocodataset.org/'+split+'/'+name,
                'https://images.cocodataset.org/'+split+'/'+name,
                'https://coco-dataset.s3.amazonaws.com/'+split+'/'+name]
        for attempt in range(6):
            if shutil.disk_usage(DATA).free < 8*1024**3:
                raise RuntimeError('COCO candidate download stopped below eight GiB free')
            try:
                trust_env = attempt >= 3
                if not hasattr(SESSIONS, 'sessions'):
                    SESSIONS.sessions = {}
                if trust_env not in SESSIONS.sessions:
                    SESSIONS.sessions[trust_env] = requests.Session()
                    SESSIONS.sessions[trust_env].trust_env = trust_env
                session = SESSIONS.sessions[trust_env]
                with session.get(urls[attempt % len(urls)], timeout=(8,30)) as response:
                    response.raise_for_status()
                    content = response.content
                facts = image_facts(content)
                if (facts['width'],facts['height']) != (image['width'],image['height']):
                    raise ValueError('Public COCO image dimensions differ from annotations')
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_suffix(destination.suffix+'.part')
                partial.write_bytes(content)
                partial.replace(destination)
                break
            except requests.RequestException:
                if attempt==5:
                    raise RuntimeError('Public COCO image unavailable after bounded retries') from None
        else:
            raise RuntimeError('COCO download did not terminate')
    facts = image_facts(content)
    if (facts['width'],facts['height']) != (image['width'],image['height']):
        raise ValueError('Cached image differs from original annotation dimensions')
    return {'path':relative,'source_image_id':image['id'],**facts}


def worker():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True)
    with (OUT/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'running','worker_script':Path(__file__).name,'pid':os.getpid(),
               'formal_training_allowed':False,'scope':'New public COCO/REFER candidate preparation only',
               'started_utc':datetime.now(timezone.utc).isoformat()}
        write_json(OUT/'status.json',state)
        try:
            rows, images, sources = candidates()
            write_rows(DATA/'coco-candidate-plan.jsonl',rows)
            write_json(OUT/'sources.json',sources)
            state.update(candidate_rows=len(rows),expected_images=len(images),completed_images=0)
            write_json(OUT/'status.json',state)
            assets, failures = [], []
            with ThreadPoolExecutor(max_workers=48) as pool:
                jobs={pool.submit(fetch,relative,image):relative for relative,image in images.items()}
                for future in as_completed(jobs):
                    try:
                        assets.append(future.result())
                    except Exception as error:
                        failures.append({'image_path':jobs[future],'error_type':type(error).__name__})
                    if (len(assets)+len(failures))%25==0 or len(assets)+len(failures)==len(images):
                        state.update(completed_images=len(assets),failed_images=len(failures),updated_utc=datetime.now(timezone.utc).isoformat())
                        write_json(OUT/'status.json',state)
                        print('COCO images',len(assets),'/',len(images),'failed',len(failures),flush=True)
            write_json(OUT/'download-failures.json',failures)
            if failures:
                raise RuntimeError('Some public COCO candidates missing; no final candidate set written')
            by_path={asset['path']:asset for asset in assets}
            for row in rows:
                row['image_pixel_sha256']=[by_path[path]['pixel_sha256'] for path in row['image_paths']]
            write_rows(DATA/'coco-candidates.jsonl',rows)
            write_rows(DATA/'coco-image-assets.jsonl',sorted(assets,key=lambda row:row['path']))
            for name in ('coco-candidates.jsonl','coco-image-assets.jsonl'):
                shutil.copy2(DATA/name,OUT/name)
            summary={'status':'complete','candidate_rows':len(rows),'images':len(images),
                     'axes':dict(Counter(row['axis'] for row in rows)),
                     'candidates_sha256':sha(DATA/'coco-candidates.jsonl'),
                     'assets_sha256':sha(DATA/'coco-image-assets.jsonl'),
                     'annotation_scope':'19,994-image COCO subset preserved in public REFER; new POPE-style construction, not official POPE',
                     'roles_frozen':False,'formal_training_allowed':False,
                     'completed_utc':datetime.now(timezone.utc).isoformat()}
            write_json(OUT/'complete.json',summary)
            state.update(summary)
        except Exception as error:
            state.update(status='failed',error_type=type(error).__name__)
            raise
        finally:
            write_json(OUT/'status.json',state)
        if os.environ.get('GH_TOKEN'):
            from publish_artifacts import publish
            publish(OUT,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker()
    else:
        token=sys.stdin.readline().strip()
        if not token: raise SystemExit('Completion-backup credential required')
        sessions=subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout
        if '.arp-v2-coco-candidates' in sessions: raise SystemExit('COCO worker already running')
        OUT.mkdir(parents=True,exist_ok=True)
        command=('source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; '
                 'exec /root/autodl-tmp/arp-venv/bin/python -u tools/prepare_coco_candidates_v2.py --worker >> '+str(OUT/'console.log')+' 2>&1')
        subprocess.run(['screen','-dmS','arp-v2-coco-candidates','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Started disjoint-image public COCO/REFER candidate preparation',flush=True)
