"""Freeze outcome-independent candidate roles and expose remaining human reviews."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import AXES, ROOT, read_json, read_rows, sha, write_json, write_rows
from independent_manifest_v2 import assign_roles, select_axis, expand_item

OUT = ROOT / 'runs/protocol-v2-data/role-manifests'
DATA = ROOT / 'data/protocol-v2'


class HashTree:
    def __init__(self):
        self.root = None

    def add(self, value, identity):
        if self.root is None:
            self.root = [value, [identity], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(identity)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [identity], {}]
                return
            node = node[2][distance]

    def near(self, value, radius):
        stack = [self.root] if self.root else []
        result = []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                result.extend((identity, distance) for identity in node[1])
            stack.extend(child for edge, child in node[2].items() if distance-radius <= edge <= distance+radius)
        return result


def wait_for_stage(relative, state):
    path = ROOT / relative / 'status.json'
    while True:
        if not path.exists():
            raise RuntimeError('Required candidate producer has not started')
        source = read_json(path)
        if source['status'] == 'complete':
            return
        if source['status'] == 'failed':
            raise RuntimeError('Candidate producer failed; do not freeze a partial universe')
        pid = source.get('pid')
        if not pid or not Path('/proc/' + str(pid)).exists():
            raise RuntimeError('Candidate producer process missing')
        state.update(status='waiting', waiting_for=str(path.relative_to(ROOT)), waiting_for_pid=pid,
                     updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(OUT / 'status.json', state)
        time.sleep(20)


def near_duplicate_candidates(assets, historical, pixel_roles):
    tree = HashTree()
    identities = {}
    for index, row in enumerate(historical):
        identity = 'historical:' + str(index)
        identities[identity] = {**row, 'role': 'historical'}
        tree.add(int(row['dhash64'],16), identity)
    representatives = {}
    for row in assets:
        representatives.setdefault(row['pixel_sha256'], row)
    pairs, counts = [], Counter()
    for pixel, row in sorted(representatives.items()):
        identity = 'new:' + pixel
        role = pixel_roles.get(pixel, 'unselected')
        for other_id, distance in tree.near(int(row['dhash64'],16),4):
            other = identities[other_id]
            if other['pixel_sha256'] == pixel:
                continue
            critical = other['role'] == 'historical' or other['role'] != role
            counts['all_candidate_pairs'] += 1
            counts['cross_role_or_historical_pairs'] += int(critical)
            if len(pairs) < 50000:
                pairs.append({'left_path':other['path'],'right_path':row['path'],
                              'left_role':other['role'],'right_role':role,'dhash_distance':distance,
                              'cross_role_or_historical':critical,'review_status':'pending',
                              'note':'Hash proximity is a review candidate, not proof of visual identity'})
        identities[identity] = {**row, 'role':role}
        tree.add(int(row['dhash64'],16),identity)
    return pairs, {**dict(counts),'stored_pairs':len(pairs),'truncated':counts['all_candidate_pairs']>len(pairs)}


def worker():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'running','worker_script':Path(__file__).name,'pid':os.getpid(),
               'started_utc':datetime.now(timezone.utc).isoformat(),'formal_training_allowed':False,
               'scope':'Candidate roles only; human near-duplicate and grounding reviews pending'}
        write_json(OUT/'status.json',state)
        try:
            wait_for_stage('runs/protocol-v2-data/candidate-index',state)
            wait_for_stage('runs/protocol-v2-data/coco-candidates',state)
            state.update(status='running',stage='global_exact_image_components')
            write_json(OUT/'status.json',state)
            hf_summary=read_json(ROOT/'runs/protocol-v2-data/candidate-index/complete.json')
            coco_summary=read_json(ROOT/'runs/protocol-v2-data/coco-candidates/complete.json')
            design={'schema':'arp.independent-candidate-roles.v2','calibration_units_per_axis':120,
                    'audit_units_per_axis':1200,'confirm_units_per_axis':1200,
                    'role_seed':'20261016:independent-manifests-v2:',
                    'hf_candidates_sha256':hf_summary['candidates_sha256'],
                    'coco_candidates_sha256':coco_summary['candidates_sha256'],
                    'training_pool_sha256':sha(ROOT/'data/training/pool.jsonl'),
                    'manifest_expansion_code_sha256':sha(ROOT/'tools/independent_manifest_v2.py'),
                    'near_duplicate_radius':4,'near_duplicate_pair_storage_cap':50000}
            if (OUT/'design.json').exists() and read_json(OUT/'design.json')!=design:
                raise ValueError('Frozen candidate-role design changed; use a new output version')
            write_json(OUT/'design.json',design)
            if (OUT/'complete.json').exists():
                done=read_json(OUT/'complete.json')
                for role,info in done['manifests'].items():
                    if sha(OUT/(role+'.jsonl'))!=info['sha256']:
                        raise ValueError('Completed role manifest changed')
                print('Frozen role manifests already checksum-verified complete',flush=True)
                return
            for file,expected in [('candidates.jsonl',hf_summary['candidates_sha256']),
                                  ('image-assets.jsonl',hf_summary['image_assets_sha256']),
                                  ('coco-candidates.jsonl',coco_summary['candidates_sha256']),
                                  ('coco-image-assets.jsonl',coco_summary['assets_sha256'])]:
                if sha(DATA/file)!=expected: raise ValueError('Candidate producer digest differs')
            historical=read_rows(DATA/'historical-image-fingerprints.jsonl')
            old_pixels={row['pixel_sha256'] for row in historical}
            old_bytes={row['source_sha256'] for row in historical}
            all_assets=read_rows(DATA/'image-assets.jsonl')+read_rows(DATA/'coco-image-assets.jsonl')
            by_path={row['path']:row for row in all_assets}
            all_rows=read_rows(DATA/'candidates.jsonl')+read_rows(DATA/'coco-candidates.jsonl')
            excluded=[]; candidates=[]
            for row in all_rows:
                facts=[by_path[path] for path in row['image_paths']]
                if any(f['pixel_sha256'] in old_pixels or f['source_sha256'] in old_bytes for f in facts):
                    excluded.append({'item_id':row['item_id'],'reason':'historical_exact_image_overlap_after_global_merge'})
                    continue
                if [f['pixel_sha256'] for f in facts]!=row['image_pixel_sha256']:
                    raise ValueError('Candidate image fingerprints differ')
                candidates.append(row)
            assigned=assign_roles(candidates)
            if len({row['item_id'] for row in assigned})!=len(assigned):
                raise ValueError('Duplicate original source IDs')
            write_rows(OUT/'excluded-items.jsonl',excluded)
            write_rows(OUT/'candidate-roles.jsonl',[{key:row[key] for key in ('item_id','axis','dataset','role','image_group','image_pixel_sha256')} for row in assigned])
            targets={'calibration':120,'audit':1200,'confirm':1200}
            manifests={}; pixel_roles={}; coverage={}; selected_assets={}
            for role,target in targets.items():
                selected=[]
                for axis in AXES:
                    pool=[row for row in assigned if row['role']==role and row['axis']==axis]
                    chosen=select_axis(pool,target)
                    selected.extend(chosen)
                rows=[prompt for row in selected for prompt in expand_item(row)]
                write_rows(OUT/(role+'.jsonl'),rows)
                manifests[role]={'sha256':sha(OUT/(role+'.jsonl')),'prompt_rows':len(rows),'original_units':len(selected)}
                coverage[role]={axis:{'units':sum(row['axis']==axis for row in selected),
                                      'image_components':len({row['image_group'] for row in selected if row['axis']==axis}),
                                      'multi_image_units':sum(row['axis']==axis and len(row['image_paths'])>1 for row in selected),
                                      'open_answer_units':sum(row['axis']==axis and row['answer_type']=='mmmu_open' for row in selected)} for axis in AXES}
                for row in selected:
                    for path,pixel in zip(row['image_paths'],row['image_pixel_sha256']):
                        if pixel in pixel_roles and pixel_roles[pixel]!=role:
                            raise ValueError('Cross-role exact pixel leakage')
                        pixel_roles[pixel]=role
                        selected_assets[path]=by_path[path]
            # Reuse explicitly approved historical training proxies, retaining
            # their identity; newly downloaded test questions never enter training.
            training=read_rows(ROOT/'data/training/pool.jsonl')
            for row in training: row.update(role='train',protocol_v2_role='historical_training_proxy')
            write_rows(OUT/'train.jsonl',training)
            manifests['train']={'sha256':sha(OUT/'train.jsonl'),'prompt_rows':len(training),
                                'original_units':len({row['bootstrap_unit'] for row in training})}
            write_rows(OUT/'selected-image-assets.jsonl',sorted(selected_assets.values(),key=lambda row:row['path']))
            state.update(stage='near_duplicate_candidate_search',updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(OUT/'status.json',state)
            pairs,pair_summary=near_duplicate_candidates(list(selected_assets.values()),historical,pixel_roles)
            write_rows(OUT/'near-duplicate-candidates.jsonl',pairs)
            summary={'status':'complete','manifests':manifests,'coverage':coverage,
                     'role_assignment':'SHA256 of minimum exact-RGB image component: calibration 8%, audit 46%, confirm 46%',
                     'no_model_outputs_used_for_selection':True,'global_exact_image_cross_role_overlap':0,
                     'historical_exact_overlap_exclusions':len(excluded),'near_duplicate_candidates':pair_summary,
                     'near_duplicate_human_review_complete':False,'grounding_blind_human_review_complete':False,
                     'independent_confirmation_accepted':False,'formal_training_allowed':False,
                     'candidate_roles_frozen':True,'completed_utc':datetime.now(timezone.utc).isoformat()}
            write_json(OUT/'complete.json',summary)
            write_json(OUT/'manifest-summary.json',summary)
            write_json(OUT/'readiness-gate.json',{'training_allowed':False,'confirmation_accepted':False,
                       'reasons':['Human review of near-duplicate candidates pending','Grounding blind-review acceptance pending','New model audit and paired gate pending']})
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
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker: worker()
    else:
        token=sys.stdin.readline().strip()
        if not token: raise SystemExit('Completion-backup credential required')
        sessions=subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout
        if '.arp-v2-role-manifests' in sessions: raise SystemExit('Candidate role worker already running')
        OUT.mkdir(parents=True,exist_ok=True)
        command=('source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/freeze_independent_candidates.py --worker >> '+str(OUT/'console.log')+' 2>&1')
        subprocess.run(['screen','-dmS','arp-v2-role-manifests','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Started candidate-role freezer; waits for completed public candidate producers',flush=True)
