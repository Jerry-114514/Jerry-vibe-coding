"""Conservative pre-outcome image isolation, retaining the original candidate split."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from experiment_common import AXES, ROOT, read_json, read_rows, sha, write_json, write_rows
from freeze_independent_candidates import HashTree
from independent_manifest_v2 import ImageComponents, rank, select_axis, expand_item

OUT = ROOT / 'runs/protocol-v2-data/clean-role-manifests-v4'
DATA = ROOT / 'data/protocol-v2'


def conservative_components(candidates, assets, historical, radius=4):
    components = ImageComponents()
    for row in candidates:
        for pixel in row['image_pixel_sha256']:
            components.union(row['image_pixel_sha256'][0], pixel)
    representatives = {row['pixel_sha256']: row for row in historical}
    representatives.update({row['pixel_sha256']: row for row in assets})
    historical_pixels = {row['pixel_sha256'] for row in historical}
    tree = HashTree()
    edges = []
    for pixel, row in sorted(representatives.items()):
        components.find(pixel)
        for other, distance in tree.near(int(row['dhash64'], 16), radius):
            components.union(pixel, other)
            edges.append({'left_pixel': other, 'right_pixel': pixel, 'distance': distance})
        tree.add(int(row['dhash64'], 16), pixel)
    historical_components = {components.find(pixel) for pixel in historical_pixels}
    assigned, excluded = [], []
    for row in candidates:
        component = components.find(row['image_pixel_sha256'][0])
        if component in historical_components:
            excluded.append({'item_id': row['item_id'], 'reason': 'exact_or_conservative_near_component_reaches_historical_image'})
            continue
        bucket = int(rank('component:' + component)[:12], 16) % 10000
        role = 'calibration' if bucket < 800 else 'audit' if bucket < 5400 else 'confirm'
        assigned.append({**row, 'role': role, 'image_group': 'conservative-image-component:' + component})
    return assigned, excluded, edges, components


def repair_role_quotas(assigned, targets=None):
    """Move complete image components only; never inspect predictions or scores."""
    def stratum(row):
        return (row['axis'], row['dataset'], row.get('subset', row.get('subject', '')),
                row['answer_key'] if row['axis'] == 'pope' else '')
    by_component = defaultdict(list)
    axis_strata = defaultdict(set)
    for row in assigned:
        by_component[row['image_group']].append(row)
        axis_strata[row['axis']].add(stratum(row)[:-1])
    needs = {}
    targets = {'calibration':120,'audit':1200,'confirm':1200} if targets is None else targets
    for role, target in targets.items():
        for axis, strata in axis_strata.items():
            if role == 'calibration' and axis == 'consistency':
                continue
            for index, key in enumerate(sorted(strata)):
                quota = target//len(strata) + int(index < target%len(strata))
                for label in ('yes','no') if axis == 'pope' else ('',):
                    needs[(role, key+(label,))] = quota//2 if axis == 'pope' else quota
    counts = Counter((row['role'],stratum(row)) for row in assigned)
    contributions = {key:Counter(stratum(row) for row in rows) for key,rows in by_component.items()}
    moves = []
    for _ in range(2000):
        deficits = [(required-counts[key],key) for key,required in needs.items() if counts[key] < required]
        if not deficits:
            return assigned,moves
        options = []
        for deficit,(recipient,key) in sorted(deficits,reverse=True):
            for group,parts in contributions.items():
                donor = by_component[group][0]['role']
                if donor == recipient or not parts[key]:
                    continue
                if any(counts[(donor,k)]-value < needs.get((donor,k),0) for k,value in parts.items()):
                    continue
                options.append((min(deficit,parts[key]),-abs(deficit-parts[key]),rank(group),group,donor,recipient))
        if not options:
            raise ValueError('Whole-component quota repair is infeasible with current public pool')
        _,_,_,group,donor,recipient = max(options)
        for key,value in contributions[group].items():
            counts[(donor,key)] -= value
            counts[(recipient,key)] += value
        for row in by_component[group]: row['role'] = recipient
        moves.append({'image_group':group,'from':donor,'to':recipient,'original_units':len(by_component[group])})
    raise RuntimeError('Quota repair iteration limit exceeded')


def legacy_consistency_calibration(historical):
    """Historical development only; new audit/confirmation exclude its whole graph."""
    by_path = {row['path']:row for row in historical}
    candidates = []
    for row in read_rows(ROOT/'data/consistency/base_items.jsonl'):
        facts = by_path[row['image_path']]
        candidates.append({'axis':'consistency','dataset':'mmbench-legacy-calibration','subset':'historical-dev',
                           'item_id':'legacy-calibration:'+row['base_item_id'],'question':row['question'],
                           'options':row['options'],'answer_key':row['answer_key'],'answer_type':'option_letter_v2',
                           'image_paths':[row['image_path']],'image_pixel_sha256':[facts['pixel_sha256']],
                           'image_group':'historical-calibration:'+facts['pixel_sha256'],'role':'calibration',
                           'width':row['width'],'height':row['height'],
                           'source_scope':'Historical development; never claimed as fresh independent calibration'})
    return select_axis(candidates,120)


def worker():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'complete.json').exists():
            done = read_json(OUT / 'complete.json')
            for role, value in done['manifests'].items():
                if sha(OUT / (role + '.jsonl')) != value['sha256']:
                    raise ValueError('Clean frozen manifest digest differs')
            print('Clean role manifests already verified complete', flush=True)
            return
        state = {'status': 'running', 'worker_script': Path(__file__).name, 'pid': os.getpid(),
                 'started_utc': datetime.now(timezone.utc).isoformat(), 'formal_training_allowed': False}
        write_json(OUT / 'status.json', state)
        try:
            hf = read_json(ROOT / 'runs/protocol-v2-data/candidate-index/complete.json')
            coco = read_json(ROOT / 'runs/protocol-v2-data/coco-candidates/complete.json')
            for name, expected in [('candidates.jsonl', hf['candidates_sha256']),
                                   ('image-assets.jsonl', hf['image_assets_sha256']),
                                   ('coco-candidates.jsonl', coco['candidates_sha256']),
                                   ('coco-image-assets.jsonl', coco['assets_sha256'])]:
                if sha(DATA / name) != expected:
                    raise ValueError('Candidate input digest changed')
            design = {'schema': 'arp.conservative-image-isolation.v4', 'dhash_radius': 4,
                      'human_near_duplicate_decisions': False,
                      'policy': 'Union every candidate image-pair within radius, union multi-image questions, exclude entire components reaching any historical image, then assign roles',
                      'outcome_dependence': False, 'targets': {'calibration': 120, 'audit': 1200, 'confirm': 1200},
                      'allocation': 'Hash roles followed by deterministic whole-component quota repair, preserving donor minima',
                      'calibration_consistency_source': '120 historical MMBench development units; preserves scarce clean AI2D test units for both 1200-unit audit and confirmation',
                      'historical_inventory_sha256': sha(DATA / 'historical-image-fingerprints.jsonl'),
                      'candidate_inputs': {'hf': hf['candidates_sha256'], 'coco': coco['candidates_sha256']},
                      'scope': 'Conservative dHash control; not a guarantee against arbitrary crops or semantic duplicates'}
            write_json(OUT / 'design.json', design)
            candidates = read_rows(DATA / 'candidates.jsonl') + read_rows(DATA / 'coco-candidates.jsonl')
            assets = read_rows(DATA / 'image-assets.jsonl') + read_rows(DATA / 'coco-image-assets.jsonl')
            historical = read_rows(DATA / 'historical-image-fingerprints.jsonl')
            assigned, excluded, edges, components = conservative_components(candidates, assets, historical)
            assigned,moves = repair_role_quotas(assigned)
            write_json(OUT/'component-role-moves.json',moves)
            write_rows(OUT / 'exclusions.jsonl', excluded)
            write_rows(OUT / 'near-component-edges.jsonl', edges)
            counts = Counter((row['role'], row['axis'], row['dataset'], row.get('subset', row.get('subject', ''))) for row in assigned)
            write_json(OUT / 'available-strata.json', [{'role': key[0], 'axis': key[1], 'dataset': key[2], 'subset': key[3], 'units': value} for key, value in sorted(counts.items())])
            by_path = {row['path']: row for row in assets}
            by_path.update({row['path']: row for row in historical})
            manifests, coverage, selected_assets, selected_components = {}, {}, {}, {}
            for role, target in design['targets'].items():
                selected = [row for axis in AXES for row in
                            (legacy_consistency_calibration(historical) if role == 'calibration' and axis == 'consistency' else
                             select_axis([r for r in assigned if r['role'] == role and r['axis'] == axis], target))]
                rows = [prompt for row in selected for prompt in expand_item(row)]
                write_rows(OUT / (role + '.jsonl'), rows)
                manifests[role] = {'sha256': sha(OUT / (role + '.jsonl')), 'prompt_rows': len(rows), 'original_units': len(selected)}
                coverage[role] = {axis: {'units': sum(r['axis'] == axis for r in selected),
                                        'image_components': len({r['image_group'] for r in selected if r['axis'] == axis}),
                                        'multi_image_units': sum(r['axis'] == axis and len(r['image_paths']) > 1 for r in selected),
                                        'open_answer_units': sum(r['axis'] == axis and r['answer_type'] == 'mmmu_open' for r in selected)} for axis in AXES}
                for row in selected:
                    group = row['image_group']
                    if group in selected_components and selected_components[group] != role:
                        raise ValueError('Conservative image component spans roles')
                    selected_components[group] = role
                    for path in row['image_paths']:
                        selected_assets[path] = by_path[path]
            train = read_rows(ROOT / 'data/training/pool.jsonl')
            for row in train:
                row.update(role='train', protocol_v2_role='historical_training_proxy')
            write_rows(OUT / 'train.jsonl', train)
            manifests['train'] = {'sha256': sha(OUT / 'train.jsonl'), 'prompt_rows': len(train),
                                  'original_units': len({r['bootstrap_unit'] for r in train})}
            write_rows(OUT / 'selected-image-assets.jsonl', sorted(selected_assets.values(), key=lambda row: row['path']))
            # Check every discovered near edge, not just pairs among selected rows.
            for edge in edges:
                if components.find(edge['left_pixel']) != components.find(edge['right_pixel']):
                    raise ValueError('Unmerged near-duplicate edge')
            done = {'status': 'complete', 'schema': design['schema'], 'manifests': manifests, 'coverage': coverage,
                    'excluded_candidate_units': len(excluded), 'conservative_near_edges': len(edges),
                    'cross_role_conservative_components': 0, 'historical_components_retained': 0,
                    'near_duplicate_human_review_complete': False,
                    'grounding_blind_human_review_complete': False, 'formal_training_allowed': False,
                    'selected_image_assets_sha256': sha(OUT / 'selected-image-assets.jsonl'),
                    'completed_utc': datetime.now(timezone.utc).isoformat()}
            write_json(OUT / 'complete.json', done)
            state.update(done)
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
        worker()
    else:
        token = sys.stdin.readline().strip()
        if not token:
            raise SystemExit('Completion-backup credential required')
        if '.arp-v3-clean-roles' in subprocess.run(['screen', '-ls'], capture_output=True, text=True).stdout:
            raise SystemExit('Clean-role worker already exists')
        OUT.mkdir(parents=True, exist_ok=True)
        command = ('source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; '
                   'exec /root/autodl-tmp/arp-venv/bin/python -u tools/freeze_clean_candidates_v3.py --worker >> ' + str(OUT / 'console.log') + ' 2>&1')
        subprocess.run(['screen', '-dmS', 'arp-v3-clean-roles', 'bash', '-c', command], env={**os.environ, 'GH_TOKEN': token}, check=True)
        print('Started conservative image-component isolation and role freezing', flush=True)
