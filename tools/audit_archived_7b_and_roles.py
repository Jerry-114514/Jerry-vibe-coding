"""Re-score the frozen 7B processor-pixel ledger and check v4 role manifests."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import tarfile

from audit_v41_archived_dataflow import digest
from processor_pixel_contract_csv import score


def read(tar, root, suffix):
    return tar.extractfile(f'{root}/{suffix}').read()


def jsonl(raw):
    return [json.loads(row) for row in raw.splitlines()]


def audit_7b(path, receipt):
    asset = next(a for a in receipt['receipt']['assets'] if a['name'] == path.name)
    assert digest(path.read_bytes()) == asset['sha256']
    root = 'runs/7b-processor-pixel-audit-20261017'
    with tarfile.open(path) as tar:
        raw = read(tar, root, 'manifest.jsonl')
        assert raw == read(tar, root, 'registration/manifest.jsonl')
        protocol = read(tar, root, 'protocol.json')
        assert digest(protocol) == receipt['protocol_sha256']
        manifest = jsonl(raw)
        lookup = {i['item_id']:i for i in manifest}
        assert len(lookup) == len(manifest) == receipt['rows_per_model']
        folders = [n.removeprefix(root+'/').removesuffix('/eval.jsonl') for n in tar.getnames()
                   if n.startswith(root+'/models/') and n.endswith('/eval.jsonl')]
        assert len(folders) == len(set(folders)) == receipt['models']
        scores = {}
        for folder in folders:
            raw_ledger = read(tar, root, folder+'/eval.jsonl')
            rows = jsonl(raw_ledger)
            marker = json.loads(read(tar, root, folder+'/scores.json'))
            assert marker['protocol_sha256'] == digest(protocol)
            assert digest(raw_ledger) == marker['ledger_sha256']
            assert len(rows) == len({r['item_id'] for r in rows}) == len(lookup) == marker['rows']
            assert {r['item_id'] for r in rows} == set(lookup)
            hits = invalid = 0
            for r in rows:
                check = score(r['raw_generation'],lookup[r['item_id']])
                assert all(r[k] == v for k,v in check.items())
                hits += check['correct_bit']
                invalid += check['parse_fail_flag']
            assert hits/len(rows) == marker['accuracy'] and invalid == marker['parse_failures']
            scores[folder.split('/')[-1]] = {'accuracy':hits/len(rows),'parse_failures':invalid,'rows':len(rows)}
        means = {arm:sum(v['accuracy'] for k,v in scores.items() if k.startswith(arm+'-'))/5
                 for arm in ('A1','A2','U')}
        return {'manifest_sha256':digest(raw),'scores':scores,'means':means,'A1_minus_U':means['A1']-means['U']}


def audit_roles(path, expected):
    root = 'runs/protocol-v2-data/clean-role-manifests-v4'
    with tarfile.open(path) as tar:
        assets = {r['path']:r for r in jsonl(read(tar, root, 'selected-image-assets.jsonl'))}
        all_roles = {}
        details = {}
        for role, meta in expected['manifests'].items():
            raw = read(tar,root,role+'.jsonl')
            assert digest(raw) == meta['sha256']
            rows = jsonl(raw)
            assert len(rows) == len({r['item_id'] for r in rows}) == meta['prompt_rows']
            ids = {(r['axis'],r['dataset'],r['bootstrap_unit']) for r in rows}
            assert len(ids) == meta['original_units']
            groups = {r['image_group'] for r in rows if 'image_group' in r}
            pixels = set()
            missing_assets = set()
            for r in rows:
                paths = r.get('image_paths') or [r['image_path']]
                known = [assets[p]['pixel_sha256'] for p in paths if p in assets]
                missing_assets.update(p for p in paths if p not in assets)
                if 'image_pixel_sha256' in r:
                    assert known == r['image_pixel_sha256']
                pixels.update(known)
            all_roles[role] = {'units':ids,'pixels':pixels,'groups':groups}
            details[role] = {'prompts':len(rows),'units':len(ids),'unit_axes':dict(Counter(x[0] for x in ids)),
                             'known_pixel_hashes':len(pixels),'paths_not_in_selected_inventory':len(missing_assets)}
        overlaps = {}
        for a,b in combinations(all_roles,2):
            overlaps[f'{a}/{b}'] = {
                'units':len(all_roles[a]['units'] & all_roles[b]['units']),
                'pixels':None if any(details[r]['paths_not_in_selected_inventory'] for r in (a,b))
                    else len(all_roles[a]['pixels'] & all_roles[b]['pixels']),
                'groups':len(all_roles[a]['groups'] & all_roles[b]['groups'])
                    if all_roles[a]['groups'] and all_roles[b]['groups'] else None,
            }
        return {'roles':details,'observed_overlaps':overlaps,
                'scope':'Archived manifests and indexed image hashes only; no pixel rehashing or near-graph reconstruction.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive_7b',type=Path)
    p.add_argument('receipt_7b',type=Path)
    p.add_argument('archive_roles',type=Path)
    p.add_argument('receipt_roles',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    result = {'7b':audit_7b(a.archive_7b,json.loads(a.receipt_7b.read_text(encoding='utf-8'))),
              'roles':audit_roles(a.archive_roles,json.loads(a.receipt_roles.read_text(encoding='utf-8')))}
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'7b_A1_minus_U':result['7b']['A1_minus_U'],**result['roles']},indent=2))
