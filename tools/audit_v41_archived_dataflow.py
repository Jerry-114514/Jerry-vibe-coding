"""CPU-only replay of the frozen 9B v4.1 ledger from the retirement archive."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tarfile

from experiment_common import AXES, score_generation, units
from qwen35_native_frontend import parse_native_box, score_native_frontend


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def direct_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / union


def replay(path, receipt):
    expected = next(a for a in receipt['receipt']['assets'] if a['name'] == path.name)
    assert path.stat().st_size == expected['bytes'] and digest(path.read_bytes()) == expected['sha256']
    with tarfile.open(path) as tar:
        names = set(tar.getnames())
        (manifest_path,) = [x for x in names if x.endswith('/formal-manifest.jsonl') and
                            not x.endswith('/registration/formal-manifest.jsonl')]
        root = manifest_path.rsplit('/', 1)[0]

        def read(name):
            return tar.extractfile(f'{root}/{name}').read()

        def jsonl(raw):
            return [json.loads(row) for row in raw.splitlines()]

        manifest_raw = read('formal-manifest.jsonl')
        assert manifest_raw == read('registration/formal-manifest.jsonl')
        protocol = read('protocol.json')
        assert digest(protocol) == receipt['protocol_sha256']
        assert json.loads(protocol)['manifest_sha256'] == digest(manifest_raw)
        manifest = jsonl(manifest_raw)
        lookup = {x['item_id']:x for x in manifest}
        assert len(lookup) == len(manifest) == 16900
        by_unit = defaultdict(list)
        for item in manifest:
            by_unit[item['axis'],item['dataset'],item['bootstrap_unit']].append(item)
        for (axis, _, _), group in by_unit.items():
            assert len(group) == (5 if axis == 'consistency' else 1)
            assert len({i['image_group'] for i in group}) == 1
        result = {'archive_sha256':expected['sha256'],'manifest_sha256':digest(manifest_raw),
                  'protocol_sha256':digest(protocol),'manifest_rows_by_axis':dict(Counter(x['axis'] for x in manifest)),
                  'original_units_by_axis':dict(Counter(k[0] for k in by_unit)), 'models':{}}
        model_names = sorted(n.split('/')[-2] for n in names if n.startswith(root+'/formal/') and n.endswith('/eval.jsonl'))
        assert len(set(model_names)) == len(model_names) == receipt['models']
        for model in model_names:
            prefix = f'formal/{model}'
            raw = read(f'{prefix}/eval.jsonl')
            scores = json.loads(read(f'{prefix}/scores.json'))
            assert digest(raw) == scores['ledger_sha256']
            ledger = jsonl(raw)
            assert len(ledger) == len({r['item_id'] for r in ledger}) == scores['rows'] == len(manifest)
            assert {r['item_id'] for r in ledger} == set(lookup)
            failures = Counter()
            per_subset = defaultdict(Counter)
            truncations = 0
            non_eos_by_axis = Counter()
            credited_non_eos_by_axis = Counter()
            direct_max_delta = 0.0
            direct_hits = 0
            for row in ledger:
                item = lookup[row['item_id']]
                assert row['axis'] == item['axis']
                ids = row['generated_token_ids']
                assert row['generation_tokens'] == len(ids) <= 128
                finish = 'eos' if ids and ids[-1] in (248046,248044) else 'length'
                assert row['finish_reason'] == finish
                truncations += finish == 'length'
                if finish == 'length':
                    non_eos_by_axis[row['axis']] += 1
                    credited_non_eos_by_axis[row['axis']] += row['correct_bit']
                if row['axis'] == 'grounding':
                    rescored = score_native_frontend(row['raw_generation'], item)
                    native = parse_native_box(row['raw_generation'])
                    if native is not None:
                        original = [v*item['width' if i%2==0 else 'height']/1000 for i,v in enumerate(native)]
                        iou = direct_iou(original,item['answer_key'])
                        direct_max_delta = max(direct_max_delta,abs(iou-rescored['iou']))
                        assert math.isclose(iou,rescored['iou'],rel_tol=0,abs_tol=1e-12)
                        direct_hits += iou >= 0.5
                    split = f"{item['dataset']}/{item.get('subset','')}"
                    per_subset[split].update({'n':1,'hits':rescored['correct_bit'],'parse_failures':rescored['parse_fail_flag']})
                else:
                    rescored = score_generation(row['raw_generation'],item)
                assert all(row[k] == v for k,v in rescored.items()), (model,row['item_id'])
                failures[row['axis']] += rescored['parse_fail_flag']
            grouped = units(manifest,ledger)
            recomputed = {axis:sum(x['correct_bit'] for x in grouped if x['axis']==axis) /
                          sum(x['axis']==axis for x in grouped) for axis in AXES}
            assert recomputed == scores['axis_scores']
            assert direct_hits == sum(r['correct_bit'] for r in ledger if r['axis']=='grounding')
            assert truncations == scores['length_capped_outputs']
            strict_units = units(manifest, [
                {**r,'correct_bit':0} if r['finish_reason'] == 'length' else r for r in ledger
            ])
            strict_scores = {axis:sum(x['correct_bit'] for x in strict_units if x['axis']==axis) /
                             sum(x['axis']==axis for x in strict_units) for axis in AXES}
            result['models'][model] = {'rows':len(ledger),'ledger_sha256':digest(raw),
                'scores':recomputed,'parse_failures':dict(failures),'non_eos_count':truncations,
                'non_eos_by_axis':dict(non_eos_by_axis),'credited_non_eos_by_axis':dict(credited_non_eos_by_axis),
                'sensitivity_non_eos_as_incorrect':strict_scores,
                'direct_iou_max_abs_delta':direct_max_delta,'grounding_subsets':dict(per_subset)}
        assert sum(x['rows'] for x in result['models'].values()) == receipt['rows']
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('receipt',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    result = replay(a.archive,json.loads(a.receipt.read_text(encoding='utf-8')))
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f"Replayed {len(result['models'])} models, {sum(x['rows'] for x in result['models'].values())} rows.")
