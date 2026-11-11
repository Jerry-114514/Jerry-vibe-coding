"""CPU audit of frozen 9B 2x2 inputs, raw evaluations and registered effects."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean
import tarfile

import numpy as np

from audit_archived_training_chain import image_overlap, table, rows
from experiment_common import AXES, ROOT, read_json, read_rows, sha, score_generation, units, write_json
from qwen35_native_frontend import native_prompt, score_native_frontend

CONDS = ('P-D','P-G','N-D','N-G')
SEEDS = (42,137)
EFFECTS = {'coordinate':[-.5,-.5,.5,.5],'teacher':[.5,-.5,.5,-.5],'interaction':[-1,1,1,-1]}


def audit_inputs(run, teacher_pool, mapping, train_chain):
    pool = {x['item_id']:x for x in teacher_pool}
    result = {}
    for seed in SEEDS:
        terms = {}
        for cond in CONDS:
            path = run/f'registration/training-inputs/{cond}-seed{seed}.jsonl'
            rel = f'runs/9b-u137-factorial-20261018/training-inputs/{path.name}'
            assert sha(path)==mapping[rel]['sha256']
            terms[cond] = read_rows(path)
        base = terms['P-D']
        with tarfile.open(train_chain) as t:
            source = json.loads(table(t,f'/formal/U-seed{seed}/train.meta.json'))
        assert len(base)==len({r['item_id'] for r in base})==600
        assert [r['item_id'] for r in base]==source['sampling_ids']
        stats = Counter()
        delta = {k:[] for k in ('prompt_tokens','gt_target_ids','teacher_target_ids')}
        eos = Counter()
        for i,pd in enumerate(base):
            common = ('item_id','axis','bootstrap_unit','image_path','source_image_sha256','answer_key','width','height','processor_grid_thw')
            source = pool[pd['item_id']]
            assert pd['image_path']==source['image_path']
            assert pd['source_image_sha256']==mapping[pd['image_path']]['sha256']
            for cond in CONDS:
                other = terms[cond][i]
                assert all(other[k]==pd[k] for k in common)
                assert other['repair_condition']==cond and other['teacher_loss_enabled']==cond.endswith('-D')
                assert other['question']==(native_prompt(source)['question'] if cond.startswith('N-') else source['question'])
                compare = terms[cond[0]+'-D'][i]
                assert all(other[k]==compare[k] for k in ('question','prompt_ids_sha256','gt_target_ids','teacher_target_ids','gt_target','teacher_target'))
                for target in ('gt_target_ids','teacher_target_ids'):
                    ids = other[target]
                    assert 0<len(ids)<=128 and other['prompt_tokens']+len(ids)<=2048
                    eos[str(ids[-1])] += 1
            nd = terms['N-D'][i]
            if pd['axis']=='grounding':
                for label in ('gt_target','teacher_target'):
                    box = json.loads(pd[label])
                    expected = [x/(pd['width'] if j%2==0 else pd['height'])*1000 for j,x in enumerate(box)]
                    assert json.loads(nd[label])==expected
                stats['grounding_rows'] += 1
                stats['grounding_prompt_hash_changed'] += nd['prompt_ids_sha256']!=pd['prompt_ids_sha256']
                stats['grounding_gt_ids_changed'] += nd['gt_target_ids']!=pd['gt_target_ids']
                for key in delta:
                    delta[key].append((nd[key]-pd[key]) if key=='prompt_tokens' else len(nd[key])-len(pd[key]))
            else:
                assert all(nd[k]==pd[k] for k in ('question','prompt_ids_sha256','gt_target_ids','teacher_target_ids'))
                stats['non_grounding_rows_identical'] += 1
        result[str(seed)] = {**stats,'target_last_token_id_counts':dict(eos),
            'normalized_minus_pixel_token_lengths':{k:{'min':min(v),'mean':mean(v),'max':max(v)} for k,v in delta.items()}}
    return result


def evaluation(run, manifest, name, dataset, old_tar, protocol_digest):
    if dataset=='development' and (name.startswith('P-D') or name=='N-seed42'):
        model = name.replace('P-D','U')
        folder = '/native-frontend-evaluation-v41/formal/'+model
        marker = json.loads(table(old_tar,folder+'/scores.json'))
        raw = table(old_tar,folder+'/eval.jsonl')
    else:
        folder = run/'evaluations'/dataset/name
        raw = (folder/'eval.jsonl').read_bytes()
        marker = read_json(folder/'scores.json')
        meta = read_json(folder/'eval.meta.json')
        assert meta['protocol_sha256']==marker['protocol_sha256']==protocol_digest
    import hashlib
    assert hashlib.sha256(raw).hexdigest()==marker['ledger_sha256']
    ledger = rows(raw)
    lookup = {r['item_id']:r for r in manifest}
    assert len(lookup)==len(manifest)==len(ledger)==marker['rows']
    for row in ledger:
        item = lookup[row['item_id']]
        scorer = score_native_frontend if item['axis']=='grounding' else score_generation
        assert all(row[k]==v for k,v in scorer(row['raw_generation'],item).items())
        ids = row['generated_token_ids']
        assert len(ids)==row['generation_tokens']<=128
        assert row['finish_reason']==('eos' if ids and ids[-1] in (248046,248044) else 'length')
    grouped = units(manifest,ledger)
    scores = {axis:mean(r['correct_bit'] for r in grouped if r['axis']==axis) for axis in AXES}
    assert scores==marker['axis_scores']
    return marker,{(r['axis'],r['dataset'],r['unit']):r for r in grouped}


def audit(cache):
    run = cache/'selective/arp-9b-122b-20261017/runs/9b-u137-factorial-20261018'
    evidence = read_json(ROOT/'docs/results/protocol-v2-20261016/9b-repair-final/report/results.json')
    assert sha(run/'protocol.json')==evidence['protocol_sha256']
    pool = read_rows(next((cache/'selective').rglob('eligible-pool.jsonl')))
    mapping = {r['path']:r for r in json.loads(gzip.decompress((cache/'map.json.gz').read_bytes()))['files']}
    manifest = {d:read_rows(run/(d+'-manifest.jsonl')) for d in ('development','heldout')}
    assert sha(run/'heldout-manifest.jsonl')==read_json(run/'heldout-novelty-verification.json')['heldout_manifest_sha256']
    inputs = audit_inputs(run,pool,mapping,next(cache.glob('*140724Z.tar.gz')))
    isolation = image_overlap(mapping,pool,{'heldout':manifest['heldout']})
    historical_isolation = image_overlap(mapping,manifest['development'],{'heldout':manifest['heldout']})
    rng = np.random.default_rng(2026101804)
    seed_indices = rng.integers(0,2,size=(10000,2))
    out = {}
    with tarfile.open(next(cache.glob('*070813Z.tar.gz'))) as old:
        for dataset, items in manifest.items():
            names = [f'{c}-seed{s}' for s in SEEDS for c in CONDS]
            values = {n:evaluation(run,items,n,dataset,old,evidence['protocol_sha256']) for n in names+['N-seed42']}
            effects = {}
            reference = values['N-seed42'][1]
            for axis in AXES:
                keys = sorted(k for k in reference if k[0]==axis)
                groups = sorted({reference[k]['image_group'] for k in keys})
                index = {g:i for i,g in enumerate(groups)}
                totals = np.zeros((len(groups),8))
                counts = np.zeros(len(groups))
                for k in keys:
                    j = index[reference[k]['image_group']]
                    counts[j] += 1
                    totals[j] += [values[n][1][k]['correct_bit'] for n in names]
                points = totals.sum(0)/counts.sum()
                draws = np.empty((10000,8))
                for start in range(0,10000,40):
                    w = rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=40)
                    draws[start:start+40] = (w@totals)/(w@counts)[:,None]
                cpoint = np.array([points[i::4].mean() for i in range(4)])
                cdraw = np.column_stack([draws[np.arange(10000)[:,None],4*seed_indices+i].mean(1) for i in range(4)])
                for effect, coeff in EFFECTS.items():
                    row = {'difference':float(cpoint@coeff),
                           'pointwise_ci95':np.quantile(cdraw@coeff,[.025,.975]).tolist(),
                           'bonferroni24_ci95':np.quantile(cdraw@coeff,[.05/48,1-.05/48]).tolist()}
                    target = evidence['datasets'][dataset]['effects'][effect][axis]
                    assert all(np.allclose(row[k],target[k],atol=1e-12,rtol=0) for k in row),(dataset,axis,effect)
                    effects.setdefault(effect,{})[axis] = row
            out[dataset] = {'effects':effects,'replayed_rows':sum(x[0]['rows'] for x in values.values())}
    return {'frozen_training_inputs':inputs,'heldout_vs_eligible_pool':isolation,'heldout_vs_development':historical_isolation,
            'datasets':out,'registered_effect_intervals_reproduced':24,
            'scope':'CPU replay of frozen token arrays and saved raw generations, not tokenizer re-encoding or training-state tensor replay.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    report=audit(a.cache)
    write_json(a.output,report)
    print(json.dumps(report['frozen_training_inputs'],indent=2))
    print('All 24 registered effects reproduced.')
