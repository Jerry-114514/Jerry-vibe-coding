"""Measure confirmation-pool contamination and capacity without freezing a new test."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import tarfile
import unicodedata

from audit_archived_image_fingerprints_and_7b_cache import from_archive
from experiment_common import AXES, read_rows, write_json
from independent_manifest_v2 import select_axis


def canon(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def text_choices(row):
    if row['axis'] not in ('consistency','mmstar'):
        return None
    q = row['question']
    opts = row.get('options')
    if not isinstance(opts,dict):
        if '\nOptions:' in q:
            q,choices = q.split('\nOptions:',1)
            choices = choices.split('\nReturn only',1)[0].strip()
            matches = re.split(r'(?:^|,\s*)([A-D])\s*:\s*',choices)
            opts = dict(zip(matches[1::2],matches[2::2]))
        else:
            opts = dict(re.findall(r'^([A-D])[.)] (.+)$',q,re.M))
            if opts:
                q = re.split(r'^([A-D])[.)] ',q,maxsplit=1,flags=re.M)[0]
    q = q.split('\nChoices:\n')[0]
    return (canon(q),tuple(sorted(canon(v) for v in opts.values()))) if opts else None


def audit(cache, *, candidate_export=None):
    manifest = {r:from_archive(cache,'094432','/'+r+'.jsonl') for r in ('calibration','audit','confirm','train')}
    known_eval = from_archive(cache,'070813','/formal-manifest.jsonl')
    old_heldout = read_rows(next((cache/'selective').rglob('heldout-manifest.jsonl')))
    raw = {r['item_id']:r for r in from_archive(cache,'085925','/candidates.jsonl')}
    original = defaultdict(list)
    for r in manifest['confirm']:
        original[r['bootstrap_unit']].append(r)
    eval_id,eval_base = {r['item_id'] for r in old_heldout},{r['bootstrap_unit'] for r in old_heldout}
    eval_group = {r['image_group'] for r in old_heldout}
    exact_old = len(eval_id & {r['item_id'] for r in manifest['confirm']})
    assert exact_old==len(old_heldout)==960
    seen_pixels = {p for r in old_heldout for p in r['image_pixel_sha256']}
    archives = []
    archived_seen = set()
    confirm_key = lambda s: s if s in original else re.sub(r':p[1-5]$','',s)

    def record(name, lines):
        ids = {r['item_id'] for r in map(json.loads,lines) if 'item_id' in r}
        matched = {i for i in ids if confirm_key(i) in original}
        archived_seen.update(confirm_key(i) for i in matched)
        archives.append({'member':name,'unique_ids':len(ids),'matched_prompt_ids':len(matched)})

    source_stamps = ('111554','105344','104906','054559','054046','093649','094134','095314','095548','103329')
    for stamp in source_stamps:
        for path in cache.glob('*'+stamp+'Z.tar.gz'):
            with tarfile.open(path) as t:
                for entry in t:
                    if not entry.isfile() or not entry.name.endswith('/ledger.jsonl'):
                        continue
                    record(entry.name,t.extractfile(entry))
    retirement = cache/'selective/arp-instance-retirement-20261020'
    for path in sorted(retirement.rglob('ledger.jsonl')):
        with path.open(encoding='utf-8') as file:
            record(path.relative_to(retirement).as_posix(),file)
    prior_groups = {original[k][0]['image_group'] for k in archived_seen}
    historical_text = set()
    for r in manifest['train']+manifest['calibration']+manifest['audit']+known_eval+old_heldout:
        item = raw.get(r.get('bootstrap_unit'),r)
        key = text_choices(item)
        if key:
            historical_text.add(key)
    reasons = Counter()
    kept = defaultdict(list)
    decisions = []
    for k,group in original.items():
        r = group[0]
        a = r['axis']
        assert len(group)==(5 if a=='consistency' else 1)
        if k in eval_base or r['image_group'] in eval_group or any(p in seen_pixels for p in r['image_pixel_sha256']):
            why = 'old_heldout_image_component'
        elif r['image_group'] in prior_groups:
            why = 'prior_calibration_or_audit_ledger_id'
        elif len(r['image_paths'])!=1 or (a in ('consistency','mmstar') and (
              not r['options'] or not set(r['options'])<=set('ABCD') or r['answer_key'] not in r['options'])):
            why = 'not_single_image_public_ABCD'
        elif text_choices(raw.get(k,r)) in historical_text:
            why = 'choice_question_and_option_set_reuse'
        else:
            kept[a].append(r)
            if candidate_export is not None:
                decisions.append({'original_id': k, 'axis': a, 'component': r['image_group'],
                                  'rows': group, 'exclusion': None})
            continue
        reasons[a,why] += 1
        if candidate_export is not None:
            decisions.append({'original_id': k, 'axis': a, 'component': r['image_group'],
                              'exclusion': why})
    if candidate_export is not None:
        candidate_export.extend(decisions)
    feasibility = {}
    for a in AXES:
        groups = Counter((r['dataset'],r['subset']) for r in kept[a])
        try:
            picks = select_axis(kept[a],120)
        except ValueError as err:
            feasibility[a] = {'feasible':False,'reason':str(err),'remaining':len(kept[a])}
        else:
            feasibility[a] = {'feasible':True,'remaining':len(kept[a]),'selected_count':len(picks),
                              'selected_components':len({x['image_group'] for x in picks})}
        feasibility[a]['remaining_strata'] = {'/'.join(k):n for k,n in sorted(groups.items())}
        if a=='mmstar' and groups:
            # Pre-outcome design alternative: capped equal quota, redistribute only
            # to available strata in a fixed alphabetical order, preserve all strata.
            wanted = 120
            q,extra = divmod(wanted,len(groups))
            quota = {k:min(groups[k],q+(i<extra)) for i,k in enumerate(sorted(groups))}
            while sum(quota.values())<wanted and any(quota[k]<groups[k] for k in quota):
                for k in quota:
                    if sum(quota.values())<wanted and quota[k]<groups[k]:
                        quota[k]+=1
            feasibility[a]['unfrozen_capped_quota_alternative'] = {
                'target_met':sum(quota.values())==wanted,
                'counts':{'/'.join(k):v for k,v in quota.items()},
                'all_subjects_retained':all(v>0 for v in quota.values())}
    return {'old_heldout_prompt_overlap':exact_old,'old_heldout_originals':len(eval_base),
            'old_heldout_components':len(eval_group),'available_compact_ledger_checks':archives,
            'missing_expected_compact_archives':[s for s in source_stamps if not list(cache.glob('*'+s+'Z.tar.gz'))],
            'archived_exposed_confirm_originals':sorted(archived_seen),
            'exclusion_counts':{a:{why:n for (ax,why),n in reasons.items() if ax==a} for a in AXES},
            'feasibility':feasibility,
            'scope':'Conservative planning exclusions only. Not a complete archived inference inventory, no new confirmation manifest or human review.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    report=audit(a.cache)
    write_json(a.output,report)
    print({k:{x:y for x,y in v.items() if x!='remaining_strata'} for k,v in report['feasibility'].items()})
    print('Old heldout overlap',report['old_heldout_prompt_overlap'],'prior ledger overlaps',len(report['archived_exposed_confirm_originals']))
