"""Post-hoc exclusion of all dHash candidates; no claim of confirmed leakage."""
import argparse
import gzip
import json
from pathlib import Path

from audit_archived_paired_intervals import aliases, load, paired
from audit_archived_image_fingerprints_and_7b_cache import from_archive
from audit_archived_training_chain import image_paths
from experiment_common import AXES, read_json, write_json


def audit(cache):
    proof = read_json(cache/'image-and-7b-cache-replay.json')
    mapping = {r['path']:r for r in json.loads(gzip.decompress((cache/'map.json.gz').read_bytes()))['files']}
    pairs = proof['images']['pairs']['historical_v41']
    fingerprints = {r['path']:r['pixel_sha256'] for r in
                    from_archive(cache,'085925','/historical-image-fingerprints.jsonl')}
    flagged = {fingerprints[p['eval_path']] for p in pairs}
    manifest, scoring = load(next(cache.glob('*070813Z.tar.gz')),'/native-frontend-evaluation-v41',
                            '/formal-manifest.jsonl','/formal/')
    excluded = {(i['axis'],i['dataset'],i['bootstrap_unit']) for i in manifest
                if any(fingerprints[p] in flagged for p in image_paths([i]))}
    identity = lambda r:(r['axis'],r['dataset'],r['unit'])
    filtered = [[r for r in model if identity(r) not in excluded] for model in scoring]
    assert excluded <= {identity(r) for r in scoring[0]}
    keys = ('A1-A2','A1-U','A1-N')
    same_grouping = aliases(manifest,mapping)
    pre = paired(scoring,AXES,2026101703,12,keys,same_grouping)
    post = paired(filtered,AXES,2026101703,12,keys,same_grouping)
    return {'excluded_image_fingerprints':len(flagged),
            'excluded_original_units':{a:sum(k[0]==a for k in excluded) for a in AXES},
            'byte_alias_clustered_full_set':pre,'byte_alias_clustered_exclude_all_candidates':post,
            'scope':'Post-hoc maximal exclusion of dHash distance<=4 candidates, not evidence that any pair is identical. Fixed full-set score remains primary.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    out = audit(a.cache)
    write_json(a.output,out)
    print(out['excluded_original_units'])
