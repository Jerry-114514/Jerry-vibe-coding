"""Independent acceptance of clean split contents and source fingerprints."""
from collections import defaultdict
from pathlib import Path

from experiment_common import ROOT, AXES, read_json, read_rows, sha, write_json
from freeze_clean_candidates_v3 import conservative_components
from index_independent_sources import image_facts

SOURCE=ROOT/'runs/protocol-v2-data/clean-role-manifests-v4'
DATA=ROOT/'data/protocol-v2'


def main():
    done=read_json(SOURCE/'complete.json')
    historical=read_rows(DATA/'historical-image-fingerprints.jsonl')
    historical_pixels={r['pixel_sha256'] for r in historical}
    candidates=read_rows(DATA/'candidates.jsonl')+read_rows(DATA/'coco-candidates.jsonl')
    assets=read_rows(DATA/'image-assets.jsonl')+read_rows(DATA/'coco-image-assets.jsonl')
    _,_,_,components=conservative_components(candidates,assets,historical)
    excluded_components={components.find(pixel) for pixel in historical_pixels}
    metadata={r['path']:r for r in read_rows(SOURCE/'selected-image-assets.jsonl')}
    if sha(SOURCE/'selected-image-assets.jsonl')!=done['selected_image_assets_sha256']:
        raise ValueError('Selected image metadata digest changed')
    pixel_role={};group_role={};checked=set();counts={}
    for role,target in [('calibration',120),('audit',1200),('confirm',1200)]:
        path=SOURCE/(role+'.jsonl')
        if sha(path)!=done['manifests'][role]['sha256']: raise ValueError('Manifest digest changed')
        rows=read_rows(path)
        if len({r['item_id'] for r in rows})!=len(rows): raise ValueError('Duplicate prompt identity')
        units=defaultdict(list)
        for row in rows:
            if row['role']!=role: raise ValueError('Wrong role tag')
            units[(row['axis'],row['bootstrap_unit'])].append(row)
            group=row['image_group']
            if group in group_role and group_role[group]!=role: raise ValueError('Image component spans roles')
            group_role[group]=role
            for relative,pixel in zip(row['image_paths'],row['image_pixel_sha256']):
                if metadata[relative]['pixel_sha256']!=pixel: raise ValueError('Image metadata differs')
                if pixel in pixel_role and pixel_role[pixel]!=role: raise ValueError('Exact image spans roles')
                pixel_role[pixel]=role
                legacy=role=='calibration' and row['dataset']=='mmbench-legacy-calibration'
                if not legacy and components.find(pixel) in excluded_components:
                    raise ValueError('New split contains a component linked to historical images')
                if not legacy and group!='conservative-image-component:'+components.find(pixel):
                    raise ValueError('Manifest image component disagrees with recomputed graph')
                if relative not in checked:
                    facts=image_facts((ROOT/relative).read_bytes())
                    if any(facts[key]!=metadata[relative][key] for key in ('source_sha256','pixel_sha256','dhash64','width','height','bytes')):
                        raise ValueError('Source image bytes, pixels or dimensions changed')
                    checked.add(relative)
        counts[role]={axis:sum(key[0]==axis for key in units) for axis in AXES}
        if any(value!=target for value in counts[role].values()): raise ValueError('Original-unit quota not met')
        for (axis,_),items in units.items():
            if len(items)!=(5 if axis=='consistency' else 1): raise ValueError('Wrong number of paraphrases')
            if len({str(r['answer_key']) for r in items})!=1: raise ValueError('Paraphrase labels disagree')
    if checked!=set(metadata): raise ValueError('Selected image inventory is not exact')
    original=read_rows(ROOT/'data/training/pool.jsonl')
    train=read_rows(SOURCE/'train.jsonl')
    if [{k:v for k,v in row.items() if k not in ('role','protocol_v2_role')} for row in train]!=original:
        raise ValueError('Historical approved training pool changed')
    report={'status':'passed','manifests':counts,'source_images_verified':len(checked),
            'cross_role_exact_image_overlap':0,'cross_role_conservative_component_overlap':0,
            'new_audit_or_confirm_components_linked_to_history':0,
            'legacy_development_calibration_explicitly_exempted':True,
            'checks':['all_frozen_manifest_hashes','exact_unit_quotas','five_paraphrases_per_original',
                      'recomputed_conservative_graph','full_source_byte_rgb_dhash_dimensions',
                      'exact_input_inventory_coverage','unchanged_approved_training_pool'],
            'human_grounding_review_complete':False,'formal_training_allowed':False}
    write_json(SOURCE/'verification.json',report)
    print(report)


if __name__=='__main__': main()
