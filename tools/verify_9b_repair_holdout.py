"""Membership/image exposure audit; never reads correctness to choose examples."""
import json
from pathlib import Path

from experiment_common import ROOT,read_json,read_rows,write_json,sha
from prepare_9b_repair import OUT,BASE,now


def verify():
    selection=read_json(OUT/'heldout-selection.json')
    targets=read_rows(OUT/'heldout-manifest.jsonl')
    wanted_ids={r['item_id'] for r in targets}
    wanted_paths=set(selection['selected_images'])
    wanted_bytes={r['source_sha256'] for r in selection['selected_images'].values()}
    inferred_ids=set();observed_paths=set();observed_bytes=set();files=[]
    for filename in ('eval.jsonl','ledger.jsonl','teacher.jsonl'):
        for path in sorted((ROOT/'runs').rglob(filename)):
            if path.is_relative_to(OUT):continue
            with path.open() as stream:
                for line in stream:
                    if not line.endswith('\n'):break # Snapshot an actively appended journal.
                    row=json.loads(line)
                    if row.get('item_id'):inferred_ids.add(row['item_id'])
                    for image in row.get('image_hashes',[]):
                        if image.get('sha256'):observed_bytes.add(image['sha256'])
                        if image.get('path'):observed_paths.add(image['path'])
                    if row.get('image_path'):observed_paths.add(row['image_path'])
                    observed_paths.update(row.get('image_paths',[]))
            files.append({'path':str(path.relative_to(ROOT)),'bytes_at_snapshot':path.stat().st_size})
    # Link inference identities back to immutable prompt manifests, including API role manifests.
    candidates=set((ROOT/'runs').rglob('*manifest*.jsonl'))
    candidates.update([ROOT/'data/manifest.jsonl',ROOT/'data/training/pool.jsonl',BASE/'eligible-pool.jsonl'])
    manifests=[]
    for path in sorted(candidates):
        if path.is_relative_to(OUT):continue
        matched=0
        for row in read_rows(path):
            if row.get('item_id') not in inferred_ids:continue
            matched+=1
            if row.get('image_path'):observed_paths.add(row['image_path'])
            observed_paths.update(row.get('image_paths',[]))
        manifests.append({'path':str(path.relative_to(ROOT)),'matched_inferred_ids':matched})
    # Paths from the selected canonical image inventory are byte-addressed; verify
    # aliases through the existing metadata instead of assuming a path is identity.
    aliases=set()
    for file in ('image-assets.jsonl','coco-image-assets.jsonl','historical-image-fingerprints.jsonl'):
        for row in read_rows(ROOT/'data/protocol-v2'/file):
            if row['path'] in observed_paths:observed_bytes.add(row['source_sha256'])
            if row.get('pixel_sha256') in {v['pixel_sha256'] for v in selection['selected_images'].values()} and row['path'] in observed_paths:
                aliases.add(row['path'])
    assert not wanted_ids&inferred_ids,'Holdout prompt identity was previously inferred'
    assert not wanted_paths&observed_paths,'Holdout image path was previously exposed'
    assert not wanted_bytes&observed_bytes,'Holdout image bytes were previously exposed'
    assert not aliases,'Holdout RGB alias was previously exposed'
    result={'status':'passed','timestamp_utc':now(),'prompt_ids_checked':len(wanted_ids),
        'image_paths_checked':len(wanted_paths),'prior_inference_ids':len(inferred_ids),
        'prior_observed_image_paths':len(observed_paths),'inference_files':files,'prompt_manifests':manifests,
        'prior_prompt_exact_overlap':0,'prior_image_path_overlap':0,'prior_image_bytes_overlap':0,
        'prior_selected_RGB_alias_overlap':0,'source_history_exact_dhash_exclusion_already_verified':True,
        'selection_did_not_consult_scores':True,'heldout_manifest_sha256':sha(OUT/'heldout-manifest.jsonl')}
    write_json(OUT/'heldout-novelty-verification.json',result)
    print({k:v for k,v in result.items() if k not in ('inference_files','prompt_manifests')},flush=True)
    return result


if __name__=='__main__':verify()
