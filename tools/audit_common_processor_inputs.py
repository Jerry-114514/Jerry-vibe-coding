"""Independent CPU audit of frozen dimensions, unchanged labels and adapters."""
from datetime import datetime,timezone
from PIL import Image

from experiment_common import ROOT,read_json,read_rows,sha,write_json
from route_b_staged import OUT as SEVEN,SEEDS
from route_b_9b_loop import OUT as NINE,MINIMAL_MANIFEST

DEST=NINE/'processor-pixel-evaluation-v3/input-audit-csv-v2'
EXTRA={'processor_width','processor_height','processor_grid_thw','processor_patch_size','frozen_prompt_tokens'}


def audit_manifest(path,original_path,only_grounding=False):
    original=read_rows(original_path)
    if only_grounding:original=[r for r in original if r['axis']=='grounding']
    actual=read_rows(path);lookup={r['item_id']:r for r in original}
    assert len(actual)==len(original)==len({r['item_id'] for r in actual})
    assert {r['item_id'] for r in actual}==set(lookup)
    patches=set();sizes=set();ground=0
    for row in actual:
        old=lookup[row['item_id']]
        if row['axis']!='grounding':assert row==old;continue
        ground+=1
        assert set(row)-set(old)<=EXTRA
        for key,value in old.items():
            if key!='question':assert row[key]==value,(row['item_id'],key)
        prefix=old['question'].rsplit('in original-image pixel coordinates.',1)[0]
        assert row['question'].startswith(prefix+'in processor-resized-image pixel coordinates.')
        width,height=row['processor_width'],row['processor_height'];grid=row['processor_grid_thw'];patch=row['processor_patch_size']
        assert grid[0]==1 and width==grid[2]*patch and height==grid[1]*patch
        assert grid[0]*grid[1]*grid[2]//4<=1280
        assert f'{width} pixels wide and {height} pixels high' in row['question']
        assert 'normalized coordinates' not in row['question']
        image=(ROOT/row['image_path']).resolve();assert image.is_relative_to((ROOT/'data').resolve())
        with Image.open(image) as source:assert source.size==(row['width'],row['height'])
        patches.add(patch);sizes.add((width,height))
    return {'rows':len(actual),'grounding_rows':ground,'source_manifest_sha256':sha(original_path),
            'measurement_manifest_sha256':sha(path),'patch_sizes':sorted(patches),'distinct_processed_sizes':len(sizes),
            'unchanged_labels_image_paths_groups_and_original_dimensions':True,
            'all_dimensions_match_frozen_grid_times_patch':True,'no_gt_based_unit_choice':True}


def run():
    seven=ROOT/'runs/7b-processor-pixel-audit-20261017';nine=NINE/'processor-pixel-evaluation-v3'
    p7=read_json(seven/'protocol.json');p9=read_json(nine/'protocol.json')
    required=p9['required7b_audit_protocol_sha256']
    lineage={'original_7b_protocol_sha256':required,'active_7b_protocol_sha256':sha(seven/'protocol.json')}
    if sha(seven/'protocol.json')!=required:
        # Accept only the externally registered syntax amendment, retaining the
        # original units, dimensions, manifest, endpoints and sampling identity.
        historical=seven/'schema-v1-history/protocol.json'
        parent=read_json(historical)
        assert sha(historical)==required
        amendment=p7['schema_amendment']
        assert amendment['parent_protocol_sha256']==required
        assert amendment['no_coordinate_value_unit_or_GT_selection_change'] is True
        assert amendment['all_models_same_grammar'] is True
        assert amendment['existing100_generations_reused_without_model_calls'] is True
        for key in ('manifest_sha256','processor_dimensions_sha256','coordinate_contract','models',
                    'rows_per_model','total_generations','N_first100_minimum_accuracy',
                    'N_first100_maximum_parse_failures','gpu_wall_hours_cap','contrasts','family_size'):
            assert p7[key]==parent[key],key
        assert (seven/'schema-registration/delivery-proof.json').exists()
        lineage['syntax_amendment_sha256']=sha(seven/'schema-amendment.json')
        lineage['original_proof_retained']=True
    for plan in (p7,p9):
        for name,digest in plan['source_sha256'].items():assert sha(ROOT/'tools'/name)==digest
    assert p7['source_sha256']['processor_pixel_contract.py']==p9['source_sha256']['processor_pixel_contract.py']
    assert sha(seven/'processor-dimensions.json')==p7['processor_dimensions_sha256']
    assert sha(nine/'processor-dimensions.json')==p9['processor_dimensions_sha256']
    assert p7['total_generations']==38400 and p7['family_size']==4 and p7['contrasts']==['A1-A2','A1-U','A1-N','U-N']
    assert p9['formal_generations']==270400 and p9['N_grounding_minimum_accuracy']==.8
    assert sha(ROOT/'tools/route_b_staged.py')==p7['original_engine_sha256']==read_json(SEVEN/'protocol.json')['source_sha256']
    manifests={'7b':audit_manifest(seven/'manifest.jsonl',ROOT/'data/manifest.jsonl',True),
               '9b_minimal':audit_manifest(nine/'minimal-manifest.jsonl',MINIMAL_MANIFEST),
               '9b_formal':audit_manifest(nine/'formal-manifest.jsonl',ROOT/'data/manifest.jsonl')}
    adapters=[]
    for base in (SEVEN,NINE):
        for seed in SEEDS:
            for arm in ('A1','A2','U'):
                folder=base/'formal'/f'{arm}-seed{seed}'
                if not (folder/'complete.json').exists():continue
                marker=read_json(folder/'complete.json');adapter=folder/'adapter/adapter_model.safetensors'
                assert marker['actual_updates']==1000 and sha(adapter)==marker['adapter_sha256']
                adapters.append({'run':str(folder.relative_to(ROOT)),'adapter_sha256':marker['adapter_sha256']})
    assert len([r for r in adapters if r['run'].startswith(str(SEVEN.relative_to(ROOT)))])==15
    assert not (SEVEN/'formal/N-seed42/adapter').exists()
    assert read_json(ROOT/'runs/protocol-v2/paired-gate-development/gate.json')['training_allowed'] is False
    from audit_route_b_completion import existing_code_unchanged
    unchanged=existing_code_unchanged()
    proof={'status':'passed','timestamp':datetime.now(timezone.utc).isoformat(),'audit_source_sha256':sha(ROOT/'tools/audit_common_processor_inputs.py'),
           '7b_protocol_sha256':sha(seven/'protocol.json'),'9b_protocol_sha256':sha(nine/'protocol.json'),
           '7b_explicit_parent_lineage':lineage,
           'manifests':manifests,'adapters':adapters,'original_python_files_unchanged':len(unchanged),
           'training_budget':read_json(NINE/'formal-training-wall-budget.json'),
           'no_gpu_or_new_predictions':True,'original_arp_gate_unpassed':True,
           'scope':'Frozen labels, dimensions, source and completed adapter identities only; does not claim pending GPU calibration passed'}
    write_json(DEST/'verification.json',proof)
    print({'status':'passed','manifests':{k:(v['rows'],v['grounding_rows']) for k,v in manifests.items()},
           'completed_adapters_verified':len(adapters),'original_python_files_unchanged':len(unchanged)},flush=True)
    return proof


if __name__=='__main__':run()
