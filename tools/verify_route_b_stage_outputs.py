"""Independently verify real updates/checkpoints/adapters/eval coverage before report."""
import gc
import math
from experiment_common import ROOT, read_json, read_rows, sha, write_json, units, score_generation
from route_b_staged import OUT, SEEDS


def verify(phase='formal'):
    import torch
    from safetensors.torch import load_file
    expected = 1000 if phase == 'formal' else 100
    seeds = SEEDS if phase == 'formal' else [42]
    protocol = read_json(OUT/'protocol.json')
    manifest_path = ROOT/protocol['eval']['formal_manifest' if phase == 'formal' else 'minimal_manifest']
    manifest = read_rows(manifest_path)
    lookup = {r['item_id']:r for r in manifest}
    jobs = [('N',42)]+[(arm,seed) for seed in seeds for arm in ('A1','A2','U')]
    for arm,seed in jobs:
        folder = OUT/phase/f'{arm}-seed{seed}'
        if arm != 'N':
            marker = read_json(folder/'complete.json')
            losses = read_rows(folder/'loss.jsonl')
            assert marker['actual_updates'] == len(losses) == expected
            assert [r['step'] for r in losses] == list(range(1,expected+1))
            assert all(r['actual_optimizer_update'] and r['grad_norm']>0 and all(math.isfinite(r[k]) for k in ('loss','ce_gt','ce_teacher','grad_norm')) for r in losses)
            assert marker['loss_sha256'] == sha(folder/'loss.jsonl')
            assert marker['adapter_sha256'] == sha(folder/'adapter/adapter_model.safetensors')
            assert read_json(folder/'train.meta.json')['protocol_sha256'] == sha(OUT/'protocol.json')
            state = torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
            assert state['step'] == expected and state['optimizer']['state']
            exported = load_file(str(folder/'adapter/adapter_model.safetensors'))
            assert set(exported) == set(state['adapter'])
            assert all(torch.equal(exported[k],state['adapter'][k]) for k in exported)
            assert all(torch.isfinite(t).all() for t in exported.values())
            del state,exported
            gc.collect()
        scored = read_rows(folder/'eval.jsonl')
        marker = read_json(folder/'scores.json')
        assert marker['ledger_sha256'] == sha(folder/'eval.jsonl') and marker['rows'] == len(manifest)
        assert read_json(folder/'eval.meta.json')['manifest_sha256'] == sha(manifest_path)
        units(manifest,scored)
        for row in scored:
            actual = score_generation(row['raw_generation'],lookup[row['item_id']])
            assert all(row[k] == actual[k] for k in ('correct_bit','parse_fail_flag','parsed_answer','iou'))
    result = {'status':'passed','phase':phase,'trained_runs':len(jobs)-1,'updates_per_trained_run':expected,
              'evaluated_models':len(jobs),'prompts_per_model':len(manifest),
              'checks':['continuous_real_updates','finite_dual_ce_gradients','resume_optimizer_and_adapter_match',
                        'frozen_hashes','exact_eval_coverage','independent_rescore']}
    write_json(OUT/(phase+'-independent-verification.json'),result)
    return result


if __name__ == '__main__':
    print(verify())
