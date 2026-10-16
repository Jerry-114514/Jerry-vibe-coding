"""CPU acceptance of the independent full-configuration memory preflight."""
import math
from experiment_common import ROOT,AXES,read_json,read_rows,sha,write_json


def main():
    folder=ROOT/'runs/protocol-v2/calibration/p14-full-7b-v2'
    done=read_json(folder/'complete.json');protocol=read_json(folder/'protocol.json')
    rows=read_rows(folder/'manifest.jsonl');losses=read_rows(folder/'loss.jsonl');teacher=read_rows(folder/'teacher.jsonl')
    if done['status']!='complete' or done['actual_real_updates']!=100 or len(losses)!=100:
        raise ValueError('Incomplete real optimizer updates')
    if [r['step'] for r in losses]!=list(range(1,101)) or [r['item_id'] for r in losses]!=[r['item_id'] for r in rows]:
        raise ValueError('Training coverage/order differs')
    if {r['item_id'] for r in teacher}!={r['item_id'] for r in rows} or len(teacher)!=100:
        raise ValueError('Teacher coverage differs')
    if any(r['returned_model']!='qwen3.5-122b-a10b' or r['finish_reason']!='stop' for r in teacher):
        raise ValueError('Wrong or truncated teacher response')
    for axis in AXES:
        if sum(r['axis']==axis for r in rows)!=25:raise ValueError('Axis coverage differs')
    for row in losses:
        if not row['actual_optimizer_update'] or not all(math.isfinite(row[k]) for k in ['loss','ce_gt','ce_teacher','grad_norm']):
            raise ValueError('Nonfinite/skipped update')
        if not math.isclose(row['loss'],.5*row['ce_gt']+.5*row['ce_teacher'],rel_tol=1e-5,abs_tol=1e-5):
            raise ValueError('Dual-CE weights differ')
        if max(row['gt_sequence_tokens'],row['teacher_sequence_tokens'])>2048:raise ValueError('Input cap exceeded')
    for field,name in [('loss_ledger_sha256','loss.jsonl'),('teacher_ledger_sha256','teacher.jsonl'),('adapter_sha256','adapter-after-100.pt')]:
        if done[field]!=sha(folder/name):raise ValueError('Artifact checksum mismatch')
    stress=read_json(folder/'shape-stress.json')
    if stress['gt_sequence_tokens']!=2048 or stress['teacher_sequence_tokens']!=2048 or stress['response_tokens']!=128:
        raise ValueError('Upper-bound shape stress absent')
    parameters=read_json(folder/'trainable-parameters.json')
    if not any('.mlp.' in n for n in parameters['names']) or not any('.self_attn.' in n for n in parameters['names']):
        raise ValueError('Missing MLP or attention adapters')
    import torch
    adapter=torch.load(folder/'adapter-after-100.pt',map_location='cpu',weights_only=True)
    matrices=[v for k,v in adapter.items() if 'lora_B' in k]
    if not matrices or not all(torch.count_nonzero(v).item()>0 for v in matrices):raise ValueError('Unchanged LoRA B')
    peak=done['peak_reserved_bytes']
    expected='freeze_configuration' if peak<=20000000000 else 'offer_memory_reduction_options' if peak<=22000000000 else 'exceeds_requested_4090_budget'
    if done['memory_decision']!=expected:raise ValueError('Wrong threshold classification')
    if protocol['lora']['target_modules']!=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
        raise ValueError('Not the requested seven-module configuration')
    write_json(folder/'verification.json',{'status':'passed','checks':['100_real_updates','balanced_training_only_coverage','122B_teacher_identity','dual_CE_weights','seven_module_scope','nonzero_adapter_updates','2048_128_shape_stress','artifact_hashes','byte_threshold_decision'],'formal_training':False})
    print('Independent full P1-4 preflight verification passed')


if __name__=='__main__':main()
