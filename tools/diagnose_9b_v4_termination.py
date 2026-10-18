"""CPU-only post-stop diagnosis. Never changes predictions or admission."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from experiment_common import ROOT, read_json, read_rows, sha, write_json
from qwen35_native_frontend import score_native_frontend


def main():
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    import torch

    base = ROOT / 'runs/route-b-9b-122b-20261017/student-experiment'
    out = base / 'native-frontend-evaluation-v4'
    model_dir = ROOT / '.model_cache/qwen35_9b'
    plan = read_json(out / 'protocol.json')
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / 'tools' / name) == digest, name
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    generation_file = model_dir / 'generation_config.json'
    generation = (GenerationConfig.from_pretrained(model_dir, local_files_only=True)
                  if generation_file.exists() else GenerationConfig.from_model_config(config))
    native_eos = tokenizer.eos_token_id
    lookup = {r['item_id']: r for r in read_rows(out / 'formal-manifest.jsonl')}
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'v4_calibration_failed_no_formal_admission',
              'posthoc_diagnostic_only': True, 'new_predictions': 0,
              'original_ledgers_unchanged': True, 'budget_reset': False,
              'stopped': read_json(out / 'stopped.json'),
              'budget': read_json(out / 'calibration-wall-budget.json'),
              'generation_config_file_present': generation_file.exists(),
              'configured_eos_token_id': generation.eos_token_id,
              'tokenizer_eos_token_id': native_eos,
              'tokenizer_eos_token': tokenizer.eos_token,
              'models': []}
    for folder in sorted((out / 'formal').iterdir()):
        marker_path = folder / 'calibration-scores.json'
        if not marker_path.exists():
            continue
        marker = read_json(marker_path)
        ledger = folder / 'eval.jsonl'
        assert sha(ledger) == marker['ledger_sha256']
        rows = read_rows(ledger)
        assert len(rows) == len({r['item_id'] for r in rows}) == marker['rows']
        diagnostic_rows = []
        for row in rows:
            ids = row['generated_token_ids']
            position = ids.index(native_eos) if native_eos in ids else None
            prefix = ids[:position] if position is not None else ids
            raw = tokenizer.decode(prefix, skip_special_tokens=True).strip()
            score = score_native_frontend(raw, lookup[row['item_id']])
            diagnostic_rows.append({'item_id': row['item_id'],
                'native_eos_position': position, 'generated_tokens': len(ids),
                'tokens_after_native_eos': len(ids)-position-1 if position is not None else None,
                'original_finish_reason': row['finish_reason'],
                'original_parse_fail_flag': row['parse_fail_flag'],
                'diagnostic_prefix_raw': raw,
                'diagnostic_prefix_parse_fail_flag': score['parse_fail_flag'],
                'diagnostic_prefix_correct_bit': score['correct_bit']})
        if folder.name != 'N-seed42':
            adapter = base / 'formal' / folder.name / 'adapter/adapter_model.safetensors'
            assert sha(adapter) == plan['adapters_sha256'][folder.name]
        result['models'].append({'model': folder.name, 'registered': marker,
            'native_eos_present_rows': sum(r['native_eos_position'] is not None for r in diagnostic_rows),
            'diagnostic_prefix_parse_failures': sum(r['diagnostic_prefix_parse_fail_flag'] for r in diagnostic_rows),
            'diagnostic_prefix_correct': sum(r['diagnostic_prefix_correct_bit'] for r in diagnostic_rows),
            'rows': diagnostic_rows})
        assert sha(ledger) == marker['ledger_sha256']
    result['generated_models'] = len(result['models'])
    result['generated_predictions'] = sum(m['registered']['rows'] for m in result['models'])
    result['failure_archive'] = read_json(out / 'calibration-proof/delivery-proof.json')
    assert not torch.cuda.is_initialized()
    result['cuda_initialized'] = False
    write_json(out / 'termination-diagnostic/diagnostic.json', result)
    if '--json' in sys.argv:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps({k:v for k,v in result.items() if k != 'models'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
