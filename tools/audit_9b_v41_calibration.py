"""Independent CPU audit of completed v4.1 pilot checkpoints; no generation."""
import json
from datetime import datetime, timezone

from experiment_common import ROOT, read_json, read_rows, sha, write_json
from qwen35_native_frontend import score_native_frontend


def audit():
    base = ROOT / 'runs/route-b-9b-122b-20261017/student-experiment'
    out = base / 'native-frontend-evaluation-v41'
    parent = base / 'native-frontend-evaluation-v4'
    plan = read_json(out / 'protocol.json')
    assert sha(parent / 'protocol.json') == plan['parent_failed_v4_protocol_sha256']
    assert sha(parent / 'stopped.json') == plan['parent_failed_v4_stop_sha256']
    assert sha(parent / 'calibration-wall-budget.json') == plan['parent_failed_v4_budget_sha256']
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / 'tools' / name) == digest, name
    assert sha(out / 'formal-manifest.jsonl') == plan['manifest_sha256']
    assert sha(out / 'calibration-selection.json') == plan['selection_sha256']
    lookup = {r['item_id']: r for r in read_rows(out / 'formal-manifest.jsonl')}
    selection = read_json(out / 'calibration-selection.json')
    reports = {}
    # calibration-proof is immutable even once the live ledger continues formally.
    for folder in sorted((out / 'calibration-proof').glob('*-seed*')):
        marker = read_json(folder / 'calibration-scores.json')
        ledger = folder / 'eval.jsonl'
        assert sha(ledger) == marker['ledger_sha256']
        rows = read_rows(ledger)
        n = folder.name == 'N-seed42'
        expected = selection['item_ids' if n else 'adapter_item_ids']
        assert [r['item_id'] for r in rows] == expected
        assert len(rows) == marker['rows'] == (100 if n else 20)
        assert not marker['optimizer_loaded'] and not marker['gradients_enabled']
        assert marker['peak_reserved_bytes'] <= plan['reserved_bytes_cap']
        assert marker['protocol_sha256'] == sha(out / 'protocol.json')
        for row in rows:
            ids = row['generated_token_ids']
            assert len(ids) == row['generation_tokens'] <= plan['max_new_tokens']
            assert ids[-1] in plan['termination_contract']['eos_token_ids']
            assert not any(i in plan['termination_contract']['eos_token_ids'] for i in ids[:-1])
            assert row['finish_reason'] == 'eos' and row['pilot_prediction']
            scored = score_native_frontend(row['raw_generation'], lookup[row['item_id']])
            assert all(row[k] == v for k, v in scored.items())
        failures = sum(r['parse_fail_flag'] for r in rows)
        accuracy = sum(r['correct_bit'] for r in rows)/len(rows)
        assert failures == marker['parse_failures'] <= (5 if n else 1)
        assert accuracy == marker['accuracy']
        assert marker['length_capped_outputs'] == 0
        if n:
            assert accuracy >= .8
        else:
            assert sha(base / 'formal' / folder.name / 'adapter/adapter_model.safetensors') == plan['adapters_sha256'][folder.name]
        reports[folder.name] = {'rows':len(rows), 'accuracy':accuracy,
                              'parse_failures':failures, 'ledger_sha256':sha(ledger)}
    result = {'status':'passed_all16' if len(reports) == 16 else 'passed_completed_subset_only',
              'models':reports, 'predictions':sum(r['rows'] for r in reports.values()),
              'protocol_sha256':sha(out / 'protocol.json'),
              'parent_failure_and_budget_unchanged':True,
              'first_eos_stopping_verified_on_every_audited_row':True,
              'new_predictions':0, 'timestamp_utc':datetime.now(timezone.utc).isoformat()}
    if len(reports) == 16:
        assert result['predictions'] == 400
        assert read_json(out / 'calibration-acceptance.json')['status'] == 'passed'
    write_json(out / 'independent-calibration-audit/result.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    audit()
