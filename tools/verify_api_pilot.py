"""Independent CPU acceptance of API ledgers; makes no API inference calls."""
import argparse
import math
from pathlib import Path
from experiment_common import ROOT, read_json, read_rows, score_generation, sha, units, write_json
from coordinate_contracts import score_normalized_grounding


def verify(folder):
    folder = Path(folder).resolve()
    if not folder.is_relative_to((ROOT / 'runs/protocol-v2/calibration').resolve()):
        raise ValueError('Unexpected experiment directory')
    complete = read_json(folder / 'complete.json')
    protocol = read_json(folder / 'protocol.json')
    manifest = read_rows(folder / 'manifest.jsonl')
    ledger = read_rows(folder / 'ledger.jsonl')
    if complete['status'] != 'complete' or len(ledger) != protocol['prompts']:
        raise ValueError('Incomplete API ledger')
    if complete['ledger_sha256'] != sha(folder / 'ledger.jsonl'):
        raise ValueError('Ledger digest mismatch')
    if complete['manifest_sha256'] != sha(folder / 'manifest.jsonl') or protocol['manifest_sha256'] != complete['manifest_sha256']:
        raise ValueError('Manifest digest mismatch')
    scored_units = units(manifest, ledger)
    items = {row['item_id']: row for row in manifest}
    score = score_normalized_grounding if protocol.get('coordinate_contract') == 'normalized_1000' else score_generation
    total = 0
    for row in ledger:
        item = items[row['item_id']]
        rescored = score(row['raw_generation'], item)
        for key in ('correct_bit', 'parsed_answer', 'parse_fail_flag', 'iou'):
            if row[key] != rescored[key]:
                raise ValueError('Independent strict rescore differs')
        if row['image_file_sha256'] != sha(ROOT / item['image_path']):
            raise ValueError('Source image changed')
        if row['checkpoint'] != protocol['model'] or row['returned_model'] != protocol['model']:
            raise ValueError('Model identity mismatch')
        usage = row['usage']
        if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
            raise ValueError('Usage accounting mismatch')
        if usage['completion_tokens'] > protocol['max_output_tokens_per_request']:
            raise ValueError('Output token limit exceeded')
        total += usage['total_tokens']
        for token in (row.get('token_logprobs') or {}).get('content', []):
            if not math.isfinite(token['logprob']):
                raise ValueError('Non-finite token log probability')
            if len(token.get('top_logprobs', [])) > protocol['top_logprobs']:
                raise ValueError('Unexpected probability output')
    if total != complete['total_reported_tokens']:
        raise ValueError('Total usage differs')
    report = {'status': 'passed', 'prompt_rows': len(ledger), 'scoring_units': len(scored_units),
              'ledger_sha256': sha(folder / 'ledger.jsonl'),
              'checks': ['exact_unique_coverage', 'manifest_and_ledger_hashes', 'independent_strict_rescore',
                         'source_image_hashes', 'returned_model_identity', 'token_usage_accounting',
                         'output_token_limits', 'finite_returned_logprobs'],
              'formal_training_allowed': False, 'confirmation': False}
    write_json(folder / 'verification.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    args = parser.parse_args()
    print(verify(args.folder))
