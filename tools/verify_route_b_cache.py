"""CPU-only, streaming audit of the immutable v1 top-50 teacher cache."""
import json
from pathlib import Path
import time

from experiment_common import ROOT, read_json, read_rows, sha, write_json
from train_distill import cache_name

OUT = ROOT / 'runs/route-b-7b-20261016'
CACHE = ROOT / 'runs/formal-20261014/teacher-cache'


def main():
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows(ROOT / 'data/training/pool.jsonl')
    complete = read_json(CACHE / 'complete.json')
    meta = read_json(CACHE / 'cache.meta.json')
    assert len(rows) == complete['items'] == 3545
    assert sha(ROOT / 'data/training/pool.jsonl') == meta['manifest_sha256'] == complete['manifest_sha256']
    assert meta['top_k'] == 50 and meta['teacher'] == 'qwen7b'
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / '.model_cache/qwen7b'), local_files_only=True)
    assert tokenizer.get_vocab() == read_json(CACHE / 'tokenizer.json')['vocab']
    generation = read_json(ROOT / '.model_cache/qwen7b/generation_config.json')
    eos = generation['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    ledger = []
    images = {}
    started = time.monotonic()
    for i, row in enumerate(rows, 1):
        filename = cache_name(row['item_id'])
        path = CACHE / filename
        assert sha(path) == complete['files'][filename], filename
        x = torch.load(path, map_location='cpu', weights_only=True)
        assert x['item_id'] == row['item_id'] and x['response'].strip()
        targets = x['targets'].reshape(-1)
        n = targets.numel()
        assert n > 0 and x['top_indices'].shape == x['top_logits'].shape == (n, 50)
        assert x['log_normalizer'].shape == (n,)
        assert torch.isfinite(x['top_logits']).all() and torch.isfinite(x['log_normalizer']).all()
        assert int(x['top_indices'].min()) >= 0 and int(x['top_indices'].max()) < len(tokenizer)
        assert torch.equal(x['input_ids'][:, x['prompt_length']:].reshape(-1), targets)
        assert ((x['top_logits'] - x['log_normalizer'][:, None]).exp().sum(-1) <= 1.0001).all()
        image = row['image_path']
        if image not in images:
            images[image] = sha(ROOT / image)
        assert x['image_sha256'] == images[image]
        # v1 stored response text, not raw generation token IDs/finish_reason.
        # Retokenized chat ends with EOS even when the generated text was capped;
        # therefore EOS alone cannot certify non-truncation of legacy generations.
        text_tokens = len(tokenizer.encode(x['response'], add_special_tokens=False))
        ledger.append({'item_id': row['item_id'], 'axis': row['axis'], 'target_tokens': n,
                       'response_text_tokens': text_tokens, 'possible_length_cap': text_tokens >= 128,
                       'retokenized_final_eos': int(targets[-1]) in eos,
                       'sha256': complete['files'][filename]})
        if i % 100 == 0:
            write_json(OUT / 'cache-audit-progress.json', {'checked': i, 'total': len(rows)})
    from experiment_common import write_rows
    write_rows(OUT / 'cache-audit.jsonl', ledger)
    ambiguous = [r['item_id'] for r in ledger if r['possible_length_cap']]
    result = {'status': 'integrity_passed' if not ambiguous else 'integrity_passed_truncation_review_required',
              'checked': len(rows), 'finite_top50': True, 'tensor_targets_untruncated': True,
              'legacy_finish_reason_absent': True, 'generation_nontruncation_certified': False,
              'possible_length_cap_ids': ambiguous, 'manifest_sha256': meta['manifest_sha256'],
              'complete_sha256': sha(CACHE / 'complete.json'), 'cache_meta_sha256': sha(CACHE / 'cache.meta.json'),
              'elapsed_seconds': time.monotonic() - started}
    write_json(OUT / 'cache-audit-complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
