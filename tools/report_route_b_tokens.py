"""CPU-only token audit of the frozen 500-prompt minimal pool (no truncation)."""
from collections import defaultdict
import statistics
from experiment_common import ROOT, AXES, read_rows, write_json
from train_distill import cache_name


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    import json
    out = ROOT / 'runs/route-b-7b-20261016'
    processor = AutoProcessor.from_pretrained(str(ROOT / '.model_cache/qwen7b'), local_files_only=True, use_fast=False)
    rows = read_rows(out / 'minimal-pool.jsonl')
    values = defaultdict(lambda: defaultdict(list))
    pending = []
    for row in rows:
        path = out / 'teacher-cache-bf16-eos' / cache_name(row['item_id'])
        if not path.exists():
            pending.append(row['item_id'])
            continue
        cached = torch.load(path, map_location='cpu', weights_only=True)
        assert cached['finish_reason'] == 'eos'
        with Image.open(ROOT / row['image_path']) as image:
            rgb = image.convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': row['question']}]}]
        gt = json.dumps(row['answer_key']) if row['axis'] == 'grounding' else str(row['answer_key'])
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': gt}]})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, add_vision_id=True)
        x = processor(text=[text], images=[rgb], return_tensors='pt', images_kwargs={
            'size': {'shortest_edge': 256*28*28, 'longest_edge': 1280*28*28},
            'min_pixels': 256*28*28, 'max_pixels': 1280*28*28})
        axis = values[row['axis']]
        counts = {'question_text': len(processor.tokenizer.encode(row['question'], add_special_tokens=False)),
                  'prompt_including_visual': cached['prompt_length'],
                  'visual': int(x['image_grid_thw'].prod(dim=1).sum().item() // 4),
                  'gt_response_including_chat_end': int(x['input_ids'].shape[1]) - cached['prompt_length'],
                  'teacher_response_including_chat_end': int(cached['targets'].numel()),
                  'teacher_raw_generated': int(cached['generation_ids'].numel()),
                  'gt_total': int(x['input_ids'].shape[1]),
                  'teacher_total': int(cached['input_ids'].shape[1])}
        for name, count in counts.items():
            axis[name].append(count)
    table = {a: {key: {'n': len(v), 'min': min(v), 'max': max(v), 'mean': statistics.mean(v)}
                 for key, v in values[a].items()} for a in AXES}
    result = {'status': 'complete' if not pending else 'partial', 'pending': len(pending),
              'units': 'tokens; padded batches are not used', 'axes': table,
              'speed_caveat': 'Two CE forwards depend on visual/prompt/response lengths; token lengths alone do not fix wall time'}
    write_json(out / 'minimal-token-distribution.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
